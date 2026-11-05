""" Whisper Encoder, Static Cross-KV Precomputation, and Autoregressive Decoder Execution (`src/llamojo/whisper/model.mojo`).\n
Implements Phase 4 of the Whisper ASR Engine in `llamojo`:
1. `WhisperState`: Pre-allocated workspace buffers (`embd_enc`, `kv_self`, `kv_cross`, `logits`)
   and scratch arenas for zero-allocation inference.
2. `whisper_encode_layer`: Executes a single bidirectional Whisper Encoder block (`attn_ln` ->
   biased Q/V & unbiased K projections -> full `n_ctx x n_ctx` multi-head self-attention ->
   output projection + residual -> `mlp_ln` -> GELU MLP + residual).
3. `whisper_build_cross_kv`: Projects the invariant encoded audio features `embd_enc` (`[n_audio_ctx, n_audio_state]`)
   through each decoder layer's `cross_attn.key` and `cross_attn.value` once per audio window,
   storing them in `state.kv_cross` (`LlamaKVCache`).
4. `whisper_encode`: End-to-end audio encoding pipeline (Log-Mel slice -> `conv1` -> `conv2` ->
   `transpose_add_pos_embd` -> `N` encoder blocks -> `ln_post` -> `whisper_build_cross_kv`).
5. `whisper_decode`: Executes single-token or multi-token prompt decoding (`d_te[token] + d_pe[pos]` ->
   `N` decoder blocks with Causal Self-Attention over `kv_self` + Cross-Attention over `kv_cross` +
   GELU MLP -> `decoder.ln` -> tied `d_te` logits projection).
"""

from std.collections import List
from std.math import sqrt, exp
from std.memory import Pointer, unsafe_memcpy, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores
from std.utils import StaticTuple
from max.algorithm import parallelize

from ..types import GGMLType, ggml_block_size, ggml_type_size, null_ptr
from ..kv_cache import LlamaKVCache
from ..backend.isa import SIMD_WIDTH, F32xW
from ..backend.kernels_norm_rope import layer_norm, gelu
from ..backend.kernels_conv import conv1d_k3_gelu, transpose_add_pos_embd
from ..backend.kernels_quant import (
    dot_quant_row,
    dequantize_row,
    matmul_quant_fused_biased,
    gemv_quant,
    gemv_quant_biased,
    gemv_gelu_biased,
)
from ..backend.kernels_attn import dot_vectors, axpy_vector, gqa_decode_head
from .audio import WhisperMel
from .loader import WhisperModel, WhisperEncoderLayer, WhisperDecoderLayer

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime Int32Ptr = Pointer[Int32, MutUntrackedOrigin]


# ─── Whisper Runtime Execution State ─────────────────────────────────────────

struct WhisperState(Movable):
    """ Mutable runtime state for Whisper encoding and decoding (`struct whisper_state`)."""
    var mel: WhisperMel
    var kv_self: LlamaKVCache
    var kv_cross: LlamaKVCache

    # Audio Encoder output: [n_audio_ctx, n_audio_state]
    var embd_enc: FloatPtr
    var n_audio_ctx: Int
    var n_audio_state: Int
    var n_text_ctx: Int
    var n_text_state: Int
    var n_vocab: Int

    # Convolutional scratch buffers
    var mel_slice: FloatPtr   # [n_mels, 2 * n_audio_ctx]
    var conv1_out: FloatPtr   # [n_audio_state, 2 * n_audio_ctx]
    var conv2_out: FloatPtr   # [n_audio_state, n_audio_ctx]

    # Decoder scratch buffers (single step / token loop)
    var dec_x: FloatPtr       # [n_text_state]
    var dec_norm: FloatPtr    # [n_text_state]
    var dec_q: FloatPtr       # [n_text_state]
    var dec_k: FloatPtr       # [n_text_state]
    var dec_v: FloatPtr       # [n_text_state]
    var dec_attn_out: FloatPtr # [n_text_state]
    var dec_proj_out: FloatPtr # [n_text_state]
    var dec_mlp_hid: FloatPtr  # [4 * n_text_state]
    var dec_att_scores: FloatPtr # [n_text_head * max(n_audio_ctx, n_text_ctx)]

    # Latest token logits [n_vocab]
    var logits: FloatPtr
    var lang_id: Int

    def __init__(out self, ref model: WhisperModel, exp_n_audio_ctx: Int = 0):
        var hp = model.hparams
        var a_ctx = exp_n_audio_ctx if (exp_n_audio_ctx > 0 and exp_n_audio_ctx <= hp.n_audio_ctx) else hp.n_audio_ctx
        var a_state = hp.n_audio_state
        var t_ctx = hp.n_text_ctx
        var t_state = hp.n_text_state
        var t_head = hp.n_text_head
        var t_head_dim = t_state // t_head

        self.mel = WhisperMel()
        self.kv_self = LlamaKVCache(t_ctx, hp.n_text_layer, t_head, t_head_dim)
        self.kv_cross = LlamaKVCache(a_ctx, hp.n_text_layer, t_head, t_head_dim)

        self.n_audio_ctx = a_ctx
        self.n_audio_state = a_state
        self.n_text_ctx = t_ctx
        self.n_text_state = t_state
        self.n_vocab = hp.n_vocab

        self.embd_enc = unsafe_alloc[Float32](a_ctx * a_state)
        self.mel_slice = unsafe_alloc[Float32](hp.n_mels * 2 * a_ctx)
        self.conv1_out = unsafe_alloc[Float32](a_state * 2 * a_ctx)
        self.conv2_out = unsafe_alloc[Float32](a_state * a_ctx)

        self.dec_x = unsafe_alloc[Float32](t_state)
        self.dec_norm = unsafe_alloc[Float32](t_state)
        self.dec_q = unsafe_alloc[Float32](t_state)
        self.dec_k = unsafe_alloc[Float32](t_state)
        self.dec_v = unsafe_alloc[Float32](t_state)
        self.dec_attn_out = unsafe_alloc[Float32](t_state)
        self.dec_proj_out = unsafe_alloc[Float32](t_state)
        self.dec_mlp_hid = unsafe_alloc[Float32](4 * t_state)

        var max_len = max(a_ctx, t_ctx)
        self.dec_att_scores = unsafe_alloc[Float32](t_head * max_len)
        self.logits = unsafe_alloc[Float32](hp.n_vocab)
        unsafe_memset_zero(self.logits, hp.n_vocab)
        self.lang_id = 0

    def reset_kv_self(mut self):
        """ Clear the autoregressive decoder self-attention KV cache for a new segment."""
        self.kv_self.clear()

    def free(mut self):
        self.mel.free()
        self.kv_self.deinit_storage()
        self.kv_cross.deinit_storage()
        if Int(self.embd_enc) != 0:
            self.embd_enc.unsafe_free()
            self.mel_slice.unsafe_free()
            self.conv1_out.unsafe_free()
            self.conv2_out.unsafe_free()
            self.dec_x.unsafe_free()
            self.dec_norm.unsafe_free()
            self.dec_q.unsafe_free()
            self.dec_k.unsafe_free()
            self.dec_v.unsafe_free()
            self.dec_attn_out.unsafe_free()
            self.dec_proj_out.unsafe_free()
            self.dec_mlp_hid.unsafe_free()
            self.dec_att_scores.unsafe_free()
            self.logits.unsafe_free()
            self.embd_enc = null_ptr[Float32]()


# ─── Helper: Embedding Row Lookup (`ggml_get_rows`) ──────────────────────────

@always_inline
def get_embedding_row(
        dst: FloatPtr,
        w_table: BytePtr,
        dtype: UInt32,
        row_idx: Int,
        dim: Int,
    ):
    """ Dequantize or copy a single row `row_idx` of length `dim` from `w_table` into `dst`."""
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var row_bytes = (dim // blk_sz) * type_sz
    var row_ptr = w_table.unsafe_offset(row_idx * row_bytes)
    dequantize_row(dtype, row_ptr, dst, dim)


# ─── Bidirectional Audio Encoder Layer (`whisper_encode_layer`) ──────────────

def whisper_encode_layer(
        x_seq: FloatPtr,
        layer: WhisperEncoderLayer,
        n_ctx: Int,
        n_state: Int,
        n_head: Int,
        eps: Float32,
        q_all: FloatPtr,
        k_all: FloatPtr,
        v_all: FloatPtr,
        attn_out_all: FloatPtr,
        workers: Int = 0,
    ):
    """ Execute one bidirectional Whisper Audio Encoder block in-place on `x_seq` (`[n_ctx, n_state]`).
    Matches `whisper_build_graph_encoder` layer loop (`whisper.cpp:2146-2285`).
    """
    var head_dim = n_state // n_head
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var n_workers = workers if workers > 0 else num_performance_cores()
    if n_workers < 1:
        n_workers = 1

    # 1. LayerNorm + QKV projections across all n_ctx frames (parallelized over frames)
    var frame_chunk = (n_ctx + n_workers - 1) // n_workers

    def qkv_worker(wid: Int) {imm}:
        var t_start = wid * frame_chunk
        var t_end = min((wid + 1) * frame_chunk, n_ctx)
        var norm_buf = unsafe_alloc[Float32](n_state)

        for t in range(t_start, t_end):
            var x_t = x_seq.unsafe_offset(t * n_state)
            layer_norm(norm_buf, x_t, layer.attn_ln_0_w, layer.attn_ln_0_b, n_state, eps)

            var q_t = q_all.unsafe_offset(t * n_state)
            var k_t = k_all.unsafe_offset(t * n_state)
            var v_t = v_all.unsafe_offset(t * n_state)

            if layer.dtype_q == layer.dtype_k and layer.dtype_q == layer.dtype_v:
                var c_tup = StaticTuple[FloatPtr, 3](q_t, k_t, v_t)
                var w_tup = StaticTuple[BytePtr, 3](layer.attn_q_w, layer.attn_k_w, layer.attn_v_w)
                var b_tup = StaticTuple[FloatPtr, 3](layer.attn_q_b, null_ptr[Float32](), layer.attn_v_b)
                matmul_quant_fused_biased[3](layer.dtype_q, c_tup, norm_buf, w_tup, b_tup, 0, n_state, n_state)
            else:
                gemv_quant_biased(q_t, layer.attn_q_w, layer.attn_q_b, layer.dtype_q, norm_buf, n_state, n_state, 1)
                gemv_quant(k_t, layer.attn_k_w, layer.dtype_k, norm_buf, n_state, n_state, 1)
                gemv_quant_biased(v_t, layer.attn_v_w, layer.attn_v_b, layer.dtype_v, norm_buf, n_state, n_state, 1)

        norm_buf.unsafe_free()

    if n_workers == 1 or n_ctx < 16:
        qkv_worker(0)
    else:
        parallelize(qkv_worker, n_workers, n_workers)

    # 2. Full Bidirectional Multi-Head Attention [n_ctx x n_ctx] (parallelized over query frames)
    def attn_worker(wid: Int) {imm}:
        var t_start = wid * frame_chunk
        var t_end = min((wid + 1) * frame_chunk, n_ctx)
        var scores = unsafe_alloc[Float32](n_ctx)

        for tq in range(t_start, t_end):
            var q_frame = q_all.unsafe_offset(tq * n_state)
            var out_frame = attn_out_all.unsafe_offset(tq * n_state)

            for h in range(n_head):
                var h_off = h * head_dim
                var q_h = q_frame.unsafe_offset(h_off)
                var out_h = out_frame.unsafe_offset(h_off)

                var max_s: Float32 = -1e30
                for tk in range(n_ctx):
                    var k_h = k_all.unsafe_offset(tk * n_state + h_off)
                    var s = dot_vectors(q_h, k_h, head_dim) * scale
                    scores[unsafe_offset=tk] = s
                    if s > max_s:
                        max_s = s

                var sum_exp: Float32 = 0.0
                for tk in range(n_ctx):
                    var e = exp(scores[unsafe_offset=tk] - max_s)
                    scores[unsafe_offset=tk] = e
                    sum_exp += e

                var inv_sum = Float32(1.0) / (sum_exp + Float32(1e-9))
                unsafe_memset_zero(out_h, head_dim)
                for tk in range(n_ctx):
                    var w = scores[unsafe_offset=tk] * inv_sum
                    var v_h = v_all.unsafe_offset(tk * n_state + h_off)
                    axpy_vector(out_h, v_h, w, head_dim)

        scores.unsafe_free()

    if n_workers == 1 or n_ctx < 16:
        attn_worker(0)
    else:
        parallelize(attn_worker, n_workers, n_workers)

    # 3. Output Projection + Residual 1 + MLP LayerNorm + GELU MLP + Residual 2 (parallelized over frames)
    def post_attn_mlp_worker(wid: Int) {imm}:
        var t_start = wid * frame_chunk
        var t_end = min((wid + 1) * frame_chunk, n_ctx)
        var proj_buf = unsafe_alloc[Float32](n_state)
        var norm_buf = unsafe_alloc[Float32](n_state)
        var mlp_hid = unsafe_alloc[Float32](4 * n_state)

        for t in range(t_start, t_end):
            var x_t = x_seq.unsafe_offset(t * n_state)
            var attn_t = attn_out_all.unsafe_offset(t * n_state)

            # Attention output projection + residual
            gemv_quant_biased(proj_buf, layer.attn_ln_1_w, layer.attn_ln_1_b, layer.dtype_out, attn_t, n_state, n_state, 1)
            for i in range(n_state):
                x_t[unsafe_offset=i] += proj_buf[unsafe_offset=i]

            # MLP LayerNorm
            layer_norm(norm_buf, x_t, layer.mlp_ln_w, layer.mlp_ln_b, n_state, eps)

            # MLP 0 (d -> 4d + bias + GELU)
            gemv_gelu_biased(mlp_hid, layer.mlp_0_w, layer.mlp_0_b, layer.dtype_mlp_0, norm_buf, 4 * n_state, n_state, 1)

            # MLP 2 (4d -> d + bias) + residual
            gemv_quant_biased(proj_buf, layer.mlp_1_w, layer.mlp_1_b, layer.dtype_mlp_1, mlp_hid, n_state, 4 * n_state, 1)
            for i in range(n_state):
                x_t[unsafe_offset=i] += proj_buf[unsafe_offset=i]

        proj_buf.unsafe_free()
        norm_buf.unsafe_free()
        mlp_hid.unsafe_free()

    if n_workers == 1 or n_ctx < 16:
        post_attn_mlp_worker(0)
    else:
        parallelize(post_attn_mlp_worker, n_workers, n_workers)


# ─── Pre-Compute Static Cross-Attention KV Cache (`whisper_build_cross_kv`) ──

def whisper_build_cross_kv(
        ref model: WhisperModel,
        mut state: WhisperState,
        workers: Int = 0,
    ):
    """ Precompute `Kcross` and `Vcross` for all `n_text_layer` decoder layers from `state.embd_enc`
    and store them in `state.kv_cross` (`[n_text_layer, n_audio_ctx, n_text_state]`).
    Matches `whisper_build_graph_cross` (`whisper.cpp:2320-2396`).
    """
    var n_ctx = state.n_audio_ctx
    var n_state = state.n_text_state
    var n_layer = model.hparams.n_text_layer
    var n_workers = workers if workers > 0 else num_performance_cores()
    if n_workers < 1:
        n_workers = 1

    var frame_chunk = (n_ctx + n_workers - 1) // n_workers

    for il in range(n_layer):
        var dl = model.layers_decoder[il]

        def cross_worker(wid: Int) {imm}:
            var t_start = wid * frame_chunk
            var t_end = min((wid + 1) * frame_chunk, n_ctx)
            for t in range(t_start, t_end):
                var enc_frame = state.embd_enc.unsafe_offset(t * n_state)
                var k_dst = state.kv_cross.get_k_ptr(il, t)
                var v_dst = state.kv_cross.get_v_ptr(il, t)

                # Kcross has no bias; Vcross has bias
                gemv_quant(k_dst, dl.cross_attn_k_w, dl.dtype_cross_k, enc_frame, n_state, n_state, 1)
                gemv_quant_biased(
                    v_dst,
                    dl.cross_attn_v_w,
                    dl.cross_attn_v_b,
                    dl.dtype_cross_v,
                    enc_frame,
                    n_state,
                    n_state,
                    1,
                )

        if n_workers == 1 or n_ctx < 16:
            cross_worker(0)
        else:
            parallelize(cross_worker, n_workers, n_workers)


# ─── Full Audio Encoder Pipeline (`whisper_encode`) ──────────────────────────

def whisper_encode(
        ref model: WhisperModel,
        mut state: WhisperState,
        mel_offset: Int = 0,
        workers: Int = 0,
    ):
    """ Run the full Whisper Audio Encoder on `state.mel` starting at frame `mel_offset`:
    1. Slice `[n_mels, 2 * n_audio_ctx]` from `state.mel`
    2. `conv1` (`K=3, S=1, P=1` + GELU) -> `conv2` (`K=3, S=2, P=1` + GELU)
    3. `transpose_add_pos_embd` (`[n_state, n_ctx]` -> `[n_ctx, n_state]` + `e_pe`)
    4. Execute all `n_audio_layer` bidirectional encoder blocks
    5. Final `encoder.ln_post` LayerNorm
    6. Pre-compute static Cross-Attention KV cache (`whisper_build_cross_kv`)
    """
    var hp = model.hparams
    var n_ctx = state.n_audio_ctx
    var n_state = state.n_audio_state
    var n_head = hp.n_audio_head
    var n_mels = hp.n_mels
    var mel_cols = 2 * n_ctx

    # 1. Extract [n_mels, 2 * n_ctx] window starting at mel_offset (matches whisper.cpp:2440-2451)
    unsafe_memset_zero(state.mel_slice, n_mels * mel_cols)
    if state.mel.n_len > 0 and Int(state.mel.data) != 0:
        var i0 = min(mel_offset, state.mel.n_len)
        var i1 = min(mel_offset + mel_cols, state.mel.n_len)
        var copy_len = i1 - i0
        if copy_len > 0:
            for j in range(n_mels):
                unsafe_memcpy(
                    dest=state.mel_slice.unsafe_offset(j * mel_cols),
                    src=state.mel.data.unsafe_offset(j * state.mel.n_len + i0),
                    count=copy_len,
                )

    # 2. Convolutional Stem: conv1 (stride=1) -> conv2 (stride=2)
    conv1d_k3_gelu(
        state.conv1_out,
        state.mel_slice,
        model.e_conv_1_w,
        model.dtype_conv_1,
        model.e_conv_1_b,
        n_mels,
        n_state,
        mel_cols,
        1,
        workers,
    )
    conv1d_k3_gelu(
        state.conv2_out,
        state.conv1_out,
        model.e_conv_2_w,
        model.dtype_conv_2,
        model.e_conv_2_b,
        n_state,
        n_state,
        mel_cols,
        2,
        workers,
    )

    # 3. Transpose [n_state, n_ctx] -> [n_ctx, n_state] and add encoder.positional_embedding
    transpose_add_pos_embd(state.embd_enc, state.conv2_out, model.e_pe, n_state, n_ctx)

    # 4. Run N_audio_layer bidirectional Transformer blocks
    var q_all = unsafe_alloc[Float32](n_ctx * n_state)
    var k_all = unsafe_alloc[Float32](n_ctx * n_state)
    var v_all = unsafe_alloc[Float32](n_ctx * n_state)
    var attn_out_all = unsafe_alloc[Float32](n_ctx * n_state)

    for il in range(hp.n_audio_layer):
        whisper_encode_layer(
            state.embd_enc,
            model.layers_encoder[il],
            n_ctx,
            n_state,
            n_head,
            hp.eps,
            q_all,
            k_all,
            v_all,
            attn_out_all,
            workers,
        )

    q_all.unsafe_free()
    k_all.unsafe_free()
    v_all.unsafe_free()
    attn_out_all.unsafe_free()

    # 5. Final Encoder Post-LayerNorm (`encoder.ln_post`)
    var norm_tmp = unsafe_alloc[Float32](n_state)
    for t in range(n_ctx):
        var frame_ptr = state.embd_enc.unsafe_offset(t * n_state)
        layer_norm(norm_tmp, frame_ptr, model.e_ln_w, model.e_ln_b, n_state, hp.eps)
        unsafe_memcpy(dest=frame_ptr, src=norm_tmp, count=n_state)
    norm_tmp.unsafe_free()

    # 6. Pre-compute Cross-Attention KV cache for all decoder layers
    whisper_build_cross_kv(model, state, workers)


# ─── Autoregressive Text Decoder (`whisper_decode_step` & `whisper_decode`) ──

def whisper_decode_step(
        ref model: WhisperModel,
        mut state: WhisperState,
        token: Int32,
        pos: Int,
        compute_logits: Bool = True,
        workers: Int = 0,
    ):
    """ Execute a single decoder step at sequence position `pos` for `token`.
    Updates `state.kv_self` at slot `pos`, attends over `state.kv_cross` (`0 .. n_audio_ctx - 1`),
    and if `compute_logits` is True, writes output logits `[n_vocab]` into `state.logits`.
    Matches `whisper_build_graph_decoder` (`whisper.cpp:2527-2905`).
    """
    var hp = model.hparams
    var n_state = state.n_text_state
    var n_head = hp.n_text_head
    var head_dim = n_state // n_head
    var n_audio_ctx = state.n_audio_ctx

    var x = state.dec_x
    var norm_buf = state.dec_norm
    var q = state.dec_q
    var k = state.dec_k
    var v = state.dec_v
    var attn_out = state.dec_attn_out
    var proj_out = state.dec_proj_out
    var mlp_hid = state.dec_mlp_hid
    var att_scores = state.dec_att_scores

    # 1. Token Embedding + Positional Embedding: cur = d_te[token] + d_pe[pos]
    get_embedding_row(x, model.d_te, model.dtype_d_te, Int(token), n_state)
    var pe_row = model.d_pe.unsafe_offset(pos * n_state)
    for i in range(n_state):
        x[unsafe_offset=i] += pe_row[unsafe_offset=i]

    # 2. Execute all N_text_layer Decoder Blocks
    for il in range(hp.n_text_layer):
        var dl = model.layers_decoder[il]

        # ── 2a. Causal Self-Attention ──
        layer_norm(norm_buf, x, dl.attn_ln_0_w, dl.attn_ln_0_b, n_state, hp.eps)

        if dl.dtype_q == dl.dtype_k and dl.dtype_q == dl.dtype_v:
            var c_tup = StaticTuple[FloatPtr, 3](q, k, v)
            var w_tup = StaticTuple[BytePtr, 3](dl.attn_q_w, dl.attn_k_w, dl.attn_v_w)
            var b_tup = StaticTuple[FloatPtr, 3](dl.attn_q_b, null_ptr[Float32](), dl.attn_v_b)
            matmul_quant_fused_biased[3](dl.dtype_q, c_tup, norm_buf, w_tup, b_tup, 0, n_state, n_state)
        else:
            gemv_quant_biased(q, dl.attn_q_w, dl.attn_q_b, dl.dtype_q, norm_buf, n_state, n_state, workers)
            gemv_quant(k, dl.attn_k_w, dl.dtype_k, norm_buf, n_state, n_state, workers)
            gemv_quant_biased(v, dl.attn_v_w, dl.attn_v_b, dl.dtype_v, norm_buf, n_state, n_state, workers)

        # Write current token K, V into kv_self at slot `pos`
        state.kv_self.write_kv(il, pos, k, v, Int32(pos), 0)

        var k_self_layer = state.kv_self.get_k_ptr(il, 0)
        var v_self_layer = state.kv_self.get_v_ptr(il, 0)

        for h in range(n_head):
            gqa_decode_head(
                q,
                k_self_layer,
                v_self_layer,
                att_scores,
                attn_out.unsafe_offset(h * head_dim),
                h,
                pos,
                n_head,
                n_head,
                head_dim,
                state.n_text_ctx,
            )

        # Self-attention output projection + residual
        gemv_quant_biased(proj_out, dl.attn_ln_1_w, dl.attn_ln_1_b, dl.dtype_out, attn_out, n_state, n_state, workers)
        for i in range(n_state):
            x[unsafe_offset=i] += proj_out[unsafe_offset=i]

        # ── 2b. Encoder-Decoder Cross-Attention ──
        layer_norm(norm_buf, x, dl.cross_attn_ln_0_w, dl.cross_attn_ln_0_b, n_state, hp.eps)
        gemv_quant_biased(
            q,
            dl.cross_attn_q_w,
            dl.cross_attn_q_b,
            dl.dtype_cross_q,
            norm_buf,
            n_state,
            n_state,
            workers,
        )

        # Attend over static kv_cross [0 .. n_audio_ctx - 1] (so effective pos = n_audio_ctx - 1)
        var k_cross_layer = state.kv_cross.get_k_ptr(il, 0)
        var v_cross_layer = state.kv_cross.get_v_ptr(il, 0)

        for h in range(n_head):
            gqa_decode_head(
                q,
                k_cross_layer,
                v_cross_layer,
                att_scores,
                attn_out.unsafe_offset(h * head_dim),
                h,
                n_audio_ctx - 1,
                n_head,
                n_head,
                head_dim,
                n_audio_ctx,
            )

        # Cross-attention output projection + residual
        gemv_quant_biased(
            proj_out,
            dl.cross_attn_ln_1_w,
            dl.cross_attn_ln_1_b,
            dl.dtype_cross_out,
            attn_out,
            n_state,
            n_state,
            workers,
        )
        for i in range(n_state):
            x[unsafe_offset=i] += proj_out[unsafe_offset=i]

        # ── 2c. Feed-Forward Network (MLP) ──
        layer_norm(norm_buf, x, dl.mlp_ln_w, dl.mlp_ln_b, n_state, hp.eps)
        gemv_gelu_biased(mlp_hid, dl.mlp_0_w, dl.mlp_0_b, dl.dtype_mlp_0, norm_buf, 4 * n_state, n_state, workers)
        gemv_quant_biased(proj_out, dl.mlp_1_w, dl.mlp_1_b, dl.dtype_mlp_1, mlp_hid, n_state, 4 * n_state, workers)
        for i in range(n_state):
            x[unsafe_offset=i] += proj_out[unsafe_offset=i]

    # 3. Final Decoder LayerNorm (`decoder.ln`) & Logits Projection (`d_te @ norm_buf`)
    if compute_logits:
        layer_norm(norm_buf, x, model.d_ln_w, model.d_ln_b, n_state, hp.eps)
        gemv_quant(state.logits, model.d_te, model.dtype_d_te, norm_buf, state.n_vocab, n_state, workers)


def whisper_decode(
        ref model: WhisperModel,
        mut state: WhisperState,
        tokens: List[Int32],
        n_past: Int = 0,
        workers: Int = 0,
    ):
    """ Decode a sequence of tokens `[tokens[0], ..., tokens[n-1]]` starting at `n_past`,
    computing output logits for the final token into `state.logits`.
    """
    var n_tokens = len(tokens)
    for i in range(n_tokens):
        var is_last = (i == n_tokens - 1)
        whisper_decode_step(model, state, tokens[i], n_past + i, is_last, workers)
