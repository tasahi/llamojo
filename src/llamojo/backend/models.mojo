""" Runtime-Dispatched Multi-Architecture Forward Passes for llamojo.

Consolidates:
- llama2.mojo/forward_worker (persistent spin-barrier worker execution)
- MojoLlama/universal_forward.mojo (GQA, MoE router, multi-layer graph)
- MojoLlama/zaya_gen_q8.mojo (residual scales & Cross-Layer Attention CCA)

Features:
- Pure Mojo 1.1.0 with zero external dependencies
- Persistent spin-barrier synchronization across worker threads
- Dense Llama/Mistral, MoE (Mixture of Experts), and CCA architectures
- Runtime-dispatched layer execution supporting all GGUF quantized formats
"""

from std.memory import Pointer, unsafe_memcpy, unsafe_memset_zero
from std.utils import StaticTuple
from max.algorithm import parallelize

from ..types import ModelConfig, GGMLType, ggml_block_size, ggml_type_size
from .threadpool import (
    LlamaPoolShared,
    LlamaPoolSharedPtr,
    llama_barrier,
    chunk_start,
    chunk_end,
)
from .kernels_norm_rope import (
    fused_rms_norm_residual,
    rms_norm,
    fused_silu_mul,
    apply_rope_norm,
    apply_rope_neox,
    apply_rope_yarn,
)
from .kernels_quant import (
    BytePtr,
    FloatPtr,
    dot_quant_row,
    dot_q4_k_q8_k_row,
    dot_q4_k_q8_k_2rows,
    dot_q4_k_q8_k_4rows,
    dot_q6_k_q8_k_row,
    dot_q6_k_q8_k_2rows,
    dot_q6_k_q8_k_4rows,
    quantize_row_q8_k,
    matmul_quant_fused,
)
from .kernels_attn import gqa_decode
from .moe import MoELayerWeights, MoEExpertWeights, forward_moe_ffn

# ─── Model Weight Descriptors ────────────────────────────────────────────────

struct LayerWeights(ImplicitlyCopyable, Copyable, Movable):
    """ Pointers to all weight tensors for a single transformer layer."""
    var attn_norm: FloatPtr
    var ffn_norm: FloatPtr
    var wq: BytePtr
    var wk: BytePtr
    var wv: BytePtr
    var wo: BytePtr
    var w_gate: BytePtr
    var w_up: BytePtr
    var w_down: BytePtr
    var w_router: FloatPtr # For MoE models
    var is_moe: Bool
    var n_experts: Int
    var n_experts_used: Int
    var exp_gates_ptr: Pointer[BytePtr, MutUntrackedOrigin]
    var exp_ups_ptr: Pointer[BytePtr, MutUntrackedOrigin]
    var exp_downs_ptr: Pointer[BytePtr, MutUntrackedOrigin]
    var dtype: UInt32
    var dtype_q: UInt32
    var dtype_k: UInt32
    var dtype_v: UInt32
    var dtype_o: UInt32
    var dtype_gate: UInt32
    var dtype_up: UInt32
    var dtype_down: UInt32

    # Gemma 4 specific layer weights & norms
    var attn_q_norm: FloatPtr
    var attn_k_norm: FloatPtr
    var post_attention_norm: FloatPtr
    var post_ffw_norm: FloatPtr
    var inp_gate: BytePtr
    var proj: BytePtr
    var post_norm: FloatPtr
    var layer_output_scale: Float32
    var dtype_inp_gate: UInt32
    var dtype_proj: UInt32

    def __init__(out self):
        var z: Int = 0
        self.attn_norm = FloatPtr(unsafe_from_address=z)
        self.ffn_norm = FloatPtr(unsafe_from_address=z)
        self.wq = BytePtr(unsafe_from_address=z)
        self.wk = BytePtr(unsafe_from_address=z)
        self.wv = BytePtr(unsafe_from_address=z)
        self.wo = BytePtr(unsafe_from_address=z)
        self.w_gate = BytePtr(unsafe_from_address=z)
        self.w_up = BytePtr(unsafe_from_address=z)
        self.w_down = BytePtr(unsafe_from_address=z)
        self.w_router = FloatPtr(unsafe_from_address=z)
        self.is_moe = False
        self.n_experts = 0
        self.n_experts_used = 0
        self.exp_gates_ptr = Pointer[BytePtr, MutUntrackedOrigin](unsafe_from_address=z)
        self.exp_ups_ptr = Pointer[BytePtr, MutUntrackedOrigin](unsafe_from_address=z)
        self.exp_downs_ptr = Pointer[BytePtr, MutUntrackedOrigin](unsafe_from_address=z)
        self.dtype = GGMLType.F32
        self.dtype_q = GGMLType.F32
        self.dtype_k = GGMLType.F32
        self.dtype_v = GGMLType.F32
        self.dtype_o = GGMLType.F32
        self.dtype_gate = GGMLType.F32
        self.dtype_up = GGMLType.F32
        self.dtype_down = GGMLType.F32

        self.attn_q_norm = FloatPtr(unsafe_from_address=z)
        self.attn_k_norm = FloatPtr(unsafe_from_address=z)
        self.post_attention_norm = FloatPtr(unsafe_from_address=z)
        self.post_ffw_norm = FloatPtr(unsafe_from_address=z)
        self.inp_gate = BytePtr(unsafe_from_address=z)
        self.proj = BytePtr(unsafe_from_address=z)
        self.post_norm = FloatPtr(unsafe_from_address=z)
        self.layer_output_scale = 1.0
        self.dtype_inp_gate = GGMLType.F32
        self.dtype_proj = GGMLType.F32

# ─── Multi-Architecture Transformer Forward Worker ────────────────────────────

def forward_layer_dense(
    layer: Int,
    pos: Int,
    lw: LayerWeights,
    cfg: ModelConfig,
    x: FloatPtr,
    xb: FloatPtr,
    xb2: FloatPtr,
    hb: FloatPtr,
    hb2: FloatPtr,
    q: FloatPtr,
    k_buf: FloatPtr,
    v_buf: FloatPtr,
    att_buf: FloatPtr,
    k_cache: FloatPtr,
    v_cache: FloatPtr,
    max_ctx: Int,
    wid: Int,
    sh: LlamaPoolSharedPtr,
):
    """Execute one transformer layer across worker threads using persistent spin barriers."""
    var workers = sh[].workers
    var dim = cfg.hidden_dim
    var hidden_dim = cfg.intermediate_dim
    var q_dim = cfg.n_heads * cfg.head_dim
    var kv_dim = cfg.n_kv_heads * cfg.head_dim
    var head_dim = cfg.head_dim
    var n_heads = cfg.n_heads
    var n_kv_heads = cfg.n_kv_heads

    var d0 = chunk_start(dim, wid, workers)
    var d1 = chunk_end(dim, wid, workers)
    var q0 = chunk_start(q_dim, wid, workers)
    var q1 = chunk_end(q_dim, wid, workers)
    var kv0 = chunk_start(kv_dim, wid, workers)
    var kv1 = chunk_end(kv_dim, wid, workers)
    var f0 = chunk_start(hidden_dim, wid, workers)
    var f1 = chunk_end(hidden_dim, wid, workers)
    var h0 = chunk_start(n_heads, wid, workers)
    var h1 = chunk_end(n_heads, wid, workers)

    var xbp = sh[].xbp.unsafe_offset(wid * sh[].xbp_dim)

    # 1. Attention RMSNorm
    if Int(lw.attn_norm) != 0:
        rms_norm(xbp, x, lw.attn_norm, dim, cfg.norm_eps)
    else:
        unsafe_memcpy(dest=xbp, src=x, count=dim)

    # Pre-quantize attention normalized input xbp to Q8_K once for all QKV projections
    if wid == 0 and Int(sh[].q8_act) != 0 and (dim % 256 == 0):
        quantize_row_q8_k(xbp, sh[].q8_act, dim)
    llama_barrier(sh)

    # 2. QKV Projections: fused or parallel row dot
    var q_ptr = q
    var k_ptr = k_buf
    var v_ptr = v_buf
    var q8_act_ptr = sh[].q8_act

    if Int(lw.wq) != 0:
        if q_dim == dim and kv_dim == dim and lw.dtype_q == lw.dtype_k and lw.dtype_q == lw.dtype_v:
            var c_qkv = StaticTuple[FloatPtr, 3](q_ptr, k_ptr, v_ptr)
            var w_qkv = StaticTuple[BytePtr, 3](lw.wq, lw.wk, lw.wv)
            matmul_quant_fused[3](lw.dtype_q, c_qkv, xbp, w_qkv, d0, d1, dim, q8_act_ptr)
        else:
            var c_q = StaticTuple[FloatPtr, 1](q_ptr)
            var w_q = StaticTuple[BytePtr, 1](lw.wq)
            matmul_quant_fused[1](lw.dtype_q, c_q, xbp, w_q, q0, q1, dim, q8_act_ptr)

            if lw.dtype_k == lw.dtype_v:
                var c_kv = StaticTuple[FloatPtr, 2](k_ptr, v_ptr)
                var w_kv = StaticTuple[BytePtr, 2](lw.wk, lw.wv)
                matmul_quant_fused[2](lw.dtype_k, c_kv, xbp, w_kv, kv0, kv1, dim, q8_act_ptr)
            else:
                var c_k = StaticTuple[FloatPtr, 1](k_ptr)
                var w_k = StaticTuple[BytePtr, 1](lw.wk)
                matmul_quant_fused[1](lw.dtype_k, c_k, xbp, w_k, kv0, kv1, dim, q8_act_ptr)

                var c_v = StaticTuple[FloatPtr, 1](v_ptr)
                var w_v = StaticTuple[BytePtr, 1](lw.wv)
                matmul_quant_fused[1](lw.dtype_v, c_v, xbp, w_v, kv0, kv1, dim, q8_act_ptr)
    else:
        # Pass-through for partial synthetic models
        if wid == 0:
            for i in range(q_dim):
                q_ptr[unsafe_offset=i] = xbp[unsafe_offset=i]
            for i in range(kv_dim):
                k_ptr[unsafe_offset=i] = xbp[unsafe_offset=i]
                v_ptr[unsafe_offset=i] = xbp[unsafe_offset=i]

    llama_barrier(sh)

    # 3. RoPE: apply in-place to worker's heads (worker 0 applies for all if 1 worker)
    if wid == 0:
        if cfg.rope_scaling_type == 2: # YaRN
            # In GGUF, rope.scaling.factor is the context extension factor (e.g. 16.0).
            # YaRN frequency scaling factor passed to apply_rope_yarn is 1.0 / factor (e.g. 1.0 / 16.0).
            var freq_scale = (Float32(1.0) / cfg.rope_freq_scale) if cfg.rope_freq_scale > 1.0 else cfg.rope_freq_scale
            apply_rope_yarn(
                q,
                k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                pos,
                cfg.rope_theta,
                freq_scale,
                cfg.rope_orig_ctx_len,
                cfg.yarn_beta_fast,
                cfg.yarn_beta_slow,
                Float32(1.0),
                cfg.yarn_attn_factor,
                cfg.rope_type != 0,
            )
        elif cfg.rope_type == 0:
            apply_rope_norm(q, k_buf, n_heads, n_kv_heads, head_dim, pos, cfg.rope_theta, cfg.rope_freq_scale)
        else:
            apply_rope_neox(q, k_buf, n_heads, n_kv_heads, head_dim, pos, cfg.rope_theta, cfg.rope_freq_scale)
    llama_barrier(sh)

    # 4. Write new K, V into cache at current slot `pos`
    if wid == 0:
        var layer_cache_k = k_cache.unsafe_offset(layer * max_ctx * kv_dim + pos * kv_dim)
        var layer_cache_v = v_cache.unsafe_offset(layer * max_ctx * kv_dim + pos * kv_dim)
        unsafe_memcpy(dest=layer_cache_k, src=k_buf, count=kv_dim)
        unsafe_memcpy(dest=layer_cache_v, src=v_buf, count=kv_dim)
    llama_barrier(sh)

    # 5. Multihead / Grouped-Query Attention for this worker's heads
    var layer_k_cache_base = k_cache.unsafe_offset(layer * max_ctx * kv_dim)
    var layer_v_cache_base = v_cache.unsafe_offset(layer * max_ctx * kv_dim)
    gqa_decode(
        q,
        layer_k_cache_base,
        layer_v_cache_base,
        att_buf,
        xb,
        pos,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        h0,
        h1,
        sliding_window=cfg.sliding_window,
    )
    llama_barrier(sh)

    # 6. Output Projection + Residual Addition
    if Int(lw.wo) != 0:
        if wid == 0 and Int(sh[].q8_act) != 0 and (q_dim % 256 == 0):
            quantize_row_q8_k(xb, sh[].q8_act, q_dim)
        llama_barrier(sh)

        var blk_sz_wo = Int(ggml_block_size(lw.dtype_o))
        var type_sz_wo = Int(ggml_type_size(lw.dtype_o))
        var bpr_wo = (q_dim // blk_sz_wo) * type_sz_wo

        if Int(sh[].q8_act) != 0 and lw.dtype_o == GGMLType.Q4_K and (q_dim % 256 == 0):
            var r = d0
            while r + 3 < d1:
                var w_row0 = lw.wo.unsafe_offset(r * bpr_wo)
                var w_row1 = lw.wo.unsafe_offset((r + 1) * bpr_wo)
                var w_row2 = lw.wo.unsafe_offset((r + 2) * bpr_wo)
                var w_row3 = lw.wo.unsafe_offset((r + 3) * bpr_wo)
                var res0: Float32 = 0.0
                var res1: Float32 = 0.0
                var res2: Float32 = 0.0
                var res3: Float32 = 0.0
                dot_q4_k_q8_k_4rows(w_row0, w_row1, w_row2, w_row3, sh[].q8_act, q_dim, res0, res1, res2, res3)
                x[unsafe_offset=r] += res0
                x[unsafe_offset=r + 1] += res1
                x[unsafe_offset=r + 2] += res2
                x[unsafe_offset=r + 3] += res3
                r += 4
            while r + 1 < d1:
                var w_row0 = lw.wo.unsafe_offset(r * bpr_wo)
                var w_row1 = lw.wo.unsafe_offset((r + 1) * bpr_wo)
                var res0: Float32 = 0.0
                var res1: Float32 = 0.0
                dot_q4_k_q8_k_2rows(w_row0, w_row1, sh[].q8_act, q_dim, res0, res1)
                x[unsafe_offset=r] += res0
                x[unsafe_offset=r + 1] += res1
                r += 2
            if r < d1:
                var w_row = lw.wo.unsafe_offset(r * bpr_wo)
                x[unsafe_offset=r] += dot_q4_k_q8_k_row(w_row, sh[].q8_act, q_dim)
        elif Int(sh[].q8_act) != 0 and lw.dtype_o == GGMLType.Q6_K and (q_dim % 256 == 0):
            var r = d0
            while r + 3 < d1:
                var w_row0 = lw.wo.unsafe_offset(r * bpr_wo)
                var w_row1 = lw.wo.unsafe_offset((r + 1) * bpr_wo)
                var w_row2 = lw.wo.unsafe_offset((r + 2) * bpr_wo)
                var w_row3 = lw.wo.unsafe_offset((r + 3) * bpr_wo)
                var res0: Float32 = 0.0
                var res1: Float32 = 0.0
                var res2: Float32 = 0.0
                var res3: Float32 = 0.0
                dot_q6_k_q8_k_4rows(w_row0, w_row1, w_row2, w_row3, sh[].q8_act, q_dim, res0, res1, res2, res3)
                x[unsafe_offset=r] += res0
                x[unsafe_offset=r + 1] += res1
                x[unsafe_offset=r + 2] += res2
                x[unsafe_offset=r + 3] += res3
                r += 4
            while r + 1 < d1:
                var w_row0 = lw.wo.unsafe_offset(r * bpr_wo)
                var w_row1 = lw.wo.unsafe_offset((r + 1) * bpr_wo)
                var res0: Float32 = 0.0
                var res1: Float32 = 0.0
                dot_q6_k_q8_k_2rows(w_row0, w_row1, sh[].q8_act, q_dim, res0, res1)
                x[unsafe_offset=r] += res0
                x[unsafe_offset=r + 1] += res1
                r += 2
            if r < d1:
                var w_row = lw.wo.unsafe_offset(r * bpr_wo)
                x[unsafe_offset=r] += dot_q6_k_q8_k_row(w_row, sh[].q8_act, q_dim)
        else:
            for r in range(d0, d1):
                var w_row = lw.wo.unsafe_offset(r * bpr_wo)
                var res_wo = dot_quant_row(lw.dtype_o, w_row, xb, q_dim)
                x[unsafe_offset=r] += res_wo
    else:
        for r in range(d0, d1):
            x[unsafe_offset=r] += xb[unsafe_offset=r]
    llama_barrier(sh)

    # 7. FFN RMSNorm (redundant into private xbp buffer)
    if Int(lw.ffn_norm) != 0:
        rms_norm(xbp, x, lw.ffn_norm, dim, cfg.norm_eps)
    else:
        unsafe_memcpy(dest=xbp, src=x, count=dim)

    # Pre-quantize FFN normalized input xbp to Q8_K once for Gate/Up projections
    if wid == 0 and Int(sh[].q8_act) != 0 and (dim % 256 == 0):
        quantize_row_q8_k(xbp, sh[].q8_act, dim)
    llama_barrier(sh)

    # 8. FFN SwiGLU / MoE execution
    if lw.is_moe and lw.n_experts > 0 and Int(lw.w_router) != 0:
        var moe = MoELayerWeights(lw.n_experts, lw.n_experts_used)
        moe.w_router = lw.w_router
        for e in range(lw.n_experts):
            var wg = lw.exp_gates_ptr[unsafe_offset=e] if Int(lw.exp_gates_ptr) != 0 else lw.w_gate
            var wu = lw.exp_ups_ptr[unsafe_offset=e] if Int(lw.exp_ups_ptr) != 0 else lw.w_up
            var wd = lw.exp_downs_ptr[unsafe_offset=e] if Int(lw.exp_downs_ptr) != 0 else lw.w_down
            moe.experts.append(MoEExpertWeights(wg, wu, wd, lw.dtype_gate))
        forward_moe_ffn(x, xbp, hb, hb2, moe, dim, hidden_dim, wid, sh)
    elif Int(lw.w_gate) != 0 and Int(lw.w_up) != 0:
        var q8_act_ptr = sh[].q8_act
        if lw.dtype_gate == lw.dtype_up:
            var c_gate_up = StaticTuple[FloatPtr, 2](hb, hb2)
            var w_gate_up = StaticTuple[BytePtr, 2](lw.w_gate, lw.w_up)
            matmul_quant_fused[2](lw.dtype_gate, c_gate_up, xbp, w_gate_up, f0, f1, dim, q8_act_ptr)
        else:
            var c_g = StaticTuple[FloatPtr, 1](hb)
            var w_g = StaticTuple[BytePtr, 1](lw.w_gate)
            matmul_quant_fused[1](lw.dtype_gate, c_g, xbp, w_g, f0, f1, dim, q8_act_ptr)

            var c_u = StaticTuple[FloatPtr, 1](hb2)
            var w_u = StaticTuple[BytePtr, 1](lw.w_up)
            matmul_quant_fused[1](lw.dtype_up, c_u, xbp, w_u, f0, f1, dim, q8_act_ptr)

        # Apply SiLU to worker's slice of hidden dimension: hb = silu(hb) * hb2
        for i in range(f0, f1):
            fused_silu_mul(hb.unsafe_offset(i), hb2.unsafe_offset(i), hb.unsafe_offset(i), 1)
        llama_barrier(sh)

        # 9. Down Projection + Residual Addition
        if Int(lw.w_down) != 0:
            if wid == 0 and Int(sh[].q8_act) != 0 and (hidden_dim % 256 == 0):
                quantize_row_q8_k(hb, sh[].q8_act, hidden_dim)
            llama_barrier(sh)

            var blk_sz_down = Int(ggml_block_size(lw.dtype_down))
            var type_sz_down = Int(ggml_type_size(lw.dtype_down))
            var bpr_down = (hidden_dim // blk_sz_down) * type_sz_down

            if Int(sh[].q8_act) != 0 and lw.dtype_down == GGMLType.Q4_K and (hidden_dim % 256 == 0):
                var r = d0
                while r + 3 < d1:
                    var w_row0 = lw.w_down.unsafe_offset(r * bpr_down)
                    var w_row1 = lw.w_down.unsafe_offset((r + 1) * bpr_down)
                    var w_row2 = lw.w_down.unsafe_offset((r + 2) * bpr_down)
                    var w_row3 = lw.w_down.unsafe_offset((r + 3) * bpr_down)
                    var res0: Float32 = 0.0
                    var res1: Float32 = 0.0
                    var res2: Float32 = 0.0
                    var res3: Float32 = 0.0
                    dot_q4_k_q8_k_4rows(w_row0, w_row1, w_row2, w_row3, sh[].q8_act, hidden_dim, res0, res1, res2, res3)
                    x[unsafe_offset=r] += res0
                    x[unsafe_offset=r + 1] += res1
                    x[unsafe_offset=r + 2] += res2
                    x[unsafe_offset=r + 3] += res3
                    r += 4
                while r + 1 < d1:
                    var w_row0 = lw.w_down.unsafe_offset(r * bpr_down)
                    var w_row1 = lw.w_down.unsafe_offset((r + 1) * bpr_down)
                    var res0: Float32 = 0.0
                    var res1: Float32 = 0.0
                    dot_q4_k_q8_k_2rows(w_row0, w_row1, sh[].q8_act, hidden_dim, res0, res1)
                    x[unsafe_offset=r] += res0
                    x[unsafe_offset=r + 1] += res1
                    r += 2
                if r < d1:
                    var w_row = lw.w_down.unsafe_offset(r * bpr_down)
                    x[unsafe_offset=r] += dot_q4_k_q8_k_row(w_row, sh[].q8_act, hidden_dim)
            elif Int(sh[].q8_act) != 0 and lw.dtype_down == GGMLType.Q6_K and (hidden_dim % 256 == 0):
                var r = d0
                while r + 3 < d1:
                    var w_row0 = lw.w_down.unsafe_offset(r * bpr_down)
                    var w_row1 = lw.w_down.unsafe_offset((r + 1) * bpr_down)
                    var w_row2 = lw.w_down.unsafe_offset((r + 2) * bpr_down)
                    var w_row3 = lw.w_down.unsafe_offset((r + 3) * bpr_down)
                    var res0: Float32 = 0.0
                    var res1: Float32 = 0.0
                    var res2: Float32 = 0.0
                    var res3: Float32 = 0.0
                    dot_q6_k_q8_k_4rows(w_row0, w_row1, w_row2, w_row3, sh[].q8_act, hidden_dim, res0, res1, res2, res3)
                    x[unsafe_offset=r] += res0
                    x[unsafe_offset=r + 1] += res1
                    x[unsafe_offset=r + 2] += res2
                    x[unsafe_offset=r + 3] += res3
                    r += 4
                while r + 1 < d1:
                    var w_row0 = lw.w_down.unsafe_offset(r * bpr_down)
                    var w_row1 = lw.w_down.unsafe_offset((r + 1) * bpr_down)
                    var res0: Float32 = 0.0
                    var res1: Float32 = 0.0
                    dot_q6_k_q8_k_2rows(w_row0, w_row1, sh[].q8_act, hidden_dim, res0, res1)
                    x[unsafe_offset=r] += res0
                    x[unsafe_offset=r + 1] += res1
                    r += 2
                if r < d1:
                    var w_row = lw.w_down.unsafe_offset(r * bpr_down)
                    x[unsafe_offset=r] += dot_q6_k_q8_k_row(w_row, sh[].q8_act, hidden_dim)
            else:
                for r in range(d0, d1):
                    var w_row = lw.w_down.unsafe_offset(r * bpr_down)
                    var res_down = dot_quant_row(lw.dtype_down, w_row, hb, hidden_dim)
                    x[unsafe_offset=r] += res_down
        llama_barrier(sh)
