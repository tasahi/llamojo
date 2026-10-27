""" Dedicated forward pass graph for Gemma 4 architecture.

Implements:
1. Input token embedding lookup + scaling (sqrt(dim) = sqrt(1536)).
2. Dual-stream Per-Layer Embeddings (PLE) extraction:
   - Token identity lookup from per_layer_token_embd scaled by sqrt(ple_dim) = 16.0
   - Context projection: inputs_embeds @ per_layer_model_proj / sqrt(dim), normalized by per_layer_proj_norm
   - Combined PLE: (proj + tok_embd) / sqrt(2.0)
3. 35-Layer Hybrid Transformer execution:
   - SWA layers (0..3, 5..8, etc.): head_dim=256, rope_base=10,000, sliding_window=512
   - Global layers (4, 9, 14, 19, 24, 29, 34): head_dim=512, rope_base=1,000,000, full context
   - Shared KV layers (15..34): reuse physical KV from layer 13 (SWA) or 14 (Global)
   - Attention pre-norm, Q/K per-head RMSNorm, unweighted V RMSNorm, RoPE, GQA, output projection, post-attention norm
   - FFN pre-norm, GELU fused gated FFN, down projection, post-FFN norm
   - Per-layer PLE gating: gelu(x @ inp_gate) * ple_slice @ proj -> post_norm -> residual
   - Layer output scalar
4. Final RMSNorm + tied LM-head dot product + logit softcapping (+/- 30.0).
"""

from std.memory import Pointer, unsafe_memcpy, unsafe_memset_zero
from std.math import sqrt
from std.utils import StaticTuple

from ..types import (
    ModelConfig,
    GGMLType,
    ggml_type_size,
    ggml_block_size,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
from .isa import SIMD_WIDTH, F32xW
from .kernels_norm_rope import (
    rms_norm,
    rms_norm_per_head,
    rms_norm_no_weight,
    apply_rope_neox,
    gelu,
    fused_gelu_mul,
    apply_logit_softcapping,
)
from .kernels_attn import gqa_decode
from .kernels_quant import (
    dot_quant_row,
    dequantize_row,
)
from .models import LayerWeights

@always_inline
def is_gemma4_swa_layer(layer: Int) -> Bool:
    """ Gemma 4 repeating pattern: 4 SWA layers followed by 1 Global layer (layers 4, 9, 14, 19, 24, 29, 34 are global)."""
    return (layer % 5) != 4

def forward_gemma4_single_token(
        token: Int32,
        pos: Int,
        cfg: ModelConfig,
        token_embd: BytePtr,
        token_embd_type: UInt32,
        per_layer_token_embd: BytePtr,
        per_layer_token_embd_type: UInt32,
        per_layer_model_proj: BytePtr,
        per_layer_model_proj_type: UInt32,
        per_layer_proj_norm: FloatPtr,
        layers: Pointer[LayerWeights, MutUntrackedOrigin],
        k_cache: FloatPtr,
        v_cache: FloatPtr,
        max_ctx: Int,
        x: FloatPtr,
        xb: FloatPtr,
        xb2: FloatPtr,
        hb: FloatPtr,
        hb2: FloatPtr,
        q: FloatPtr,
        k_buf: FloatPtr,
        v_buf: FloatPtr,
        att_buf: FloatPtr,
        ple_buf: FloatPtr,
        ple_tok_buf: FloatPtr,
        ple_proj_buf: FloatPtr,
        gate_buf: FloatPtr,
        proj_buf: FloatPtr,
    ):
    """ Execute pure-Mojo forward pass for a single token in Gemma 4."""
    var dim = cfg.hidden_dim
    var ple_dim = cfg.ple_dim if cfg.ple_dim > 0 else 256
    var n_layers = cfg.n_layers
    var ple_total = n_layers * ple_dim
    var eps = cfg.norm_eps

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Input Token Embedding & Scaling
    # ─────────────────────────────────────────────────────────────────────────
    var blk_sz_emb = Int(ggml_block_size(token_embd_type))
    var type_sz_emb = Int(ggml_type_size(token_embd_type))
    var bpr_emb = (dim // blk_sz_emb) * type_sz_emb
    var row_ptr = token_embd.unsafe_offset(Int(token) * bpr_emb)
    dequantize_row(token_embd_type, row_ptr, x, dim)

    var embd_scale = sqrt(Float32(dim))
    for i in range(dim):
        x[unsafe_offset=i] *= embd_scale

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Dual-Stream Per-Layer Embeddings (PLE)
    # ─────────────────────────────────────────────────────────────────────────
    if Int(per_layer_token_embd) != 0 and Int(per_layer_model_proj) != 0:
        # 2a. Token identity lookup: per_layer_token_embd[token] -> ple_tok_buf
        var blk_sz_ple = Int(ggml_block_size(per_layer_token_embd_type))
        var type_sz_ple = Int(ggml_type_size(per_layer_token_embd_type))
        var bpr_ple = (ple_total // blk_sz_ple) * type_sz_ple
        var ple_row_ptr = per_layer_token_embd.unsafe_offset(Int(token) * bpr_ple)
        dequantize_row(per_layer_token_embd_type, ple_row_ptr, ple_tok_buf, ple_total)

        var tok_embd_scale = sqrt(Float32(ple_dim)) # sqrt(256) = 16.0
        for i in range(ple_total):
            ple_tok_buf[unsafe_offset=i] *= tok_embd_scale

        # 2b. Context projection: x @ per_layer_model_proj (dim -> ple_total)
        var blk_sz_proj = Int(ggml_block_size(per_layer_model_proj_type))
        var type_sz_proj = Int(ggml_type_size(per_layer_model_proj_type))
        var bpr_proj = (dim // blk_sz_proj) * type_sz_proj
        for r in range(ple_total):
            var w_row = per_layer_model_proj.unsafe_offset(r * bpr_proj)
            ple_proj_buf[unsafe_offset=r] = dot_quant_row(per_layer_model_proj_type, w_row, x, dim)

        var proj_scale = Float32(1.0) / sqrt(Float32(dim))
        for i in range(ple_total):
            ple_proj_buf[unsafe_offset=i] *= proj_scale

        # Normalize each layer's slice with per_layer_proj_norm
        for l in range(n_layers):
            var slice_ptr = ple_proj_buf.unsafe_offset(l * ple_dim)
            rms_norm(slice_ptr, slice_ptr, per_layer_proj_norm, ple_dim, eps)

        # 2c. Stream fusion: (proj + tok_embd) / sqrt(2.0)
        var fusion_scale = Float32(1.0) / sqrt(Float32(2.0))
        for i in range(ple_total):
            ple_buf[unsafe_offset=i] = (ple_proj_buf[unsafe_offset=i] + ple_tok_buf[unsafe_offset=i]) * fusion_scale

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Sequential 35-Layer Transformer Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    for l in range(n_layers):
        var lw = layers[unsafe_offset=l]
        var is_swa = is_gemma4_swa_layer(l)
        var head_dim = 256 if is_swa else 512
        var n_heads = cfg.n_heads
        var n_kv_heads = cfg.n_kv_heads
        var q_dim = n_heads * head_dim
        var kv_dim = n_kv_heads * head_dim
        var rope_theta = Float32(10000.0) if is_swa else Float32(1000000.0)

        # --- A. Pre-Attention RMSNorm ---
        rms_norm(xb, x, lw.attn_norm, dim, eps)

        # --- B. Query Projection ---
        var blk_sz_q = Int(ggml_block_size(lw.dtype_q))
        var type_sz_q = Int(ggml_type_size(lw.dtype_q))
        var bpr_q = (dim // blk_sz_q) * type_sz_q
        for r in range(q_dim):
            var w_row = lw.wq.unsafe_offset(r * bpr_q)
            q[unsafe_offset=r] = dot_quant_row(lw.dtype_q, w_row, xb, dim)

        # Q per-head RMSNorm
        if Int(lw.attn_q_norm) != 0:
            rms_norm_per_head(q, lw.attn_q_norm, n_heads, head_dim, eps)

        # Q RoPE (for Global layers, max_rot_dim=128 cuts off the upper 384 dims as in rope_freqs)
        var max_rot = 256 if is_swa else 128
        apply_rope_neox(q, k_buf, n_heads, 0, head_dim, pos, rope_theta, Float32(1.0), max_rot)

        # --- C. Key / Value Projection & Cache ---
        var phys_layer = l
        if l < 15:
            # Active KV layer: compute K & V
            var blk_sz_k = Int(ggml_block_size(lw.dtype_k))
            var type_sz_k = Int(ggml_type_size(lw.dtype_k))
            var bpr_k = (dim // blk_sz_k) * type_sz_k
            for r in range(kv_dim):
                var w_row = lw.wk.unsafe_offset(r * bpr_k)
                k_buf[unsafe_offset=r] = dot_quant_row(lw.dtype_k, w_row, xb, dim)

            var blk_sz_v = Int(ggml_block_size(lw.dtype_v))
            var type_sz_v = Int(ggml_type_size(lw.dtype_v))
            var bpr_v = (dim // blk_sz_v) * type_sz_v
            for r in range(kv_dim):
                var w_row = lw.wv.unsafe_offset(r * bpr_v)
                v_buf[unsafe_offset=r] = dot_quant_row(lw.dtype_v, w_row, xb, dim)

            # K per-head RMSNorm
            if Int(lw.attn_k_norm) != 0:
                rms_norm_per_head(k_buf, lw.attn_k_norm, n_kv_heads, head_dim, eps)

            # V unweighted RMSNorm
            rms_norm_no_weight(v_buf, v_buf, kv_dim, eps)

            # K RoPE (for Global layers, max_rot_dim=128 cuts off the upper 384 dims as in rope_freqs)
            apply_rope_neox(q, k_buf, 0, n_kv_heads, head_dim, pos, rope_theta, Float32(1.0), max_rot)

            # Store to KV cache at slot pos
            var cache_k = k_cache.unsafe_offset(l * max_ctx * 512 + pos * 512)
            var cache_v = v_cache.unsafe_offset(l * max_ctx * 512 + pos * 512)
            unsafe_memcpy(dest=cache_k, src=k_buf, count=kv_dim)
            unsafe_memcpy(dest=cache_v, src=v_buf, count=kv_dim)
        else:
            # Reused KV layer: reuse layer 13 for SWA, layer 14 for Global
            phys_layer = 13 if is_swa else 14

        # --- D. Attention Computation ---
        var layer_k_cache_base = k_cache.unsafe_offset(phys_layer * max_ctx * 512)
        var layer_v_cache_base = v_cache.unsafe_offset(phys_layer * max_ctx * 512)
        var sw_len = 512 if is_swa else 0
        gqa_decode(
            q,
            layer_k_cache_base,
            layer_v_cache_base,
            att_buf,
            xb2,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            0,
            n_heads,
            sliding_window=sw_len,
            attn_scale=Float32(1.0), # Gemma 4 uses 1.0 attention scale
            stride_kv=512,           # Gemma 4 KV cache slots are padded to 512 floats
        )

        # --- E. Attention Output Projection + Post-Attention Norm + Residual ---
        var blk_sz_wo = Int(ggml_block_size(lw.dtype_o))
        var type_sz_wo = Int(ggml_type_size(lw.dtype_o))
        var bpr_wo = (q_dim // blk_sz_wo) * type_sz_wo
        for r in range(dim):
            var w_row = lw.wo.unsafe_offset(r * bpr_wo)
            xb[unsafe_offset=r] = dot_quant_row(lw.dtype_o, w_row, xb2, q_dim)

        if Int(lw.post_attention_norm) != 0:
            rms_norm(xb, xb, lw.post_attention_norm, dim, eps)

        for i in range(dim):
            x[unsafe_offset=i] += xb[unsafe_offset=i]

        # --- F. Feed-Forward Network (GELU Gated) ---
        # Gemma 4 has per-layer FFN intermediate dims (layers 0..14: 6144, layers 15..34: 12288)
        var hidden_dim = 6144 if l < 15 else 12288
        rms_norm(xb, x, lw.ffn_norm, dim, eps)

        var blk_sz_g = Int(ggml_block_size(lw.dtype_gate))
        var type_sz_g = Int(ggml_type_size(lw.dtype_gate))
        var bpr_g = (dim // blk_sz_g) * type_sz_g
        for r in range(hidden_dim):
            var w_row = lw.w_gate.unsafe_offset(r * bpr_g)
            hb[unsafe_offset=r] = dot_quant_row(lw.dtype_gate, w_row, xb, dim)

        var blk_sz_u = Int(ggml_block_size(lw.dtype_up))
        var type_sz_u = Int(ggml_type_size(lw.dtype_up))
        var bpr_u = (dim // blk_sz_u) * type_sz_u
        for r in range(hidden_dim):
            var w_row = lw.w_up.unsafe_offset(r * bpr_u)
            hb2[unsafe_offset=r] = dot_quant_row(lw.dtype_up, w_row, xb, dim)

        for i in range(hidden_dim):
            fused_gelu_mul(hb.unsafe_offset(i), hb2.unsafe_offset(i), hb.unsafe_offset(i), 1)

        var blk_sz_d = Int(ggml_block_size(lw.dtype_down))
        var type_sz_d = Int(ggml_type_size(lw.dtype_down))
        var bpr_d = (hidden_dim // blk_sz_d) * type_sz_d
        for r in range(dim):
            var w_row = lw.w_down.unsafe_offset(r * bpr_d)
            xb[unsafe_offset=r] = dot_quant_row(lw.dtype_down, w_row, hb, hidden_dim)

        if Int(lw.post_ffw_norm) != 0:
            rms_norm(xb, xb, lw.post_ffw_norm, dim, eps)

        for i in range(dim):
            x[unsafe_offset=i] += xb[unsafe_offset=i]

        # --- G. Per-Layer Embedding Residual Injection ---
        if Int(lw.inp_gate) != 0 and Int(lw.proj) != 0:
            # 1. Gate: gelu(x @ inp_gate)
            var blk_sz_ig = Int(ggml_block_size(lw.dtype_inp_gate))
            var type_sz_ig = Int(ggml_type_size(lw.dtype_inp_gate))
            var bpr_ig = (dim // blk_sz_ig) * type_sz_ig
            for r in range(ple_dim):
                var w_row = lw.inp_gate.unsafe_offset(r * bpr_ig)
                gate_buf[unsafe_offset=r] = dot_quant_row(lw.dtype_inp_gate, w_row, x, dim)

            for i in range(ple_dim):
                gate_buf[unsafe_offset=i] = gelu(gate_buf[unsafe_offset=i])

            # 2. Modulate: gate * ple_slice
            var ple_slice = ple_buf.unsafe_offset(l * ple_dim)
            for i in range(ple_dim):
                gate_buf[unsafe_offset=i] *= ple_slice[unsafe_offset=i]

            # 3. Projection: modulated @ proj (ple_dim -> dim)
            var blk_sz_pr = Int(ggml_block_size(lw.dtype_proj))
            var type_sz_pr = Int(ggml_type_size(lw.dtype_proj))
            var bpr_pr = (ple_dim // blk_sz_pr) * type_sz_pr
            for r in range(dim):
                var w_row = lw.proj.unsafe_offset(r * bpr_pr)
                proj_buf[unsafe_offset=r] = dot_quant_row(lw.dtype_proj, w_row, gate_buf, ple_dim)

            # 4. Post-norm + residual
            if Int(lw.post_norm) != 0:
                rms_norm(proj_buf, proj_buf, lw.post_norm, dim, eps)

            for i in range(dim):
                x[unsafe_offset=i] += proj_buf[unsafe_offset=i]

        # --- H. Layer Output Scalar ---
        if lw.layer_output_scale != 1.0:
            for i in range(dim):
                x[unsafe_offset=i] *= lw.layer_output_scale
