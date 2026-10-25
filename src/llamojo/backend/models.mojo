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
    apply_rope_neox,
)
from .kernels_quant import (
    BytePtr,
    FloatPtr,
    dot_quant_row,
    matmul_quant_fused,
)
from .kernels_attn import gqa_decode

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
    var dtype: UInt32

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
        self.dtype = GGMLType.F32

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
    var kv_dim = cfg.n_kv_heads * cfg.head_dim
    var head_dim = cfg.head_dim
    var n_heads = cfg.n_heads
    var n_kv_heads = cfg.n_kv_heads

    var d0 = chunk_start(dim, wid, workers)
    var d1 = chunk_end(dim, wid, workers)
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

    # 2. QKV Projections: fused or parallel row dot
    var q_ptr = q
    var k_ptr = k_buf
    var v_ptr = v_buf

    if Int(lw.wq) != 0:
        if kv_dim == dim:
            var c_qkv = StaticTuple[FloatPtr, 3](q_ptr, k_ptr, v_ptr)
            var w_qkv = StaticTuple[BytePtr, 3](lw.wq, lw.wk, lw.wv)
            matmul_quant_fused[3](lw.dtype, c_qkv, xbp, w_qkv, d0, d1, dim)
        else:
            var c_q = StaticTuple[FloatPtr, 1](q_ptr)
            var w_q = StaticTuple[BytePtr, 1](lw.wq)
            matmul_quant_fused[1](lw.dtype, c_q, xbp, w_q, d0, d1, dim)

            var c_kv = StaticTuple[FloatPtr, 2](k_ptr, v_ptr)
            var w_kv = StaticTuple[BytePtr, 2](lw.wk, lw.wv)
            matmul_quant_fused[2](lw.dtype, c_kv, xbp, w_kv, kv0, kv1, dim)
    else:
        # Pass-through for partial synthetic models
        if wid == 0:
            for i in range(dim):
                q_ptr[unsafe_offset=i] = xbp[unsafe_offset=i]
            for i in range(kv_dim):
                k_ptr[unsafe_offset=i] = xbp[unsafe_offset=i]
                v_ptr[unsafe_offset=i] = xbp[unsafe_offset=i]

    llama_barrier(sh)

    # 3. RoPE: apply in-place to worker's heads (worker 0 applies for all if 1 worker)
    if wid == 0:
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
    )
    llama_barrier(sh)

    # 6. Output Projection + Residual Addition
    if Int(lw.wo) != 0:
        var blk_sz_wo = Int(ggml_block_size(lw.dtype))
        var type_sz_wo = Int(ggml_type_size(lw.dtype))
        var bpr_wo = (dim // blk_sz_wo) * type_sz_wo
        for r in range(d0, d1):
            var w_row = lw.wo.unsafe_offset(r * bpr_wo)
            var res_wo = dot_quant_row(lw.dtype, w_row, xb, dim)
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

    # 8. FFN SwiGLU: compute gate (hb) and up (hb2) projections
    if Int(lw.w_gate) != 0 and Int(lw.w_up) != 0:
        var c_gate_up = StaticTuple[FloatPtr, 2](hb, hb2)
        var w_gate_up = StaticTuple[BytePtr, 2](lw.w_gate, lw.w_up)
        matmul_quant_fused[2](lw.dtype, c_gate_up, xbp, w_gate_up, f0, f1, dim)

        # Apply SiLU to worker's slice of hidden dimension: hb = silu(hb) * hb2
        for i in range(f0, f1):
            fused_silu_mul(hb.unsafe_offset(i), hb2.unsafe_offset(i), hb.unsafe_offset(i), 1)
        llama_barrier(sh)

        # 9. Down Projection + Residual Addition
        if Int(lw.w_down) != 0:
            var blk_sz_down = Int(ggml_block_size(lw.dtype))
            var type_sz_down = Int(ggml_type_size(lw.dtype))
            var bpr_down = (hidden_dim // blk_sz_down) * type_sz_down
            for r in range(d0, d1):
                var w_row = lw.w_down.unsafe_offset(r * bpr_down)
                var res_down = dot_quant_row(lw.dtype, w_row, hb, hidden_dim)
                x[unsafe_offset=r] += res_down
        llama_barrier(sh)
