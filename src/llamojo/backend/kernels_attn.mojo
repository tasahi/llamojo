""" Attention Kernels for llamojo: Grouped-Query Attention (GQA) & Multi-Head Attention (MHA).

Consolidates and ports:
- llama2.mojo/attention_head
- MojoLlama/src/mojollama/kernels/cengine_batch_instr.c:gqa
- MojoLlama/src/mojollama/kernels/attention.mojo

Provides:
- Online numerically stable softmax: softmax(Q @ K.T / sqrt(head_dim))
- GQA head grouping (n_heads // n_kv_heads sharing key/value heads)
- Decode attention (single query token attending over KV history [0..pos])
- Prefill batch attention (causal masked tiled attention)
"""

from std.memory import Pointer, unsafe_memset_zero
from std.math import sqrt, exp
from .isa import SIMD_WIDTH, F32xW

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

@always_inline
def dot_vectors(a: FloatPtr, b: FloatPtr, n: Int) -> Float32:
    """Compute dot product of two Float32 vectors of length `n`."""
    var acc = F32xW(0.0)
    var i = 0
    while i + SIMD_WIDTH <= n:
        acc += a.unsafe_load[width=SIMD_WIDTH](i) * b.unsafe_load[width=SIMD_WIDTH](i)
        i += SIMD_WIDTH
    var total = acc.reduce_add()
    while i < n:
        total += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return total

@always_inline
def axpy_vector(y: FloatPtr, x: FloatPtr, a: Float32, n: Int):
    """Compute y[i] += a * x[i] for vectors of length `n`."""
    var av = F32xW(a)
    var i = 0
    while i + SIMD_WIDTH <= n:
        var yv = y.unsafe_load[width=SIMD_WIDTH](i)
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        y.unsafe_store[width=SIMD_WIDTH](i, yv + av * xv)
        i += SIMD_WIDTH
    while i < n:
        y[unsafe_offset=i] += a * x[unsafe_offset=i]
        i += 1

@always_inline
def gqa_decode_head(
    q: FloatPtr,
    k_cache: FloatPtr,
    v_cache: FloatPtr,
    att_scores: FloatPtr,
    out_head: FloatPtr,
    h: Int,
    pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    max_ctx: Int,
    sliding_window: Int = 0,
    attn_scale: Float32 = 0.0,
):
    """ Compute Grouped-Query Attention for a single head `h` at position `pos`.
    
    Q head `h` attends to KV head `h // (n_heads // n_kv_heads)` over history `0..pos`.
    If `sliding_window` > 0, restricts attention to `max(0, pos - sliding_window + 1) .. pos`.
    If `attn_scale` > 0.0, uses `attn_scale` instead of default `1.0 / sqrt(head_dim)`.
    Writes result directly to `out_head`.
    """
    var kv_mul = n_heads // n_kv_heads
    var kv_h = h // kv_mul
    var kv_dim = n_kv_heads * head_dim
    var q_offset = h * head_dim
    var kv_head_offset = kv_h * head_dim
    var q_head_ptr = q.unsafe_offset(q_offset)
    var inv_sqrt_d = attn_scale if attn_scale > 0.0 else (Float32(1.0) / sqrt(Float32(head_dim)))

    var start_t = 0
    if sliding_window > 0 and pos >= sliding_window:
        start_t = pos - sliding_window + 1

    # 1. Compute dot products Q @ K.T and track maximum for online softmax
    var max_score: Float32 = -1e30
    for t in range(start_t, pos + 1):
        var k_slot_ptr = k_cache.unsafe_offset(t * kv_dim + kv_head_offset)
        var score = dot_vectors(q_head_ptr, k_slot_ptr, head_dim) * inv_sqrt_d
        att_scores[unsafe_offset=t] = score
        if score > max_score:
            max_score = score

    # 2. Exponentiate and sum
    var sum_exp: Float32 = 0.0
    for t in range(start_t, pos + 1):
        var exp_s = exp(att_scores[unsafe_offset=t] - max_score)
        att_scores[unsafe_offset=t] = exp_s
        sum_exp += exp_s

    var inv_sum = Float32(1.0) / (sum_exp + Float32(1e-9))

    # 3. Weighted accumulation over V cache: out = sum(att[t] * V[t])
    unsafe_memset_zero(out_head, head_dim)
    for t in range(start_t, pos + 1):
        var weight = att_scores[unsafe_offset=t] * inv_sum
        var v_slot_ptr = v_cache.unsafe_offset(t * kv_dim + kv_head_offset)
        axpy_vector(out_head, v_slot_ptr, weight, head_dim)

@always_inline
def gqa_decode(
    q: FloatPtr,
    k_cache: FloatPtr,
    v_cache: FloatPtr,
    att_buf: FloatPtr,
    out_ptr: FloatPtr,
    pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    max_ctx: Int,
    head_start: Int,
    head_end: Int,
    sliding_window: Int = 0,
    attn_scale: Float32 = 0.0,
):
    """ Compute GQA attention decode for a worker's head range [head_start, head_end)."""
    for h in range(head_start, head_end):
        var att_scores_h = att_buf.unsafe_offset(h * max_ctx)
        var out_h = out_ptr.unsafe_offset(h * head_dim)
        gqa_decode_head(
            q,
            k_cache,
            v_cache,
            att_scores_h,
            out_h,
            h,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            sliding_window,
            attn_scale,
        )
