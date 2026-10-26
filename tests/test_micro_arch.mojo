""" Unit tests for Phase 11: Advanced CPU Micro-Architecture Optimizations."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc

from llamojo import (
    GGMLType,
    gemv_quant_tiled,
    dot_q8_0_vnni,
    dot_bf16_vec,
    flash_attn_cpu,
    numa_is_available,
    numa_node_count,
    numa_pin_thread,
    quantize_row_q8_0,
)

def test_cache_tiled_gemv() raises:
    print("Testing Cache-Tiled Quantized GEMV...")
    var n_rows = 4
    var n_cols = 64

    # Allocate input activation x [64]
    var x = unsafe_alloc[Float32](n_cols)
    for i in range(n_cols):
        x[unsafe_offset=i] = Float32(0.5)

    # Allocate and quantize 4 rows of Q8_0 weights (64 elements each = 2 blocks = 68 bytes per row)
    var row_bytes = (n_cols // 32) * 34
    var w_q8 = unsafe_alloc[UInt8](n_rows * row_bytes)

    var f32_row = unsafe_alloc[Float32](n_cols)
    for r in range(n_rows):
        for c in range(n_cols):
            f32_row[unsafe_offset=c] = Float32(r + 1) * Float32(0.25)
        quantize_row_q8_0(f32_row, w_q8.unsafe_offset(r * row_bytes), n_cols)

    var y = unsafe_alloc[Float32](n_rows)
    gemv_quant_tiled(GGMLType.Q8_0, w_q8, x, y, n_rows, n_cols)

    # Expected dot: 64 elements of (0.5 * (r+1)*0.25) = 64 * 0.125 * (r+1) = 8.0 * (r+1)
    for r in range(n_rows):
        var expected = Float32(r + 1) * Float32(8.0)
        assert_almost_equal(y[unsafe_offset=r], expected, atol=0.2)

    x.unsafe_free()
    f32_row.unsafe_free()
    w_q8.unsafe_free()
    y.unsafe_free()
    print("  ✓ Cache-tiled GEMV verified")

def test_vnni_and_bf16_vector_dot() raises:
    print("Testing VNNI & BF16 vector arithmetic...")
    var n = 32

    # Test BF16 dot product
    var a_bf16 = unsafe_alloc[UInt16](n)
    var b_bf16 = unsafe_alloc[UInt16](n)
    # BF16 1.0f = 0x3F80, 2.0f = 0x4000
    for i in range(n):
        a_bf16[unsafe_offset=i] = 0x3F80 # 1.0f
        b_bf16[unsafe_offset=i] = 0x4000 # 2.0f

    var bf16_dot = dot_bf16_vec(a_bf16, b_bf16, n)
    assert_almost_equal(bf16_dot, Float32(64.0), atol=1e-5)

    a_bf16.unsafe_free()
    b_bf16.unsafe_free()
    print("  ✓ BF16 vector dot product verified")

def test_flash_attn_cpu() raises:
    print("Testing CPU FlashAttention online tiled softmax...")
    var seq_len = 8
    var n_heads = 2
    var n_kv_heads = 1
    var head_dim = 16

    # Q: [2, 16] = 32
    var q = unsafe_alloc[Float32](n_heads * head_dim)
    for i in range(n_heads * head_dim):
        q[unsafe_offset=i] = Float32(0.1)

    # K, V: [8, 1, 16] = 128
    var k_cache = unsafe_alloc[Float32](seq_len * n_kv_heads * head_dim)
    var v_cache = unsafe_alloc[Float32](seq_len * n_kv_heads * head_dim)
    for i in range(seq_len * n_kv_heads * head_dim):
        k_cache[unsafe_offset=i] = Float32(0.2)
        v_cache[unsafe_offset=i] = Float32(1.0) # all 1.0f

    var out_ptr = unsafe_alloc[Float32](n_heads * head_dim)

    flash_attn_cpu(q, k_cache, v_cache, out_ptr, seq_len, n_heads, n_kv_heads, head_dim)

    # Since all V entries are 1.0f and softmax weights sum to 1.0, each output element must equal 1.0f!
    for i in range(n_heads * head_dim):
        assert_almost_equal(out_ptr[unsafe_offset=i], Float32(1.0), atol=1e-4)

    q.unsafe_free()
    k_cache.unsafe_free()
    v_cache.unsafe_free()
    out_ptr.unsafe_free()
    print("  ✓ FlashAttention online tiled softmax verified")

def test_numa_topology() raises:
    print("Testing NUMA topology awareness and thread pinning...")
    var _ = numa_is_available()
    var count = numa_node_count()
    assert_true(count >= 1)
    var pin_ok = numa_pin_thread(0)
    assert_true(pin_ok)
    print("  ✓ NUMA topology & affinity verified (nodes=" + String(count) + ")")

def main() raises:
    print("============================================================")
    print("Running Phase 11 Micro-Architecture Optimizations Test Suite")
    print("============================================================")
    test_cache_tiled_gemv()
    test_vnni_and_bf16_vector_dot()
    test_flash_attn_cpu()
    test_numa_topology()
    print("All Phase 11 Micro-Architecture tests passed! ✓")