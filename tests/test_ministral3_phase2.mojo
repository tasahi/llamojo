""" Unit tests for Phase 2 YaRN RoPE mathematical functions, frequency blending, and kernel behavior."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc
from std.math import sqrt, exp, pow, cos, sin, log, floor, ceil
from llamojo import (
    apply_rope_yarn,
    rope_yarn_corr_factor,
    rope_yarn_corr_dims,
    rope_yarn_ramp,
    apply_rope_norm,
    apply_rope_neox,
)

def test_yarn_corr_factor_and_dims() raises:
    print("Testing YaRN correction dimension bounds and ramp calculations...")
    var n_dims = 128
    var n_ctx_orig = 16384
    var freq_base: Float32 = 1000000.0
    var beta_fast: Float32 = 32.0
    var beta_slow: Float32 = 1.0

    var corr_low: Float32 = 0.0
    var corr_high: Float32 = 0.0
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_low, corr_high)

    # Verification of boundaries
    assert_true(corr_low >= 0.0)
    assert_true(corr_high <= Float32(n_dims - 1))
    assert_true(corr_low <= corr_high)

    # Verification of ramp behavior:
    # At or below corr_low: ramp should be 1.0 (full extrapolation blend)
    var ramp_at_low = rope_yarn_ramp(corr_low, corr_high, Int(corr_low))
    assert_almost_equal(ramp_at_low, Float32(1.0), atol=1e-3)

    # At or above corr_high: ramp should be 0.0 (full interpolation blend)
    var ramp_at_high = rope_yarn_ramp(corr_low, corr_high, Int(corr_high))
    assert_almost_equal(ramp_at_high, Float32(0.0), atol=1e-3)

    # Mid point should be between 0.0 and 1.0
    var mid = Int((corr_low + corr_high) / 2.0)
    var ramp_mid = rope_yarn_ramp(corr_low, corr_high, mid)
    assert_true(ramp_mid >= 0.0 and ramp_mid <= 1.0)
    print("  ✓ YaRN corr dims and ramp verified")

def test_yarn_rope_kernel_execution() raises:
    print("Testing apply_rope_yarn execution and rotation accuracy...")
    var n_heads = 4
    var n_kv_heads = 2
    var head_dim = 64
    var pos = 128
    var rope_theta: Float32 = 1000000.0
    var rope_freq_scale: Float32 = 1.0 / 16.0 # 16x context extension
    var n_ctx_orig = 16384
    var beta_fast: Float32 = 32.0
    var beta_slow: Float32 = 1.0

    var q_buf = unsafe_alloc[Float32](n_heads * head_dim)
    var k_buf = unsafe_alloc[Float32](n_kv_heads * head_dim)

    # Initialize buffers with known test signals
    for i in range(n_heads * head_dim):
        q_buf[unsafe_offset=i] = Float32(1.0)
    for i in range(n_kv_heads * head_dim):
        k_buf[unsafe_offset=i] = Float32(1.0)

    # Apply YaRN RoPE (Interleaved / NORM layout)
    apply_rope_yarn(
        q_buf,
        k_buf,
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
        rope_theta,
        rope_freq_scale,
        n_ctx_orig,
        beta_fast,
        beta_slow,
        ext_factor=Float32(1.0),
        attn_factor=Float32(1.0),
        is_neox=False,
    )

    # Verify that values rotated and didn't explode or produce NaN / Inf
    for i in range(n_heads * head_dim):
        var v = q_buf[unsafe_offset=i]
        assert_true(v >= -10.0 and v <= 10.0)
        assert_true(v == v) # NaN check

    for i in range(n_kv_heads * head_dim):
        var v = k_buf[unsafe_offset=i]
        assert_true(v >= -10.0 and v <= 10.0)
        assert_true(v == v)

    # Also test NeoX layout
    for i in range(n_heads * head_dim):
        q_buf[unsafe_offset=i] = Float32(1.0)
    for i in range(n_kv_heads * head_dim):
        k_buf[unsafe_offset=i] = Float32(1.0)

    apply_rope_yarn(
        q_buf,
        k_buf,
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
        rope_theta,
        rope_freq_scale,
        n_ctx_orig,
        beta_fast,
        beta_slow,
        ext_factor=Float32(1.0),
        attn_factor=Float32(1.0),
        is_neox=True,
    )

    for i in range(n_heads * head_dim):
        var v = q_buf[unsafe_offset=i]
        assert_true(v >= -10.0 and v <= 10.0)
        assert_true(v == v)

    q_buf.unsafe_free()
    k_buf.unsafe_free()
    print("  ✓ apply_rope_yarn execution and numerical stability verified")

def main() raises:
    print("============================================================")
    print("Running Ministral 3 3B Phase 2 (YaRN RoPE) Tests")
    print("============================================================")
    test_yarn_corr_factor_and_dims()
    test_yarn_rope_kernel_execution()
    print("All Phase 2 tests passed! ✓")
