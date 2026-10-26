"""Unit tests for Phase 2: Quantized Matmul Kernels and Fused Norm / SwiGLU / RoPE."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, bitcast
from std.utils import StaticTuple

from llamojo import (
    GGMLType,
    fused_rms_norm_residual,
    rms_norm,
    fused_silu_mul,
    gelu,
    fused_gelu_mul,
    rms_norm_per_head,
    rms_norm_no_weight,
    apply_logit_softcapping,
    apply_rope_neox,
    k4_scale,
    quantize_row_q8_0,
    dot_f32_row,
    dot_q4_0_row,
    dot_q8_0_row,
    dot_q4_k_row,
    dot_mxfp4_row,
    dot_quant_row,
    matmul_quant_fused,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

def test_fused_norm_and_swiglu() raises:
    print("Testing fused_rms_norm_residual and fused_silu_mul...")
    var n = 64
    var x = unsafe_alloc[Float32](n)
    var res = unsafe_alloc[Float32](n)
    var weight = unsafe_alloc[Float32](n)
    var out_norm = unsafe_alloc[Float32](n)
    var out_res = unsafe_alloc[Float32](n)

    for i in range(n):
        x[unsafe_offset=i] = 1.0
        res[unsafe_offset=i] = 2.0
        weight[unsafe_offset=i] = 2.0

    fused_rms_norm_residual(x, res, weight, out_norm, out_res, n)

    # x + res = 3.0 everywhere. RMS = sqrt(3^2) = 3.0. normed = 3.0 / 3.0 * 2.0 = 2.0
    for i in range(n):
        assert_almost_equal(out_res[unsafe_offset=i], Float32(3.0), atol=1e-4)
        assert_almost_equal(out_norm[unsafe_offset=i], Float32(2.0), atol=1e-4)

    # Test SwiGLU (silu(gate) * up)
    var gate = unsafe_alloc[Float32](n)
    var up = unsafe_alloc[Float32](n)
    var out_swiglu = unsafe_alloc[Float32](n)
    for i in range(n):
        gate[unsafe_offset=i] = 0.0 # silu(0) = 0.0 / (1 + 1) = 0.0
        up[unsafe_offset=i] = 5.0
    fused_silu_mul(gate, up, out_swiglu, n)
    for i in range(n):
        assert_almost_equal(out_swiglu[unsafe_offset=i], Float32(0.0), atol=1e-5)

    gate[unsafe_offset=0] = 100.0 # silu(100) approx 100.0
    up[unsafe_offset=0] = 2.0
    fused_silu_mul(gate, up, out_swiglu, n)
    assert_almost_equal(out_swiglu[unsafe_offset=0], Float32(200.0), atol=1e-2)

    x.unsafe_free()
    res.unsafe_free()
    weight.unsafe_free()
    out_norm.unsafe_free()
    out_res.unsafe_free()
    gate.unsafe_free()
    up.unsafe_free()
    out_swiglu.unsafe_free()
    print("  ✓ Fused RMSNorm & SwiGLU passed")

    # Test GELU, GELU-GLU, Per-Head RMSNorm, Unweighted RMSNorm, and Softcapping
    print("Testing GELU, Per-Head RMSNorm, and Logit Softcapping...")
    var gate_gelu = unsafe_alloc[Float32](4)
    var up_gelu = unsafe_alloc[Float32](4)
    var out_gelu = unsafe_alloc[Float32](4)
    gate_gelu[unsafe_offset=0] = 0.0 # gelu(0) = 0.0
    gate_gelu[unsafe_offset=1] = 1.0 # gelu(1.0) approx 0.8413447
    gate_gelu[unsafe_offset=2] = -1.0 # gelu(-1.0) approx -0.158655
    gate_gelu[unsafe_offset=3] = 10.0 # gelu(10) approx 10.0
    for i in range(4):
        up_gelu[unsafe_offset=i] = 2.0
    fused_gelu_mul(gate_gelu, up_gelu, out_gelu, 4)
    assert_almost_equal(out_gelu[unsafe_offset=0], Float32(0.0), atol=1e-5)
    assert_almost_equal(out_gelu[unsafe_offset=1], Float32(0.8413447 * 2.0), atol=1e-3)
    assert_almost_equal(out_gelu[unsafe_offset=2], Float32(-0.158655 * 2.0), atol=1e-3)
    assert_almost_equal(out_gelu[unsafe_offset=3], Float32(20.0), atol=1e-2)
    gate_gelu.unsafe_free()
    up_gelu.unsafe_free()
    out_gelu.unsafe_free()

    # Test per-head RMSNorm
    var q_buf = unsafe_alloc[Float32](8) # 2 heads of dim 4
    var w_head = unsafe_alloc[Float32](4)
    for i in range(4):
        w_head[unsafe_offset=i] = 1.0
        q_buf[unsafe_offset=i] = 2.0     # Head 0: RMS = 2.0 -> normed = 1.0
        q_buf[unsafe_offset=4 + i] = 4.0 # Head 1: RMS = 4.0 -> normed = 1.0
    rms_norm_per_head(q_buf, w_head, n_heads=2, head_dim=4, eps=1e-6)
    for i in range(8):
        assert_almost_equal(q_buf[unsafe_offset=i], Float32(1.0), atol=1e-3)
    q_buf.unsafe_free()
    w_head.unsafe_free()

    # Test unweighted RMSNorm
    var v_buf = unsafe_alloc[Float32](4)
    var v_out = unsafe_alloc[Float32](4)
    for i in range(4):
        v_buf[unsafe_offset=i] = 3.0
    rms_norm_no_weight(v_out, v_buf, 4, eps=1e-6)
    for i in range(4):
        assert_almost_equal(v_out[unsafe_offset=i], Float32(1.0), atol=1e-3)
    v_buf.unsafe_free()
    v_out.unsafe_free()

    # Test logit softcapping
    var logits = unsafe_alloc[Float32](3)
    logits[unsafe_offset=0] = 0.0   # 30 * tanh(0) = 0
    logits[unsafe_offset=1] = 300.0 # 30 * tanh(10) approx 30.0
    logits[unsafe_offset=2] = -300.0 # 30 * tanh(-10) approx -30.0
    apply_logit_softcapping(logits, 3, cap=Float32(30.0))
    assert_almost_equal(logits[unsafe_offset=0], Float32(0.0), atol=1e-4)
    assert_almost_equal(logits[unsafe_offset=1], Float32(30.0), atol=1e-3)
    assert_almost_equal(logits[unsafe_offset=2], Float32(-30.0), atol=1e-3)
    logits.unsafe_free()
    print("  ✓ GELU, Per-Head RMSNorm & Softcapping passed")

def test_rope_rotation() raises:
    print("Testing apply_rope_neox in-place rotary embeddings...")
    var head_dim = 16
    var n_heads = 2
    var n_kv_heads = 1
    var q = unsafe_alloc[Float32](n_heads * head_dim)
    var k = unsafe_alloc[Float32](n_kv_heads * head_dim)

    for i in range(n_heads * head_dim):
        q[unsafe_offset=i] = 1.0
    for i in range(n_kv_heads * head_dim):
        k[unsafe_offset=i] = 1.0

    # At pos 0: angle = 0, cos(0) = 1, sin(0) = 0. Values should be identical
    apply_rope_neox(q, k, n_heads, n_kv_heads, head_dim, pos=0)
    for i in range(n_heads * head_dim):
        assert_almost_equal(q[unsafe_offset=i], Float32(1.0), atol=1e-5)
    for i in range(n_kv_heads * head_dim):
        assert_almost_equal(k[unsafe_offset=i], Float32(1.0), atol=1e-5)

    # At pos 1: values should rotate
    apply_rope_neox(q, k, n_heads, n_kv_heads, head_dim, pos=1)
    # Check norm preservation of rotated pairs
    for h in range(n_heads):
        var q_h = q.unsafe_offset(h * head_dim)
        for i in range(head_dim // 2):
            var q0 = q_h[unsafe_offset=i]
            var q1 = q_h[unsafe_offset=i + head_dim // 2]
            var r2 = q0 * q0 + q1 * q1
            assert_almost_equal(r2, Float32(2.0), atol=1e-4)

    q.unsafe_free()
    k.unsafe_free()
    print("  ✓ RoPE rotation passed")

def test_q8_quantization_and_dot() raises:
    print("Testing quantize_row_q8_0 and dot_q8_0_row...")
    var n = 64
    var x = unsafe_alloc[Float32](n)
    var q8 = unsafe_alloc[UInt8]((n // 32) * 34)

    for i in range(n):
        x[unsafe_offset=i] = Float32(i + 1) * 0.1

    quantize_row_q8_0(x, q8, n)

    # Compute dot product against self
    var dot_val = dot_q8_0_row(q8, x, n)

    # Expected exact sum(x_i^2)
    var expected: Float32 = 0.0
    for i in range(n):
        expected += x[unsafe_offset=i] * x[unsafe_offset=i]

    # Q8 quantization error should be < 1%
    var diff = dot_val - expected
    if diff < 0:
        diff = -diff
    var rel_err = diff / expected
    assert_true(rel_err < 0.01)

    x.unsafe_free()
    q8.unsafe_free()
    print("  ✓ Q8_0 activation quantization and dot product passed")

def test_k4_scale_and_q4k_dot() raises:
    print("Testing k4_scale bit unpacking and dot_q4_k_row...")
    # Verify k4_scale unpacking
    var scales = unsafe_alloc[UInt8](12)
    for i in range(12):
        scales[unsafe_offset=i] = UInt8(i * 17)

    var d0: UInt8 = 0
    var m0: UInt8 = 0
    k4_scale(0, scales, d0, m0)
    assert_equal(d0, scales[unsafe_offset=0] & UInt8(63))
    assert_equal(m0, scales[unsafe_offset=4] & UInt8(63))

    var d4: UInt8 = 0
    var m4: UInt8 = 0
    k4_scale(4, scales, d4, m4)
    var exp_d4 = (scales[unsafe_offset=8] & UInt8(0x0F)) | ((scales[unsafe_offset=0] >> UInt8(6)) << UInt8(4))
    assert_equal(d4, exp_d4)

    # Test Q4_K dot product (256 elements = 144 bytes)
    var w_q4k = unsafe_alloc[UInt8](144)
    var x = unsafe_alloc[Float32](256)
    # Set d = 1.0 (0x3C00 in f16), dmin = 0.0
    w_q4k[unsafe_offset=0] = UInt8(0x00)
    w_q4k[unsafe_offset=1] = UInt8(0x3C)
    w_q4k[unsafe_offset=2] = UInt8(0x00)
    w_q4k[unsafe_offset=3] = UInt8(0x00)
    for i in range(4, 16):
        w_q4k[unsafe_offset=i] = UInt8(1) # scale = 1
    for i in range(16, 144):
        w_q4k[unsafe_offset=i] = UInt8(0x22) # nibbles = 2 (low) and 2 (high)
    for i in range(256):
        x[unsafe_offset=i] = 1.0

    var q4k_dot = dot_q4_k_row(w_q4k, x, 256)
    # nibble 2 * d(1.0) * scale(1.0) * x(1.0) * 256 elements = 512.0
    assert_almost_equal(q4k_dot, Float32(512.0), atol=1e-2)

    scales.unsafe_free()
    w_q4k.unsafe_free()
    x.unsafe_free()
    print("  ✓ k4_scale unpacking & Q4_K dot product passed")

def test_mxfp4_dot() raises:
    print("Testing dot_mxfp4_row...")
    # 32 elements per block: 16B nibbles + 1B E8M0 scale (byte 16)
    var w_mx = unsafe_alloc[UInt8](17)
    var x = unsafe_alloc[Float32](32)
    # E8M0: 127 = 2^(127-127) = 1.0 scale
    w_mx[unsafe_offset=16] = UInt8(127)
    for i in range(16):
        w_mx[unsafe_offset=i] = UInt8(0x33) # lo=3, hi=3
    for i in range(32):
        x[unsafe_offset=i] = 2.0

    var res = dot_mxfp4_row(w_mx, x, 32)
    # 32 elements * 3 * 2.0 * 1.0 = 192.0
    assert_almost_equal(res, Float32(192.0), atol=1e-3)

    w_mx.unsafe_free()
    x.unsafe_free()
    print("  ✓ MXFP4 dot product passed")

def test_fused_multimatrix_dispatch() raises:
    print("Testing matmul_quant_fused for QKV (n=3) and Gate/Up (n=2)...")
    var nc = 64
    var nr = 4
    var x = unsafe_alloc[Float32](nc)
    for i in range(nc):
        x[unsafe_offset=i] = 1.0

    # 3 weight matrices for Q, K, V (Q8_0 format: 2 blocks of 34 bytes = 68 bytes per row)
    var bytes_per_row = (nc // 32) * 34
    var total_w_bytes = nr * bytes_per_row
    var w_q = unsafe_alloc[UInt8](total_w_bytes)
    var w_k = unsafe_alloc[UInt8](total_w_bytes)
    var w_v = unsafe_alloc[UInt8](total_w_bytes)

    # Initialize weights: scale = 1.0 (0x3C00), int8 values
    for r in range(nr):
        for b in range(nc // 32):
            var bo = r * bytes_per_row + b * 34
            w_q[unsafe_offset=bo] = UInt8(0x00); w_q[unsafe_offset=bo+1] = UInt8(0x3C)
            w_k[unsafe_offset=bo] = UInt8(0x00); w_k[unsafe_offset=bo+1] = UInt8(0x3C)
            w_v[unsafe_offset=bo] = UInt8(0x00); w_v[unsafe_offset=bo+1] = UInt8(0x3C)
            for j in range(32):
                w_q[unsafe_offset=bo + 2 + j] = UInt8(1)
                w_k[unsafe_offset=bo + 2 + j] = UInt8(2)
                w_v[unsafe_offset=bo + 2 + j] = UInt8(3)

    var c_q = unsafe_alloc[Float32](nr)
    var c_k = unsafe_alloc[Float32](nr)
    var c_v = unsafe_alloc[Float32](nr)

    var c_tuple = StaticTuple[FloatPtr, 3](c_q, c_k, c_v)
    var w_tuple = StaticTuple[BytePtr, 3](w_q, w_k, w_v)

    matmul_quant_fused[3](
        GGMLType.Q8_0,
        c_tuple,
        x,
        w_tuple,
        row_start=0,
        row_end=nr,
        nc=nc,
    )

    for r in range(nr):
        # Q: 64 elements * 1 = 64.0
        assert_almost_equal(c_q[unsafe_offset=r], Float32(64.0), atol=1e-3)
        # K: 64 elements * 2 = 128.0
        assert_almost_equal(c_k[unsafe_offset=r], Float32(128.0), atol=1e-3)
        # V: 64 elements * 3 = 192.0
        assert_almost_equal(c_v[unsafe_offset=r], Float32(192.0), atol=1e-3)

    x.unsafe_free()
    w_q.unsafe_free()
    w_k.unsafe_free()
    w_v.unsafe_free()
    c_q.unsafe_free()
    c_k.unsafe_free()
    c_v.unsafe_free()
    print("  ✓ matmul_quant_fused QKV (n=3) passed")

def main() raises:
    print("============================================================")
    print("Running Phase 2 Kernels Test Suite")
    print("============================================================")
    test_fused_norm_and_swiglu()
    test_rope_rotation()
    test_q8_quantization_and_dot()
    test_k4_scale_and_q4k_dot()
    test_mxfp4_dot()
    test_fused_multimatrix_dispatch()
    print("All Phase 2 Kernel tests passed! ✓")

