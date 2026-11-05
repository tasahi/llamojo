""" Unit tests for Whisper Phase 1 Core Kernels.\n
Tests:
1. `layer_norm` and `fused_layer_norm_residual` (with weight, bias, and nullable variants)
2. `matmul_quant_fused_biased` (Q/V with bias, K without bias matching Whisper self/cross attention)
3. `gemv_quant_biased` and `gemv_gelu_biased` (matching Whisper MLP and output projections)
4. `conv1d_f32` and `conv1d_k3_gelu` (matching Whisper `conv1` stride=1 and `conv2` stride=2)
5. `transpose_add_pos_embd` (channel-first `[C, T]` -> sequence-first `[T, C]` + `e_pe`)
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.utils import StaticTuple

from llamojo import (
    GGMLType,
    null_ptr,
    layer_norm,
    fused_layer_norm_residual,
    gelu,
    matmul_quant_fused_biased,
    gemv_quant_biased,
    gemv_gelu_biased,
    conv1d_f32,
    conv1d_k3_gelu,
    transpose_add_pos_embd,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime UInt16Ptr = Pointer[UInt16, MutUntrackedOrigin]


def test_layer_norm() raises:
    print("Testing layer_norm and fused_layer_norm_residual...")
    var n = 64
    var x = unsafe_alloc[Float32](n)
    var weight = unsafe_alloc[Float32](n)
    var bias = unsafe_alloc[Float32](n)
    var out_ptr = unsafe_alloc[Float32](n)

    # Set alternating values: half = 2.0, half = 4.0 -> mean = 3.0, var = 1.0, std = 1.0
    for i in range(n):
        x[unsafe_offset=i] = 2.0 if (i % 2 == 0) else 4.0
        weight[unsafe_offset=i] = 2.0
        bias[unsafe_offset=i] = 0.5

    layer_norm(out_ptr, x, weight, bias, n, 1e-5)

    for i in range(n):
        var expected: Float32 = (-1.0 * 2.0 + 0.5) if (i % 2 == 0) else (1.0 * 2.0 + 0.5)
        assert_almost_equal(out_ptr[unsafe_offset=i], expected, atol=1e-4)

    # Test fused_layer_norm_residual
    var res = unsafe_alloc[Float32](n)
    var out_res = unsafe_alloc[Float32](n)
    for i in range(n):
        x[unsafe_offset=i] = 1.0 if (i % 2 == 0) else 3.0
        res[unsafe_offset=i] = 1.0  # x + res = 2.0 or 4.0

    fused_layer_norm_residual(x, res, weight, bias, out_ptr, out_res, n, 1e-5)
    for i in range(n):
        var exp_res: Float32 = 2.0 if (i % 2 == 0) else 4.0
        var exp_norm: Float32 = -1.5 if (i % 2 == 0) else 2.5
        assert_almost_equal(out_res[unsafe_offset=i], exp_res, atol=1e-4)
        assert_almost_equal(out_ptr[unsafe_offset=i], exp_norm, atol=1e-4)

    x.unsafe_free()
    res.unsafe_free()
    out_res.unsafe_free()
    weight.unsafe_free()
    bias.unsafe_free()
    out_ptr.unsafe_free()
    print("  ✓ layer_norm & fused_layer_norm_residual passed")


def test_biased_quant_projections() raises:
    print("Testing matmul_quant_fused_biased, gemv_quant_biased, and gemv_gelu_biased...")
    var dim = 64
    var x = unsafe_alloc[Float32](dim)
    var wq = unsafe_alloc[Float32](dim * dim)
    var wk = unsafe_alloc[Float32](dim * dim)
    var wv = unsafe_alloc[Float32](dim * dim)
    var bq = unsafe_alloc[Float32](dim)
    var bv = unsafe_alloc[Float32](dim)

    var q_out = unsafe_alloc[Float32](dim)
    var k_out = unsafe_alloc[Float32](dim)
    var v_out = unsafe_alloc[Float32](dim)

    for i in range(dim):
        x[unsafe_offset=i] = 1.0
        bq[unsafe_offset=i] = 0.25
        bv[unsafe_offset=i] = -0.5

    for r in range(dim):
        for c in range(dim):
            wq[unsafe_offset=r * dim + c] = 0.5
            wk[unsafe_offset=r * dim + c] = 1.0
            wv[unsafe_offset=r * dim + c] = 2.0

    # Whisper pattern: Q has bias, K has NO bias (null_ptr), V has bias
    var c_tuple = StaticTuple[FloatPtr, 3](q_out, k_out, v_out)
    var w_tuple = StaticTuple[BytePtr, 3](
        wq.unsafe_bitcast[UInt8](),
        wk.unsafe_bitcast[UInt8](),
        wv.unsafe_bitcast[UInt8](),
    )
    var b_tuple = StaticTuple[FloatPtr, 3](bq, null_ptr[Float32](), bv)

    matmul_quant_fused_biased[3](GGMLType.F32, c_tuple, x, w_tuple, b_tuple, 0, dim, dim)

    for r in range(dim):
        # q: 64 * 0.5 + 0.25 = 32.25
        assert_almost_equal(q_out[unsafe_offset=r], Float32(32.25), atol=1e-4)
        # k: 64 * 1.0 + 0.0 = 64.0
        assert_almost_equal(k_out[unsafe_offset=r], Float32(64.0), atol=1e-4)
        # v: 64 * 2.0 - 0.5 = 127.5
        assert_almost_equal(v_out[unsafe_offset=r], Float32(127.5), atol=1e-4)

    # Test gemv_quant_biased and gemv_gelu_biased
    var mlp_out = unsafe_alloc[Float32](dim)
    for r in range(dim):
        for c in range(dim):
            wq[unsafe_offset=r * dim + c] = Float32(1.0) / Float32(dim)
        bq[unsafe_offset=r] = 0.0  # w @ x + b = 1.0

    gemv_gelu_biased(mlp_out, wq.unsafe_bitcast[UInt8](), bq, GGMLType.F32, x, dim, dim, 1)
    var expected_gelu1 = gelu(Float32(1.0))
    for r in range(dim):
        assert_almost_equal(mlp_out[unsafe_offset=r], expected_gelu1, atol=1e-4)

    gemv_quant_biased(mlp_out, wk.unsafe_bitcast[UInt8](), bv, GGMLType.F32, x, dim, dim, 1)
    for r in range(dim):
        assert_almost_equal(mlp_out[unsafe_offset=r], Float32(63.5), atol=1e-4)

    x.unsafe_free()
    wq.unsafe_free()
    wk.unsafe_free()
    wv.unsafe_free()
    bq.unsafe_free()
    bv.unsafe_free()
    q_out.unsafe_free()
    k_out.unsafe_free()
    v_out.unsafe_free()
    mlp_out.unsafe_free()
    print("  ✓ matmul_quant_fused_biased, gemv_quant_biased & gemv_gelu_biased passed")


def test_conv1d_and_pos_embd() raises:
    print("Testing conv1d_f32, conv1d_k3_gelu (stride 1 & 2, F32 & F16), and transpose_add_pos_embd...")
    var c_in = 4
    var c_out = 8
    var t_in = 32

    var inp = unsafe_alloc[Float32](c_in * t_in)
    var w_f32 = unsafe_alloc[Float32](c_out * c_in * 3)
    var w_f16 = unsafe_alloc[UInt16](c_out * c_in * 3)
    var bias = unsafe_alloc[Float32](c_out)

    var out_ref_s1 = unsafe_alloc[Float32](c_out * t_in)
    var out_fast_s1 = unsafe_alloc[Float32](c_out * t_in)
    var out_f16_s1 = unsafe_alloc[Float32](c_out * t_in)

    var t_out_s2 = t_in // 2
    var out_ref_s2 = unsafe_alloc[Float32](c_out * t_out_s2)
    var out_fast_s2 = unsafe_alloc[Float32](c_out * t_out_s2)

    for i in range(c_in * t_in):
        inp[unsafe_offset=i] = Float32((i % 7) - 3) * 0.25

    # 0x3800 in IEEE-754 FP16 is exactly 0.5
    for i in range(c_out * c_in * 3):
        w_f32[unsafe_offset=i] = 0.5
        w_f16[unsafe_offset=i] = 0x3800

    for oc in range(c_out):
        bias[unsafe_offset=oc] = Float32(oc) * 0.1

    # 1. Compare general conv1d_f32 vs SIMD conv1d_k3_gelu for stride = 1 (F32 & F16)
    conv1d_f32(
        out_ref_s1, inp, w_f32.unsafe_bitcast[UInt8](), GGMLType.F32,
        bias, c_in, c_out, t_in, 3, 1, 1, True, 1
    )
    conv1d_k3_gelu(
        out_fast_s1, inp, w_f32.unsafe_bitcast[UInt8](), GGMLType.F32,
        bias, c_in, c_out, t_in, 1, 1
    )
    conv1d_k3_gelu(
        out_f16_s1, inp, w_f16.unsafe_bitcast[UInt8](), GGMLType.F16,
        bias, c_in, c_out, t_in, 1, 1
    )

    for i in range(c_out * t_in):
        assert_almost_equal(out_fast_s1[unsafe_offset=i], out_ref_s1[unsafe_offset=i], atol=1e-5)
        assert_almost_equal(out_f16_s1[unsafe_offset=i], out_ref_s1[unsafe_offset=i], atol=1e-5)

    # 2. Compare general conv1d_f32 vs SIMD conv1d_k3_gelu for stride = 2 (Whisper conv2 downsampling)
    conv1d_f32(
        out_ref_s2, inp, w_f32.unsafe_bitcast[UInt8](), GGMLType.F32,
        bias, c_in, c_out, t_in, 3, 2, 1, True, 1
    )
    conv1d_k3_gelu(
        out_fast_s2, inp, w_f32.unsafe_bitcast[UInt8](), GGMLType.F32,
        bias, c_in, c_out, t_in, 2, 1
    )

    for i in range(c_out * t_out_s2):
        assert_almost_equal(out_fast_s2[unsafe_offset=i], out_ref_s2[unsafe_offset=i], atol=1e-5)

    # 3. Test transpose_add_pos_embd ([c_out, t_out_s2] -> [t_out_s2, c_out] + pos_embd)
    var pos_embd = unsafe_alloc[Float32](t_out_s2 * c_out)
    var seq_out = unsafe_alloc[Float32](t_out_s2 * c_out)
    for t in range(t_out_s2):
        for c in range(c_out):
            pos_embd[unsafe_offset=t * c_out + c] = Float32(t * 10 + c)

    transpose_add_pos_embd(seq_out, out_fast_s2, pos_embd, c_out, t_out_s2)

    for t in range(t_out_s2):
        for c in range(c_out):
            var expected = out_fast_s2[unsafe_offset=c * t_out_s2 + t] + Float32(t * 10 + c)
            assert_almost_equal(seq_out[unsafe_offset=t * c_out + c], expected, atol=1e-5)

    inp.unsafe_free()
    w_f32.unsafe_free()
    w_f16.unsafe_free()
    bias.unsafe_free()
    out_ref_s1.unsafe_free()
    out_fast_s1.unsafe_free()
    out_f16_s1.unsafe_free()
    out_ref_s2.unsafe_free()
    out_fast_s2.unsafe_free()
    pos_embd.unsafe_free()
    seq_out.unsafe_free()
    print("  ✓ conv1d_f32, conv1d_k3_gelu (F32/F16, stride 1/2) & transpose_add_pos_embd passed")


def main() raises:
    print("=== Running Whisper Phase 1 Core Kernel Tests ===")
    test_layer_norm()
    test_biased_quant_projections()
    test_conv1d_and_pos_embd()
    print("=== All Whisper Phase 1 Core Kernel Tests Passed! ===")
