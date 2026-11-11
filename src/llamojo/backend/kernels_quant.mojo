""" Unified Quantized Matrix-Vector and Multi-Matrix Multiplication Kernels.\n
Consolidates, ports and optimizes:
- MojoLlama/src/mojollama/mojo_engine/mojo_quant_kernels.mojo
- MojoLlama/src/mojollama/kernels/kquants.mojo
- MojoLlama/src/mojollama/kernels/optimized_kernels.mojo
- MojoLlama/src/mojollama/mojo_engine/simd_mxfp4.mojo
- llama2.mojo/matmul_rows fused StaticTuple dispatch

Features:
- Pure Mojo 1.1.0 (Pointer[..., MutUntrackedOrigin], no Python, no C compiler)
- F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4 dot products
- Exact bit-ops for Q4_K / Q5_K / Q6_K matching upstream llama.cpp k4_scale
- Multi-matrix horizontal fusion: matmul_quant_fused[n] (fusing QKV n=3 and Gate/Up n=2)
- Fast input activation quantization to Q8_0 (quantize_row_q8_0)
"""

from std.memory import Pointer, bitcast
from std.memory.alloc import unsafe_alloc
from std.utils import StaticTuple
from std.sys.info import simd_width_of, num_performance_cores, num_logical_cores
from std.ffi import external_call
from max.algorithm import parallelize

from .isa import (
    SIMD_WIDTH,
    F32xW,
    f16_to_f32,
    bf16_to_f32,
    load_f16_scale,
)
from .kernels_norm_rope import gelu
from ..types import GGMLType, ggml_type_size, ggml_block_size

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

# ─── Exact K-Quant Bit Unpacking Helpers (cengine_batch_instr.c:k4_scale) ─────

@always_inline
def k4_scale(j: Int, scales_ptr: BytePtr, mut d_out: UInt8, mut m_out: UInt8):
    """ Exact bit-unpacking for 12-byte scales array in Q4_K and Q5_K superblocks."""
    if j < 4:
        d_out = scales_ptr[unsafe_offset=j] & UInt8(63)
        m_out = scales_ptr[unsafe_offset=j + 4] & UInt8(63)
    else:
        var q_j4 = scales_ptr[unsafe_offset=j + 4]
        var q_jm4 = scales_ptr[unsafe_offset=j - 4]
        var q_j = scales_ptr[unsafe_offset=j]
        d_out = (q_j4 & UInt8(0x0F)) | ((q_jm4 >> UInt8(6)) << UInt8(4))
        m_out = (q_j4 >> UInt8(4)) | ((q_j >> UInt8(6)) << UInt8(4))

# ─── Fast Q8_0 Activation Vector Quantization ────────────────────────────────


@always_inline
def quantize_row_q8_k(x: FloatPtr, y: BytePtr, k: Int):
    """ Quantize Float32 vector to Q8_K intermediate block format (292 bytes per 256 values).
    Matches llama.cpp quantize_row_q8_K_ref bit-for-bit.
    """
    var nb = k // 256
    for i in range(nb):
        var xo = i * 256
        var yo = i * 292
        var amax: Float32 = 0.0
        var max_val: Float32 = 0.0
        for j in range(256):
            var v = x[unsafe_offset=xo + j]
            var av = v if v >= 0 else -v
            if av > amax:
                amax = av
                max_val = v
        if amax == 0.0:
            y.unsafe_offset(yo).unsafe_bitcast[Float32]()[unsafe_offset=0] = 0.0
            for j in range(256 + 32):
                y[unsafe_offset=yo + 4 + j] = 0
            continue

        var iscale = Float32(-127.0) / max_val
        y.unsafe_offset(yo).unsafe_bitcast[Float32]()[unsafe_offset=0] = Float32(1.0) / iscale

        var qs_ptr = y.unsafe_offset(yo + 4)
        var bsums_ptr = y.unsafe_offset(yo + 4 + 256).unsafe_bitcast[Int16]()

        for j in range(256):
            var v = Int32(x[unsafe_offset=xo + j] * iscale + (Float32(0.5) if x[unsafe_offset=xo + j] * iscale >= 0 else Float32(-0.5)))
            if v > 127: v = 127
            elif v < -128: v = -128
            qs_ptr[unsafe_offset=j] = bitcast[DType.uint8](Int8(v))

        for j in range(16):
            var sum: Int32 = 0
            for ii in range(16):
                sum += Int32(bitcast[DType.int8](qs_ptr[unsafe_offset=j * 16 + ii]))
            bsums_ptr[unsafe_offset=j] = Int16(sum)

@always_inline
def quantize_row_q8_0(x: FloatPtr, q8: BytePtr, n: Int):
    """ Quantize Float32 vector to Q8_0 format (34 bytes per 32 values) in pure Mojo."""
    var nb = n // 32
    for b in range(nb):
        var xo = b * 32
        var bo = b * 34
        var amax: Float32 = 0.0
        for j in range(32):
            var v = x[unsafe_offset=xo + j]
            if v < 0:
                v = -v
            if v > amax:
                amax = v
        if amax == 0.0:
            amax = 1.0

        var d = amax / Float32(127.0)
        var inv_d = Float32(127.0) / amax

        # Encode f16 scale
        var f16_bits = bitcast[DType.uint16](Float16(d))
        q8[unsafe_offset=bo] = UInt8(f16_bits & 0xFF)
        q8[unsafe_offset=bo + 1] = UInt8((f16_bits >> 8) & 0xFF)

        for j in range(32):
            var qv = Int32(x[unsafe_offset=xo + j] * inv_d)
            if qv > 127:
                qv = 127
            elif qv < -128:
                qv = -128
            q8[unsafe_offset=bo + 2 + j] = bitcast[DType.uint8](Int8(qv))

# ─── Single-Row Quantized Dot Product Kernels ─────────────────────────────────

@always_inline
def dot_f32_row(w: FloatPtr, x: FloatPtr, nc: Int) -> Float32:
    """ Float32 dot product with SIMD accumulation."""
    var acc = F32xW(0.0)
    var i = 0
    while i + SIMD_WIDTH <= nc:
        acc += w.unsafe_load[width=SIMD_WIDTH](i) * x.unsafe_load[width=SIMD_WIDTH](i)
        i += SIMD_WIDTH
    var total = acc.reduce_add()
    while i < nc:
        total += w[unsafe_offset=i] * x[unsafe_offset=i]
        i += 1
    return total


@always_inline
def dot_f16_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ F16 weights dot product against Float32 vector with SIMD Float16 -> Float32 acceleration."""
    var w_f16 = w.unsafe_bitcast[Float16]()
    var acc = F32xW(0.0)
    var i = 0
    while i + SIMD_WIDTH <= nc:
        var w_vec = w_f16.unsafe_load[width=SIMD_WIDTH](i).cast[DType.float32]()
        acc += w_vec * x.unsafe_load[width=SIMD_WIDTH](i)
        i += SIMD_WIDTH
    var total = acc.reduce_add()
    while i < nc:
        total += Float32(w_f16[unsafe_offset=i]) * x[unsafe_offset=i]
        i += 1
    return total

@always_inline
def dot_bf16_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ BF16 weights dot product against Float32 vector."""
    var total: Float32 = 0.0
    for i in range(nc):
        var lo = UInt16(w[unsafe_offset=i * 2])
        var hi = UInt16(w[unsafe_offset=i * 2 + 1])
        var wf = bf16_to_f32((hi << 8) | lo)
        total += wf * x[unsafe_offset=i]
    return total


@always_inline
def dot_q4_0_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q4_0 dot product: 32 elements per 18 bytes [2B scale, 16B nibbles]."""
    var bpr = nc // 32
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 18
        var d = load_f16_scale(w, bo)
        var xo = blk * 32
        var sum_i: Float32 = 0.0
        for j in range(16):
            var p = w[unsafe_offset=bo + 2 + j]
            var lo = Float32(Int32(p & 0x0F) - 8)
            var hi = Float32(Int32(p >> 4) - 8)
            sum_i += lo * x[unsafe_offset=xo + j]
            sum_i += hi * x[unsafe_offset=xo + j + 16]
        total += sum_i * d
    return total


@always_inline
def dot_q4_1_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q4_1 dot product: 32 elements per 20 bytes [2B d, 2B m, 16B nibbles]."""
    var bpr = nc // 32
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 20
        var d = load_f16_scale(w, bo)
        var m = load_f16_scale(w, bo + 2)
        var xo = blk * 32
        var sum_d: Float32 = 0.0
        var sum_m: Float32 = 0.0
        for j in range(16):
            var p = w[unsafe_offset=bo + 4 + j]
            var lo = Float32(p & 0x0F)
            var hi = Float32(p >> 4)
            var x_lo = x[unsafe_offset=xo + j]
            var x_hi = x[unsafe_offset=xo + j + 16]
            sum_d += lo * x_lo + hi * x_hi
            sum_m += x_lo + x_hi
        total += sum_d * d + sum_m * m
    return total


@always_inline
def dot_q5_0_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q5_0 dot product: 32 elements per 22 bytes [2B d, 4B qh, 16B ql]."""
    var bpr = nc // 32
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 22
        var d = load_f16_scale(w, bo)
        var qh = UInt32(w[unsafe_offset=bo + 2]) | (UInt32(w[unsafe_offset=bo + 3]) << 8) | (UInt32(w[unsafe_offset=bo + 4]) << 16) | (UInt32(w[unsafe_offset=bo + 5]) << 24)
        var xo = blk * 32
        var sum_i: Float32 = 0.0
        for j in range(16):
            var p = w[unsafe_offset=bo + 6 + j]
            var lo_h = Float32(((qh >> UInt32(j)) & 1) << 4)
            var hi_h = Float32(((qh >> UInt32(j + 16)) & 1) << 4)
            var lo = Float32(p & 0x0F) + lo_h - Float32(16)
            var hi = Float32(p >> 4) + hi_h - Float32(16)
            sum_i += lo * x[unsafe_offset=xo + j] + hi * x[unsafe_offset=xo + j + 16]
        total += sum_i * d
    return total


@always_inline
def dot_q8_0_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q8_0 dot product: 32 elements per 34 bytes [2B d, 32B int8]."""
    var bpr = nc // 32
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 34
        var d = load_f16_scale(w, bo)
        var xo = blk * 32
        var sum_i: Float32 = 0.0
        for j in range(32):
            var qs = bitcast[DType.int8](w[unsafe_offset=bo + 2 + j])
            sum_i += Float32(qs) * x[unsafe_offset=xo + j]
        total += sum_i * d
    return total


@always_inline
def dot_q4_k_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q4_K dot product (256-element superblock, 144 bytes). Bit-exact with llama.cpp with SIMD."""
    var bpr = nc // 256
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 144
        var d = load_f16_scale(w, bo)
        var dm = load_f16_scale(w, bo + 2)
        var scales_ptr = w.unsafe_offset(bo + 4) # 12 bytes scales
        var qs_ptr = w.unsafe_offset(bo + 16)     # 128 bytes qs (256 nibbles)
        var xo = blk * 256

        var is_idx = 0
        var q_off = 0
        for s in range(0, 256, 64):
            var sc1: UInt8 = 0
            var sc2: UInt8 = 0
            var m1: UInt8 = 0
            var m2: UInt8 = 0
            k4_scale(is_idx, scales_ptr, sc1, m1)
            k4_scale(is_idx + 1, scales_ptr, sc2, m2)

            var d1 = d * Float32(sc1)
            var mm1 = dm * Float32(m1)
            var d2 = d * Float32(sc2)
            var mm2 = dm * Float32(m2)

            var acc = SIMD[DType.float32, 8](0.0)
            var d1_v = SIMD[DType.float32, 8](d1)
            var mm1_v = SIMD[DType.float32, 8](mm1)
            var d2_v = SIMD[DType.float32, 8](d2)
            var mm2_v = SIMD[DType.float32, 8](mm2)

            for g in range(4):
                var j = g * 8
                var b_v = qs_ptr.unsafe_load[width=8](q_off + j)
                var lo_f32 = (b_v & UInt8(0x0F)).cast[DType.float32]()
                var hi_f32 = (b_v >> UInt8(4)).cast[DType.float32]()

                var x1_v = x.unsafe_load[width=8](xo + s + j)
                var x2_v = x.unsafe_load[width=8](xo + s + 32 + j)

                acc += (lo_f32 * d1_v - mm1_v) * x1_v + (hi_f32 * d2_v - mm2_v) * x2_v

            total += acc.reduce_add()
            is_idx += 2
            q_off += 32
    return total


@always_inline
def dot_q5_k_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q5_K dot product (256-element superblock, 176 bytes). Bit-exact with llama.cpp."""
    var bpr = nc // 256
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 176
        var d = load_f16_scale(w, bo)
        var dm = load_f16_scale(w, bo + 2)
        var scales_ptr = w.unsafe_offset(bo + 4) # 12 bytes
        var qh_ptr = w.unsafe_offset(bo + 16)     # 32 bytes
        var ql_ptr = w.unsafe_offset(bo + 48)     # 128 bytes
        var xo = blk * 256

        var is_idx = 0
        for s in range(0, 256, 64):
            var sc1: UInt8 = 0
            var sc2: UInt8 = 0
            var m1: UInt8 = 0
            var m2: UInt8 = 0
            k4_scale(is_idx, scales_ptr, sc1, m1)
            k4_scale(is_idx + 1, scales_ptr, sc2, m2)

            var d1 = d * Float32(sc1)
            var mm1 = dm * Float32(m1)
            var d2 = d * Float32(sc2)
            var mm2 = dm * Float32(m2)

            for j in range(32):
                var idx = s + j
                var lo1 = Float32(ql_ptr[unsafe_offset=idx // 2] >> UInt8((idx % 2) * 4) & UInt8(0x0F))
                var hi1 = Float32(((qh_ptr[unsafe_offset=idx // 8] >> UInt8(idx % 8)) & UInt8(1)) << UInt8(4))
                var val1 = (lo1 + hi1) * d1 - mm1
                total += val1 * x[unsafe_offset=xo + idx]

                var idx2 = s + 32 + j
                var lo2 = Float32(ql_ptr[unsafe_offset=idx2 // 2] >> UInt8((idx2 % 2) * 4) & UInt8(0x0F))
                var hi2 = Float32(((qh_ptr[unsafe_offset=idx2 // 8] >> UInt8(idx2 % 8)) & UInt8(1)) << UInt8(4))
                var val2 = (lo2 + hi2) * d2 - mm2
                total += val2 * x[unsafe_offset=xo + idx2]

            is_idx += 2
    return total


@always_inline
def dot_q6_k_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ Q6_K dot product (256-element superblock, 210 bytes). Bit-exact with llama.cpp with SIMD."""
    var bpr = nc // 256
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 210
        var ql_ptr = w.unsafe_offset(bo)         # 128 bytes
        var qh_ptr = w.unsafe_offset(bo + 128)   # 64 bytes
        var sc_ptr = w.unsafe_offset(bo + 192)   # 16 bytes
        var d = load_f16_scale(w, bo + 208)      # 2 bytes
        var xo = blk * 256

        var ql_off = 0
        var qh_off = 0
        var sc_off = 0
        for n in range(0, 256, 128):
            var acc1 = SIMD[DType.float32, 8](0.0)
            var acc2 = SIMD[DType.float32, 8](0.0)
            var acc3 = SIMD[DType.float32, 8](0.0)
            var acc4 = SIMD[DType.float32, 8](0.0)

            for g in range(4):
                var l = g * 8
                var is_idx = l // 16

                var s1 = Float32(Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 0]))) * d
                var s2 = Float32(Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 2]))) * d
                var s3 = Float32(Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 4]))) * d
                var s4 = Float32(Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 6]))) * d

                var ql0_v = ql_ptr.unsafe_load[width=8](ql_off + l)
                var ql32_v = ql_ptr.unsafe_load[width=8](ql_off + l + 32)
                var qh_v = qh_ptr.unsafe_load[width=8](qh_off + l)

                var q1_u8 = (ql0_v & UInt8(0x0F)) | (((qh_v >> UInt8(0)) & UInt8(3)) << UInt8(4))
                var q1_f32 = (q1_u8.cast[DType.int32]() - 32).cast[DType.float32]()

                var q2_u8 = (ql32_v & UInt8(0x0F)) | (((qh_v >> UInt8(2)) & UInt8(3)) << UInt8(4))
                var q2_f32 = (q2_u8.cast[DType.int32]() - 32).cast[DType.float32]()

                var q3_u8 = (ql0_v >> UInt8(4)) | (((qh_v >> UInt8(4)) & UInt8(3)) << UInt8(4))
                var q3_f32 = (q3_u8.cast[DType.int32]() - 32).cast[DType.float32]()

                var q4_u8 = (ql32_v >> UInt8(4)) | (((qh_v >> UInt8(6)) & UInt8(3)) << UInt8(4))
                var q4_f32 = (q4_u8.cast[DType.int32]() - 32).cast[DType.float32]()

                var x1_v = x.unsafe_load[width=8](xo + n + l + 0)
                var x2_v = x.unsafe_load[width=8](xo + n + l + 32)
                var x3_v = x.unsafe_load[width=8](xo + n + l + 64)
                var x4_v = x.unsafe_load[width=8](xo + n + l + 96)

                acc1 += q1_f32 * (x1_v * s1)
                acc2 += q2_f32 * (x2_v * s2)
                acc3 += q3_f32 * (x3_v * s3)
                acc4 += q4_f32 * (x4_v * s4)

            total += (acc1 + acc2 + acc3 + acc4).reduce_add()
            ql_off += 64
            qh_off += 32
            sc_off += 8
    return total


@always_inline
def dot_q6_k_q8_k_row(w: BytePtr, q8: BytePtr, nc: Int) -> Float32:
    """ Hardware-accelerated integer-domain SIMD dot product using AVX2 pmadd.ub.sw."""
    var nb = nc // 256
    var total: Float32 = 0.0
    var ones = SIMD[DType.int16, 16](1)

    for blk in range(nb):
        var bo = blk * 210
        var yo = blk * 292
        var d_w = load_f16_scale(w, bo + 208)
        var d_y = q8.unsafe_offset(yo).unsafe_bitcast[Float32]()[unsafe_offset=0]
        var d = d_w * d_y

        var ql_ptr = w.unsafe_offset(bo)
        var qh_ptr = w.unsafe_offset(bo + 128)
        var sc_ptr = w.unsafe_offset(bo + 192)
        var qs_ptr = q8.unsafe_offset(yo + 4)

        var sumi: Int32 = 0
        var ql_off = 0
        var qh_off = 0
        var sc_off = 0

        for n in range(0, 256, 128):
            for g in range(2):
                var l = g * 16
                var is_idx = l // 16

                var s1 = Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 0]))
                var s2 = Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 2]))
                var s3 = Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 4]))
                var s4 = Int32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 6]))

                var ql0_v = ql_ptr.unsafe_load[width=16](ql_off + l)
                var ql32_v = ql_ptr.unsafe_load[width=16](ql_off + l + 32)
                var qh_v = qh_ptr.unsafe_load[width=16](qh_off + l)

                var q1_u = (ql0_v & UInt8(0x0F)) | (((qh_v >> UInt8(0)) & UInt8(3)) << UInt8(4))
                var q2_u = (ql32_v & UInt8(0x0F)) | (((qh_v >> UInt8(2)) & UInt8(3)) << UInt8(4))
                var q3_u = (ql0_v >> UInt8(4)) | (((qh_v >> UInt8(4)) & UInt8(3)) << UInt8(4))
                var q4_u = (ql32_v >> UInt8(4)) | (((qh_v >> UInt8(6)) & UInt8(3)) << UInt8(4))

                var y1 = qs_ptr.unsafe_load[width=16](n + l).cast[DType.int8]()
                var y2 = qs_ptr.unsafe_load[width=16](n + l + 32).cast[DType.int8]()
                var y3 = qs_ptr.unsafe_load[width=16](n + l + 64).cast[DType.int8]()
                var y4 = qs_ptr.unsafe_load[width=16](n + l + 96).cast[DType.int8]()

                var u32_1 = q1_u.join(q2_u)
                var s32_1 = y1.join(y2)

                var p16_1 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](u32_1, s32_1)
                var p32_1 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_1, ones)

                var dot1 = p32_1.slice[4, offset=0]().reduce_add() - Int32(32) * y1.cast[DType.int32]().reduce_add()
                var dot2 = p32_1.slice[4, offset=4]().reduce_add() - Int32(32) * y2.cast[DType.int32]().reduce_add()

                sumi += dot1 * s1 + dot2 * s2

                var u32_2 = q3_u.join(q4_u)
                var s32_2 = y3.join(y4)

                var p16_2 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](u32_2, s32_2)
                var p32_2 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_2, ones)

                var dot3 = p32_2.slice[4, offset=0]().reduce_add() - Int32(32) * y3.cast[DType.int32]().reduce_add()
                var dot4 = p32_2.slice[4, offset=4]().reduce_add() - Int32(32) * y4.cast[DType.int32]().reduce_add()

                sumi += dot3 * s3 + dot4 * s4

            ql_off += 64
            qh_off += 32
            sc_off += 8

        total += Float32(sumi) * d
    return total

@always_inline
def dot_q4_k_q8_k_row(w: BytePtr, q8: BytePtr, nc: Int) -> Float32:
    """ Hardware-accelerated integer-domain SIMD dot product using AVX2 pmadd.ub.sw."""
    var bpr = nc // 256
    var total: Float32 = 0.0
    var ones = SIMD[DType.int16, 16](1)

    for blk in range(bpr):
        var bo = blk * 144
        var yo = blk * 292

        var d_w = load_f16_scale(w, bo)
        var dm_w = load_f16_scale(w, bo + 2)
        var d_y = q8.unsafe_offset(yo).unsafe_bitcast[Float32]()[unsafe_offset=0]

        var d = d_w * d_y
        var dm = dm_w * d_y

        var scales_ptr = w.unsafe_offset(bo + 4)
        var qs_w = w.unsafe_offset(bo + 16)
        var qs_y = q8.unsafe_offset(yo + 4)

        var is_idx = 0
        var q_off = 0
        var sumi: Float32 = 0.0

        for s in range(0, 256, 64):
            var sc1: UInt8 = 0
            var sc2: UInt8 = 0
            var m1: UInt8 = 0
            var m2: UInt8 = 0
            k4_scale(is_idx, scales_ptr, sc1, m1)
            k4_scale(is_idx + 1, scales_ptr, sc2, m2)

            var d1 = d * Float32(sc1)
            var mm1 = dm * Float32(m1)
            var d2 = d * Float32(sc2)
            var mm2 = dm * Float32(m2)

            for g in range(2):
                var j = g * 16
                var b_v = qs_w.unsafe_load[width=16](q_off + j)
                var lo_16 = (b_v & UInt8(0x0F)).cast[DType.uint8]()
                var hi_16 = (b_v >> UInt8(4)).cast[DType.uint8]()

                var y1_16 = qs_y.unsafe_load[width=16](s + j).cast[DType.int8]()
                var y2_16 = qs_y.unsafe_load[width=16](s + 32 + j).cast[DType.int8]()

                var lo_u8 = lo_16.join(SIMD[DType.uint8, 16](0))
                var y1 = y1_16.join(SIMD[DType.int8, 16](0))
                var hi_u8 = hi_16.join(SIMD[DType.uint8, 16](0))
                var y2 = y2_16.join(SIMD[DType.int8, 16](0))

                var p16_1 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](lo_u8, y1)
                var p16_2 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](hi_u8, y2)

                var p32_1 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_1, ones)
                var p32_2 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_2, ones)

                var dot1 = p32_1.slice[4, offset=0]().reduce_add()
                var sum_y1 = y1_16.cast[DType.int32]().reduce_add()

                var dot2 = p32_2.slice[4, offset=0]().reduce_add()
                var sum_y2 = y2_16.cast[DType.int32]().reduce_add()

                sumi += d1 * Float32(dot1) - mm1 * Float32(sum_y1) + d2 * Float32(dot2) - mm2 * Float32(sum_y2)

            is_idx += 2
            q_off += 32

        total += sumi
    return total

@always_inline
def dot_q6_k_q8_k_2rows(
        w0: BytePtr,
        w1: BytePtr,
        q8: BytePtr,
        nc: Int,
        mut res0: Float32,
        mut res1: Float32,
    ):
    """ Tiled 2-row hardware-accelerated integer-domain SIMD dot product using AVX2 pmadd.ub.sw."""
    var nb = nc // 256
    var total0: Float32 = 0.0
    var total1: Float32 = 0.0
    var ones = SIMD[DType.int16, 16](1)

    for blk in range(nb):
        var bo = blk * 210
        var yo = blk * 292

        var d_y = q8.unsafe_offset(yo).unsafe_bitcast[Float32]()[unsafe_offset=0]
        var d0 = load_f16_scale(w0, bo + 208) * d_y
        var d1 = load_f16_scale(w1, bo + 208) * d_y

        var ql0_ptr = w0.unsafe_offset(bo)
        var qh0_ptr = w0.unsafe_offset(bo + 128)
        var sc0_ptr = w0.unsafe_offset(bo + 192)

        var ql1_ptr = w1.unsafe_offset(bo)
        var qh1_ptr = w1.unsafe_offset(bo + 128)
        var sc1_ptr = w1.unsafe_offset(bo + 192)

        var qs_ptr = q8.unsafe_offset(yo + 4)

        var sumi0: Int32 = 0
        var sumi1: Int32 = 0
        var ql_off = 0
        var qh_off = 0
        var sc_off = 0

        for n in range(0, 256, 128):
            var y1 = qs_ptr.unsafe_load[width=16](n + 0).cast[DType.int8]()
            var y2 = qs_ptr.unsafe_load[width=16](n + 32).cast[DType.int8]()
            var y3 = qs_ptr.unsafe_load[width=16](n + 64).cast[DType.int8]()
            var y4 = qs_ptr.unsafe_load[width=16](n + 96).cast[DType.int8]()

            var y1_2 = qs_ptr.unsafe_load[width=16](n + 16).cast[DType.int8]()
            var y2_2 = qs_ptr.unsafe_load[width=16](n + 48).cast[DType.int8]()
            var y3_2 = qs_ptr.unsafe_load[width=16](n + 80).cast[DType.int8]()
            var y4_2 = qs_ptr.unsafe_load[width=16](n + 112).cast[DType.int8]()

            for g in range(2):
                var l = g * 16
                var is_idx = l // 16

                var y_a = y1 if g == 0 else y1_2
                var y_b = y2 if g == 0 else y2_2
                var y_c = y3 if g == 0 else y3_2
                var y_d = y4 if g == 0 else y4_2

                var s32_1 = y_a.join(y_b)
                var s32_2 = y_c.join(y_d)

                var sum_ya = y_a.cast[DType.int32]().reduce_add()
                var sum_yb = y_b.cast[DType.int32]().reduce_add()
                var sum_yc = y_c.cast[DType.int32]().reduce_add()
                var sum_yd = y_d.cast[DType.int32]().reduce_add()

                # Row 0
                var s1_0 = Int32(bitcast[DType.int8](sc0_ptr[unsafe_offset=sc_off + is_idx + 0]))
                var s2_0 = Int32(bitcast[DType.int8](sc0_ptr[unsafe_offset=sc_off + is_idx + 2]))
                var s3_0 = Int32(bitcast[DType.int8](sc0_ptr[unsafe_offset=sc_off + is_idx + 4]))
                var s4_0 = Int32(bitcast[DType.int8](sc0_ptr[unsafe_offset=sc_off + is_idx + 6]))

                var ql0_v0 = ql0_ptr.unsafe_load[width=16](ql_off + l)
                var ql32_v0 = ql0_ptr.unsafe_load[width=16](ql_off + l + 32)
                var qh_v0 = qh0_ptr.unsafe_load[width=16](qh_off + l)

                var q1_u0 = (ql0_v0 & UInt8(0x0F)) | (((qh_v0 >> UInt8(0)) & UInt8(3)) << UInt8(4))
                var q2_u0 = (ql32_v0 & UInt8(0x0F)) | (((qh_v0 >> UInt8(2)) & UInt8(3)) << UInt8(4))
                var q3_u0 = (ql0_v0 >> UInt8(4)) | (((qh_v0 >> UInt8(4)) & UInt8(3)) << UInt8(4))
                var q4_u0 = (ql32_v0 >> UInt8(4)) | (((qh_v0 >> UInt8(6)) & UInt8(3)) << UInt8(4))

                var u32_1_0 = q1_u0.join(q2_u0)
                var u32_2_0 = q3_u0.join(q4_u0)

                var p16_1_0 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](u32_1_0, s32_1)
                var p16_2_0 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](u32_2_0, s32_2)

                var p32_1_0 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_1_0, ones)
                var p32_2_0 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_2_0, ones)

                var dot1_0 = p32_1_0.slice[4, offset=0]().reduce_add() - Int32(32) * sum_ya
                var dot2_0 = p32_1_0.slice[4, offset=4]().reduce_add() - Int32(32) * sum_yb
                var dot3_0 = p32_2_0.slice[4, offset=0]().reduce_add() - Int32(32) * sum_yc
                var dot4_0 = p32_2_0.slice[4, offset=4]().reduce_add() - Int32(32) * sum_yd

                sumi0 += dot1_0 * s1_0 + dot2_0 * s2_0 + dot3_0 * s3_0 + dot4_0 * s4_0

                # Row 1
                var s1_1 = Int32(bitcast[DType.int8](sc1_ptr[unsafe_offset=sc_off + is_idx + 0]))
                var s2_1 = Int32(bitcast[DType.int8](sc1_ptr[unsafe_offset=sc_off + is_idx + 2]))
                var s3_1 = Int32(bitcast[DType.int8](sc1_ptr[unsafe_offset=sc_off + is_idx + 4]))
                var s4_1 = Int32(bitcast[DType.int8](sc1_ptr[unsafe_offset=sc_off + is_idx + 6]))

                var ql0_v1 = ql1_ptr.unsafe_load[width=16](ql_off + l)
                var ql32_v1 = ql1_ptr.unsafe_load[width=16](ql_off + l + 32)
                var qh_v1 = qh1_ptr.unsafe_load[width=16](qh_off + l)

                var q1_u1 = (ql0_v1 & UInt8(0x0F)) | (((qh_v1 >> UInt8(0)) & UInt8(3)) << UInt8(4))
                var q2_u1 = (ql32_v1 & UInt8(0x0F)) | (((qh_v1 >> UInt8(2)) & UInt8(3)) << UInt8(4))
                var q3_u1 = (ql0_v1 >> UInt8(4)) | (((qh_v1 >> UInt8(4)) & UInt8(3)) << UInt8(4))
                var q4_u1 = (ql32_v1 >> UInt8(4)) | (((qh_v1 >> UInt8(6)) & UInt8(3)) << UInt8(4))

                var u32_1_1 = q1_u1.join(q2_u1)
                var u32_2_1 = q3_u1.join(q4_u1)

                var p16_1_1 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](u32_1_1, s32_1)
                var p16_2_1 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](u32_2_1, s32_2)

                var p32_1_1 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_1_1, ones)
                var p32_2_1 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_2_1, ones)

                var dot1_1 = p32_1_1.slice[4, offset=0]().reduce_add() - Int32(32) * sum_ya
                var dot2_1 = p32_1_1.slice[4, offset=4]().reduce_add() - Int32(32) * sum_yb
                var dot3_1 = p32_2_1.slice[4, offset=0]().reduce_add() - Int32(32) * sum_yc
                var dot4_1 = p32_2_1.slice[4, offset=4]().reduce_add() - Int32(32) * sum_yd

                sumi1 += dot1_1 * s1_1 + dot2_1 * s2_1 + dot3_1 * s3_1 + dot4_1 * s4_1

            ql_off += 64
            qh_off += 32
            sc_off += 8

        total0 += Float32(sumi0) * d0
        total1 += Float32(sumi1) * d1

    res0 = total0
    res1 = total1

@always_inline
def dot_q4_k_q8_k_2rows(
        w0: BytePtr,
        w1: BytePtr,
        q8: BytePtr,
        nc: Int,
        mut res0: Float32,
        mut res1: Float32,
    ):
    """ Tiled 2-row hardware-accelerated integer-domain SIMD dot product using AVX2 pmadd.ub.sw."""
    var bpr = nc // 256
    var total0: Float32 = 0.0
    var total1: Float32 = 0.0
    var ones = SIMD[DType.int16, 16](1)

    for blk in range(bpr):
        var bo = blk * 144
        var yo = blk * 292

        var d_y = q8.unsafe_offset(yo).unsafe_bitcast[Float32]()[unsafe_offset=0]

        var d0 = load_f16_scale(w0, bo) * d_y
        var dm0 = load_f16_scale(w0, bo + 2) * d_y

        var d1 = load_f16_scale(w1, bo) * d_y
        var dm1 = load_f16_scale(w1, bo + 2) * d_y

        var scales_ptr0 = w0.unsafe_offset(bo + 4)
        var qs_w0 = w0.unsafe_offset(bo + 16)

        var scales_ptr1 = w1.unsafe_offset(bo + 4)
        var qs_w1 = w1.unsafe_offset(bo + 16)

        var qs_y = q8.unsafe_offset(yo + 4)

        var is_idx = 0
        var q_off = 0
        var sumi0: Float32 = 0.0
        var sumi1: Float32 = 0.0

        for s in range(0, 256, 64):
            var sc1_0: UInt8 = 0; var sc2_0: UInt8 = 0
            var m1_0: UInt8 = 0; var m2_0: UInt8 = 0
            k4_scale(is_idx, scales_ptr0, sc1_0, m1_0)
            k4_scale(is_idx + 1, scales_ptr0, sc2_0, m2_0)

            var d1_0 = d0 * Float32(sc1_0)
            var mm1_0 = dm0 * Float32(m1_0)
            var d2_0 = d0 * Float32(sc2_0)
            var mm2_0 = dm0 * Float32(m2_0)

            var sc1_1: UInt8 = 0; var sc2_1: UInt8 = 0
            var m1_1: UInt8 = 0; var m2_1: UInt8 = 0
            k4_scale(is_idx, scales_ptr1, sc1_1, m1_1)
            k4_scale(is_idx + 1, scales_ptr1, sc2_1, m2_1)

            var d1_1 = d1 * Float32(sc1_1)
            var mm1_1 = dm1 * Float32(m1_1)
            var d2_1 = d1 * Float32(sc2_1)
            var mm2_1 = dm1 * Float32(m2_1)

            for g in range(2):
                var j = g * 16
                var b_v0 = qs_w0.unsafe_load[width=16](q_off + j)
                var lo_16_0 = (b_v0 & UInt8(0x0F)).cast[DType.uint8]()
                var hi_16_0 = (b_v0 >> UInt8(4)).cast[DType.uint8]()

                var b_v1 = qs_w1.unsafe_load[width=16](q_off + j)
                var lo_16_1 = (b_v1 & UInt8(0x0F)).cast[DType.uint8]()
                var hi_16_1 = (b_v1 >> UInt8(4)).cast[DType.uint8]()

                var y1_16 = qs_y.unsafe_load[width=16](s + j).cast[DType.int8]()
                var y2_16 = qs_y.unsafe_load[width=16](s + 32 + j).cast[DType.int8]()

                var y1 = y1_16.join(SIMD[DType.int8, 16](0))
                var y2 = y2_16.join(SIMD[DType.int8, 16](0))
                var sum_y1 = y1_16.cast[DType.int32]().reduce_add()
                var sum_y2 = y2_16.cast[DType.int32]().reduce_add()

                # Row 0
                var p16_1_0 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](lo_16_0.join(SIMD[DType.uint8, 16](0)), y1)
                var p16_2_0 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](hi_16_0.join(SIMD[DType.uint8, 16](0)), y2)
                var p32_1_0 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_1_0, ones)
                var p32_2_0 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_2_0, ones)
                var dot1_0 = p32_1_0.slice[4, offset=0]().reduce_add()
                var dot2_0 = p32_2_0.slice[4, offset=0]().reduce_add()

                sumi0 += d1_0 * Float32(dot1_0) - mm1_0 * Float32(sum_y1) + d2_0 * Float32(dot2_0) - mm2_0 * Float32(sum_y2)

                # Row 1
                var p16_1_1 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](lo_16_1.join(SIMD[DType.uint8, 16](0)), y1)
                var p16_2_1 = external_call["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](hi_16_1.join(SIMD[DType.uint8, 16](0)), y2)
                var p32_1_1 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_1_1, ones)
                var p32_2_1 = external_call["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](p16_2_1, ones)
                var dot1_1 = p32_1_1.slice[4, offset=0]().reduce_add()
                var dot2_1 = p32_2_1.slice[4, offset=0]().reduce_add()

                sumi1 += d1_1 * Float32(dot1_1) - mm1_1 * Float32(sum_y1) + d2_1 * Float32(dot2_1) - mm2_1 * Float32(sum_y2)

            is_idx += 2
            q_off += 32

        total0 += sumi0
        total1 += sumi1

    res0 = total0
    res1 = total1

@always_inline
def dequantize_row(dtype: UInt32, weight_row_ptr: BytePtr, dst: FloatPtr, nc: Int):
    """ Dequantize a single row into dst Float32 buffer."""
    if dtype == GGMLType.F32:
        var src_f32 = weight_row_ptr.unsafe_bitcast[Float32]()
        for i in range(nc):
            dst[unsafe_offset=i] = src_f32[unsafe_offset=i]
    elif dtype == GGMLType.F16:
        for i in range(nc):
            dst[unsafe_offset=i] = f16_to_f32(weight_row_ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i])
    elif dtype == GGMLType.BF16:
        for i in range(nc):
            dst[unsafe_offset=i] = bf16_to_f32(weight_row_ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i])
    elif dtype == GGMLType.Q8_0:
        var bpr = nc // 32
        for blk in range(bpr):
            var bo = blk * 34
            var d = load_f16_scale(weight_row_ptr, bo)
            var xo = blk * 32
            for j in range(32):
                var qs = bitcast[DType.int8](weight_row_ptr[unsafe_offset=bo + 2 + j])
                dst[unsafe_offset=xo + j] = Float32(qs) * d
    elif dtype == GGMLType.Q4_0:
        var bpr = nc // 32
        for blk in range(bpr):
            var bo = blk * 18
            var d = load_f16_scale(weight_row_ptr, bo)
            var xo = blk * 32
            for j in range(16):
                var p = weight_row_ptr[unsafe_offset=bo + 2 + j]
                var lo = Float32(Int32(p & 0x0F) - 8)
                var hi = Float32(Int32(p >> 4) - 8)
                dst[unsafe_offset=xo + j] = lo * d
                dst[unsafe_offset=xo + j + 16] = hi * d
    elif dtype == GGMLType.Q4_K:
        var bpr = nc // 256
        for blk in range(bpr):
            var bo = blk * 144
            var d = load_f16_scale(weight_row_ptr, bo)
            var dm = load_f16_scale(weight_row_ptr, bo + 2)
            var scales_ptr = weight_row_ptr.unsafe_offset(bo + 4)
            var qs_ptr = weight_row_ptr.unsafe_offset(bo + 16)
            var xo = blk * 256
            var is_idx = 0
            var q_off = 0
            for s in range(0, 256, 64):
                var sc1: UInt8 = 0
                var sc2: UInt8 = 0
                var m1: UInt8 = 0
                var m2: UInt8 = 0
                k4_scale(is_idx, scales_ptr, sc1, m1)
                k4_scale(is_idx + 1, scales_ptr, sc2, m2)
                var d1 = d * Float32(sc1)
                var dm1 = dm * Float32(m1)
                var d2 = d * Float32(sc2)
                var dm2 = dm * Float32(m2)
                for j in range(32):
                    var p = qs_ptr[unsafe_offset=q_off + j]
                    var q1 = Float32(p & 0x0F)
                    var q2 = Float32(p >> 4)
                    dst[unsafe_offset=xo + s + j] = d1 * q1 - dm1
                    dst[unsafe_offset=xo + s + 32 + j] = d2 * q2 - dm2
                is_idx += 2
                q_off += 32
    elif dtype == GGMLType.Q6_K:
        var bpr = nc // 256
        for blk in range(bpr):
            var bo = blk * 210
            var ql_ptr = weight_row_ptr.unsafe_offset(bo)
            var qh_ptr = weight_row_ptr.unsafe_offset(bo + 128)
            var sc_ptr = weight_row_ptr.unsafe_offset(bo + 192)
            var d = load_f16_scale(weight_row_ptr, bo + 208)
            var xo = blk * 256

            var ql_off = 0
            var qh_off = 0
            var sc_off = 0
            for n in range(0, 256, 128):
                for l in range(32):
                    var is_idx = l // 16
                    var q1 = Float32(Int32((ql_ptr[unsafe_offset=ql_off + l + 0] & 0x0F) | (((qh_ptr[unsafe_offset=qh_off + l] >> 0) & 3) << 4)) - 32)
                    var q2 = Float32(Int32((ql_ptr[unsafe_offset=ql_off + l + 32] & 0x0F) | (((qh_ptr[unsafe_offset=qh_off + l] >> 2) & 3) << 4)) - 32)
                    var q3 = Float32(Int32((ql_ptr[unsafe_offset=ql_off + l + 0] >> 4) | (((qh_ptr[unsafe_offset=qh_off + l] >> 4) & 3) << 4)) - 32)
                    var q4 = Float32(Int32((ql_ptr[unsafe_offset=ql_off + l + 32] >> 4) | (((qh_ptr[unsafe_offset=qh_off + l] >> 6) & 3) << 4)) - 32)

                    var s1 = Float32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 0])) * d
                    var s2 = Float32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 2])) * d
                    var s3 = Float32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 4])) * d
                    var s4 = Float32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 6])) * d

                    dst[unsafe_offset=xo + n + l + 0] = s1 * q1
                    dst[unsafe_offset=xo + n + l + 32] = s2 * q2
                    dst[unsafe_offset=xo + n + l + 64] = s3 * q3
                    dst[unsafe_offset=xo + n + l + 96] = s4 * q4

                ql_off += 64
                qh_off += 32
                sc_off += 8
    else:
        for i in range(nc):
            dst[unsafe_offset=i] = Float32(0.0)


@always_inline
def dot_mxfp4_row(w: BytePtr, x: FloatPtr, nc: Int) -> Float32:
    """ MXFP4 dot product: 32 elements per 17 bytes [16B nibbles, 1B E8M0 scale]."""
    var bpr = nc // 32
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = blk * 17
        var eb = w[unsafe_offset=bo + 16]
        if eb == 0:
            continue
        var ei = Int(eb) - 127
        var sf: Float32 = 1.0
        if ei >= 0:
            for _ in range(ei):
                sf *= 2.0
        else:
            for _ in range(-ei):
                sf *= 0.5
        if sf > 1e20:
            sf = 1e20

        var xo = blk * 32
        var sum_i: Float32 = 0.0
        for j in range(16):
            var p = w[unsafe_offset=bo + j]
            var lo = Int32(p & 0x0F)
            if lo > 7:
                lo -= 16
            var hi = Int32(p >> 4)
            if hi > 7:
                hi -= 16
            sum_i += Float32(lo) * x[unsafe_offset=xo + j * 2] + Float32(hi) * x[unsafe_offset=xo + j * 2 + 1]
        total += sum_i * sf
    return total


# ─── Multi-Format Dispatcher for a Single Row ────────────────────────────────

@always_inline
def dot_quant_row(
        dtype: UInt32,
        weight_row_ptr: BytePtr,
        x: FloatPtr,
        nc: Int,
    ) -> Float32:
    """ Compute dot product between one row of quantized weights and Float32 input."""
    if dtype == GGMLType.F32:
        return dot_f32_row(weight_row_ptr.unsafe_bitcast[Float32](), x, nc)
    elif dtype == GGMLType.F16:
        return dot_f16_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.BF16:
        return dot_bf16_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q4_0:
        return dot_q4_0_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q4_1:
        return dot_q4_1_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q5_0:
        return dot_q5_0_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q8_0:
        return dot_q8_0_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q4_K:
        return dot_q4_k_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q5_K:
        return dot_q5_k_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.Q6_K:
        return dot_q6_k_row(weight_row_ptr, x, nc)
    elif dtype == GGMLType.MXFP4:
        return dot_mxfp4_row(weight_row_ptr, x, nc)
    else:
        # Fallback to F32
        return dot_f32_row(weight_row_ptr.unsafe_bitcast[Float32](), x, nc)

# ─── Horizontal Multi-Matrix Fusion (QKV n=3, Gate/Up n=2) ───────────────────

@always_inline
def matmul_quant_fused[ n: Int, ](
        dtype: UInt32,
        C: StaticTuple[FloatPtr, n],
        x: FloatPtr,
        W: StaticTuple[BytePtr, n],
        row_start: Int,
        row_end: Int,
        nc: Int,
    ):
    """ Fused multi-matrix projection across worker row range [row_start, row_end).\n    
    Shares input vector `x` loads across registers for `n` projections simultaneously.
    Used for QKV (n=3) and SwiGLU Gate/Up (n=2).
    """
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var bytes_per_row = (nc // blk_sz) * type_sz

    for r in range(row_start, row_end):
        comptime for k in range(n):
            var row_w_ptr = W[k].unsafe_offset(r * bytes_per_row)
            var res = dot_quant_row(dtype, row_w_ptr, x, nc)
            C[k][unsafe_offset=r] = res


@always_inline
def matmul_quant_fused_biased[ n: Int, ](
        dtype: UInt32,
        C: StaticTuple[FloatPtr, n],
        x: FloatPtr,
        W: StaticTuple[BytePtr, n],
        B: StaticTuple[FloatPtr, n],
        row_start: Int,
        row_end: Int,
        nc: Int,
    ):
    """ Fused multi-matrix projection with optional per-projection bias vector `B[k]`.
    If `Int(B[k]) != 0`, adds `B[k][r]` in-register before writing to `C[k][r]`.
    Ideal for Whisper QKV (where Q and V have bias and K has null bias).
    """
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var bytes_per_row = (nc // blk_sz) * type_sz

    for r in range(row_start, row_end):
        comptime for k in range(n):
            var row_w_ptr = W[k].unsafe_offset(r * bytes_per_row)
            var res = dot_quant_row(dtype, row_w_ptr, x, nc)
            if Int(B[k]) != 0:
                res += B[k][unsafe_offset=r]
            C[k][unsafe_offset=r] = res


@always_inline
def gemv_quant_biased(
        dst: FloatPtr,
        w: BytePtr,
        bias: FloatPtr,
        dtype: UInt32,
        x: FloatPtr,
        rows: Int,
        cols: Int,
        workers: Int = 0,
    ):
    """ Perform matrix-vector product with bias `dst = w @ x + bias` using multi-threaded row chunking.
    Supports nullable `bias` (if `Int(bias) == 0`, behaves identically to `gemv_quant`).
    """
    gemv_quant(dst, w, dtype, x, rows, cols, workers)
    if Int(bias) != 0:
        var i = 0
        while i + SIMD_WIDTH <= rows:
            var dv = dst.unsafe_load[width=SIMD_WIDTH](i)
            var bv = bias.unsafe_load[width=SIMD_WIDTH](i)
            dst.unsafe_store[width=SIMD_WIDTH](i, dv + bv)
            i += SIMD_WIDTH
        while i < rows:
            dst[unsafe_offset=i] += bias[unsafe_offset=i]
            i += 1


@always_inline
def gemv_gelu_biased(
        dst: FloatPtr,
        w: BytePtr,
        bias: FloatPtr,
        dtype: UInt32,
        x: FloatPtr,
        rows: Int,
        cols: Int,
        workers: Int = 0,
    ):
    """ Fused linear projection + bias + GELU activation (`dst = gelu(w @ x + bias)`),
    used in Whisper Encoder & Decoder MLP first layer (`mlp.0`).
    """
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var bpr = (cols // blk_sz) * type_sz
    var has_bias = Int(bias) != 0

    var n_workers = workers if workers > 0 else max(1, num_logical_cores())
    if n_workers <= 1 or rows < 512:
        for r in range(rows):
            var w_row = w.unsafe_offset(r * bpr)
            var val = dot_quant_row(dtype, w_row, x, cols)
            if has_bias:
                val += bias[unsafe_offset=r]
            dst[unsafe_offset=r] = gelu(val)
        return

    var chunk = (rows + n_workers - 1) // n_workers

    def worker_fn(wid: Int) {imm}:
        var r_start = wid * chunk
        var r_end = min((wid + 1) * chunk, rows)
        for r in range(r_start, r_end):
            var w_row = w.unsafe_offset(r * bpr)
            var val = dot_quant_row(dtype, w_row, x, cols)
            if has_bias:
                val += bias[unsafe_offset=r]
            dst[unsafe_offset=r] = gelu(val)

    parallelize(worker_fn, n_workers, n_workers)

# ─── Parallel Quantized GEMV ──────────────────────────────────────────────────

@always_inline
def gemv_quant(
        dst: FloatPtr,
        w: BytePtr,
        dtype: UInt32,
        x: FloatPtr,
        rows: Int,
        cols: Int,
        workers: Int = 0,
    ):
    """ Perform matrix-vector product `dst = w @ x` with multi-threaded row chunking."""
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var bpr = (cols // blk_sz) * type_sz

    var n_workers = workers if workers > 0 else max(1, num_logical_cores())
    if dtype == GGMLType.Q6_K and (cols % 256 == 0):
        var q8_bytes = (cols // 256) * 292
        var q8_buf = unsafe_alloc[UInt8](q8_bytes)
        quantize_row_q8_k(x, q8_buf, cols)

        if n_workers <= 1 or rows < 512:
            var r = 0
            while r + 1 < rows:
                var w_row0 = w.unsafe_offset(r * bpr)
                var w_row1 = w.unsafe_offset((r + 1) * bpr)
                dot_q6_k_q8_k_2rows(w_row0, w_row1, q8_buf, cols, dst[unsafe_offset=r], dst[unsafe_offset=r + 1])
                r += 2
            if r < rows:
                var w_row = w.unsafe_offset(r * bpr)
                dst[unsafe_offset=r] = dot_q6_k_q8_k_row(w_row, q8_buf, cols)
            q8_buf.unsafe_free()
            return

        var chunk = (((rows + n_workers - 1) // n_workers) + 1) & ~1

        def worker_fn_q8(wid: Int) {imm}:
            var r_start = wid * chunk
            var r_end = min((wid + 1) * chunk, rows)
            var r = r_start
            while r + 1 < r_end:
                var w_row0 = w.unsafe_offset(r * bpr)
                var w_row1 = w.unsafe_offset((r + 1) * bpr)
                dot_q6_k_q8_k_2rows(w_row0, w_row1, q8_buf, cols, dst[unsafe_offset=r], dst[unsafe_offset=r + 1])
                r += 2
            if r < r_end:
                var w_row = w.unsafe_offset(r * bpr)
                dst[unsafe_offset=r] = dot_q6_k_q8_k_row(w_row, q8_buf, cols)

        parallelize(worker_fn_q8, n_workers, n_workers)
        q8_buf.unsafe_free()
        return

    if dtype == GGMLType.Q4_K and (cols % 256 == 0):
        var q8_bytes = (cols // 256) * 292
        var q8_buf = unsafe_alloc[UInt8](q8_bytes)
        quantize_row_q8_k(x, q8_buf, cols)

        if n_workers <= 1 or rows < 512:
            var r = 0
            while r + 1 < rows:
                var w_row0 = w.unsafe_offset(r * bpr)
                var w_row1 = w.unsafe_offset((r + 1) * bpr)
                dot_q4_k_q8_k_2rows(w_row0, w_row1, q8_buf, cols, dst[unsafe_offset=r], dst[unsafe_offset=r + 1])
                r += 2
            if r < rows:
                var w_row = w.unsafe_offset(r * bpr)
                dst[unsafe_offset=r] = dot_q4_k_q8_k_row(w_row, q8_buf, cols)
            q8_buf.unsafe_free()
            return

        var chunk = (((rows + n_workers - 1) // n_workers) + 1) & ~1

        def worker_fn_q4(wid: Int) {imm}:
            var r_start = wid * chunk
            var r_end = min((wid + 1) * chunk, rows)
            var r = r_start
            while r + 1 < r_end:
                var w_row0 = w.unsafe_offset(r * bpr)
                var w_row1 = w.unsafe_offset((r + 1) * bpr)
                dot_q4_k_q8_k_2rows(w_row0, w_row1, q8_buf, cols, dst[unsafe_offset=r], dst[unsafe_offset=r + 1])
                r += 2
            if r < r_end:
                var w_row = w.unsafe_offset(r * bpr)
                dst[unsafe_offset=r] = dot_q4_k_q8_k_row(w_row, q8_buf, cols)

        parallelize(worker_fn_q4, n_workers, n_workers)
        q8_buf.unsafe_free()
        return

    if n_workers <= 1 or rows < 512:
        for r in range(rows):
            var w_row = w.unsafe_offset(r * bpr)
            dst[unsafe_offset=r] = dot_quant_row(dtype, w_row, x, cols)
        return

    var chunk = (rows + n_workers - 1) // n_workers

    def worker_fn(wid: Int) {imm}:
        var r_start = wid * chunk
        var r_end = min((wid + 1) * chunk, rows)
        for r in range(r_start, r_end):
            var w_row = w.unsafe_offset(r * bpr)
            dst[unsafe_offset=r] = dot_quant_row(dtype, w_row, x, cols)

    parallelize(worker_fn, n_workers, n_workers)

@always_inline
def gemv_gate_up_fused(
        hb: FloatPtr,
        w_gate: BytePtr,
        dtype_gate: UInt32,
        w_up: BytePtr,
        dtype_up: UInt32,
        xb: FloatPtr,
        hidden_dim: Int,
        dim: Int,
        workers: Int = 0,
    ):
    """ Fused Gate and Up GEMVs with inline fused GELU activation in a single pass."""
    var blk_sz_g = Int(ggml_block_size(dtype_gate))
    var type_sz_g = Int(ggml_type_size(dtype_gate))
    var bpr_g = (dim // blk_sz_g) * type_sz_g

    var blk_sz_u = Int(ggml_block_size(dtype_up))
    var type_sz_u = Int(ggml_type_size(dtype_up))
    var bpr_u = (dim // blk_sz_u) * type_sz_u

    var n_workers = workers if workers > 0 else max(1, num_logical_cores())
    if n_workers <= 1 or hidden_dim < 512:
        for r in range(hidden_dim):
            var w_g_row = w_gate.unsafe_offset(r * bpr_g)
            var w_u_row = w_up.unsafe_offset(r * bpr_u)
            var g = dot_quant_row(dtype_gate, w_g_row, xb, dim)
            var u = dot_quant_row(dtype_up, w_u_row, xb, dim)
            hb[unsafe_offset=r] = gelu(g) * u
        return

    var chunk = (hidden_dim + n_workers - 1) // n_workers

    def worker_fn(wid: Int) {imm}:
        var r_start = wid * chunk
        var r_end = min((wid + 1) * chunk, hidden_dim)
        for r in range(r_start, r_end):
            var w_g_row = w_gate.unsafe_offset(r * bpr_g)
            var w_u_row = w_up.unsafe_offset(r * bpr_u)
            var g = dot_quant_row(dtype_gate, w_g_row, xb, dim)
            var u = dot_quant_row(dtype_up, w_u_row, xb, dim)
            hb[unsafe_offset=r] = gelu(g) * u

    parallelize(worker_fn, n_workers, n_workers)
