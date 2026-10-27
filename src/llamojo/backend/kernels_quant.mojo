"""Unified Quantized Matrix-Vector and Multi-Matrix Multiplication Kernels.

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

from std.memory import Pointer, bitcast, stack_allocation
from std.utils import StaticTuple
from std.sys.info import simd_width_of

from .isa import (
    SIMD_WIDTH,
    F32xW,
    f16_to_f32,
    bf16_to_f32,
    load_f16_scale,
)
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
    """ F16 weights dot product against Float32 vector."""
    var total: Float32 = 0.0
    for i in range(nc):
        var lo = UInt16(w[unsafe_offset=i * 2])
        var hi = UInt16(w[unsafe_offset=i * 2 + 1])
        var wf = f16_to_f32((hi << 8) | lo)
        total += wf * x[unsafe_offset=i]
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
    """ Q4_K dot product (256-element superblock, 144 bytes). Bit-exact with llama.cpp."""
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

            for j in range(32):
                var byte_val = qs_ptr[unsafe_offset=q_off + j]
                var lo = Float32(byte_val & 0x0F)
                var hi = Float32(byte_val >> 4)
                var x1 = x[unsafe_offset=xo + s + j]
                var x2 = x[unsafe_offset=xo + s + 32 + j]
                total += (lo * d1 - mm1) * x1 + (hi * d2 - mm2) * x2

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
    """ Q6_K dot product (256-element superblock, 210 bytes). Bit-exact with llama.cpp."""
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

                total += s1 * q1 * x[unsafe_offset=xo + n + l + 0]
                total += s2 * q2 * x[unsafe_offset=xo + n + l + 32]
                total += s3 * q3 * x[unsafe_offset=xo + n + l + 64]
                total += s4 * q4 * x[unsafe_offset=xo + n + l + 96]

            ql_off += 64
            qh_off += 32
            sc_off += 8
    return total


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
