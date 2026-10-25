""" ISA & SIMD Dispatch for llamojo (Mojo 1.1.0).

Ported from MojoLlama's `kernels/isa_dispatch.mojo` to Mojo 1.1.0.
Provides compile-time target ISA detection (AVX-512, AVX2, FMA, NEON),
parametric SIMD widths, and hardware-accelerated F16/BF16 -> F32 conversion.
"""

from std.sys.info import CompilationTarget, simd_width_of, num_performance_cores
from std.memory import Pointer, bitcast
from std.collections import List

# ─── ISA Feature Detection ───────────────────────────────────────────────────

@always_inline
def has_avx512f() -> Bool:
    """Check if target supports AVX-512F (foundation)."""
    return CompilationTarget.has_avx512f()

@always_inline
def has_avx512vnni() -> Bool:
    """Check if target supports AVX-512-VNNI (int8 dot product)."""
    return CompilationTarget.has_vnni()

@always_inline
def has_avx2() -> Bool:
    """Check if target supports AVX2."""
    return CompilationTarget.has_avx2()

@always_inline
def has_fma() -> Bool:
    """Check if target supports FMA."""
    return CompilationTarget.has_fma()

# ─── SIMD Width Selection ────────────────────────────────────────────────────

comptime SIMD_WIDTH_AVX2: Int = 8      # 256 bits / 32 = 8 floats
comptime SIMD_WIDTH_AVX512: Int = 16   # 512 bits / 32 = 16 floats
comptime SIMD_WIDTH_NEON: Int = 4      # 128 bits / 32 = 4 floats (ARM)

def get_simd_width() -> Int:
    """Select hardware SIMD width in Float32 lanes."""
    comptime if CompilationTarget.has_avx512f():
        return SIMD_WIDTH_AVX512
    elif CompilationTarget.has_avx2():
        return SIMD_WIDTH_AVX2
    else:
        return max(4, simd_width_of[Float32]())

comptime SIMD_WIDTH: Int = get_simd_width()
comptime F32xW = SIMD[DType.float32, SIMD_WIDTH]
comptime I8xW = SIMD[DType.int8, SIMD_WIDTH]
comptime U8xW = SIMD[DType.uint8, SIMD_WIDTH]
comptime I32xW = SIMD[DType.int32, SIMD_WIDTH]

# ─── ISA Info Strings ────────────────────────────────────────────────────────

def isa_info() -> String:
    """Return human-readable ISA feature string."""
    var parts = List[String]()
    comptime if CompilationTarget.has_vnni():
        parts.append("AVX-512-VNNI")
    comptime if CompilationTarget.has_avx512f():
        parts.append("AVX-512F")
    comptime if CompilationTarget.has_avx2():
        parts.append("AVX2")
    comptime if CompilationTarget.has_fma():
        parts.append("FMA")
    if len(parts) == 0:
        return "scalar/NEON"
    var res = String("")
    for i in range(len(parts)):
        if i > 0:
            res += "|"
        res += parts[i]
    return res

def simd_info() -> String:
    """Return SIMD configuration summary string."""
    return (
        "SIMD width="
        + String(SIMD_WIDTH)
        + " floats ("
        + String(SIMD_WIDTH * 32)
        + " bits) ["
        + isa_info()
        + "], cores="
        + String(num_performance_cores())
    )

# ─── Byte Swap & Float16 / BFloat16 Conversions ──────────────────────────────

@always_inline
def bswap16(x: UInt16) -> UInt16:
    """Byte-swap a 16-bit value."""
    return ((x & 0xFF) << 8) | ((x >> 8) & 0xFF)

@always_inline
def f16_to_f32(h: UInt16) -> Float32:
    """Convert IEEE 754 half-precision (UInt16 bits) to Float32."""
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s = (UInt32(h) >> 15) & 1
        var e = (UInt32(h) >> 10) & 0x1F
        var m = UInt32(h) & 0x3FF
        if e == 0:
            if m == 0:
                return Float32(bitcast[DType.float32](s << 31))
            var mm = m
            var c: UInt32 = 0
            while mm > 0:
                mm >>= 1
                c += 1
            var sh = 24 - c
            var vm = m << (sh + 13)
            return Float32(
                bitcast[DType.float32](
                    (s << 31) | ((UInt32(113 - sh)) << 23) | (vm & 0x7FFFFF)
                )
            )
        if e == 31:
            return Float32(bitcast[DType.float32]((s << 31) | 0x7F800000 | (m << 13)))
        return Float32(bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13)))

@always_inline
def bf16_to_f32(b: UInt16) -> Float32:
    """Convert BFloat16 (UInt16 bits) to Float32 by shifting to upper 16 bits."""
    return Float32(bitcast[DType.float32](UInt32(b) << 16))

@always_inline
def load_f16_scale(ptr: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> Float32:
    """Load a little-endian f16 quantization scale from packed byte pointer."""
    var lo = UInt16(ptr[unsafe_offset=offset])
    var hi = UInt16(ptr[unsafe_offset=offset + 1])
    return f16_to_f32((hi << 8) | lo)

@always_inline
def saturating_cast_i8(x: Float32) -> Int8:
    """Clamp Float32 to [-128, 127] and cast to Int8."""
    if x > 127.0:
        return Int8(127)
    elif x < -128.0:
        return Int8(-128)
    else:
        return Int8(x)

@always_inline
def saturating_cast_u8(x: Float32) -> UInt8:
    """Clamp Float32 to [0, 255] and cast to UInt8."""
    if x > 255.0:
        return UInt8(255)
    elif x < 0.0:
        return UInt8(0)
    else:
        return UInt8(x)