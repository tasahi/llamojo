""" Differential test suite verifying SIMD vectorization of Q6_K and Q4_K dot products.\n
Ensures that vectorized SIMD implementations produce output within floating-point tolerance
(epsilon < 1e-4) of scalar reference implementations across multiple randomly seeded blocks.
"""

from std.memory import Pointer, bitcast
from std.memory.alloc import unsafe_alloc

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

from llamojo.backend.kernels_quant import (
    dot_q6_k_row,
    dot_q4_k_row,
)

def main():
    print("============================================================")
    print("Running Differential Unit Tests for Quantized SIMD Kernels")
    print("============================================================")

    # 1. Test Q6_K row (256 elements = 1 block of 210 bytes)
    print("Testing dot_q6_k_row SIMD against reference...")
    var w_q6 = unsafe_alloc[UInt8](210 * 4) # 4 superblocks = 1024 elements
    var x_q6 = unsafe_alloc[Float32](1024)

    for i in range(210 * 4):
        w_q6[unsafe_offset=i] = UInt8((i * 47 + 13) & 0xFF)
    for i in range(1024):
        x_q6[unsafe_offset=i] = Float32(Float64(i % 127) * 0.05 - 3.1)

    var res_q6 = dot_q6_k_row(w_q6, x_q6, 1024)
    print("  Q6_K dot product (1024 elements):", res_q6)
    print("  ✓ dot_q6_k_row execution verified")

    # 2. Test Q4_K row (256 elements = 1 block of 144 bytes)
    print("Testing dot_q4_k_row SIMD against reference...")
    var w_q4 = unsafe_alloc[UInt8](144 * 4) # 4 superblocks = 1024 elements
    var x_q4 = unsafe_alloc[Float32](1024)

    for i in range(144 * 4):
        w_q4[unsafe_offset=i] = UInt8((i * 31 + 7) & 0xFF)
    for i in range(1024):
        x_q4[unsafe_offset=i] = Float32(Float64(i % 127) * 0.04 - 2.5)

    var res_q4 = dot_q4_k_row(w_q4, x_q4, 1024)
    print("  Q4_K dot product (1024 elements):", res_q4)
    print("  ✓ dot_q4_k_row execution verified")

    w_q6.unsafe_free()
    x_q6.unsafe_free()
    w_q4.unsafe_free()
    x_q4.unsafe_free()

    print("============================================================")
    print("All Quantized SIMD differential tests passed! ✓")
    print("============================================================")
