---
type: reference
title: "Gemma 4 Acceleration: Phase 2 Implementation Documentation"
description: "Multi-threaded GEMV row partitioning with dynamic work-stealing and cache affinity."
status: completed
tags: [acceleration, phase2, multithreading, thread-pool, gemv-parallel]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Acceleration: Phase 2 Implementation Documentation

This document records the design, implementation, differential verification, and benchmark results of **Phase 2: Explicit SIMD Vectorization of Dominant K-Quants (`Q6_K` and `Q4_K`)** for Google's **Gemma 4** (`gemma4`) model in **llamojo**.

---

## 1. Executive Summary & Gains

In Phase 1, analysis revealed that the compiler's LLVM loop vectorizer could not automatically vectorize quantized dot products because of non-contiguous bit shifts and scalar masking operations.

In **Phase 2**, we implemented hand-vectorized SIMD routines using Mojo's native `SIMD[DType.float32, 8]`, `SIMD[DType.uint8, 8]`, and vector arithmetic:
- **Generation Time (16 tokens)**: Dropped from **23,196 ms** to **5,549 ms** (AOT binary) / **6,943 ms**.
- **Throughput**: Increased from **0.71 tok/s** to **2.88 tok/s** (**> 4.0x speedup** on single-core throughput!).
- **Numerical Parity**: 100% exact token sequence preserved (`", in a land far, far away, there lived a little bird named Pip"`), with differential errors under $10^{-5}$ compared to the scalar reference.

---

## 2. Kernel Design & SIMD Vectorization

### A. Vectorizing `dot_q4_k_row` ([`src/llamojo/backend/kernels_quant.mojo`](../../src/llamojo/backend/kernels_quant.mojo))

In standard `Q4_K`, each 256-element superblock contains 128 bytes of packed 4-bit nibbles.
In Phase 2, we load 8 packed bytes at a time into `SIMD[DType.uint8, 8]`:
```mojo
for g in range(4):
    var j = g * 8
    var b_v = qs_ptr.unsafe_load[width=8](q_off + j)
    var lo_f32 = (b_v & UInt8(0x0F)).cast[DType.float32]()
    var hi_f32 = (b_v >> UInt8(4)).cast[DType.float32]()

    var x1_v = x.unsafe_load[width=8](xo + s + j)
    var x2_v = x.unsafe_load[width=8](xo + s + 32 + j)

    acc += (lo_f32 * d1_v - mm1_v) * x1_v + (hi_f32 * d2_v - mm2_v) * x2_v
```
- **8 low nibbles** and **8 high nibbles** are masked and shifted simultaneously in 8-wide SIMD registers.
- The scale and min vectors (`d1_v, mm1_v, d2_v, mm2_v`) broadcast scalar parameters across registers.
- Floating-point multiply-accumulate operations fuse cleanly into AVX2 FMA pipelines.

### B. Vectorizing `dot_q6_k_row` ([`src/llamojo/backend/kernels_quant.mojo`](../../src/llamojo/backend/kernels_quant.mojo))

In `Q6_K`, each 256-element superblock contains:
- 128 bytes of low 4 bits (`ql`)
- 64 bytes of high 2 bits (`qh`)
- 16 bytes of scales (`sc`)
- 2 bytes of base scale (`d`)

We vectorize the 32-element inner loop in 4 steps of 8 elements using 8-wide SIMD loads:
```mojo
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
```
- Vectorized unpacking eliminates 32 individual branchy scalar extractions per sub-block.
- All accumulators (`acc1..acc4`) stay in registers throughout the group loop and reduce via a single `.reduce_add()` at the end of each block.

---

## 3. Verification & Differential Testing

### A. Dedicated Differential Test (`tests/test_quant_simd.mojo`)
A new differential test suite was created in [`tests/test_quant_simd.mojo`](../../tests/test_quant_simd.mojo) to verify:
1. `dot_q6_k_row`: absolute difference vs scalar reference $< 7.7 \times 10^{-6}$.
2. `dot_q4_k_row`: absolute difference vs scalar reference $< 1.5 \times 10^{-6}$.
- Result: **All tests passed! ✓**

### B. Full Test Regression Suite
- `tests/test_kernels.mojo`: **Passed ✓** (RMSNorm, SwiGLU, GELU, RoPE, Q8_0, Q4_K, MXFP4, fused matmuls).
- `tests/test_context_decode.mojo`: **Passed ✓** (PageTable, KV cache, full decode pipeline).

---

## 4. Benchmark Progression

| Optimization Stage | Generation Time (16 tokens) | Throughput (tok/s) | Speedup vs Baseline |
|---|---|---|---|
| **Baseline (Scalar loops, generic AOT)** | 22,497 ms | 0.71 tok/s | 1.00x |
| **Phase 1 (Compiler flags `-O3 --target-cpu alderlake`)** | 23,196 ms | 0.69 – 0.71 tok/s | ~1.00x |
| **Phase 2 (SIMD Vectorized `Q6_K` & `Q4_K` AVX2)** | **5,549 ms** | **2.88 tok/s** | **4.06x** |
| *Target: Upstream `llama.cpp` (Multi-threaded)* | *3,158 ms* | *5.07 tok/s* | *7.14x* |

---

## 5. Phase Status

- **Phase 2: Explicit SIMD Vectorization**: **COMPLETED**
- **Next Phase**: **Phase 3: Multi-Core GEMV Row Parallelization (`parallel_for`)** to scale across available CPU cores and close the remaining gap to `llama.cpp`.

