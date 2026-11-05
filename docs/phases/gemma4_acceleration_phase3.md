---
type: reference
title: "Gemma 4 Acceleration: Phase 3 Implementation Documentation"
description: "Memory layout alignment, cacheline padding, and non-temporal streaming stores for prompt ingest."
status: completed
tags: [acceleration, phase3, cache-alignment, memory-layout, streaming-stores]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Acceleration: Phase 3 Implementation Documentation

This document records the design, implementation, micro-benchmarking, and verification of **Phase 3: Multi-Core GEMV Row Parallelization** for Google's **Gemma 4** (`gemma4`) model in **llamojo**.

---

## 1. Executive Summary & Goals

The objective of Phase 3 was to introduce multi-threaded row parallelization into `llamojo`'s quantized matrix-vector multiplications (`gemv_quant`) to leverage multi-core CPU hardware without introducing race conditions, heap corruption, or numerical drift.

### Key Results
- **Generation Time (16 tokens)**: Reduced to **5,258 ms** (down from **23,196 ms** baseline and **5,549 ms** in Phase 2).
- **Throughput**: Reached **3.04 tok/s** (up from **0.71 tok/s** baseline, representing an overall **4.28x speedup**).
- **Parity & Regression**: 100% token sequence exactness preserved (`", in a land far, far away, there lived a little bird named Pip"`).
- **Hardware Profile**: 2 vCPUs on WSL2 host.

---

## 2. Technical Design & Implementation

### A. Dynamic Row-Chunked GEMV (`gemv_quant`)
In [`src/llamojo/backend/kernels_quant.mojo`](../../src/llamojo/backend/kernels_quant.mojo), we implemented `gemv_quant`:
```mojo
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

    var n_workers = workers if workers > 0 else num_performance_cores()
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
```

### B. Parallel Dispatch Thresholding
Dispatching thread workers incurs a small synchronization cost (~10–30 $\mu$s).
- Matrices with `rows < 512` (such as smaller KV projections or low-dimensional PLE vectors) bypass thread dispatch and execute sequentially on the caller thread.
- Wide projections with large row counts benefit significantly:
  - FFN Gate and Up projections: $12288 \times 1536$
  - FFN Down projections: $1536 \times 12288$
  - PLE Context projection: $8960 \times 1536$
  - Attention Query projection: $4096 \times 1536$ (Global) / $2048 \times 1536$ (SWA)
  - Attention Output projection: $1536 \times 4096$

### C. Integration into Gemma 4 Forward Pass
In [`src/llamojo/backend/gemma4.mojo`](../../src/llamojo/backend/gemma4.mojo):
- Replaced serial row loops for `per_layer_model_proj`, `wq`, `wo`, `w_gate`, `w_up`, and `w_down` with unified calls to `gemv_quant`.

---

## 3. Micro-Benchmarking & Validation

### A. Isolated Steady-State Micro-Benchmark
Testing isolated GEMV on a $12288 \times 1536$ `Q6_K` matrix over 50 iterations:
- Single-thread execution: **3.96 ms**
- 2-thread parallel execution: **2.94 ms**
- **Speedup**: **1.35x** on 2 virtual cores with zero memory contention.

### B. Regression Suite
- `tests/test_quant_simd.mojo`: **Passed ✓**
- `tests/test_kernels.mojo`: **Passed ✓**
- `tests/test_context_decode.mojo`: **Passed ✓**
- `./tools/compare_gemma4.sh`: **Passed ✓** (exact matching token generation).

---

## 4. Benchmark Progression Summary

| Optimization Stage | Generation Time (16 tokens) | Throughput (tok/s) | Speedup vs Baseline |
|---|---|---|---|
| **Baseline (Single-threaded scalar)** | 22,497 ms | 0.71 tok/s | 1.00x |
| **Phase 1 (Compiler flags `-O3 --target-cpu`)** | 23,196 ms | 0.69 – 0.71 tok/s | ~1.00x |
| **Phase 2 (SIMD Vectorization Q6_K & Q4_K)** | 5,549 ms | 2.88 tok/s | 4.06x |
| **Phase 3 (Multi-Core GEMV Parallelization)** | **5,258 ms** | **3.04 tok/s** | **4.28x** |
| *Target: Upstream `llama.cpp`* | *2,645 ms* | *6.05 tok/s* | *8.52x* |

---

## 5. Next Steps

- **Phase 3 Status**: **COMPLETED**
- **Next Phase**: **Phase 4: Fused Horizontal GEMVs (Gate/Up & QKV)**
  - Interleave memory loads of $x$ to compute `Gate` and `Up` projections concurrently in a single pass over memory, cutting FFN memory bandwidth consumption in half.

