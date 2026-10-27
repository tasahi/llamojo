# llamojo Acceleration Phase 6: 2-Row GEMV Register Tiling

## Executive Summary
In Phase 6 of the `llamojo` acceleration plan, we implemented **2-row register tiling** for both `Q6_K` and `Q4_K` matrix-vector multiplications (`gemv_quant`).

By computing two rows $(r, r+1)$ concurrently, the activation vector $y \in \text{Q8\_K}$ is loaded into SIMD registers once and reused across both rows. This halves the memory and L1/L2 cache bandwidth required for activation streaming during matrix multiplication, boosting kernel throughput and maintaining bit-for-bit exactness (`Diff: 0.0`).

---

## 1. Architectural Motivation & Analysis
In single-row quantized GEMV dot products (`dot_q6_k_q8_k_row` and `dot_q4_k_q8_k_row`), every row $r$ streams through the quantized activation vector $q8\_buf$:
- For every 256 weights, 256 8-bit activations are loaded into SIMD registers and cast from `Int8` to `Int16`.
- Over large matrix projections (e.g. Gemma 4's FFN hidden dimension of 12,288), the activation buffer was loaded and unpacked 12,288 times per token.
- By unrolling the row loop by 2 (`r += 2`), activations are loaded into SIMD registers once and multiplied against weights for row 0 and row 1 in parallel.

---

## 2. Implementation Details

### A. 2-Row Tiled Kernels (`kernels_quant.mojo`)
1. **`dot_q6_k_q8_k_2rows`**:
   - Simultaneously unpacks 256 weights for `w0` and `w1`.
   - Shared 16-wide activation vector loads (`qs_ptr.unsafe_load[width=16]`).
   - Shared scale factor scaling and separate integer accumulation: `sumi0`, `sumi1`.
   - Produces outputs `res0` and `res1` in registers.

2. **`dot_q4_k_q8_k_2rows`**:
   - Simultaneously unpacks 256 weights for `w0` and `w1` (low/high nibbles and scales).
   - Shared 16-wide activation vector loads and horizontal dot-products.
   - Computes both row sums with zero extra activation cache traffic.

### B. Dispatch Integration in `gemv_quant`
- In both single-thread and multi-core `parallelize` paths, `chunk` sizes are aligned to even row counts:
  ```mojo
  var chunk = (((rows + n_workers - 1) // n_workers) + 1) & ~1
  ```
- Workers execute in pairs (`while r + 1 < r_end: dot_..._2rows(...)`), with an odd-row fallback to `dot_..._row` if necessary.

---

## 3. Micro-Benchmark Verification
We evaluated the numerical equivalence and speedup using dedicated differential benchmarks:
```
Ref0: -102281.22 Tiled0: -102281.22 Diff: 0.0
Ref1: -35312.402 Tiled1: -35312.402 Diff: 0.0
Separate 2 rows time : 1.75e-05 us
Tiled 2 rows time    : 1.45e-05 us
Speedup              : ~1.20x in isolated kernel execution
```

---

## 4. End-to-End Benchmark & Parity Verification

### Regression Test Suite
All regression tests passed with 100% success:
- `tests/test_kernels.mojo`: **PASSED**
- `tests/test_context_decode.mojo`: **PASSED**

### End-to-End Generation Comparison (64 tokens)
```
Prompt: "Once upon a time"
Temperature: 0.0 (Greedy)

Output:
, in a land far, far away, there lived a curious little squirrel named Squeaky. Squeaky had a bushy tail and a twitchy nose, and he loved to explore every nook and cranny of the forest.

One sunny morning, Squeaky was hopping through the trees when he spotted something
```
- **Story Parity**: Exact canonical "Squeaky the squirrel" narrative preserved.
- **Compiled Binary Throughput**: **2.50 tok/s** (25,637 ms for 64 tokens), up from uncompiled/scalar baselines.
