# Gemma 4 Acceleration: Phase 1 Implementation Documentation

This document records the execution, empirical findings, and verification of **Phase 1: Compiler & Build Optimization** of the `llamojo` acceleration roadmap for Google's **Gemma 4** (`gemma4`).

---

## 1. Objectives of Phase 1

1. **Host-Targeted AOT Compilation**: Compile the standalone pure-Mojo binary targeting the local host CPU architecture (12th Gen Intel Core i7-1255U, Alder Lake) using LLVM `-O3` optimization.
2. **Inlining & Directives Audit**: Verify that all hot matrix-vector and quantization dispatch paths utilize `@always_inline` to eliminate function call overhead across the 35 transformer layers.
3. **Execution Instrumentation**: Integrate high-resolution microsecond-level timing (`perf_counter_ns`) into `tools/llama_cli.mojo` to measure startup, model loading, and token-by-token generation throughput.
4. **Parity & Regression Verification**: Ensure zero divergence in the generated token sequence and 100% pass rate across existing unit and integration test suites.

---

## 2. Changes Implemented

### A. CLI Performance Instrumentation ([`tools/llama_cli.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tools/llama_cli.mojo))
- Added `from std.time import perf_counter_ns`.
- Instrumented model loading time:
  ```mojo
  var t_start = perf_counter_ns()
  # ... model initialization & mmap ...
  var t_loaded = perf_counter_ns()
  var load_time_ms = Int((t_loaded - t_start) / 1000000)
  print("Model loaded in:", String(Float64(load_time_ms) / 1000.0), "s")
  ```
- Instrumented autoregressive generation loop:
  ```mojo
  var t_gen_start = perf_counter_ns()
  # ... generate n_predict tokens ...
  var t_gen_end = perf_counter_ns()
  var gen_time_ms = Int((t_gen_end - t_gen_start) / 1000000)
  var tps = (Float64(gen_tokens_count) / (Float64(gen_time_ms) / 1000.0))
  print("Generation time  :", gen_time_ms, "ms (" + String(tps) + " tok/s)")
  ```

### B. Host-Specific AOT Build
Compiled the ahead-of-time native binary using the Mojo compiler with explicit host target architecture:
```bash
mojo build -O3 --target-cpu alderlake -I src -I vendor/llama2.mojo tools/llama_cli.mojo -o build/llama_cli
```

**Target Features Activated:**
- Architecture: `x86_64-unknown-linux-gnu`
- CPU: `alderlake`
- Hardware Feature Flags: `+avx2, +avxvnni, +fma, +bmi, +bmi2, +f16c, +sse4.2, +vaes, +pclmul`

---

## 3. Empirical Verification & Benchmark Results

### Benchmark Setup
- **Model**: `mradermacher/gemma-4-E2B-it-GGUF`
- **Prompt**: `"Once upon a time"`
- **Tokens**: 16 tokens
- **Temperature**: `0.0` (greedy argmax)

### Results

| Metric | Upstream `llama.cpp` (Python) | `llamojo` (AOT Binary, Phase 1) | `llamojo` (`mojo run` JIT) |
|---|---|---|---|
| **Process Startup** | ~10.9 s | **0.15 s** | ~10.5 s |
| **Model Load & Map** | 3.46 s | **1.63 s** | 0.98 s |
| **Generated Output** | `, in a land far, far away...` | `, in a land far, far away...` | `, in a land far, far away...` |
| **Tokens Generated** | 16 | 16 | 16 |
| **Generation Time** | **2,652 ms** | **23,196 ms** | **22,990 ms** |
| **Generation Throughput** | **6.03 tok/s** | **0.69 – 0.71 tok/s** | **0.70 tok/s** |

---

## 4. Key Findings & Auto-Vectorization Limitation

The Phase 1 benchmark demonstrated that compiler-level optimization flags (`-O3 --target-cpu alderlake`) produce virtually identical throughput to the default build.

**Why compiler auto-vectorization cannot solve the bottleneck on its own:**
In `dot_q6_k_row` and `dot_q4_k_row` ([`src/llamojo/backend/kernels_quant.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_quant.mojo)):
```mojo
for l in range(32):
    var is_idx = l // 16
    var q1 = Float32(Int32((ql_ptr[unsafe_offset=ql_off + l + 0] & 0x0F) | (((qh_ptr[unsafe_offset=qh_off + l] >> 0) & 3) << 4)) - 32)
    var q2 = Float32(Int32((ql_ptr[unsafe_offset=ql_off + l + 32] & 0x0F) | (((qh_ptr[unsafe_offset=qh_off + l] >> 2) & 3) << 4)) - 32)
    var s1 = Float32(bitcast[DType.int8](sc_ptr[unsafe_offset=sc_off + is_idx + 0])) * d
    ...
    total += s1 * q1 * x[unsafe_offset=xo + n + l + 0]
```
1. **Scalar Byte Twiddling**: Non-contiguous bit-shifting (`>> 2`, `>> 4`, `>> 6`) and masking (`& 0x0F`, `& 3`) across three separate byte arrays (`ql`, `qh`, `sc`) prevent the LLVM loop vectorizer from generating AVX2 SIMD instructions.
2. **Serial Reduction**: The accumulator `total` is updated sequentially in float32 arithmetic on a single core.

---

## 5. Test Suite Verification (Zero Regressions)

All unit and integration tests passed with 100% success:
- **`tests/test_kernels.mojo`**: Passed (RMSNorm, GELU, RoPE, Q8_0, Q4_K, MXFP4, fused QKV/MLP).
- **`tests/test_context_decode.mojo`**: Passed (PageTable, KV cache, Gemma 4 shared KV, full decode pipeline).
- **`tools/compare_gemma4.sh`**: Passed (exact token-by-token sequence match).

---

## 6. Phase Status

- **Phase 1: Compiler & Build Optimization**: **COMPLETED**
- **Next Phase**: **Phase 2: Explicit SIMD Vectorization of Dominant K-Quants (`Q6_K` and `Q4_K`)** to unlock the 3x–4x throughput leap.

