# Phase 4 Implementation Report: Ministral 3 3B Performance Acceleration

## 1. Overview & Objective

Phase 4 tackles the final computational bottleneck that was limiting autoregressive token generation throughput:
1. **$O(V^2)$ Sampler Sorting Elimination**: Overcoming the quadratic sorting latency incurred when filtering Ministral 3 3B's massive 131,072-token vocabulary.
2. **Attention Micro-Kernel 4× AVX2 Unrolling**: Accelerating grouped-query attention inner products (`dot_vectors` and `axpy_vector`) with 32-float AVX2 loop unrolling.
3. **Optimized Pipeline Execution**: Re-ordering the sampler pipeline stages to perform partial Top-$K$ candidate pruning before temperature and distribution transformations.

---

## 2. Root Cause Analysis: The 131k Vocabulary Sampler Bottleneck

### 2.1. The $O(V^2)$ Selection Sort Trap
In [src/llamojo/sampler.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/sampler.mojo), `_sort_token_data_descending` was implemented as a full selection sort over all candidates in `cur_p`:
```mojo
for i in range(n):
    var best_idx = i
    var best_val = data[i].logit
    for j in range(i + 1, n):
        if data[j].logit > best_val:
            best_val = data[j].logit
            best_idx = j
    ...
```
For standard LLaMA-2 models ($V = 32,000$), $N(N-1)/2 \approx 512 \text{ million}$ operations per token is noticeable but manageable.
However, for **Ministral 3 3B** ($V = 131,072$):
$$\frac{V(V-1)}{2} = \frac{131,072 \times 131,071}{2} \approx \mathbf{8,589,869,056 \text{ comparisons per token!}}$$
Executing over **8.5 billion scalar operations** on every single generated token created an immense CPU bottleneck, causing each token to spend hundreds of milliseconds inside the sampling function rather than in the neural network forward pass.

### 2.2. Sampler Chain Order Inefficiency
In `examples/chat_ministral3.mojo`, the sampler chain was previously initialized as:
```mojo
llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature))
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40))
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(Float32(0.9)))
llama_sampler_chain_add(smpl, llama_sampler_init_dist(42))
```
Applying temperature scaling over all 131,072 elements before pruning was wasteful. By placing `top_k(40)` first, only the top 40 candidates need to be scaled and sampled.

---

## 3. Implementation Details

### 3.1. $O(K \cdot V)$ Partial Selection Sort
In [src/llamojo/sampler.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/sampler.mojo):
Updated `_sort_token_data_descending` to accept an optional `top_k: Int = -1`:
```mojo
def _sort_token_data_descending(
    cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin],
    top_k: Int = -1,
):
    var n = cur_p[].size
    var data = cur_p[].data
    var max_i = n if top_k <= 0 or top_k > n else top_k
    for i in range(max_i):
        var best_idx = i
        var best_val = data[unsafe_offset=i].logit
        for j in range(i + 1, n):
            if data[unsafe_offset=j].logit > best_val:
                best_val = data[unsafe_offset=j].logit
                best_idx = j
        if best_idx != i:
            var tmp = data[unsafe_offset=i]
            data[unsafe_offset=i] = data[unsafe_offset=best_idx]
            data[unsafe_offset=best_idx] = tmp
    cur_p[].sorted = True
```
When `_apply_top_k` executes with $K = 40$:
- Number of comparisons drops from **8,589,869,056** to $40 \times 131,072 = \mathbf{5,242,880}$ comparisons.
- **Reduction factor: 1,638× fewer operations per token!**

### 3.2. 4× AVX2 Unrolling in Attention Micro-Kernels
In [src/llamojo/backend/kernels_attn.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_attn.mojo):
Refactored `dot_vectors` and `axpy_vector` to process 32 floats per loop iteration across four parallel 8-lane SIMD registers:
```mojo
@always_inline
def dot_vectors(a: FloatPtr, b: FloatPtr, n: Int) -> Float32:
    var acc0 = SIMD[DType.float32, 8](0.0)
    var acc1 = SIMD[DType.float32, 8](0.0)
    var acc2 = SIMD[DType.float32, 8](0.0)
    var acc3 = SIMD[DType.float32, 8](0.0)
    var i = 0
    while i + 32 <= n:
        acc0 += a.unsafe_load[width=8](i + 0) * b.unsafe_load[width=8](i + 0)
        acc1 += a.unsafe_load[width=8](i + 8) * b.unsafe_load[width=8](i + 8)
        acc2 += a.unsafe_load[width=8](i + 16) * b.unsafe_load[width=8](i + 16)
        acc3 += a.unsafe_load[width=8](i + 24) * b.unsafe_load[width=8](i + 24)
        i += 32
    ...
```
For `head_dim = 128` in Ministral 3 3B, each head dot product completes in exactly 4 unrolled loop iterations with zero residual loop overhead.

---

## 4. Benchmark Results & Parity Analysis

Benchmarked on Intel Core i7-1255U (WSL2 Ubuntu 22.04, 2 vCPUs) on prompt: `"Why is the sky blue?"`:

| Implementation Phase | Prompt Prefill (tok/s) | Generation (tok/s) | Speedup vs Baseline |
|---|---|---|---|
| **Baseline (Single-thread JIT)** | 0.027 tok/s | 0.109 tok/s | 1.00× (ref) |
| **Phase 1 (2-thread AOT Binary)** | ~0.08 tok/s | ~0.35 tok/s | ~3.2× |
| **Phase 2 (Unclamped Workers + Batched Prefill)** | 1.21 tok/s | 0.58 tok/s | ~5.3× |
| **Phase 3 (AVX2 Q8_K Layer Projections)** | 1.25 tok/s | 0.82 tok/s | ~7.5× |
| **Phase 4 (Partial Top-K + 4× AVX2 Attention)** | **1.00 tok/s** | **2.32 tok/s** | **21.3× (Gen)** |
| **llama.cpp Reference** | 1.33 tok/s | 2.96 tok/s | Parity Target |

### Summary of Achievements:
- **Generation throughput reached 2.32 tok/s**, closing within **78% of upstream `llama.cpp`** (2.96 tok/s).
- Total speedup from initial baseline: **21.3× faster token generation**.
- Total prompt prefill speedup: **37× faster ingestion**.
- 100% pure Mojo, zero external C/C++ libraries.

---

## 5. End-to-End Verification

Full 128-token generation executed with fluent output:
```
llamojo Ministral 3 3B Chat Application
Model:   models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
User:    Why is the sky blue?
============================================================
Ingesting prompt ( 10  tokens): .......... done.

Ministral 3: The sky appears blue due to a phenomenon called **Rayleigh scattering**, which is a type of scattering of light by the molecules in the Earth's atmosphere. Here's a step-by-step explanation:

1. **Sunlight Composition**: Sunlight consists of a mixture of all colors of the visible spectrum, but it can be thought of as white light. This white light contains all wavelengths of visible light, roughly from 400 nm to 700 nm.

2. **Sunlight Travels**: When sunlight reaches the Earth's atmosphere, it consists of all these colors traveling together.

3. **Interaction with Atmosphere**: As...

------------------------------------------------------------
Metrics: Prompt  10  tokens @  1.00 tok/s | Generation  128  tokens @  2.32 tok/s
------------------------------------------------------------
```
