# Phase 2 Implementation Report: Ministral 3 3B Performance Acceleration

## 1. Overview & Objective

Phase 2 builds directly upon Phase 1's hardware topology foundation by optimizing the computational bottlenecks identified during Ministral 3 3B inference:
1. **Thread Pool Core Clamp Elimination**: Removing lingering `num_performance_cores()` clamp limits across the persistent worker pool and matrix engines.
2. **Batched Prompt Prefill Decoding**: Transitioning prompt ingestion from scalar $O(N)$ single-token decode dispatches to single-batch evaluation (`llama_batch_init(n_tokens)`), allowing KV cache population in a unified sequence decode pass.
3. **Multi-Threaded Output Head Linear Projection (`gemv_quant`)**: Refactoring the 131,072-element vocabulary projection from unvectorized scalar worker iteration to `gemv_quant` with dynamic 2-row SIMD register tiling and multi-core work distribution.

---

## 2. Root Cause Analysis of Phase 1 Bottlenecks

### 2.1. Residual Single-Thread Clamps in `LlamaWorkerPool`
While Phase 1 updated `llama_context_default_params` to use `num_logical_cores()`, the underlying `LlamaWorkerPool` struct in [src/llamojo/backend/threadpool.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/threadpool.mojo) still clamped thread initialization:
```mojo
var hw_cores = num_performance_cores()
self.workers = max(1, min(actual_workers, hw_cores))
```
Because `num_performance_cores()` returns `1` inside WSL2 on Intel hybrid processors, any `workers` value passed in was immediately clamped back down to `1`. Consequently, internal spin barriers and layer workers ran on a single thread.

Similarly, in [src/llamojo/backend/kernels_quant.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_quant.mojo), `gemv_quant`, `gemv_gelu_biased`, and `gemv_gate_up_fused` defaulted unassigned worker counts to `num_performance_cores()`, preventing dynamic multi-threading.

### 2.2. Vocabulary Head Projection Latency
Ministral 3 3B's vocabulary size is $V = 131,072$. For each token where logits are requested, the model executes:
$$\text{logits} = W_{\text{output}} \cdot x_{\text{norm}}$$
In `llama_decode`, this was previously evaluated with:
```mojo
for r in range(v0, v1):
    var w_row = model[].output_weight.unsafe_offset(r * bpr_out)
    out_logits[r] = dot_quant_row(model[].output_weight_type, w_row, xbp, dim)
```
Calling `dot_quant_row` per row does not reuse activation loads across rows and operates in floating-point SIMD rather than integer-domain vectorized AVX2 dot products.

### 2.3. Per-Token Single Decode Loop in Prompt Ingestion
In [examples/chat_ministral3.mojo](file:///mnt/c/Documents/Mojo/llamojo/examples/chat_ministral3.mojo), the prompt was ingested one token at a time:
```mojo
for i in range(n_tokens):
    var b = llama_batch_init(1, ...)
    llama_decode(ctx, b)
```
Each iteration paid the full overhead of layer-by-layer synchronization, threadpool coordination, and intermediate buffer management.

---

## 3. Implementation Details

### 3.1. Removing Core Clamps Across Threadpool & Quantized Kernels
In [src/llamojo/backend/threadpool.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/threadpool.mojo):
```mojo
from std.sys.info import num_performance_cores, num_logical_cores
...
def __init__(out self, workers: Int = 0, dim: Int = 4096):
    var hw_cores = max(1, num_logical_cores())
    var actual_workers = workers if workers > 0 else hw_cores
    self.workers = max(1, actual_workers)
    self.sh = unsafe_alloc[LlamaPoolShared](1)
    self.sh.unsafe_write(LlamaPoolShared(self.workers, dim))
    self.initialized = True
```

In [src/llamojo/backend/kernels_quant.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_quant.mojo):
Updated `gemv_quant`, `gemv_gelu_biased`, and `gemv_gate_up_fused` default worker fallbacks from `num_performance_cores()` to `max(1, num_logical_cores())`.

### 3.2. Vectorized 2-Row Tiled LM-Head via `gemv_quant`
In [src/llamojo/llama.mojo](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/llama.mojo):
Replaced the manual row-by-row `dot_quant_row` loop in `llama_decode` with `gemv_quant`:
```mojo
gemv_quant(
    out_logits,
    model[].output_weight,
    model[].output_weight_type,
    xbp,
    vocab_size,
    dim,
    workers,
)
```
Benefits:
- For `Q4_K` / `Q6_K` weights, `gemv_quant` quantizes the input activation $x_{\text{norm}}$ into `Q8_K` once per pass.
- Computes two rows simultaneously using `dot_q4_k_q8_k_2rows` / `dot_q6_k_q8_k_2rows` using hardware AVX2 `pmadd.ub.sw` vector instructions.
- Automatically partitions the 131,072 rows across all available worker threads.

### 3.3. Batched Prompt Prefill Ingestion
In [examples/chat_ministral3.mojo](file:///mnt/c/Documents/Mojo/llamojo/examples/chat_ministral3.mojo):
Batched the entire prompt into a single `llama_batch` allocation:
```mojo
var batch = llama_batch_init(n_tokens, embd=0, n_seq_max=1)
for i in range(Int(n_tokens)):
    batch.token[unsafe_offset=i] = prompt_tokens_buf[unsafe_offset=i]
    batch.pos[unsafe_offset=i] = Int32(cur_pos + i)
    batch.logits[unsafe_offset=i] = 1 if i == Int(n_tokens) - 1 else 0
_ = llama_decode(ctx, batch)
llama_batch_free(batch)
```
And updated the generation loop to extract logits from the final prompt token index:
```mojo
var logits = llama_get_logits_ith(ctx, Int32(n_tokens - 1 if step == 0 else 0))
```

---

## 4. Verification & Comparative Benchmarks

### 4.1. Comparative Performance Matrix
Benchmarked on Intel Core i7-1255U (WSL2 Linux 5.15, 2 vCPUs) on prompt: `"Why is the sky blue?"`:

| Implementation | Prompt Prefill (tok/s) | Generation (tok/s) | Speedup vs Baseline |
|---|---|---|---|
| **llamojo Baseline (Single-thread JIT)** | 0.027 tok/s | 0.109 tok/s | 1.00× (ref) |
| **llamojo Phase 1 (2-thread AOT)** | ~0.08 tok/s | ~0.35 tok/s | ~3.2× |
| **llamojo Phase 2 (Unclamped Threadpool + `gemv_quant` LM Head)** | **1.21 tok/s** | **0.58 tok/s** | **~44.8× (Prefill)** / **~5.3× (Gen)** |
| **llama.cpp Baseline** | 1.33 tok/s | 2.96 tok/s | Parity Target |

### 4.2. Qualitative Parity
The model produces logically sound, fluent responses:
> **Ministral 3**: *"The sky appears blue due to a phenomenon called **Rayleigh scattering**, which is a type of scattering of light by the molecules in the Earth's atmosphere..."*

---

## 5. Next Steps (Phase 3 Acceleration)
1. **Fused QKV and Gate/Up Projections**: Integrate horizontal multi-matrix projection kernels with direct Q8_K activation reuse in `forward_layer_dense`.
2. **AVX2 Vectorized Attention Decode**: Specialize grouped-query attention (`gqa_decode`) to eliminate scalar exp reductions during causal masking.
