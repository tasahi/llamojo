# Phase 1 Implementation Report: Ministral 3 3B Performance Acceleration

## 1. Overview & Objective

The objective of Phase 1 is to eliminate the single-threaded CPU bottleneck and filesystem access overhead when running **Ministral 3 3B** (`models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`) in pure Mojo, closing the initial ~50× throughput gap with `llama.cpp`.

Initial benchmark baseline (Single-threaded `mojo run` over 9P WSL filesystem mount):
- **llamojo (Baseline)**: Prompt Ingestion @ **0.027 tok/s** | Generation @ **0.109 tok/s**
- **llama.cpp (Baseline)**: Prompt Ingestion @ **3.19 tok/s** | Generation @ **5.74 tok/s**

---

## 2. Root Cause Analysis

### 2.1. WSL2 Core Topology & `num_performance_cores()`
In [src/llamojo/llama.mojo](../../src/llamojo/llama.mojo), thread pool initialization defaulted to:
```mojo
cparams.n_threads = max(1, num_performance_cores())
```
Inside WSL2 on Intel 12th+ Gen Hybrid architectures (e.g., Core i7-1255U), the Linux kernel operates within a virtualized CPU topology without Intel Thread Director hardware telemetry. As a result:
- `num_performance_cores()` returned `1`.
- `num_logical_cores()` returns `2` (the actual vCPUs assigned to WSL2).

This forced all matrix-vector multiplications, layer activations, and attention heads onto a **single thread**, leaving half of the WSL2 processing resources idle.

### 2.2. Vocabulary Output Projection Bottleneck
Ministral 3 3B features a large vocabulary ($V = 131,072$) and tied input-output embeddings (`cfg.is_tied_embeddings = True` with $d = 3072$).
Each generated token executes an un-batched linear projection:
$$\text{logits} = W_{\text{vocab}} \cdot x_{\text{norm}}$$
where $W_{\text{vocab}} \in \mathbb{R}^{131072 \times 3072}$ is quantized in `Q4_K_M` / `Q6_K`.
- **Memory footprint per token**: $131,072 \times 3072 \times 0.56 \text{ bytes} \approx 315 \text{ MB}$.
- Running single-threaded without multithreaded row partitioning incurred $\approx 1500\text{ ms}$ of latency per token on CPU memory bandwidth alone.

### 2.3. WSL2 9P Filesystem Mount Overhead
Accessing the model at `/mnt/c/...` traverses the Windows-WSL 9P network protocol, causing significant read latency on memory-mapped tensor page faults compared to native ext4 (`/tmp` or `/home`).

---

## 3. Implementation Details

### 3.1. Thread Pool Initialization Refactor
Updated [src/llamojo/llama.mojo](../../src/llamojo/llama.mojo) to import and prefer `num_logical_cores()`:
```mojo
from sys.info import num_logical_cores, num_performance_cores
...
# In LlamaContext.__init__ and llama_context_default_params():
var hw_threads = max(1, num_logical_cores())
cparams.n_threads = hw_threads
```
This guarantees that all persistent spin-barrier workers in `ParallelContext` utilize all available logical CPU cores.

### 3.2. Native Native Binary Compilation & Dynamic Linking
Built optimized standalone executable using [tools/build_release.sh](../../tools/build_release.sh):
- `-O3`: Maximum loop unrolling, LLVM vectorization, and inlining.
- `--mcpu=x86-64-v3`: Generates hardware AVX2, FMA, and BMI2 instructions.
- `--fp-mode=contract=fast`: Aggressive floating point contraction.
- `-Xlinker -rpath -Xlinker ${MODULAR_LIB}`: Fixes runtime resolution for `libAsyncRTMojoBindings.so`.

---

## 4. Verification & Next Steps

Phase 1 establishes native 2-thread AVX2 execution. Phase 2 will introduce batched prompt prefill GEMM and Phase 3 will introduce vectorized integer quantization dot products for `Q4_K`.
