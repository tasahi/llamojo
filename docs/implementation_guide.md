# `llamojo` Implementation & Architecture Documentation

**Target Repository:** `/mnt/c/Documents/Mojo/llamojo`  
**Mojo Toolchain:** Mojo 1.1.0 (MAX 26.6.0)  
**Primary Integration Assets:**
- `vendor/llama2.mojo` (`/mnt/c/Documents/Mojo/llama2.mojo`)
- `vendor/MojoLlama` (`/mnt/c/Documents/Mojo/MojoLlama`)

---

## 1. Overview & Goals

`llamojo` is a pure Mojo implementation of `llama.cpp` (`libllama`). Its purpose is to deliver drop-in `llama.h` API compatibility with **zero external compute dependencies**:
- **Zero Python Runtime**: No `from python import Python` or `PythonObject`.
- **Zero External C/C++ Compute Runtimes**: No OpenMP, OpenBLAS, or `libllama.so`.
- **Zero Disk Duplication**: No extracting GGUF tensors into temporary binary files on disk. Direct in-place virtual memory mapping via POSIX `mmap64`.
- **Minimal System Call Boundary**: Relies solely on standard POSIX libc/kernel syscalls (`open64`, `close`, `lseek64`, `mmap64`, `munmap`, `mlock`, `sched_yield`).

---

## 2. Source Code Organization

```text
/mnt/c/Documents/Mojo/llamojo/
├── README.md                          # Repository documentation & quickstart
├── mojo.toml                          # Project manifest & module include search paths
├── run-tests.sh                       # Automated test suite runner
├── build/
│   └── llamojo.mojoc                  # Precompiled binary package (198 KB)
├── docs/
│   ├── llama_cpp_pure_mojo_study.md   # Architectural gap analysis & multi-phase roadmap
│   └── implementation_guide.md        # This technical implementation guide
├── vendor/                            # Sublibrary symlinks / Git submodules
│   ├── llama2.mojo -> /mnt/c/Documents/Mojo/llama2.mojo
│   └── MojoLlama -> /mnt/c/Documents/Mojo/MojoLlama
├── src/
│   └── llamojo/
│       ├── __init__.mojo              # Re-exports public library surface
│       ├── llama.mojo                 # Public llama.h API implementation
│       ├── types.mojo                 # GGML types, llama.h structs & ModelConfig
│       ├── gguf.mojo                  # Pure Mojo GGUF v2/v3 parser, mmap & synthetic writer
│       └── backend/
│           ├── __init__.mojo          # Backend subpackage re-exports
│           ├── isa.mojo               # Hardware feature detection & F16/BF16 conversions
│           ├── threadpool.mojo        # Persistent spin-barrier worker pool & Matrix
│           ├── kernels_norm_rope.mojo # Fused RMSNorm+Residual, SwiGLU, and RoPE
│           └── kernels_quant.mojo     # Quantized matmul kernels & fused QKV/Gate-Up
└── tests/
    ├── test_gguf.mojo                 # Tests GGUF parsing, mmap vs heap, and types
    ├── test_llama_api.mojo            # Tests llama.h API & multi-worker thread pool
    └── test_kernels.mojo              # Tests quantized dot products, fused norm, RoPE
```

---

## 3. Sublibrary Architecture & Loading

In Mojo 1.1.0, sublibraries are imported without code changes or wrapper overhead:

1. **Vendor Directory (`vendor/`)**:
   Links existing codebases directly into the project workspace:
   ```bash
   ln -sfn /mnt/c/Documents/Mojo/llama2.mojo vendor/llama2.mojo
   ln -sfn /mnt/c/Documents/Mojo/MojoLlama vendor/MojoLlama
   ```
2. **Search Path Inclusion (`-I`)**:
   Configured in `mojo.toml` and CLI commands:
   `-I src -I vendor/llama2.mojo -I vendor/MojoLlama/src`.
3. **Selective Reuse & Upgrades**:
   - `llama2.mojo` is already on Mojo 1.1.0 and is imported directly in [`src/llamojo/backend/threadpool.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/threadpool.mojo).
   - `MojoLlama`'s pure Mojo algorithms (GGUF parsing, ISA detection, K-quant bit layouts) were upgraded to Mojo 1.1.0 and integrated into `src/llamojo/`.
4. **Binary Package Precompilation (`.mojoc`)**:
   ```bash
   mojo precompile src/llamojo -I vendor/llama2.mojo -o build/llamojo.mojoc
   ```

---

## 4. Subsystems Detail

### 4.1 Upstream GGML Types & Sizing (`src/llamojo/types.mojo`)
- Implements exact numeric IDs from upstream `ggml.h`:
  - `F32` (0), `F16` (1), `Q4_0` (2), `Q4_1` (3), `Q5_0` (6), `Q5_1` (7), `Q8_0` (8), `Q8_1` (9), `Q2_K` (10), `Q3_K` (11), `Q4_K` (12), `Q5_K` (13), `Q6_K` (14), `Q8_K` (15), `BF16` (30), `MXFP4` (39).
- Accurately calculates block size (`ggml_block_size`) and byte size (`ggml_type_size`), correcting discrepancies present in legacy files.
- Provides `null_ptr[T]()` to model zero-initialized pointers in Mojo 1.1.0 without crashing non-nullability constraints.

### 4.2 Zero-Copy In-Place GGUF Loader (`src/llamojo/gguf.mojo`)
- Reads the 24-byte header (`magic`, `version`, `n_tensors`, `n_kv`).
- Parses all scalar types (`UINT8`..`FLOAT64`, `BOOL`, `STRING`) and arrays into `GGUFValue`.
- Maps files into memory via POSIX `mmap64` (`PROT_READ`, `MAP_PRIVATE`) with an automatic `pread64` fallback for non-mmap environments.
- Directly binds tensor pointers to `data_offset + tensor.offset`, enabling zero disk allocation and instant loading.
- Exposes `as_matrix()` on `GGUFTensorInfo` to create non-owning `llama2.Matrix` views.

### 4.3 Persistent Spin-Barrier Thread Pool (`src/llamojo/backend/threadpool.mojo`)
- Eliminates the ~200 µs thread wakeup penalty of repeated thread-pool dispatches by running worker threads continuously through an `Atomic[Int]` sense-reversing barrier with `sched_yield` backoff.
- Coordinates worker thread chunks through `chunk_start` and `chunk_end`.
- Manages per-worker private scratchpad memory (`xbp`) across layers.

### 4.4 Fused Normalization & RoPE (`src/llamojo/backend/kernels_norm_rope.mojo`)
- **`fused_rms_norm_residual`**: Computes residual addition and RMS normalization in a single pass, writing the un-normalized sum to `out_residual` and the normalized vector to `out_normed`.
- **`fused_silu_mul`**: SwiGLU activation ($\text{SiLU}(\text{gate}) \times \text{up}$) without intermediate buffer allocations.
- **`rope_head`**: Interleaved-pair RoPE rotation for `llama2.c`.
- **`apply_rope_neox`**: NeoX / half-split RoPE rotation for Llama-3, Qwen, and Mistral.

### 4.5 Quantized Matrix Kernels (`src/llamojo/backend/kernels_quant.mojo`)
- Pure Mojo dot products for `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, and `MXFP4`.
- **Bit-exact `k4_scale`**: Unpacks 12-byte scales for `Q4_K` and `Q5_K` using the exact bit-shift logic from `cengine_batch_instr.c:k4_scale` and upstream `llama.cpp`.
- **Activation Quantization**: `quantize_row_q8_0` dynamically quantizes activation vectors into `Q8_0` blocks.
- **Horizontal Fusion (`matmul_quant_fused[n]`)**: Fuses multiple projections (QKV with $n=3$, Gate/Up with $n=2$) into a single row loop, sharing input register loads across projections.

### 4.6 Public `llama.h` Compatibility API (`src/llamojo/llama.mojo`)
- `llama_backend_init` / `llama_backend_free`: Runtime initialization.
- `llama_model_default_params` / `llama_context_default_params`: Standard parameters matching `llama.cpp`.
- `llama_model_load_from_file` / `llama_model_free`: Lifecycle management returning opaque `llama_model_p` handles.
- Model metadata inspection functions: `llama_model_n_embd`, `llama_model_n_layer`, `llama_model_n_head`, `llama_model_n_head_kv`, `llama_model_n_ctx_train`, `llama_model_n_vocab`, `llama_model_desc`, `llama_model_meta_val_str`.

---

## 5. Verification & Testing

All modules are verified via the automated test runner [`run-tests.sh`](file:///mnt/c/Documents/Mojo/llamojo/run-tests.sh):

| Test Script | Validated Subsystems | Status |
| :--- | :--- | :--- |
| [`tests/test_gguf.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_gguf.mojo) | GGUF v3 parsing, `mmap` vs `heap` binding, non-owning `Matrix` views, GGML type block sizes | Passed |
| [`tests/test_llama_api.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_llama_api.mojo) | `llama.h` API, model inspection, persistent spin-barrier multi-worker pool | Passed |
| [`tests/test_kernels.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_kernels.mojo) | Fused RMSNorm+Res, SwiGLU, RoPE rotation, Q8_0 quant, `k4_scale`, Q4_K / MXFP4 dot, multi-matrix fusion ($n=3$) | Passed |
| `mojo precompile` | Clean binary package generation ([`build/llamojo.mojoc`](file:///mnt/c/Documents/Mojo/llamojo/build/llamojo.mojoc)) | Passed (0 warnings) |

