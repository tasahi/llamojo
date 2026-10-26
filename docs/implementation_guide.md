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

### 4.7 KV Cache & PagedAttention Subsystem (`src/llamojo/kv_cache.mojo`)
- **`PageTable`**: Block allocation manager organizing key/value context memory into discrete pages with physical-to-logical translation tables, free list reclamation, and multi-sequence tracking.
- **`LlamaKVCache`**: Manages sliding-window or non-contiguous KV cells across all model layers.
- **Sequence Manipulation**: Implements `llama_kv_cache_clear`, `llama_kv_cache_seq_rm`, `llama_kv_cache_seq_cp`, `llama_kv_cache_seq_keep`, and `llama_kv_cache_seq_add` for multi-turn conversational pruning and prefix caching.

### 4.8 Transformer Decode Engine (`src/llamojo/llama.mojo`)
- **`llama_decode`**: Full multi-layer auto-regressive transformer forward pass executing:
  - Fused RMSNorm on input hidden states.
  - Multi-threaded quantized matmul for queries, keys, and values.
  - NeoX / interleaved rotary positional embeddings (RoPE).
  - PagedAttention and Grouped-Query Attention (`gqa_decode`).
  - Output projection matmul and residual addition.
  - FFN pre-norm followed by fused Gate/Up SwiGLU projection.
  - Down projection matmul and final residual accumulation.
  - Output norm and final vocabulary projection to logits buffer (`llama_get_logits`, `llama_get_logits_ith`).

### 4.9 GGUF Vocabulary & Tokenizer Engine (`src/llamojo/vocab.mojo`)
- **Metadata Extraction**: Extracts vocabulary tokens, scores, token types, and special token markers (`bos`, `eos`, `eot`, `pad`) directly from GGUF metadata dictionaries.
- **BPE Tokenization**: Byte-Pair Encoding implementation supporting recursive merges, SentencePiece whitespace markers (` `, U+2581), and hexadecimal byte fallbacks (`<0xXX>`).
- **SentencePiece Unigram Tokenization (Phase 7)**: Implements dynamic programming Viterbi trellis search finding the maximum-likelihood token segmentation based on vocabulary log-probabilities, supporting byte fallbacks (`<0xXX>`) and whitespace marker handling.
- **Unified Dispatcher**: `tokenize_text` automatically delegates to BPE or SPM based on detected `tokenizer.ggml.model` metadata (`VOCAB_TYPE_SPM`, `VOCAB_TYPE_BPE`, etc.).
- **Public API**: `llama_tokenize`, `llama_token_to_piece`, `llama_detokenize`, and query helpers (`llama_vocab_bos`, `llama_vocab_eos`, `llama_vocab_is_eog`, `llama_vocab_n_tokens`, `llama_vocab_type`, `llama_vocab_get_score`).

### 4.10 Composable Sampling Pipeline (`src/llamojo/sampler.mojo`)
- **Chain Architecture**: Tagged-union `llama_sampler` struct supporting arbitrary chained filters with `llama_sampler_chain_init` and `llama_sampler_chain_add`.
- **Supported Stages**:
  - Repetition, frequency, and presence penalties (`llama_sampler_init_penalties`).
  - Temperature scaling (`llama_sampler_init_temp`).
  - Top-K filtering (`llama_sampler_init_top_k`).
  - Softmax probability calculation and Top-P / Nucleus sampling (`llama_sampler_init_top_p`).
  - Min-P noise-threshold filtering (`llama_sampler_init_min_p`).
  - Greedy argmax (`llama_sampler_init_greedy`) and PRNG categorical distribution sampling (`llama_sampler_init_dist`).
  - Context acceptance tracking (`llama_sampler_accept`) and memory teardown (`llama_sampler_free`).

### 4.11 Multi-threaded Block Quantizer & GGUF Converter (`src/llamojo/quantize.mojo`)
- **Pure-Mojo Quantizers**: Multi-threaded parallel block encoders (`parallelize`) for `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, and `Q6_K`.
- **IEEE 754 Float16 Encoder**: Pure-Mojo `encode_f16` for block scale factors.
- **Full GGUF Metadata Mirroring (Phase 5)**: Preserves all arbitrary key-value pairs (chat templates, tokenizer tokens and scores, architecture parameters) from input models, dynamically writes target `general.file_type = ftype`, and updates `general.quantization_version = 2`.
- **`llama_model_quantize`**: Full streaming model converter reading source GGUF files and emitting quantized v3 files with 32-byte aligned metadata headers and aligned tensor payload buffers.

### 4.12 C Foreign Function Interface ABI Export (`src/llamojo/c_api.mojo`)
- **Direct Linkable C-ABI (`libllama.so`)**: Exposes native `@export` symbols with standard `abi("C")` calling conventions matching `llama.h` function signatures for zero-wrapper foreign language integration (C, C++, Rust, `llama-cpp-python`).
- **Comprehensive API Surface**: Covers backend system lifecycle, model and context instantiation, batch execution, KV cache manipulation, BPE tokenization/detokenization, composable sampler pipeline stages, and quantization.

### 4.13 CLI Utilities & HTTP Inference Server (`tools/`)
- [`tools/llama_cli.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tools/llama_cli.mojo): Standalone interactive command-line generation tool.
- [`tools/llama_bench.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tools/llama_bench.mojo): Micro-benchmarking harness measuring prompt processing and generation throughput (tok/s).
- [`tools/llama_quantize.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tools/llama_quantize.mojo): CLI interface for model quantization.
- [`tools/llama_server.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tools/llama_server.mojo): Zero-dependency HTTP inference server using POSIX socket syscalls.

---

## 5. Verification & Testing

All modules are verified via the automated test runner [`run-tests.sh`](file:///mnt/c/Documents/Mojo/llamojo/run-tests.sh):

| Test Script | Validated Subsystems | Status |
| :--- | :--- | :--- |
| [`tests/test_gguf.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_gguf.mojo) | GGUF v3 parsing, `mmap` vs `heap` binding, non-owning `Matrix` views, GGML type block sizes | Passed |
| [`tests/test_llama_api.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_llama_api.mojo) | `llama.h` API, model inspection, persistent spin-barrier multi-worker pool | Passed |
| [`tests/test_kernels.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_kernels.mojo) | Fused RMSNorm+Res, SwiGLU, RoPE rotation, Q8_0 quant, `k4_scale`, Q4_K / MXFP4 dot, multi-matrix fusion ($n=3$) | Passed |
| [`tests/test_context_decode.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_context_decode.mojo) | KV Cache, PageTable allocation, sequence manipulation, `llama_batch`, multi-layer `llama_decode` forward pass | Passed |
| [`tests/test_vocab_sampler.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_vocab_sampler.mojo) | GGUF vocabulary BPE & SentencePiece Unigram (Viterbi trellis) tokenization, detokenization, composable sampler pipeline (penalties, temp, top-k, top-p, greedy), block quantizers (Q4_0, Q8_0, Q4_K, Q6_K), and full metadata preservation serialization | Passed |
| [`tests/test_c_abi.c`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_c_abi.c) | Direct dynamic linking of `build/libllama.so` with `gcc -lllama`, verifying C-ABI function signatures, raw opaque pointers, backend lifecycle, capabilities query, and greedy sampler creation | Passed |
| `mojo build --emit shared-lib` | Clean native shared library generation ([`build/libllama.so`](file:///mnt/c/Documents/Mojo/llamojo/build/libllama.so) — 447 KB) | Passed (0 errors) |
| `mojo precompile` | Clean binary package generation ([`build/llamojo.mojoc`](file:///mnt/c/Documents/Mojo/llamojo/build/llamojo.mojoc) — 371 KB) | Passed (0 errors) |
