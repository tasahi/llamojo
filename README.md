# `llamojo`: Pure Mojo Implementation of `llama.cpp` (`libllama`)

A pure Mojo implementation of the `llama.cpp` library API and runtime, designed with **zero external compute dependencies** (zero Python runtime, zero C/C++ compiler or OpenMP/OpenBLAS runtime, and zero external `libllama.so`).

`llamojo` combines:
1. High-efficiency persistent thread-pooling and atomic spin barriers from [`llama2.mojo`](../llama2.mojo).
2. Deep GGUF metadata parsing, quantization types, and multi-architecture kernels from [`MojoLlama`](../MojoLlama).
3. Exact `llama.h` function signatures, parameter structs, and opaque handles (`llama_model_p`, `llama_model_load_from_file`, `llama_model_params`, etc.).

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Public llama.h API (src/llamojo/llama.mojo)          │
│  llama_backend_init | llama_model_load_from_file | llama_model_free    │
│  llama_model_default_params | llama_context_default_params             │
│  llama_model_n_embd | llama_model_n_layer | llama_model_desc           │
├────────────────────────────────────────────────────────────────────────┤
│                          Core Subsystems                               │
│  • types.mojo: llama.h types + upstream ggml.h enum & block sizes      │
│  • gguf.mojo: In-place GGUF v2/v3 parser + zero-copy POSIX mmap64      │
│  • vocab.mojo: GGUF vocab loader, detokenizer, & llama.h tokenizer API │
│  • vocab_gpt2.mojo: GPT-2 / ModernBERT byte-level BPE encoding engine  │
│  • vocab_gemma.mojo: Gemma 2 / Gemma 4 / mmBERT / SPM BPE engine       │
├────────────────────────────────────────────────────────────────────────┤
│                     Backend Compute & Synchronization                  │
│  • backend/threadpool.mojo: Persistent atomic spin-barrier pool        │
│  • backend/isa.mojo: Comptime AVX-512, VNNI, AVX2, F16/BF16 bitcasts   │
│  • backend/kernels_norm_rope.mojo: Fused RMSNorm+Res, SwiGLU, RoPE     │
│  • backend/kernels_quant.mojo: Q4_0..Q6_K, MXFP4, fused QKV/Gate-Up    │
├──────────────────────────────────────┬─────────────────────────────────┤
│        Mojo Standard Library         │       POSIX / Kernel ABI        │
│  • std.memory (Pointer, alloc)       │  • mmap64, munmap, mlock        │
│  • std.atomic (Atomic[Int])          │  • open64, close, lseek64       │
│  • max.algorithm (parallelize)       │  • sched_yield                  │
└──────────────────────────────────────┴─────────────────────────────────┘
```

---

## Sublibrary Integration Strategy

The project links existing codebases directly under `vendor/`:
- `vendor/llama2.mojo` $\rightarrow$ `/mnt/c/Documents/Mojo/llama2.mojo`
- `vendor/MojoLlama` $\rightarrow$ `/mnt/c/Documents/Mojo/MojoLlama`

In Mojo 1.1.0, these are incorporated without modifying upstream repos:
1. **Search Path Inclusion (`-I`)**:
   Modules are imported via `-I src -I vendor/llama2.mojo -I vendor/MojoLlama/src`.
2. **Direct Sublibrary Imports**:
   [`src/llamojo/backend/threadpool.mojo`](./src/llamojo/backend/threadpool.mojo) imports `Matrix`, `BufferPtrFloat32`, `nelts`, `chunk_start`, `chunk_end`, `spin_wait`, `ForwardArgs`, `PoolShared`, and `barrier` directly from `llama2`.
3. **Precompilation**:
   The entire library precompiles into a standalone binary artifact (`build/llamojo.mojoc`) via `mojo precompile`.

---

## Implemented Subsystems (Phases 1, 2 & 3)

### 1. Types & Data Layouts ([`src/llamojo/types.mojo`](./src/llamojo/types.mojo))
- Upstream `ggml.h` quantization enum parity: `F32 (0)`, `F16 (1)`, `Q4_0 (2)`, `Q4_1 (3)`, `Q5_0 (6)`, `Q5_1 (7)`, `Q8_0 (8)`, `Q8_1 (9)`, `Q2_K (10)`, `Q3_K (11)`, `Q4_K (12)`, `Q5_K (13)`, `Q6_K (14)`, `Q8_K (15)`, `BF16 (30)`, `MXFP4 (39)`.
- Verified block element sizes (`ggml_block_size`) and block byte sizes (`ggml_type_size`).
- Full `llama.h` ABI parameter structs:
  - `llama_model_params`: `n_gpu_layers`, `split_mode`, `use_mmap`, `use_mlock`, `check_tensors`.
  - `llama_context_params`: `n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `flash_attn`.
  - `llama_model_quantize_params`, `llama_batch`, `llama_token_data`, `llama_token_data_array`.
- Dynamic runtime model configuration: [`ModelConfig`](./src/llamojo/types.mojo#L323-L361).

### 2. Zero-Copy GGUF Loader & Parser ([`src/llamojo/gguf.mojo`](./src/llamojo/gguf.mojo))
- Full GGUF v2/v3 binary spec implementation in 100% pure Mojo.
- Eliminates Python dictionaries and intermediate `/tmp/*.bin` extraction.
- **POSIX `mmap64` tensor binding**: Tensor data pointers map directly to the file descriptor offset (`data_offset + tensor_offset`), with `pread64` fallback for non-mmap environments.
- Non-owning `Matrix` view construction via [`GGUFTensorInfo.as_matrix()`](./src/llamojo/gguf.mojo#L231-L243).
- Embedded GGUF v3 synthetic writer ([`write_synthetic_gguf`](./src/llamojo/gguf.mojo#L607-L741)) for offline unit testing.

### 3. Persistent Spin-Barrier Worker Pool ([`src/llamojo/backend/threadpool.mojo`](./src/llamojo/backend/threadpool.mojo))
- Reusable sense-reversing spin barrier over `std.atomic.Atomic[Int]` with `sched_yield` backoff.
- Dispatches worker threads once per layer/pass instead of thousands of individual thread-pool wakeups.
- Wraps shared state into [`LlamaWorkerPool`](./src/llamojo/backend/threadpool.mojo#L79-L115), managing per-worker scratchpads (`xbp`) aligned to hardware core counts.

### 4. Fused Normalization, SwiGLU & RoPE ([`src/llamojo/backend/kernels_norm_rope.mojo`](./src/llamojo/backend/kernels_norm_rope.mojo))
- [`fused_rms_norm_residual`](./src/llamojo/backend/kernels_norm_rope.mojo#L24-L65): Fuses residual connection addition and RMS normalization into a single memory pass, computing $x + \text{residual} \rightarrow \text{out\_res}$ and scaling into $\text{out\_normed}$.
- [`fused_silu_mul`](./src/llamojo/backend/kernels_norm_rope.mojo#L108-L119): SwiGLU activation ($\text{SiLU}(\text{gate}) \times \text{up}$) without intermediate buffer allocations.
- [`rope_head`](./src/llamojo/backend/kernels_norm_rope.mojo#L123-L144): Interleaved pair RoPE rotation for `llama2.c`.
- [`apply_rope_neox`](./src/llamojo/backend/kernels_norm_rope.mojo#L146-L184): NeoX / half-split RoPE rotation for Llama-3, Qwen, and Mistral models.

### 5. Unified Quantized Kernels & Multi-Matrix Dispatch ([`src/llamojo/backend/kernels_quant.mojo`](./src/llamojo/backend/kernels_quant.mojo))
- High-performance dot products across all major formats:
  - FP32, FP16, BF16
  - Block quants: `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`
  - K-quants: `Q4_K`, `Q5_K`, `Q6_K`
  - Microscaling: `MXFP4` (16-byte nibbles + E8M0 scale byte)
- Bit-exact scale unpacking: [`k4_scale`](./src/llamojo/backend/kernels_quant.mojo#L34-L44) matches upstream `cengine_batch_instr.c` and `llama.cpp` for 12-byte scales packing.
- On-the-fly activation quantization: [`quantize_row_q8_0`](./src/llamojo/backend/kernels_quant.mojo#L48-L77).
- Horizontal multi-matrix projection fusion: [`matmul_quant_fused[n]`](./src/llamojo/backend/kernels_quant.mojo#L309-L336) shares activation register loads across QKV ($n=3$) and Gate/Up ($n=2$).

### 6. KV Cache & PagedAttention PageTable ([`src/llamojo/kv_cache.mojo`](./src/llamojo/kv_cache.mojo))
- Unified KV Cache supporting contiguous Float32 and quantized `Q8_0` buffers.
- Sequence metadata per slot (`LlamaKVCell: pos, seq_id`) implementing sequence manipulation:
  - `llama_kv_cache_clear`, `llama_kv_cache_seq_rm`, `llama_kv_cache_seq_cp`, `llama_kv_cache_seq_keep`, `llama_kv_cache_seq_add`.
- PagedAttention `PageTable` translation layer for fragmented dynamic batching.

### 7. Attention & Multi-Architecture Forward Engine ([`src/llamojo/backend/kernels_attn.mojo`](./src/llamojo/backend/kernels_attn.mojo) & [`src/llamojo/backend/models.mojo`](./src/llamojo/backend/models.mojo))
- Online numerically stable Grouped-Query Attention (`gqa_decode` / `gqa_decode_head`) supporting arbitrary head counts (`n_heads // n_kv_heads`).
- Full multi-layer forward pass (`forward_layer_dense`) coordinating worker thread slices through the persistent spin-barrier pool.

### 8. Public `llama.h` Context & Inference API ([`src/llamojo/llama.mojo`](./src/llamojo/llama.mojo))
- Context initialization and activation scratchpad workspace: `llama_init_from_model`, `llama_free`.
- Batch management: `llama_batch_init`, `llama_batch_get_one`, `llama_batch_free`.
- Inference engine: `llama_decode`, `llama_encode`.
- Logits & embeddings extraction: `llama_get_logits`, `llama_get_logits_ith`, `llama_get_embeddings`.
- Model reflection: `llama_model_n_embd`, `llama_model_n_layer`, `llama_model_n_head`, `llama_model_n_head_kv`, `llama_model_n_ctx_train`, `llama_model_n_vocab`, `llama_model_desc`, `llama_model_meta_val_str`.

### 9. Modular Vocabulary & Multi-Architecture Tokenization ([`src/llamojo/vocab.mojo`](./src/llamojo/vocab.mojo), [`vocab_gpt2.mojo`](./src/llamojo/vocab_gpt2.mojo), [`vocab_gemma.mojo`](./src/llamojo/vocab_gemma.mojo))
- **[`src/llamojo/vocab.mojo`](./src/llamojo/vocab.mojo)**: GGUF vocabulary metadata loader (`llama_vocab`), model-type router (`tokenize_text`), UTF-8 piece decoder (`token_to_piece_str`), and public `llama.h` functions (`llama_tokenize`, `llama_detokenize`, `llama_token_to_piece`).
- **[`src/llamojo/vocab_gpt2.mojo`](./src/llamojo/vocab_gpt2.mojo)**: GPT-2 / LLaMA-3 / Qwen / English ModernBERT byte-level BPE engine with 256-entry Unicode byte remapping (`get_gpt2_byte_table`), regex-equivalent chunk splitting (`split_gpt2_chunks`), and rank-based merging (`tokenize_gpt2_bpe`).
- **[`src/llamojo/vocab_gemma.mojo`](./src/llamojo/vocab_gemma.mojo)**: SentencePiece and Gemma-family (`gemma2`, `gemma4`, `mmbert`, LLaMA SPM) BPE engine with doubly-linked UTF-8 symbol priority-queue merging (`_SpmSymbol`, `_SpmBigram`), configurable `\u2581` space normalization (`normalize_spm_spaces` with `add_space_prefix=False` for Gemma/mmBERT), and `<0xXX>` hexadecimal byte fallback (`encode_hex_byte_fallback`).

---

## Building and Testing

### Prerequisites
- Mojo compiler 1.1.0 (MAX >= 26.6.0)

### Running the Test Suite
Execute the test runner script:
```bash
./run-tests.sh
```
Or run individual test modules:
```bash
# Test 1: GGUF v3 parser, mmap binding, and GGML types
mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_gguf.mojo

# Test 2: llama.h public API and persistent worker pool
mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_llama_api.mojo

# Test 3: Quantized kernels, fused norm/SwiGLU, and RoPE
mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_kernels.mojo
```

### Precompiling the Binary Package
```bash
mojo precompile src/llamojo -I vendor/llama2.mojo -o build/llamojo.mojoc
```
Produces `build/llamojo.mojoc` with zero warnings and zero runtime dependencies.
