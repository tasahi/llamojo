# Pure Mojo Implementation of `llama.cpp` (`libllama`): Architectural Study & Gap Analysis

**Target Repository:** `/mnt/c/Documents/Mojo/llamojo`  
**Source Assets Analyzed:**
- [`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo) (`/mnt/c/Documents/Mojo/llama2.mojo`)
- [`MojoLlama`](file:///mnt/c/Documents/Mojo/MojoLlama/README.md) (`/mnt/c/Documents/Mojo/MojoLlama`)

---

## 1. Executive Summary

This study evaluates how to construct a **pure Mojo implementation of the `llama.cpp` library (`libllama`)** that preserves the exact C/Mojo API call names, parameter signatures, opaque handle semantics, and core runtime functionalities, while achieving **minimal external integrations** (zero Python runtime, zero external C/C++ compute libraries such as OpenBLAS/OpenMP, and zero dependency on `libllama.so`).

By combining the architectural strengths of **`llama2.mojo`** (idiomatic Mojo 1.1 memory management, zero-overhead sense-reversing atomic spin barriers, and multi-matrix SIMD horizontal fusion) with **`MojoLlama`** (broad GGUF quantization support, multi-architecture forward passes including MoE and Mamba-2 SSM, K-quant/MXFP4 kernels, and full sampling pipelines), `llamojo` can implement ~80% of `llama.cpp`'s core CPU inference and quantization surface natively in Mojo today, with a clear, self-contained path for the remaining 20% (in-place `mmap` GGUF loading, Paged/Unified KV cache management, and GBNF grammar state machines).

---

## 2. Inventory & Comparative Analysis of Existing Codebases

### 2.1 `/mnt/c/Documents/Mojo/llama2.mojo`
[`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo) is a clean, single-file (1,104 LOC) pure Mojo port of Karpathy's `llama2.c` updated for **Mojo 1.1.0 (MAX 26.6.0)**.

| Subsystem | Implementation Details | Reusable Value for `llamojo` |
| :--- | :--- | :--- |
| **Memory & Tensor Wrapper** | [`Matrix`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L26-L146) struct using `Pointer[Float32, MutUntrackedOrigin]` and `unsafe_alloc[Float32]`. Supports non-owning views/slices (`allocated = 0`) and 1D/2D/3D indexing. | High — provides a clean foundation for a native `ggml_tensor` equivalent without Python or C `malloc` wrappers. |
| **Thread Synchronization** | [`PoolShared`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L663-L678) & [`barrier()`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L682-L695) implementing a sense-reversing spin barrier over `std.atomic.Atomic[Int]` with `sched_yield` backoff ([`spin_wait`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L656-L659)). | **Critical** — solves the ~200 µs thread-pool wakeup penalty of calling `parallelize` 37 times per token by dispatching workers **once** per forward pass ([`Transformer.transformer`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L877-L908)) and synchronizing stages in ~4 µs. |
| **Fused SIMD Matmul** | [`matmul_rows[n]`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L431-L466) takes a `StaticTuple` of $n$ weight pointers and computes $n$ matrix-vector products simultaneously (`matmul_rows[3]` for QKV, `matmul_rows[2]` for `w1`/`w3`), sharing input vector loads across registers. | High — directly applicable to `llama_decode` QKV and SwiGLU gate/up projections across all quantization types. |
| **Tokenizer (BPE)** | [`Tokenizer`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L180-L229) & [`bpe_encode`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L928-L960) using native Mojo `Dict[String, Int]` and `List[String]` for merge-score priority lookup. | Moderate — proves pure-Mojo BPE encoding works without Python, though it must be extended to read vocabulary tables from GGUF metadata rather than `tokenizer.bin`. |

### 2.2 `/mnt/c/Documents/Mojo/MojoLlama`
[`MojoLlama`](file:///mnt/c/Documents/Mojo/MojoLlama/README.md) is a large, multi-engine platform containing both a legacy Python/C/OpenMP stack and a newer **Pure Mojo Engine** (`src/mojollama/mojo_engine/` and `src/mojollama/kernels/`). For a pure Mojo `llama.cpp` implementation, we strictly extract from the pure Mojo modules and port the remaining C algorithms (`cengine_batch_instr.c`) into Mojo:

| Module | File Reference | Capabilities & Reusable Value for `llamojo` |
| :--- | :--- | :--- |
| **GGUF Types & Sizing** | [`gguf/types.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/gguf/types.mojo#L1-L156) | Defines `GGUF_MAGIC`, `GGUFValueType`, `GGMLType` (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q8_K, IQ1–IQ4), [`ggml_type_size`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/gguf/types.mojo#L47-L99), and [`ggml_block_size`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/gguf/types.mojo#L101-L153). *(Note: Fix minor block byte-size discrepancies in `types.mojo` using the verified sizes in `mojo_quant_kernels.mojo`).* |
| **Native Binary GGUF Parser** | [`mojo_engine/gguf_extract.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_extract.mojo#L51-L155)<br>[`mojo_engine/gguf_scanner.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_scanner.mojo#L33-L105) | Pure Mojo binary header, KV metadata, and tensor info parser using direct POSIX file descriptors (`open`, `lseek`, `read`) with zero Python dependencies (unlike [`gguf/reader.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/gguf/reader.mojo), which relies on `PythonObject`). |
| **ISA & SIMD Dispatch** | [`kernels/isa_dispatch.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/isa_dispatch.mojo#L15-L165) | Compile-time target detection (`CompilationTarget.has_avx512f()`, `has_avx512vnni()`, `has_avx2()`, `has_fma()`) and hardware-accelerated `f16_to_f32` via `bitcast[DType.float16]`. |
| **Quantized Matmul Kernels** | [`mojo_engine/mojo_quant_kernels.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/mojo_quant_kernels.mojo#L46-L510)<br>[`kernels/kquants.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/kquants.mojo#L44-L548)<br>[`kernels/optimized_kernels.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/optimized_kernels.mojo#L47-L278)<br>[`mojo_engine/simd_mxfp4.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/simd_mxfp4.mojo#L4-L50) | Pure Mojo dequantization and dot-product kernels for **F32, F16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, and MXFP4**, including 4-row register blocking (`q4_mm_regblock`) and batched Q8_0 FMA (`_mm_q8_batch` in [`tinyllama_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/tinyllama_gen_q8.mojo#L110-L200)). |
| **Quantization & GGUF Writer** | [`mojo_engine/mojo_quantize.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/mojo_quantize.mojo#L54-L454) | Pure Mojo parallel quantizers (`q_q4`, `q_q5`, `q_q8`, `q_q4k`, `q_q5k`, `q_q6k`, `q_mx`) and binary GGUF file writer. Directly implements `llama_model_quantize`. |
| **Multi-Arch Forward Pass** | [`mojo_engine/universal_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L259-L443)<br>[`mojo_engine/zaya_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/zaya_gen_q8.mojo#L1-L705)<br>[`mojo_engine/ssm_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/ssm_forward.mojo#L82-L201) | Supports Dense GQA (Llama 2/3, Mistral, Gemma), Mixture of Experts ([`moe_router`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L276-L309), [`moe_ffn`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L311-L332)), Cross-Layer Attention (CCA), partial RoPE, learned residual scales, and Mamba-2 Selective State Space Models (`ssm_forward`). |
| **Sampling Pipeline** | [`kernels/sampling.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/sampling.mojo#L22-L419)<br>[`mojo_engine/sampling.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/sampling.mojo#L47-L120) | Full `llama.cpp`-ordered sampling chain: repetition penalty, temperature scaling, min-heap top-k (`topk_extract`), nucleus top-p (`top_p_filter`), min-p (`min_p_filter`), greedy argmax, and PCG LCG random sampling. |
| **HTTP Server** | [`server.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/server.mojo#L1-L207) | Zero-dependency HTTP server written in pure Mojo over POSIX sockets (`socket`, `bind`, `listen`, `accept`). Provides a standalone foundation for `llama-server`. |

---

## 3. Mapping `llama.cpp` API Surface to `llamojo`

To maintain `llama.cpp` API compatibility (function names, parameter lists, and struct layouts) while writing pure Mojo, `llamojo` should expose both:
1. **Native Mojo API**: Functions and structs with identical names (`llama_model_load_from_file`, `llama_init_from_model`, `llama_decode`, etc.) importable by any Mojo application.
2. **C ABI Export Layer (Optional/Zero-Cost)**: Using `@export("llama_...", abi="C")` on the exact same functions so external C/Rust/Python/Go callers can link against `liblamojo.so` as a drop-in replacement for `libllama.so`.

### 3.1 Core `llama.h` Types & Parameter Structs

```mojo
# Opaque handles (mirroring llama.h pointers via typed Mojo pointers)
comptime llama_model_p = Pointer[llama_model, MutUntrackedOrigin]
comptime llama_context_p = Pointer[llama_context, MutUntrackedOrigin]
comptime llama_sampler_p = Pointer[llama_sampler, MutUntrackedOrigin]
comptime llama_vocab_p = Pointer[llama_vocab, MutUntrackedOrigin]

comptime llama_token = Int32
comptime llama_pos = Int32
comptime llama_seq_id = Int32

@fieldwise_init
struct llama_model_params(Copyable, Movable):
    var n_gpu_layers: Int32
    var split_mode: Int32
    var main_gpu: Int32
    var vocab_only: Bool
    var use_mmap: Bool
    var use_mlock: Bool
    var check_tensors: Bool

@fieldwise_init
struct llama_context_params(Copyable, Movable):
    var n_ctx: UInt32
    var n_batch: UInt32
    var n_ubatch: UInt32
    var n_seq_max: UInt32
    var n_threads: Int32
    var n_threads_batch: Int32
    var rope_scaling_type: Int32
    var rope_freq_base: Float32
    var rope_freq_scale: Float32
    var type_k: UInt32          # GGMLType for K cache (F16, F32, Q8_0, Q4_0)
    var type_v: UInt32          # GGMLType for V cache
    var embeddings: Bool
    var offload_kqv: Bool
    var flash_attn: Bool

@fieldwise_init
struct llama_batch(Copyable, Movable):
    var n_tokens: Int32
    var token: Pointer[llama_token, MutUntrackedOrigin]
    var embd: Pointer[Float32, MutUntrackedOrigin]
    var pos: Pointer[llama_pos, MutUntrackedOrigin]
    var n_seq_id: Pointer[Int32, MutUntrackedOrigin]
    var seq_id: Pointer[Pointer[llama_seq_id, MutUntrackedOrigin], MutUntrackedOrigin]
    var logits: Pointer[Int8, MutUntrackedOrigin]
```

### 3.2 Function-by-Function API Mapping & Source Synthesis

| `llama.cpp` API Function | Source Asset to Use | Implementation Strategy in `llamojo` |
| :--- | :--- | :--- |
| `llama_backend_init()`<br>`llama_backend_free()` | [`isa_dispatch.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/isa_dispatch.mojo)<br>[`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L849-L858) | Initialize global ISA capability flags, thread count defaults (`num_performance_cores()`), and global timing state. |
| `llama_model_default_params()`<br>`llama_context_default_params()`<br>`llama_model_quantize_default_params()` | New boilerplate | Return default-initialized `llama_model_params`, `llama_context_params`, and `llama_model_quantize_params` structs matching `llama.cpp` defaults. |
| `llama_model_load_from_file(path_model, params)`<br>`llama_model_free(model)` | [`gguf_scanner.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_scanner.mojo)<br>[`gguf_extract.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_extract.mojo)<br>[`llama2.mojo:Matrix`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L55-L60) | Parse GGUF v3 header, KV dictionary, and tensor table in pure Mojo. Instead of dumping `.bin` files to `/tmp` (as `gguf_extract.mojo` does), map the file directly into virtual memory via POSIX `mmap` (`use_mmap = True`) or `pread` into aligned heap allocations, and construct non-owning tensor descriptors pointing to `mmap_base + data_start_offset + tensor_offset`. |
| `llama_init_from_model(model, params)`<br>`llama_free(ctx)` | [`engine_v3.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L69-L81)<br>[`llama2.mojo:RunState`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L270-L295)<br>[`llama2.mojo:PoolShared`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L663-L678) | Allocate activation scratchbuffers (`x`, `xb`, `hb`, `q`, `k`, `v`, `att`), output `logits` buffer (`n_batch * vocab_size`), KV cache (F32, F16, or Q8_0 via [`engine_v3.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L360-L376)), and persistent spin-barrier worker state (`PoolShared`). |
| `llama_batch_init(n_tokens, embd, n_seq_max)`<br>`llama_batch_get_one(tokens, n_tokens)`<br>`llama_batch_free(batch)` | [`tinyllama_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/tinyllama_gen_q8.mojo#L431-L438) | Allocate or wrap `llama_batch` arrays. `llama_batch_get_one` provides a zero-allocation view over a contiguous `tokens` array with auto-incrementing positions. |
| `llama_decode(ctx, batch)`<br>`llama_encode(ctx, batch)` | [`llama2.mojo:forward_worker`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L727-L843)<br>[`universal_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L344-L443)<br>[`ssm_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/ssm_forward.mojo#L82-L201) | Combine `llama2.mojo`'s single-dispatch spin-barrier worker execution with `universal_forward.mojo`'s runtime architecture/quantization dispatch. Only compute LM-head logits for batch indices where `batch.logits[i] != 0` (or the final token in `llama_batch_get_one`). |
| `llama_get_logits(ctx)`<br>`llama_get_logits_ith(ctx, i)`<br>`llama_get_embeddings(ctx)` | [`llama2.mojo:RunState`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L278)<br>[`engine_v3.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L107) | Return raw `Pointer[Float32, MutUntrackedOrigin]` to the context's logits buffer or the $i$-th batch token's logits/embeddings slice. |
| `llama_kv_cache_clear(ctx)`<br>`llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)`<br>`llama_kv_cache_seq_cp(...)`<br>`llama_kv_cache_seq_keep(...)`<br>`llama_kv_cache_seq_add(...)` | [`cengine_batch_instr.c:PageTable`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/cengine_batch_instr.c#L160-L200)<br>[`rope_kv_cache.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/rope_kv_cache.mojo#L35-L46) | Port the `PageTable` struct from `cengine_batch_instr.c` into pure Mojo and add per-slot `(seq_id, pos)` metadata matching `llama.cpp`'s `llama_kv_cache_unified`. |
| `llama_tokenize(...)`<br>`llama_token_to_piece(...)`<br>`llama_detokenize(...)`<br>`llama_chat_apply_template(...)` | [`llama2.mojo:Tokenizer`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L180-L229)<br>[`llama2.mojo:bpe_encode`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L928-L960)<br>[`zaya_gen_q8.mojo:decode_token`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/zaya_gen_q8.mojo#L241-L250) | Unify `llama2.mojo`'s `Dict`-backed BPE merger with GGUF `tokenizer.ggml.tokens`, `scores`, `token_type`, and `merges` arrays; add a WPM/Unigram path for SentencePiece/Gemma models. |
| `llama_sampler_chain_init(...)`<br>`llama_sampler_chain_add(...)`<br>`llama_sampler_init_top_k(...)`<br>`llama_sampler_init_top_p(...)`<br>`llama_sampler_init_min_p(...)`<br>`llama_sampler_init_temp(...)`<br>`llama_sampler_init_penalties(...)`<br>`llama_sampler_init_greedy()`<br>`llama_sampler_init_dist(seed)`<br>`llama_sampler_sample(smpl, ctx, idx)` | [`kernels/sampling.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/sampling.mojo#L98-L419)<br>[`mojo_engine/sampling.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/sampling.mojo#L47-L120) | Wrap existing functions (`apply_repetition_penalty`, `top_k_filter` / `topk_extract`, `top_p_filter`, `min_p_filter`, `apply_temperature`, `sample_greedy`, `sample_random`) into acomposable `llama_sampler` struct operating on `llama_token_data_array`. |
| `llama_model_quantize(fname_inp, fname_out, params)` | [`mojo_engine/mojo_quantize.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/mojo_quantize.mojo#L54-L454) | Wire `mojo_quantize.mojo`'s parallel block quantizers (`Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4`) into a streaming GGUF-to-GGUF tensor transformation pipeline. |

---

## 4. Gap Analysis: What Is Missing and Needs to Be Built

Despite extensive coverage across the two repositories, **eight specific technical gaps** stand between the current files and a complete, production-grade `llama.cpp` replacement in pure Mojo.

### Gap 1: Direct In-Place GGUF Loading via `mmap` (Eliminating Intermediate `.bin` Extraction)
- **Current State**:
  - [`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L312-L365) reads flat, headerless `llama2.c` FP32 binary checkpoints sequentially.
  - [`MojoLlama/src/mojollama/gguf/reader.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/src/mojollama/gguf/reader.mojo#L1-L136) uses `Python.evaluate("open")` and returns a `PythonObject` dictionary (violating zero-integration).
  - [`MojoLlama/src/mojollama/mojo_engine/gguf_extract.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_extract.mojo#L51-L220) is pure Mojo, but it extracts every tensor into separate `/tmp/<tensor>.bin` files on disk, which `tinyllama_gen_q8.mojo` and `zaya_gen_q8.mojo` then load individually.
- **What `llama.cpp` Does**: `llama_model_load_from_file` parses the GGUF header and tensor info directory in-place, calls `mmap()` on the file descriptor, and binds each tensor's data pointer directly to `mmap_base + data_offset + tensor_offset` with zero disk duplication and instant startup.
- **How to Close in Pure Mojo**:
  - Combine the pure Mojo KV/tensor parser from [`gguf_scanner.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_scanner.mojo#L62-L90) and [`gguf_extract.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_extract.mojo#L78-L154) into a native `GGUFContext` struct storing metadata in Mojo `Dict[String, GGUFValue]` and tensor headers in `Dict[String, GGUFTensorInfo]`.
  - Use a single POSIX `@extern("mmap")` call (or `MapViewOfFile` on Windows) to map the GGUF file into memory and assign `Pointer[UInt8, MutExternalOrigin]` views directly to the model's weight matrices.

### Gap 2: Dynamic Runtime Model Configuration (Replacing `comptime` Hardcoding)
- **Current State**:
  - [`universal_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L26-L47), [`zaya_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/zaya_gen_q8.mojo#L20-L39), and [`tinyllama_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/tinyllama_gen_q8.mojo#L11-L24) hardcode `comptime NE`, `NH`, `NK`, `HD`, `NL`, `FF`, `NV`, and `QT`, requiring recompilation for every model.
  - Conversely, [`llama2.mojo:Config`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L230-L269) and [`engine_v3.mojo:ModelConfig`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L33-L45) load dimensions dynamically at runtime.
- **How to Close in Pure Mojo**:
  - Use [`engine_v3.mojo:ModelConfig`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L33-L45) populated dynamically from GGUF metadata (`{arch}.embedding_length`, `{arch}.block_count`, `{arch}.attention.head_count`, `{arch}.attention.head_count_kv`, `{arch}.rope.freq_base`, etc.).
  - Keep `comptime` specialization **only** for hardware SIMD width (`nelts` / `SIMD_WIDTH`), register tile sizes (`RPW`), and quantized block constants (`QK`, `QB`), while passing matrix dimensions (`nr`, `nc`, `n_heads`, `head_dim`) as runtime parameters—just as [`llama2.mojo:ForwardArgs`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L571-L646) does.

### Gap 3: Pre-Allocated Workspace & Persistent Thread Pool Across All Kernels
- **Current State**:
  - [`universal_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L268-L269) and [`ssm_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/ssm_forward.mojo#L104-L108) call `@extern("malloc")` inside the per-layer hot path (and often leak those buffers), while also invoking `parallelize` on every individual matrix multiplication (causing 7–10 ms of thread wakeup overhead per token, as diagnosed in [`agents-optimization.md`](file:///mnt/c/Documents/Mojo/llama2.mojo/agents-optimization.md#L15-L19)).
  - Furthermore, [`universal_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L345) stores `weight_addrs` as `UnsafePointer[Int32, ...]` instead of `Int64`, which truncates 64-bit pointers!
- **How to Close in Pure Mojo**:
  - Adopt [`llama2.mojo`'s `RunState`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L270-L295) + [`PoolShared`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L663-L695) pattern globally inside `llama_context`: allocate all layer scratchpads once during `llama_init_from_model` using `unsafe_alloc`, and run the entire multi-layer forward pass inside a single `parallelize` invocation synchronized via the sense-reversing atomic `barrier(sh)`.

### Gap 4: Quantized Activation Dot Products ($W_{\text{quant}} \times X_{\text{Q8\_0 / Q8\_K}}$) & Missing K/IQ Quants
- **Current State**:
  - All Mojo kernels in `MojoLlama` dequantize weights to `Float32` on the fly and multiply against a `Float32` activation vector. As noted in [`SUMMARY.md`](file:///mnt/c/Documents/Mojo/MojoLlama/SUMMARY.md#L16-L36), scalar nibble extraction for `Q4_0` in Mojo was 32% slower than `F16` without SIMD byte shuffles, and [`kquants.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/kquants.mojo#L539-L545) has slight layout mismatches between `Q4_K_S` and `Q4_K_M` scale unpacking compared to the reference C struct in [`cengine_batch_instr.c`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/cengine_batch_instr.c#L26-L60).
  - `Q2_K`, `Q3_K`, `IQ1_*`–`IQ4_*`, and `BF16` are listed in [`gguf/types.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/gguf/types.mojo#L20-L46) but lack compute kernels.
- **What `llama.cpp` Does**: `llama.cpp` quantizes the input activation vector `x` **once** per layer to `Q8_0` or `Q8_K` (which [`mojo_engine/sampling.mojo:quantize_row_q8_0`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/sampling.mojo#L25-L46) already implements in pure Mojo!) and then computes integer dot products between $W_{\text{Q4\_0 / Q4\_K}}$ and $X_{\text{Q8\_0 / Q8\_K}}$ before scaling once per block.
- **How to Close in Pure Mojo**:
  - Use [`quantize_row_q8_0`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/sampling.mojo#L25-L46) before each matmul stage so workers compute integer SIMD dot products (`Int16` / `Int32` accumulation) rather than converting every 4-bit weight to `Float32`.
  - Port the exact 12-byte scale unpacking function `k4_scale` from [`cengine_batch_instr.c`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/cengine_batch_instr.c#L56-L60) into `mojo_quant_kernels.mojo` to guarantee bit-exact `Q4_K` and `Q5_K` parity with `llama.cpp`.
  - Add `Q2_K`, `Q3_K`, `IQ4_NL`, `IQ4_XS`, and `BF16` block decoders.

### Gap 5: General `llama_batch` Prefill, Multi-Sequence Routing, & KV Cache Management
- **Current State**:
  - [`tinyllama_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/tinyllama_gen_q8.mojo#L110-L200) supports fixed `comptime B = 2, 4, 8` lockstep decoding where all sequences are at the exact same `pos`.
  - [`engine_v3.mojo:prefill`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L327-L356) is an incomplete stub.
  - [`cengine_batch_instr.c`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/cengine_batch_instr.c#L160-L200) implements `PageTable` (`page_table_init`, `page_table_alloc`, `page_table_free`, `map_page`) in C, not yet ported to Mojo.
- **How to Close in Pure Mojo**:
  - Port `PageTable` from `cengine_batch_instr.c` into `src/llamojo/kv_cache.mojo` and add cell metadata (`pos: llama_pos`, `seq_id_mask: UInt64`) to support `llama_kv_cache_seq_rm`, `seq_cp`, `seq_keep`, and `seq_add` (sliding window / prefix caching).
  - Generalize `_mm_q8_batch` and `q4_mm_regblock` into a runtime-tiled GEMM kernel (tiling `batch.n_tokens` in groups of 4 or 8 rows) so prompt prefill (`n_tokens > 1`) reuses weight loads across tokens and achieves high prompt-processing tok/s.

### Gap 6: Complete GGUF-Driven Tokenizer & Detokenizer
- **Current State**:
  - [`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L180-L229) implements BPE merge encoding using `Dict[String, Int]`, while [`zaya_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/zaya_gen_q8.mojo#L241-L250) and [`tinyllama_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/tinyllama_gen_q8.mojo#L306-L318) implement `▁` (U+2581) SentencePiece space-stripping decoding from a custom `/tmp/vocab.bin`.
- **How to Close in Pure Mojo**:
  - Populate the `llama_vocab` struct directly from the GGUF KV arrays (`tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.token_type`, `tokenizer.ggml.merges`, `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`).
  - Support the two primary `tokenizer.ggml.model` modes in pure Mojo:
    1. `"llama"` (SentencePiece BPE using `scores` priority queue, directly adapting [`llama2.mojo:bpe_encode`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L928-L960) + `<0xXX>` byte fallback).
    2. `"gpt2"` (Byte-level BPE for Llama-3, Qwen, DeepSeek using `merges` rank map).

### Gap 7: Composable `llama_sampler` Chain & Grammar Constrained Decoding
- **Current State**:
  - [`kernels/sampling.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/sampling.mojo#L361-L419) executes a fixed pipeline via `SamplingConfig`.
  - It lacks `llama.cpp`'s composable `llama_sampler_chain` abstraction, Mirostat (`v1`/`v2`), XTC, Typical-P, Dynamic Temperature, and `llama_sampler_init_grammar` (GBNF).
- **How to Close in Pure Mojo**:
  - Define `llama_token_data` (`id: llama_token, logit: Float32, p: Float32`) and `llama_token_data_array` (`data: Pointer[llama_token_data], size: Int, selected: Int64, sorted: Bool`).
  - Implement `llama_sampler` as a tagged union / variant struct in Mojo where `llama_sampler_chain` holds a `List[llama_sampler]` and iterates `apply()` across each stage.
  - Add a pure-Mojo recursive-descent GBNF parser and pushdown automaton filter (~300 LOC) to mask invalid token IDs before softmax.

### Gap 8: LoRA Adapter Application & State Save/Load
- **Current State**: Absent from the pure Mojo files (handled via external Python scripts in `MojoLlama`).
- **How to Close in Pure Mojo**:
  - Implement `llama_adapter_lora_init` using the same GGUF loader (`mmap`) and apply $y \leftarrow y + \text{scale} \cdot (B \cdot (A \cdot x))$ as an auxiliary low-rank matmul pass after linear projections.
  - Implement `llama_state_get_size`, `llama_state_get_data`, and `llama_state_set_data` by serializing `RunState` KV cache buffers and RNG state via `unsafe_memcpy`.

---

## 5. Strategy for Minimal Integrations (Pure Mojo Architecture)

To achieve **minimal external integrations**, `llamojo` should enforce a strict **Three-Layer Zero-Dependency Boundary**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Public API Layer (llama.h parity)                    │
│  llama_model_load_from_file | llama_init_from_model | llama_decode      │
│  llama_tokenize | llama_sampler_sample | llama_model_quantize           │
│  (Native Mojo API + optional @export(..., abi="C") for C ABI drop-in)   │
├─────────────────────────────────────────────────────────────────────────┤
│                     Pure Mojo Core Engine (100% Mojo)                   │
│  • GGUF v3 Parser & Writer (gguf_scanner + mojo_quantize)               │
│  • Tokenizer & Sampler Chain (llama2.mojo BPE + kernels/sampling.mojo)  │
│  • Persistent Spin-Barrier Thread Pool (llama2.mojo PoolShared)         │
│  • Multi-Arch Graph: Dense GQA, MoE, CCA, Mamba-2 SSM, MLA              │
│  • SIMD Kernels: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q2_K-Q6_K, MXFP4│
├──────────────────────────────────────────────┬──────────────────────────┤
│          Mojo Standard Library               │   POSIX / OS Syscalls    │
│  • std.memory (Pointer, unsafe_alloc, memcpy)│  • open, close, lseek    │
│  • std.atomic (Atomic[Int] spin barriers)    │  • mmap, munmap, mlock   │
│  • std.sys.info (CompilationTarget, cores)   │  • sched_yield           │
│  • std.math & SIMD (fma, vectorize, bitcast) │  • socket, bind, listen  │
│  • max.algorithm (parallelize worker launch) │    (server binary only)  │
└──────────────────────────────────────────────┴──────────────────────────┘
```

### Key Rules to Minimize Integrations:
1. **Zero Python Interop (`from python import Python` is forbidden)**:
   - Replace `src/mojollama/gguf/reader.mojo` and `mojollama_engine.mojo` (which use `Python` / `numpy`) with the pure Mojo binary parser from [`gguf_scanner.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_scanner.mojo) + `mmap`.
   - Replace `Python` timing/printing in [`optimized_kernels.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/optimized_kernels.mojo#L370-L372) with `std.time.perf_counter_ns()`.
2. **Zero External C/C++ Compute Libraries (No OpenMP, No OpenBLAS, No `libllama.so`)**:
   - All multi-threading uses Mojo's `parallelize` + [`llama2.mojo`'s `Atomic` spin barrier](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L682-L695).
   - All heap allocations use Mojo's `unsafe_alloc` (`std.memory.alloc`), eliminating `@extern("malloc")` and `@extern("free")`.
   - All math operations (`sin`, `cos`, `exp`, `sqrt`, `pow`, `fma`) use Mojo's `std.math` and `SIMD` intrinsics, eliminating `@extern("sinf")`, `@extern("expf")`, etc.
3. **Only 4 OS Syscalls via `@extern` (for Zero-Copy I/O)**:
   - Only `mmap`, `munmap`, `mlock`, and `sched_yield` (plus `open`/`close`/`fstat` if not using `std.os`) are called via `std.ffi.external_call`. These are part of the OS kernel/libc ABI present on every POSIX system and require zero third-party packages.

### 5.1 Including `llama2.mojo` and `MojoLlama` as Sublibraries

In Mojo 1.1.0, external Mojo repositories can be integrated as sublibraries using four complementary mechanisms:

1. **Vendor Directory / Git Submodules (`vendor/`)**:
   - Mount or submodule both repositories under `/mnt/c/Documents/Mojo/llamojo/vendor/`:
     - `vendor/llama2.mojo` $\rightarrow$ `/mnt/c/Documents/Mojo/llama2.mojo`
     - `vendor/MojoLlama` $\rightarrow$ `/mnt/c/Documents/Mojo/MojoLlama`
   - For Git repositories:
     ```bash
     git submodule add https://github.com/tairov/llama2.mojo.git vendor/llama2.mojo
     git submodule add <mojollama-repo-url> vendor/MojoLlama
     ```

2. **Compiler Module Search Path (`mojo -I <path>`)**:
   - Mojo resolves top-level modules (`.mojo` files) and packages (directories containing `__init__.mojo`) via `-I` flags:
     ```bash
     mojo run -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_llama_api.mojo
     ```
   - Because [`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo) is a single-file module already upgraded to Mojo 1.1.0, any module in `llamojo` can directly import its core types and kernels without executing its `main()` entry point:
     ```mojo
     from llama2 import (
         Matrix, BufferPtrFloat32, nelts,
         chunk_start, chunk_end, spin_wait,
         ForwardArgs, PoolShared, PoolSharedPtr, barrier,
         rmsnorm, softmax, matmul_rows, rope_head, silu_mul, add, dot, axpy, argmax,
     )
     ```
   - Similarly, [`MojoLlama/src/mojollama`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/__init__.mojo) is structured as a Mojo package hierarchy (`mojollama.gguf`, `mojollama.kernels`, `mojollama.mojo_engine`).

3. **Precompiled Mojo Packages (`.mojoc` / `.mojopkg` via `mojo precompile`)**:
   - Any package directory containing an `__init__.mojo` can be precompiled into a portable binary archive for faster downstream builds:
     ```bash
     mojo precompile src/llamojo -I vendor/llama2.mojo -o build/llamojo.mojoc
     ```
   - Downstream consumers can then import `llamojo` simply by placing `llamojo.mojoc` in their `-I` search path.

4. **Mojo 1.1.0 Facade & Adapter Submodules (`src/llamojo/backend/` & `src/llamojo/gguf.mojo`)**:
   - While `llama2.mojo` targets **Mojo 1.1.0** (`def`, `comptime`, `Pointer[..., MutUntrackedOrigin]`, `unsafe_alloc`), `MojoLlama` contains a mix of Mojo 0.26 (`fn`, `alias`, `PythonObject`) and Mojo 1.0b1 (`UnsafePointer`, `.load()`, `.store()`) syntax.
   - To maintain strict **Mojo 1.1.0 compatibility** and **zero Python dependency**, `llamojo` directly imports `llama2` as a live sublibrary in [`src/llamojo/backend/threadpool.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/threadpool.mojo) while porting `MojoLlama`'s pure-Mojo GGUF parser, quantization sizing tables, and ISA dispatch into Mojo 1.1.0 modules under `src/llamojo/`.

---

## 6. Proposed Directory Structure & Implementation Roadmap for `llamojo`

### 6.1 Target Repository Layout (`/mnt/c/Documents/Mojo/llamojo`)

```text
/mnt/c/Documents/Mojo/llamojo/
├── docs/
│   └── llama_cpp_pure_mojo_study.md   # This architectural study
├── src/
│   └── llamojo/
│       ├── __init__.mojo              # Re-exports public llama.h-compatible API
│       ├── llama.mojo                 # Public API functions (llama_model_load_from_file, llama_decode, etc.)
│       ├── types.mojo                 # llama_token, llama_pos, llama_batch, llama_model_params, GGMLType
│       ├── gguf.mojo                  # Pure Mojo GGUF v3 parser + mmap tensor binder + GGUF writer
│       ├── vocab.mojo                 # Pure Mojo BPE & SentencePiece tokenizer/detokenizer + chat templates
│       ├── kv_cache.mojo              # Unified & Paged KV cache (F32, F16, Q8_0, Q4_0) + seq_* ops
│       ├── sampler.mojo               # llama_sampler_chain, top_k, top_p, min_p, penalties, mirostat, grammar
│       ├── quantize.mojo              # llama_model_quantize (parallel block quantizers for all GGUF types)
│       └── backend/
│           ├── threadpool.mojo        # Persistent worker pool + sense-reversing Atomic spin barrier
│           ├── isa.mojo               # Comptime AVX-512 / AVX2 / NEON detection & F16C bitcast helpers
│           ├── kernels_quant.mojo     # Q4_0, Q4_1, Q5_0, Q8_0, Q2_K–Q6_K, IQ4, MXFP4, F16, BF16 matmuls
│           ├── kernels_norm_rope.mojo # Fused RMSNorm+Residual, LayerNorm, RoPE (normal/NeoX/YaRN), SiLU/GeGLU
│           ├── kernels_attn.mojo      # Flash/Tiled GQA decode & prefill attention with online softmax
│           └── models.mojo            # Runtime-dispatched forward pass (Llama/Mistral/Gemma/MoE/ZAYA/SSM)
├── tools/
│   ├── llama_cli.mojo                 # Pure Mojo drop-in for `llama-cli` (ported from llama2.mojo + mojollama_cli)
│   ├── llama_bench.mojo               # Pure Mojo drop-in for `llama-bench`
│   ├── llama_quantize.mojo            # Pure Mojo drop-in for `llama-quantize`
│   └── llama_server.mojo              # Pure Mojo drop-in for `llama-server` (ported from server.mojo)
└── tests/
    ├── test_gguf.mojo
    ├── test_kernels.mojo
    ├── test_tokenizer.mojo
    ├── test_sampler.mojo
    └── test_llama_api.mojo
```

### 6.2 Phased Implementation Plan

#### Phase 1: Core Types, Zero-Copy GGUF Loader, & Persistent Spin-Barrier Pool
1. Port [`gguf/types.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/gguf/types.mojo) and [`gguf_scanner.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/gguf_scanner.mojo) into `src/llamojo/gguf.mojo`, adding full KV array parsing and POSIX `mmap` tensor pointer binding.
2. Extract [`Matrix`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L26-L146), [`PoolShared`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L663-L678), and [`barrier()`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L682-L695) from [`llama2.mojo`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo) into `src/llamojo/backend/threadpool.mojo`.
3. Implement `llama_backend_init`, `llama_model_default_params`, `llama_model_load_from_file`, and `llama_model_free`.

#### Phase 2: Unified Quantized Kernels & Fused Multi-Matrix Dispatch
1. Merge [`mojo_quant_kernels.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/mojo_quant_kernels.mojo), [`optimized_kernels.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/optimized_kernels.mojo), [`kquants.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/kquants.mojo), and [`simd_mxfp4.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/simd_mxfp4.mojo) into `src/llamojo/backend/kernels_quant.mojo`.
2. Fix the `Q4_K` / `Q6_K` scale unpacking using the exact bit-ops from [`cengine_batch_instr.c:k4_scale`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/cengine_batch_instr.c#L56-L60) and generalize [`llama2.mojo:matmul_rows[n]`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L431-L466) + [`tinyllama_gen_q8.mojo:_mm_q8_2out_batch`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/tinyllama_gen_q8.mojo#L202-L270) so any quant format can fuse QKV (`n=3`) and Gate/Up (`n=2`) across worker row slices.
3. Integrate [`fused_kernels.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/fused_kernels.mojo) (`fused_rms_norm_residual`, `fused_silu_mul`) and [`rope_kv_cache.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/rope_kv_cache.mojo) into `kernels_norm_rope.mojo`.

#### Phase 3: Context, KV Cache, `llama_batch`, & Multi-Architecture `llama_decode`
1. Port [`engine_v3.mojo:KVCache`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/engine_v3.mojo#L360-L376) and [`cengine_batch_instr.c:PageTable`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/cengine_batch_instr.c#L160-L200) into `src/llamojo/kv_cache.mojo`, implementing `llama_kv_cache_clear` and `llama_kv_cache_seq_*`.
2. Unify [`llama2.mojo:forward_worker`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L727-L843), [`universal_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/universal_forward.mojo#L344-L443) (Dense + MoE), [`zaya_gen_q8.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/zaya_gen_q8.mojo) (CCA + residual scales), and [`ssm_forward.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/ssm_forward.mojo) (Mamba-2 SSM) into a runtime-configured `llama_decode(ctx, batch)` engine.
3. Implement `llama_init_from_model`, `llama_batch_init`, `llama_batch_get_one`, `llama_get_logits_ith`, and `llama_get_embeddings`.

#### Phase 4: GGUF Tokenizer, Composable `llama_sampler` Chain, & CLI/Server Tools
1. Adapt [`llama2.mojo:Tokenizer`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L180-L229) and [`bpe_encode`](file:///mnt/c/Documents/Mojo/llama2.mojo/llama2.mojo#L928-L960) to read directly from `GGUFContext` vocabulary arrays, exposing `llama_tokenize`, `llama_token_to_piece`, and `llama_detokenize`.
2. Refactor [`kernels/sampling.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/kernels/sampling.mojo#L98-L419) and [`mojo_engine/sampling.mojo:topk_extract`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/sampling.mojo#L47-L89) into the `llama_sampler_chain_*` and `llama_sampler_init_*` APIs, adding GBNF grammar filtering.
3. Wire [`mojo_quantize.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/mojo_engine/mojo_quantize.mojo) to `llama_model_quantize` and [`server.mojo`](file:///mnt/c/Documents/Mojo/MojoLlama/src/mojollama/server.mojo) to `tools/llama_server.mojo`.

