# `llamojo` Pure-CPU Implementation Plan: Path to Full `llama.cpp` Parity

This roadmap outlines the remaining implementation stages to elevate `llamojo` from its current Phase 4 status to complete pure-CPU feature parity with upstream `llama.cpp` (`libllama`), ordered strictly from **easiest / lowest risk** to **most complex / architecturally involved**.

---

## Roadmap Overview

1. **Stage 1: Complete GGUF Metadata Mirroring in Quantizer** (Easiest)
2. **Stage 2: C Foreign Function Interface ABI Export (`libllama.so`)**
3. **Stage 3: SentencePiece Unigram Tokenizer & Tokenizer Variants**
4. **Stage 4: Dynamic LoRA Adapter Ingestion & Online Merging**
5. **Stage 5: Full GBNF Grammar & JSON-Schema Parsing Filter**
6. **Stage 6: Sparse Mixture-of-Experts (MoE) Routing & Forward Pass**
7. **Stage 7: Advanced CPU Micro-Architecture Optimizations** (Most Complex)

---

## Stage 1: Full GGUF Metadata Mirroring in Quantizer (Easiest)
*Complexity: Low | Risk: Very Low*

### Objectives
- When executing `llama_model_quantize`, preserve all original metadata key-value pairs (model architecture parameters, tokenizer configs, chat templates, quantization history) rather than emitting minimal placeholder headers.
- Update `general.quantization_version` and `general.file_type`.

### Tasks
1. Iterate over `gguf.metadata` dictionary in `quantize.mojo`.
2. Serialize KV pairs into the GGUF v3 wire format (keys as Pascal-style strings, values formatted by `GGUFValueType`).
3. Correctly calculate header padding and 32-byte alignment for tensor payloads.
4. Add unit test verifying that quantized models retain identical tokenizer metadata and architecture tags.

---

## Stage 2: C ABI Compatibility Layer (`libllama.so` Export)
*Complexity: Low to Moderate | Risk: Low*

### Objectives
- Expose native `@export` functions matching the canonical `llama.h` function signatures so external C/C++, Rust, or Python programs (`ctypes`, `cffi`, `llama-cpp-python`) can link against `llamojo`.

### Tasks
1. Create `src/llamojo/c_api.mojo`:
   - `@export("llama_backend_init") def llama_backend_init_c() -> None`
   - `@export("llama_model_load_from_file") def llama_model_load_from_file_c(...)`
   - `@export("llama_decode") def llama_decode_c(...)`
   - `@export("llama_sampler_sample") def llama_sampler_sample_c(...)`
2. Configure build script or compiler flags to output a shared library (`build/libllama.so`).
3. Verify basic C-header integration with a small C dummy binary test.

---

## Stage 3: SentencePiece Unigram Tokenizer & Byte-Fallback Refinement
*Complexity: Moderate | Risk: Low*

### Objectives
- Add support for models that do not use BPE (e.g., Gemma 1, T5, or SentencePiece Unigram language models).

### Tasks
1. Implement the Viterbi algorithm for Unigram probability scoring in `src/llamojo/vocab.mojo`.
2. Read `tokenizer.ggml.model` to branch between `"llama"` / `"gpt2"` (BPE) and `"llama"` (SentencePiece Unigram).
3. Validate tokenization and detokenization parity against Hugging Face tokenizer reference outputs.

---

## Stage 4: Dynamic LoRA Adapter Support
*Complexity: Moderate | Risk: Medium*

### Objectives
- Enable applying Low-Rank Adaptation (LoRA) checkpoint files (`.gguf` adapters) at runtime without rewriting the base model file on disk.

### Tasks
1. Implement `llama_lora_adapter_init(model, path_lora)`:
   - Parse LoRA $A$ and $B$ low-rank matrices from adapter GGUF files.
2. In-memory weight merging or on-the-fly scaled addition:
   $$W_{\text{effective}} = W_{\text{base}} + \frac{\alpha}{r} (A \times B)$$
3. Support dynamic LoRA scaling factor adjustments during context evaluation.

---

## Stage 5: Full GBNF Grammar Engine & JSON Schema Constraint Masking
*Complexity: High | Risk: Medium*

### Objectives
- Transform the current stub in `llama_sampler_init_grammar` into a full LR(1) grammar constraint engine that enforces strictly valid JSON or schema-conforming generations.

### Tasks
1. Implement a GBNF rule parser in Mojo that compiles grammar strings into pushdown automaton (PDA) state transitions.
2. At each token selection step, traverse active grammar states to build a valid token bitmask.
3. Apply negative infinite logits (`-1e10`) to any candidate token in `llama_token_data_array` that violates the grammar.
4. Support standard grammar presets (e.g., JSON object, JSON array, regex, integer/float).

---

## Stage 6: Sparse Mixture-of-Experts (MoE) Routing & Forward Pass
*Complexity: High | Risk: Medium*

### Objectives
- Complete the wiring of MoE layers in `src/llamojo/llama.mojo` to support models like Mixtral 8x7B, Qwen1.5-MoE, and DeepSeek-V2/V3.

### Tasks
1. Wire `ffn_gate_inp.weight` router projection to compute softmax gating probabilities over top-$k$ experts.
2. Connect backend MoE kernel dispatch (`moe.mojo` / `forward_layer_moe`):
   - Only evaluate the selected top-$k$ experts per token.
   - Accumulate weighted expert outputs into the hidden state.
3. Implement parallel thread assignment across active experts to maximize CPU cache locality.

---

## Stage 7: Advanced CPU Micro-Architecture Optimizations (Most Complex)
*Complexity: Very High | Risk: High*

### Objectives
- Reach or exceed upstream `llama.cpp` AVX-512 and AMX throughput via cache blocking, AVX-512 VNNI / BF16 SIMD intrinsics, and NUMA-aware memory binding.

### Tasks
1. **L1/L2/L3 Cache Tiling**:
   - Reorganize GEMV and GEMM loops into cache-blocked tiles to prevent cache thrashing on CPUs with small L2/L3 caches.
2. **AVX-512 VNNI & BF16 Hardware Acceleration**:
   - Optimize int8 dot products (`vpdpbusd` instruction via Mojo inline LLVM IR/assembly).
   - Leverage native `bfloat16` instructions on newer Intel/AMD architectures (Zen 4+, Sapphire Rapids+).
3. **NUMA Node Pinning**:
   - Implement `llama_numa_init` using `sched_setaffinity` and `set_mempolicy` / `mbind` syscalls to ensure thread memory locality on multi-socket CPU workstations and servers.
4. **FlashAttention CPU Implementation**:
   - Fused online softmax attention without materializing the full $S \times S$ score matrix in memory.

