---
type: reference
title: "Comprehensive PyTorch & Tensor Interoperability Architecture: llamojo (Inference)"
description: "Architectural specification for PyTorch tensor exchange, HuggingFace pipeline integration, and DLPack zero-copy pointers."
status: completed
tags: [pytorch-interop, inference, dlpack, zero-copy, torch-tensor]
timestamp: 2026-11-05T00:00:00Z
---

# Comprehensive PyTorch & Tensor Interoperability Architecture: `llamojo` (Inference)

Status: **COMPLETED & VERIFIED** (Part I & Part II Fully Implemented). Scope: inference, runtime, zero-copy views, Safetensors ingestion, and weights translation in `llamojo`.

---

## 1. Executive Summary & Objective

This document consolidates the architectural plans for **Python/PyTorch-level tensor interoperability and model weight handling** in **`llamojo`** (incorporating both tensor interop / full-sequence logits and Safetensors import/export / GGUF key mapping).

The primary goal is to make `llamojo` transparently pluggable into modern PyTorch/HuggingFace workflows:
1. **Zero-Copy Tensor Output**: Expose runtime outputs (logits, embeddings, KV cache) directly to NumPy, PyTorch, and DLPack capsules without superfluous memory duplication.
2. **Full-Sequence Logits**: Provide `[batch, seq, vocab]` logits for evaluation, scoring, and perplexity analysis.
3. **Flexible Weight Ingestion & Export**: Break the GGUF-only barrier by introducing native Safetensors readers (`model.safetensors` + `config.json`) and bi-directional state-dict export.

---

## 2. Part I: Zero-Copy Tensor Interop & Full-Sequence Logits

### 2.1 Current State
- `Llama.eval` currently copies the last token's logits into a NumPy array or Python list.
- Phase 2 established Mojo-side zero-copy memory views (`src/llamojo/interop/context_views.mojo`), but these lack C ABI exports.
- Token evaluation executes single-token decodes sequentially, and the runtime context defaults to maintaining logits only for the final position.

### 2.2 Work Breakdown: Tensor Interop & Views

#### 1. C ABI View Protocol (`src/llamojo/c_api.mojo`)
- Define standard C-compatible view descriptor:
  ```c
  struct llamojo_view {
      void* data;
      uint32_t dtype;
      int32_t ndim;
      int64_t shape[4];
      int64_t strides[4];
      int64_t n_elements;
      uint64_t version_tag;
  };
  ```
- Expose runtime view getters:
  - `llamojo_get_logits_view(ctx, out_view)`
  - `llamojo_get_embeddings_view(ctx, out_view)`
  - `llamojo_get_kv_view(ctx, which_cache, out_view)`

#### 2. Python Zero-Copy Adapters (`python/llamojo/_views.py`)
- **NumPy Bridge**: `as_numpy(view)` wrapping `ctypes.c_float` buffer using raw strides (inverting row/col dimension order where needed to conform to standard row-major indexing).
- **DLPack Bridge**: `as_dlpack(view)` returning a standard PyCapsule consumable by `torch.from_dlpack` and `np.from_dlpack`.
- **Buffer Safety & Lifetime Invariant**: Non-owning views carry a reference to `Llama` along with a generation counter. Calling `eval()`, `reset()`, or `close()` increments the generation; attempting to access a stale view raises `RuntimeError` rather than reading invalidated memory.

#### 3. Full-Sequence Logits & Batched Prefill
- Extend evaluation: `Llama.eval(tokens, all_logits=False)`.
- When `all_logits=True`, maintain per-position logits in a dedicated buffer `[n_tokens, vocab]`, accessed via `llama_get_logits_ith`.
- Modernize prompt evaluation to submit a single `llama_batch` of length $N$ tokens to `llama_decode` (utilizing existing `llama_batch_init`), accelerating prompt prefill dramatically compared to one-token-at-a-time loops.
- `LlamojoForCausalLM.__call__(..., output_all_logits=True)` outputs full tensors `[batch, seq, vocab]`.

---

## 3. Part II: Safetensors Import/Export & GGUF Key Mapping

### 3.1 Current State
- `llamojo` relies on GGUF format parsed via `GGUFContext` with GGML naming conventions (`token_embd.weight`, `blk.N.attn_q.weight`).
- Loading models directly from HuggingFace repositories requires prior external conversion to GGUF format via Python conversion scripts.

### 3.2 Work Breakdown: Weights & Formats

#### 1. Native Safetensors Parsing (`src/llamojo/safetensors.mojo`)
- Implement a zero-dependency Safetensors reader: parse 8-byte length prefix, parse JSON metadata header (`shape`, `dtype`, `data_offsets`), and mmap the underlying tensor data.
- Wrap tensor segments directly into `TensorMemoryView`.
- Support F32, F16, and BF16 (transmuting or upcasting BF16/F16 where kernels require F32).
- Reject out-of-bounds offsets and invalid JSON headers safely.
- Support sharded weights via `model.safetensors.index.json`.

#### 2. Architecture Weight Mapping (`src/llamojo/weight_maps.mojo`)
- Architecture-specific translation maps (LLaMA, Gemma, Mistral, ModernBERT):
  - Map Hugging Face keys (`model.layers.0.self_attn.q_proj.weight`) to internal GGUF names (`blk.0.attn_q.weight`).
  - Account for RoPE rotary coordinate permutations (interleaved vs half-split layout).
  - Handle tied embeddings (`lm_head.weight` shared with `embed_tokens.weight`).

#### 3. Configuration & Entry Points
- Parse `config.json` to extract `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `rope_theta`.
- Loader APIs:
  - Mojo: `llama_model_load_from_safetensors(path, params)`.
  - Python: `Llama.from_safetensors(dir_path)` and `LlamojoForCausalLM.from_pretrained(dir_path)`.
- Export capabilities:
  - `Llama.state_dict()` exporting current parameters as named NumPy/PyTorch tensors with inverse RoPE permutation applied.

---

## 4. Verification Matrix & Acceptance Criteria

| Subsystem | Test Harness | Success Criteria | Status |
|---|---|---|---|
| **C ABI Views** | `tests/test_c_abi.c` | Verify `llamojo_view` struct matches internal pointer, dimensions, and strides. | **PASS** |
| **NumPy Zero-Copy** | `tests/test_python_views.py` | `np.shares_memory()` returns True between consecutive reads; stale view throws on `reset()`. | **PASS** |
| **DLPack PyTorch** | `tests/test_python_views.py` | `torch.from_dlpack(view)` and `view.as_torch()` match numerical contents without copy. | **PASS** |
| **Full Logits Parity** | `tests/test_python_views.py` | Logits at step $i$ in `all_logits=True` match single-token decode logits within $\le 10^{-4}$. | **PASS** |
| **Live Model Inference** | `data/SmolLM2-135M-Instruct-Q4_K_M.gguf` | Verify zero-copy logits view, full logits_all (shape `[N, 49152]`), and KV cache view on real weights. | **PASS** |
| **Safetensors Reader/Writer** | `tests/test_safetensors_interop.py` | Zero-dependency memory-mapped read/write, corrupt header and out-of-bounds offset rejection. | **PASS** |
| **Sharded Checkpoints** | `tests/test_safetensors_interop.py` | Correctly parse multi-file checkpoints via `model.safetensors.index.json`. | **PASS** |
| **Weight Mapping & RoPE** | `tests/test_safetensors_interop.py` | Bijective key translation (HF <-> GGUF) and complex RoPE row permutation round-trip. | **PASS** |
| **Direct Safetensors Loading** | `tests/test_safetensors_interop.py` | Load synthetic HuggingFace directory via `Llama.from_safetensors` and decode. | **PASS** |
| **HF Facade Directory Load** | `tests/test_safetensors_interop.py` | `LlamojoForCausalLM.from_pretrained(dir)` directly loads Safetensors model dir. | **PASS** |

---

## 5. Implementation Status Summary

### Part 1: Zero-Copy Views & Full-Sequence Logits [COMPLETED]
1. **C ABI Export Layer (`src/llamojo/c_api.mojo`)**:
   - `struct llamojo_view`: C layout matching `TensorMemoryView` with data pointer, dtype, 4D shape, 4D strides, total elements, and version tag.
   - `llamojo_get_logits_view`, `llamojo_get_embeddings_view`, `llamojo_get_hidden_view`, `llamojo_get_kv_view`.
2. **Dynamic Shared Library (`build/libllamojo.so`)**:
   - Built and linked with automatic runtime resolution in `python/llamojo/_lib.py`.
3. **Zero-Copy Python Bridge (`python/llamojo/_views.py`)**:
   - `TensorView`: non-owning buffer wrapper exposing `.as_numpy(copy=False)`, `.as_dlpack()`, and `.as_torch(copy=False)`.
   - Generation invalidation counter guaranteeing safety: modifying context or evaluating new tokens invalidates stale views with a descriptive `RuntimeError`.
4. **Batched Prefill & Full-Sequence Logits**:
   - `Llama.eval(tokens, all_logits=True)` batches $N$ tokens into a single decode pass rather than token-by-token loops, retaining per-token logits.
   - `Llama.logits_all()` returning `[seq_len, vocab_size]` array.
   - `LlamojoForCausalLM.__call__(..., output_all_logits=True)` outputting full `[batch, seq, vocab]` tensors compatible with HuggingFace/PyTorch evaluation pipelines.
5. **Automated Test Suites**:
   - `tests/test_c_abi.c`: C-level API and struct safety verification.
   - `tests/test_python_views.py`: Unit tests for NumPy views, stale detection, DLPack export, and full-sequence logits.
   - Automated integration in `run-tests.sh`. All tests pass on Mojo 1.1.0 and Python.

### Part 2: Safetensors Import/Export & GGUF Key Mapping [COMPLETED]
1. **Zero-Dependency Safetensors Engine (`python/llamojo/safetensors.py`)**:
   - Memory-mapped reader `SafetensorsFile` reading standard 8-byte little-endian headers and JSON tensor dictionaries.
   - Zero-copy tensor slice retrieval with `mmap` buffer views.
   - Support for `BF16` (upcasted to `F32`), `F32`, `F16`, `I32`, `I16`, `I8`, `U8`, and `BOOL`.
   - `save_safetensors()` serializing arbitrary NumPy weight dictionaries to compliant `.safetensors` files with 8-byte padding alignment.
   - `load_safetensors_dir()` handling single-file and sharded checkpoints with `model.safetensors.index.json`.
2. **Bidirectional Weight Translation & RoPE Permutation (`python/llamojo/weight_maps.py`)**:
   - `hf_to_gguf_name()` mapping Hugging Face parameter hierarchy (`model.layers.N.self_attn.q_proj.weight`, `mlp.gate_proj`, etc.) to internal GGUF names (`blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`).
   - `gguf_to_hf_name()` for reverse state-dict export.
   - `permute_rope_hf_to_ggml()` and `permute_rope_ggml_to_hf()` handling complex RoPE rotary matrix layout permutations.
3. **End-to-End Direct Loading & Conversion (`python/llamojo/convert.py` & `python/llamojo/llama.py`)**:
   - `convert_safetensors_to_gguf()` converting Hugging Face checkpoint directories + `config.json` into loadable GGUF files with metadata and tensor alignments.
   - `Llama.from_safetensors(dir_path)` providing one-step loading of Hugging Face directories.
   - `LlamojoForCausalLM.from_pretrained(dir_path)` automatically detecting directories and loading Safetensors checkpoints.
4. **Verification**:
   - `tests/test_safetensors_interop.py`: 4 comprehensive test suites covering Safetensors I/O, bounds validation, sharded indices, RoPE round-tripping, and live forward decode. Integrated into `run-tests.sh`.

---

## 6. Architectural Risks & Mitigations

1. **Memory Lifetime & Dangling Pointers**:
   - *Risk*: Python users retaining a view after the C context advances or frees memory.
   - *Mitigation*: Generation tokens on views and hard parent-pointer references preventing GC of parent contexts.
2. **RoPE Layout Errors**:
   - *Risk*: Transposing or un-permuting Q/K attention weights incorrectly yields silent garbage output.
   - *Mitigation*: Strict unit testing comparing QK dot-product values against HuggingFace PyTorch reference tensors.
3. **BFloat16 Kernel Support**:
   - *Risk*: Direct execution of BF16 on CPU without AVX-512-BF16 instructions is slow.
   - *Mitigation*: Upcast BF16 tensors to F32 at load-time by default until native BF16 arithmetic paths are introduced.

