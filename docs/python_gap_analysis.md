---
type: architecture-decision
title: "Comprehensive Functional Gap Analysis: llamojo vs. llama-cpp-python"
description: "Exhaustive parity audit comparing llamojo Python bindings with llama-cpp-python across all methods, parameters, and flags."
status: completed
tags: [gap-analysis, llama-cpp-python, parity-audit, api-comparison]
timestamp: 2026-11-05T00:00:00Z
---

# Comprehensive Functional Gap Analysis: `llamojo` vs. `llama-cpp-python`

## Executive Summary
This document provides a thorough audit of the functionality provided by the original `llama-cpp-python` library compared against `llamojo`'s native Python bindings.

Through automated reflection across `llama_cpp.Llama` and `llamojo.Llama`, we track which capabilities are **fully implemented**, which are **partially supported**, and which represent **pending functional gaps**.

---

## 1. High-Level Comparison Matrix

| Functional Category | `llama-cpp-python` | `llamojo` Status | Assessment |
|---|:---:|:---:|---|
| **Core Text Inference (`create_completion`, `__call__`)** | Yes | **Full** | Complete parity (prompt tokens, greedy/temp, stop sequences, OpenAI JSON response). |
| **Streaming (`stream=True`)** | Yes | **Full** | Complete parity (token piece generator, stop sequence interruption). |
| **Chat Completion (`create_chat_completion`)** | Yes | **Full** | ChatML, Jinja2 auto-detection, and assistant role responses. |
| **Embeddings (`embed`, `create_embedding`)** | Yes | **Full** | Vector generation and normalization, OpenAI list schema. |
| **Tokenizer (`tokenize`, `detokenize`, `token_to_piece`)** | Yes | **Full** | SPM, BPE, byte fallback, special token control. |
| **Low-Level Step Control (`eval`, `sample`, `generate`, `reset`)** | Yes | **Full** | Step-by-step token generation and KV cache management. |
| **GBNF Grammar Constraints (`LlamaGrammar`)** | Yes | **Full** | C-ABI backed GBNF parser, bitmask filtering, pushdown automaton state advancement. |
| **LoRA Adapters (`lora_path`, `apply_lora`)** | Yes | **Full** | Ingestion of GGUF adapter pairs and online additive weight merging. |
| **HuggingFace Hub Ingestion (`Llama.from_pretrained`)** | Yes | **Full** | Downloading `.gguf` files via `huggingface-hub` with automatic variant and glob resolution. |
| **Context / Model Properties (`n_ctx()`, `n_embd()`, `n_vocab()`)** | Yes | **Full** | Dual support as both properties (`llm.n_ctx`) and callable zero-arg methods (`llm.n_ctx()`), plus `model`, `ctx`, and `set_seed`. |
| **Logit Processors / Custom Biases (`logit_bias`, `logits_processor`)** | Yes | **Full** | `logit_bias` dict, `LogitsProcessorList`, `MinTokensLogitsProcessor`, zero-copy NumPy array views in C memory. |
| **Model Chat Template Engines (Jinja2 / GGUF metadata templates)** | Yes | **Full** | Automatic GGUF `tokenizer.chat_template` extraction, `Jinja2ChatFormatter`, built-in templates (`llama-3`, `gemma`, `llama-2`, `mistral-instruct`, `chatml`), `register_chat_format`, and streaming chat chunks. |
| **Logprob Computation (`logprobs=...`)** | Yes | **Full** | Numerically stable `logits_to_logprobs`, `logprobs=k` extraction for top alternatives, text offsets, and streaming chunk logprobs. |
| **KV Cache Serialization (`save_state`, `load_state`)** | Yes | **Full** | Persisting and restoring KV cache state between turns or process restarts via C-ABI `llama_state_*` and `LlamaState`. |
| **OpenAI Compatible Web Server (`python -m llamojo.server`)** | Yes | **Pending** | FastAPI/Uvicorn server hosting `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. |
| **GPU Acceleration (CUDA, Metal, Vulkan)** | Yes | **By Design (N/A)**| `llamojo` is intentionally an ultra-optimized, dependency-free pure CPU inference engine. |

---

## 2. Recently Closed Gaps

### Closed: KV Cache State Serialization (`save_state` & `load_state`)
- **Status**: **Fully Implemented**
- **Details**: Added `llama_state_get_size`, `llama_state_get_data`, and `llama_state_set_data` to `src/llamojo/llama.mojo` and exported through `c_api.mojo`.
- **Capabilities**:
  - Implemented `LlamaState` in `python/llamojo/llama.py` container holding `input_ids`, `scores`, `n_tokens`, `llama_state` bytes, `llama_state_size`, and `seed`.
  - Serializes a structured binary payload (`LLAMOJOS` magic signature, cell metadata table, and complete active $K$ and $V$ activation buffers).
  - Calling `state = llm.save_state()` followed by `llm.reset()` and `llm.load_state(state)` restores exact token generation trajectories deterministically.
- **Reference**: [`docs/python_kv_cache_serialization.md`](./python_kv_cache_serialization.md)

### Closed: Logprob Computation (`logprobs=...`)
- **Status**: **Fully Implemented**
- **Details**: Numerically stable log softmax over context logits and top-$k$ candidate token extraction.
- **Capabilities**:
  - Top-$k$ alternative tokens returned with text offsets, token pieces, and logprob scores.
  - Full compatibility with both non-streaming (`create_completion`) and chunked streaming (`stream=True`).
- **Reference**: [`docs/python_logprobs.md`](./python_logprobs.md)

### Closed: Model Chat Template Engines (Jinja2 & Built-in Formats)
- **Status**: **Fully Implemented**
- **Details**: Added `chat_format.py` supporting Jinja2 templates and built-in formatters.
- **Capabilities**:
  - Automatic extraction of `tokenizer.chat_template` from GGUF metadata via `extract_chat_template_from_gguf`.
  - Built-in presets for ChatML, LLaMA-2/3, Mistral, and Gemma.
  - Streaming chat completion chunks (`create_chat_completion(..., stream=True)`).
- **Reference**: [`docs/python_chat_templates.md`](./python_chat_templates.md)

### Closed: HuggingFace Hub Integration (`Llama.from_pretrained`)
- **Status**: **Fully Implemented**
- **Details**: Added `Llama.from_pretrained(repo_id, filename=None, ...)` to `python/llamojo/llama.py`.
- **Capabilities**:
  - Direct download and local cache management via `huggingface_hub.hf_hub_download`.
  - Wildcard / glob patterns (e.g. `filename="*Q4_K_M.gguf"`).
  - Automatic variant detection prioritizing `Q4_K_M`, `Q4_0`, and `Q8_0` when `filename` is omitted.
- **Reference**: [`docs/python_from_pretrained.md`](./python_from_pretrained.md)

### Closed: Context & Model Property Accessors (`n_ctx()`, `n_embd()`, `n_vocab()`)
- **Status**: **Fully Implemented**
- **Details**: Implemented `_CallableInt` wrappers for model and context dimension indicators in `python/llamojo/llama.py`.
- **Capabilities**:
  - Dual support: accessible as properties (`llm.n_ctx`) and callable zero-arg methods (`llm.n_ctx()`).
  - Covers `n_ctx`, `n_embd`, `n_vocab`, `n_head`, `n_layer`, `n_batch`, and `n_threads`.
  - Added properties `llm.model` and `llm.ctx` returning ctypes pointers.
  - Added `llm.set_seed(seed: int)`.

### Closed: Logit Processors & Custom Biases (`logit_bias`, `logits_processor`)
- **Status**: **Fully Implemented**
- **Details**: Direct in-place logits manipulation and callback chaining in `python/llamojo/llama.py`.
- **Capabilities**:
  - `logit_bias: Optional[Dict[int, float]]`: Applied in $O(K)$ time via direct pointer index manipulation.
  - `LogitsProcessorList`: Chain of callbacks receiving `(input_ids, scores)`.
  - Zero-copy NumPy view (`np.ctypeslib.as_array`) into context logits memory buffer when NumPy is installed.
  - `MinTokensLogitsProcessor`: Built-in processor to prevent early EOS generation.
  - Fully wired into `create_completion`, `__call__`, `sample`, `generate`, and streaming (`stream=True`).
- **Reference**: [`docs/python_logit_processors.md`](./python_logit_processors.md)

---

## 3. Detailed Breakdown of Remaining Pending Gaps

### Gap 1: OpenAI-Compatible Web Server (`python -m llamojo.server`)
- **Upstream Feature**:
  Built-in CLI server for hosting local OpenAI-compatible REST API endpoints:
  - `GET /v1/models`
  - `POST /v1/completions`
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
- **Current `llamojo` Behavior**:
  `llamojo` is currently used as a Python library. All core endpoint generation methods (`create_completion`, `create_chat_completion`, `create_embedding`) are already implemented and return 100% compliant OpenAI JSON schemas.
- **Implementation Effort**:
  **Low (Python-only)**. Requires adding a `python/llamojo/server.py` module leveraging `fastapi` and `uvicorn`, routing incoming HTTP POST requests directly to `llm.create_completion` and `llm.create_chat_completion`.

---

## 4. Summary Roadmap

With KV Cache serialization (`save_state`/`load_state`), Jinja2 Chat Templates, Logprobs, HuggingFace Hub downloads, Logit Biases, and GBNF Grammars completely closed:
1. **Inference Core Parity**: **100% Complete** with `llama-cpp-python`.
2. **Next Enhancement**: Implement the optional standalone CLI web server (`python -m llamojo.server`).
