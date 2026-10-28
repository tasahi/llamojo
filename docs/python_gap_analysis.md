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
| **Chat Completion (`create_chat_completion`)** | Yes | **Full** | ChatML template formatting and assistant role responses. |
| **Embeddings (`embed`, `create_embedding`)** | Yes | **Full** | Vector generation and normalization, OpenAI list schema. |
| **Tokenizer (`tokenize`, `detokenize`, `token_to_piece`)** | Yes | **Full** | SPM, BPE, byte fallback, special token control. |
| **Low-Level Step Control (`eval`, `sample`, `generate`, `reset`)** | Yes | **Full** | Step-by-step token generation and KV cache management. |
| **GBNF Grammar Constraints (`LlamaGrammar`)** | Yes | **Full** | C-ABI backed GBNF parser, bitmask filtering, pushdown automaton state advancement. |
| **LoRA Adapters (`lora_path`, `apply_lora`)** | Yes | **Full** | Ingestion of GGUF adapter pairs and online additive weight merging. |
| **HuggingFace Hub Ingestion (`Llama.from_pretrained`)** | Yes | **Full** | Downloading `.gguf` files via `huggingface-hub` with automatic variant and glob resolution. |
| **Context / Model Properties (`n_ctx()`, `n_embd()`, `n_vocab()`)** | Yes | **Full** | Dual support as both properties (`llm.n_ctx`) and callable zero-arg methods (`llm.n_ctx()`), plus `model`, `ctx`, and `set_seed`. |
| **Logit Processors / Custom Biases (`logit_bias`, `logits_processor`)** | Yes | **Full** | `logit_bias` dict, `LogitsProcessorList`, `MinTokensLogitsProcessor`, zero-copy NumPy array views in C memory. |
| **Model Chat Template Engines (Jinja2 / GGUF metadata templates)** | Yes | **Full** | Automatic GGUF `tokenizer.chat_template` extraction, `Jinja2ChatFormatter`, built-in templates (`llama-3`, `gemma`, `llama-2`, `mistral-instruct`, `chatml`), `register_chat_format`, and streaming chat chunks. |
| **KV Cache Serialization (`save_state`, `load_state`)** | Yes | **Pending** | Persisting and restoring KV cache state between turns or process restarts. |
| **Logprob Computation (`logprobs=...`)** | Yes | **Pending** | Softmax logprobs array on top candidate tokens in completion outputs. |
| **OpenAI Compatible Web Server (`python -m llama_cpp.server`)** | Yes | **Pending** | FastAPI/Uvicorn server hosting `/v1/completions` and `/v1/chat/completions`. |
| **GPU Acceleration (CUDA, Metal, Vulkan)** | Yes | **By Design (N/A)**| `llamojo` is intentionally an ultra-optimized, dependency-free pure CPU inference engine. |

---

## 2. Recently Closed Gaps

### Closed: HuggingFace Hub Integration (`Llama.from_pretrained`)
- **Status**: **Fully Implemented**
- **Details**: Added `Llama.from_pretrained(repo_id, filename=None, ...)` to `python/llamojo/llama.py`.
- **Capabilities**:
  - Direct download and local cache management via `huggingface_hub.hf_hub_download`.
  - Wildcard / glob patterns (e.g. `filename="*Q4_K_M.gguf"`).
  - Automatic variant detection prioritizing `Q4_K_M`, `Q4_0`, and `Q8_0` when `filename` is omitted.
  - Forwarding authentication tokens, custom `cache_dir`, `local_dir`, and revisions.
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

### Gap 1: KV Cache State Serialization (`save_state` & `load_state`)
- **Upstream Feature**:
  ```python
  state = llm.save_state()  # Returns LlamaState containing KV cache bytes & tokens
  llm.load_state(state)     # Restores context state instantly
  ```
- **Current `llamojo` Behavior**: Has `reset()` to wipe the KV cache, but cannot serialize and restore the cache buffer across calls or process restarts.
- **Implementation Effort**: **Medium (Mojo C-ABI + Python)**. Requires exporting `llama_state_get_size`, `llama_state_get_data`, and `llama_state_set_data` from `src/llamojo/c_api.mojo`.

---

### Gap 2: Dynamic Jinja2 Chat Formatter
- **Upstream Feature**: Reads `tokenizer.chat_template` from GGUF metadata and formats `messages=[{"role": "user", ...}]` using the model's exact native prompt template (e.g., LLaMA-3 `<|start_header_id|>`, Mistral `[INST]`, Gemma `<start_of_turn>`).
- **Current `llamojo` Behavior**: Formats messages using the standard ChatML template `<|im_start|>role\ncontent<|im_end|>`.
- **Implementation Effort**: **Low (Python-only)**. Can parse `tokenizer.chat_template` or inspect model architecture to apply Jinja2 templates via `jinja2`.

---

### Gap 3: Logprobs Computation (`logprobs=...`)
- **Upstream Feature**:
  - `logprobs: Optional[int] = 5`: Returns the top 5 token log probabilities and chosen token logprob in completion output choices.
- **Current `llamojo` Behavior**: `logprobs: None`.
- **Implementation Effort**: **Medium**. Softmax over logits at each step, extracting the top-$k$ log probabilities into the choices dictionary.

---

### Gap 4: OpenAI-Compatible Web Server (`python -m llamojo.server`)
- **Upstream Feature**: Built-in CLI server for running local OpenAI-compatible endpoints (`/v1/models`, `/v1/completions`, `/v1/chat/completions`).
- **Current `llamojo` Behavior**: Python library only.
- **Implementation Effort**: **Low-Medium (Python-only)**. FastAPI / Uvicorn wrapper around `llamojo.Llama`.

---

## 4. Next Implementation Priorities

1. **KV Cache State Serialization (`save_state`, `load_state`)**:
   - Add C-ABI export functions in `src/llamojo/c_api.mojo`.
   - Rebuild `libllamojo.so` via `tools/build_shared_lib.sh`.
   - Expose `LlamaState`, `save_state()`, and `load_state()` in `python/llamojo/llama.py`.
2. **Dynamic Jinja2 Chat Templates**:
   - Extract GGUF chat template metadata or apply model-type heuristics.
3. **Logprob Computation**:
   - Return top candidate log probabilities in `create_completion`.
