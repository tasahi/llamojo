# Comprehensive Functional Gap Analysis: `llamojo` vs. `llama-cpp-python`

## Executive Summary
This document provides a thorough audit of the functionality provided by the original `llama-cpp-python` library compared against `llamojo`'s native Python bindings.

Through automated reflection across `llama_cpp.Llama` and `llamojo.Llama`, we identified which capabilities are **fully implemented**, which are **partially supported**, and which represent **pending functional gaps**.

---

## 1. High-Level Comparison Matrix

| Functional Category | `llama-cpp-python` | `llamojo` Status | Assessment |
|---|:---:|:---:|---|
| **Core Text Inference (`create_completion`, `__call__`)** | Yes | **Full** | Complete parity (prompt tokens, greedy/temp, stop sequences, OpenAI JSON response). |
| **Streaming (`stream=True`)** | Yes | **Full** | Complete parity (token piece generator, stop sequence interruption). |
| **Chat Completion (`create_chat_completion`)** | Yes | **Full** | Basic ChatML template formatting and assistant role responses. |
| **Embeddings (`embed`, `create_embedding`)** | Yes | **Full** | Vector generation and normalization, OpenAI list schema. |
| **Tokenizer (`tokenize`, `detokenize`, `token_to_piece`)** | Yes | **Full** | SPM, BPE, byte fallback, special token control. |
| **Low-Level Step Control (`eval`, `sample`, `generate`, `reset`)** | Yes | **Full** | Step-by-step token generation and KV cache management. |
| **GBNF Grammar Constraints (`LlamaGrammar`)** | Yes | **Full** | C-ABI backed GBNF parser, bitmask filtering, pushdown automaton state advancement. |
| **LoRA Adapters (`lora_path`, `apply_lora`)** | Yes | **Full** | Ingestion of GGUF adapter pairs and online additive weight merging. |
| **HuggingFace Hub Ingestion (`Llama.from_pretrained`)** | Yes | **Full** | High convenience utility for downloading `.gguf` files via `huggingface-hub`. |
| **Context / Model Properties (`n_ctx()`, `n_embd()`, `n_vocab()`)** | Yes | **Full** | Dual support as both properties (`llm.n_ctx`) and callable zero-arg methods (`llm.n_ctx()`), plus `model`, `ctx`, and `set_seed`. |
| **Logit Processors / Custom Biases (`logit_bias`, `logits_processor`)** | Yes | **Full** | `logit_bias` dict, `LogitsProcessorList`, `MinTokensLogitsProcessor`, zero-copy NumPy array views in C memory. |
| **KV Cache Serialization (`save_state`, `load_state`)** | Yes | **Pending** | Persisting and restoring KV cache state between turns or process restarts. |
| **Model Chat Template Engines (Jinja2 / GGUF metadata templates)** | Yes | **Partial** | `llamojo` currently defaults to standard ChatML; lacks dynamic Jinja2 template rendering from GGUF metadata. |
| **Context / Model Properties (`n_ctx()`, `n_embd()`, `n_vocab()`)** | Yes | **Partial** | Exposed as instance variables (`llm.n_ctx`) rather than zero-arg accessor methods (`llm.n_ctx()`). |
| **Logprob Computation (`logprobs=...`)** | Yes | **Pending** | Softmax logprobs array on top candidate tokens in completion outputs. |
| **Logit Processors / Custom Biases (`logit_bias=...`)** | Yes | **Pending** | Per-token additive biasing prior to sampling. |
| **OpenAI Compatible Web Server (`python -m llama_cpp.server`)** | Yes | **Pending** | FastAPI/Uvicorn server hosting `/v1/completions` and `/v1/chat/completions`. |
| **GPU Acceleration (CUDA, Metal, Vulkan)** | Yes | **By Design (N/A)**| `llamojo` is intentionally an ultra-optimized, dependency-free pure CPU inference engine. |

---

## 2. Detailed Breakdown of Pending Functional Gaps

### Gap 1: HuggingFace Hub Integration (`Llama.from_pretrained`)
- **Upstream Feature**:
  ```python
  llm = Llama.from_pretrained(
      repo_id="mradermacher/SmolLM2-135M-Instruct-GGUF",
      filename="SmolLM2-135M-Instruct-Q4_K_M.gguf"
  )
  ```
- **Current `llamojo` Behavior**: Requires local file path `Llama(model_path="path/to/model.gguf")`.
- **Implementation Effort**: **Very Low (Python-only)**. Can be implemented in `python/llamojo/llama.py` using `huggingface_hub.hf_hub_download` if installed.

---

### Gap 2: KV Cache State Serialization (`save_state` & `load_state`)
- **Upstream Feature**:
  ```python
  state = llm.save_state()  # Returns LlamaState containing KV cache bytes & tokens
  llm.load_state(state)     # Restores context state instantly
  ```
- **Current `llamojo` Behavior**: Has `reset()` to wipe KV cache, but cannot serialize and restore the cache buffer across calls.
- **Implementation Effort**: **Low-Medium**. Requires exporting `llama_state_get_size`, `llama_state_get_data`, and `llama_state_set_data` from `c_api.mojo`.

---

### Gap 3: Dynamic Jinja2 Chat Formatter
- **Upstream Feature**: Reads `tokenizer.chat_template` from GGUF metadata and formats `messages=[{"role": "user", ...}]` using the model's exact native prompt template (e.g., LLaMA-3 `<|start_header_id|>`, Mistral `[INST]`, Gemma `<start_of_turn>`).
- **Current `llamojo` Behavior**: Formats messages using the standard ChatML template `<|im_start|>role\ncontent<|im_end|>`.
- **Implementation Effort**: **Low (Python-only)**. Can parse `tokenizer.chat_template` or inspect model type to apply the appropriate jinja2 / regex template.

---

### Gap 4: Logprobs & Logit Biasing (`logprobs` & `logit_bias`)
- **Upstream Feature**:
  - `logprobs: Optional[int] = 5`: Returns the top 5 token log probabilities alongside the generated token.
  - `logit_bias: Optional[Dict[int, float]]`: Injects additive logit penalties/boosts before sampling.
- **Current `llamojo` Behavior**: `logprobs: None`; sampling applies greedy/temperature/top_k/top_p/grammar without per-token user logit bias dictionary.
- **Implementation Effort**: **Low**. Can be injected directly into the sampling logits array before `llama_sampler_sample_token`.

---

### Gap 5: Method vs. Attribute Compatibility (`n_ctx()`, `n_embd()`, `n_vocab()`)
- **Upstream Feature**: In `llama-cpp-python`, `llm.n_ctx()` can be called as a method (or accessed as property), and properties `llm.model` / `llm.ctx` return underlying ctypes pointers.
- **Current `llamojo` Behavior**: Exposes attributes `llm.n_ctx`, `llm.n_embd`, `llm.n_vocab`, `llm.n_head`.
- **Implementation Effort**: **Trivial**. Add method aliases / property getters so both `llm.n_ctx` and `llm.n_ctx()` work transparently.

---

## 3. Recommended Roadmap for Closing the Gaps

1. **Quick Wins (Pure Python)**:
   - Add `Llama.from_pretrained(repo_id, filename)` via `huggingface_hub`.
   - Add method aliases (`n_ctx()`, `n_embd()`, `n_vocab()`, `set_seed()`).
   - Add `logit_bias` dictionary support in `create_completion`.
   - Add model-aware chat templates (Gemma, LLaMA-3, Mistral, ChatML).
2. **Intermediate (Mojo C-ABI + Python)**:
   - Export `save_state` and `load_state` from `c_api.mojo`.
   - Implement `logprobs` calculation in completion responses.

