---
type: reference
title: "llamojo Python Integration Phase 2: High-Level llama_cpp-Compatible Python Wrapper"
description: "Development of ctypes-driven Python wrapper matching llama-cpp-python Llama class interface."
status: completed
tags: [python-integration, phase2, ctypes, llama-cpp-python, api-wrapper]
timestamp: 2026-11-05T00:00:00Z
---

# llamojo Python Integration Phase 2: High-Level `llama_cpp`-Compatible Python Wrapper

## Executive Summary
In Phase 2 of the Python integration plan, we implemented the high-level Python package **`llamojo`** that exposes an identical interface to `llama-cpp-python` (`from llamojo import Llama`).

This enables Python applications, pipelines, and frameworks to swap out `llama_cpp` for `llamojo` with zero code modifications at the callsite.

---

## 1. Package Structure

```
python/
  └── llamojo/
      ├── __init__.py      # Package export: Llama
      ├── _lib.py          # Low-level ctypes dynamic library loader for libllamojo.so
      └── llama.py         # High-level Llama class implementing OpenAI/llama_cpp schemas
```

---

## 2. API Capabilities

The implemented `Llama` class supports the primary surface of `llama-cpp-python`:

### A. Initialization
```python
from llamojo import Llama

llm = Llama(
    model_path="data/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    n_ctx=512,
    n_batch=512,
    n_threads=2,
    n_gpu_layers=0,
)
```
- Automatically resolves `libllamojo.so` from build outputs or environment paths (`LLAMOJO_LIB_PATH`).
- Initializes context, models, and vocabularies safely via native pointers.

### B. Text Completion (`create_completion` and `__call__`)
Supports both standard and shorthand execution with OpenAI-compatible responses:
```python
# Function call or callable instance
response = llm(
    "Once upon a time",
    max_tokens=64,
    temperature=0.0,
    stop=["\n", "The end"],
    echo=False,
)

print(response["choices"][0]["text"])
print(response["usage"])
```
**Schema Returned**:
```json
{
  "id": "cmpl-b5a87158-7451-4e78-...",
  "object": "text_completion",
  "created": 1793188207,
  "model": "data/SmolLM2-135M-Instruct-Q4_K_M.gguf",
  "choices": [
    {
      "text": ", I was a young man with dreams...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 64,
    "total_tokens": 68
  }
}
```

### C. Streaming Generation (`stream=True`)
Iterates over token pieces in real time:
```python
for chunk in llm("Once upon a time", max_tokens=32, stream=True):
    print(chunk["choices"][0]["text"], end="", flush=True)
```

### D. Chat Completion (`create_chat_completion`)
Supports multi-turn chat messages:
```python
chat = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=32,
)
print(chat["choices"][0]["message"]["content"])
```

### E. Tokenizer Utilities
- `tokens = llm.tokenize("Some text", add_bos=True)`
- `raw_bytes = llm.detokenize(tokens)`
- `piece = llm.token_to_piece(token_id)`

---

## 3. Test Suite & Verification

A dedicated unit test suite was added to [`tests/test_python_bindings.py`](../../tests/test_python_bindings.py) covering:
1. `test_tokenization`: Validates forward and reverse token piece decoding.
2. `test_completion_schema`: Validates return dictionary matching OpenAI/llama_cpp specs.
3. `test_callable_and_stop`: Validates callable `__call__` syntax and early termination on `stop` substrings.
4. `test_streaming`: Validates generator chunk yield sequence.
5. `test_chat_completion`: Validates message formatting and assistant response structure.

### Test Execution Results
```bash
PYTHONPATH=python python tests/test_python_bindings.py
```
```
.....
----------------------------------------------------------------------
Ran 5 tests in 11.739s

OK
```
All unit tests passed with 100% success.
