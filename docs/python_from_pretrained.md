---
type: reference
title: "Hugging Face Hub Integration: Llama.from_pretrained"
description: "Design and ctypes implementation for downloading, caching, and instantiating GGUF models directly from Hugging Face Hub."
status: completed
tags: [huggingface-hub, from-pretrained, caching, gguf-download, python-api]
timestamp: 2026-11-05T00:00:00Z
---

# Hugging Face Hub Integration: `Llama.from_pretrained`

## Overview
`llamojo` now includes direct drop-in support for downloading and loading `.gguf` weights from the Hugging Face Hub via `Llama.from_pretrained(...)`, matching the behaviour of `llama-cpp-python`.

---

## Usage Examples

### 1. Specific Model File
```python
from llamojo import Llama

llm = Llama.from_pretrained(
    repo_id="mradermacher/SmolLM2-135M-Instruct-GGUF",
    filename="SmolLM2-135M-Instruct-Q4_K_M.gguf",
    n_ctx=512,
)

response = llm("Tell me a three-sentence story about space.", max_tokens=64)
print(response["choices"][0]["text"])
```

### 2. Wildcard / Glob Pattern Matching
```python
llm = Llama.from_pretrained(
    repo_id="mradermacher/SmolLM2-135M-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_ctx=512,
)
```

### 3. Automatic Variant Resolution
When `filename` is omitted, `Llama.from_pretrained` inspects the repository file tree and selects the most appropriate `.gguf` file (prioritizing `Q4_K_M`, `Q4_0`, or `Q8_0`):
```python
llm = Llama.from_pretrained(
    repo_id="mradermacher/SmolLM2-135M-Instruct-GGUF",
    n_ctx=512,
)
```

### 4. Custom Cache Directory & Authentication
All standard Hugging Face download options (`cache_dir`, `local_dir`, `token`, `revision`, etc.) are forwarded cleanly to `huggingface_hub.hf_hub_download`, while model parameters (`n_ctx`, `n_batch`, `lora_path`, etc.) are forwarded directly to the `Llama` instance:
```python
llm = Llama.from_pretrained(
    repo_id="private-org/custom-model-gguf",
    filename="model-q4_0.gguf",
    token="hf_...",
    cache_dir="/custom/cache/models",
    n_ctx=1024,
    n_threads=4,
)
```

---

## Verification
The feature is validated in `tests/test_python_bindings.py`:
- Explicit filename resolution
- Wildcard glob pattern resolution
- Auto-selection of quantized GGUF weights
- Passing 6/6 test cases in the python bindings suite.

