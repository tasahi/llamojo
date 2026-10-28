# Logit Processors & Custom Biases in `llamojo`

## Overview
`llamojo` implements full API compatibility with `llama-cpp-python`'s logits manipulation pipeline:
1. `logit_bias: Optional[Dict[int, float]]`: Token-level additive bias dictionary.
2. `logits_processor: Optional[Union[LogitsProcessorList, Sequence[LogitsProcessor]]]`: Sequential list/chain of processor callbacks modifying output scores prior to sampling.
3. `LogitsProcessorList` and `MinTokensLogitsProcessor` classes exposed directly from `llamojo`.

---

## 1. Using `logit_bias`
Pass a dictionary mapping token IDs to positive boosts or negative penalties:

```python
from llamojo import Llama

llm = Llama("data/SmolLM2-135M-Instruct-Q4_K_M.gguf")

# Find token ID
tok_id = llm.tokenize("banana", add_bos=False)[0]

# Strongly boost the token to force it
res = llm(
    "Once upon a time in a jungle, there was a",
    max_tokens=10,
    logit_bias={tok_id: 100.0},
)
print(res["choices"][0]["text"])

# Forbid a specific token
res_forbidden = llm(
    "1 2 3",
    max_tokens=5,
    logit_bias={tok_id: -100.0},
)
```

---

## 2. Custom `LogitsProcessorList`
Implement custom logits processors matching the signature `processor(input_ids: Sequence[int], scores: Sequence[float]) -> Sequence[float]`.

```python
import numpy as np
from llamojo import Llama, LogitsProcessorList

llm = Llama("data/SmolLM2-135M-Instruct-Q4_K_M.gguf")

def ban_first_word_processor(input_ids, scores):
    # If using numpy, scores is a zero-copy float32 ndarray view directly into C memory
    banned_tok = 2947
    scores[banned_tok] = -np.inf
    return scores

procs = LogitsProcessorList([ban_first_word_processor])

res = llm(
    "The secret word is",
    max_tokens=16,
    logits_processor=procs,
)
```

---

## 3. `MinTokensLogitsProcessor`
Ensure a minimum generation length before allowing the model to produce `EOS`:

```python
from llamojo import Llama, MinTokensLogitsProcessor, LogitsProcessorList

llm = Llama("data/SmolLM2-135M-Instruct-Q4_K_M.gguf")

min_proc = MinTokensLogitsProcessor(min_tokens=20, token_eos=llm.token_eos())
procs = LogitsProcessorList([min_proc])

res = llm("Hi!", max_tokens=64, logits_processor=procs)
```

---

## Performance & Memory
- Direct pointer manipulation (`ctypes.LP_c_float`) applies `logit_bias` in $O(K)$ time where $K$ is the number of biased tokens.
- When `numpy` is installed, processors receive a zero-copy `np.ndarray` view (`np.ctypeslib.as_array`) of the underlying context logits memory buffer, eliminating unnecessary allocations.
- Fully supported in `create_completion`, `__call__`, `sample`, `generate`, and streaming (`stream=True`).

