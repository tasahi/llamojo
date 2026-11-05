---
type: reference
title: "Logprob Computation in llamojo"
description: "High-performance token logprob extraction, top-k candidate ranking, and OpenAI API compliant schema formatting."
status: completed
tags: [logprobs, token-probabilities, top-k, openai-compliance]
timestamp: 2026-11-05T00:00:00Z
---

# Logprob Computation in `llamojo`

## Overview
`llamojo` implements log probability calculation matching `llama-cpp-python`'s OpenAI completion specification:
- `logprobs: Optional[int]`: When specified (e.g. `logprobs=5`), returns the chosen token's log probability alongside the top-$k$ candidate token alternatives at each step.
- `Llama.logits_to_logprobs(logits, axis=-1)`: Static helper function computing numerically stable log softmax.

---

## 1. Non-Streaming Output Schema
In standard completion calls (`stream=False`), `choices[0]["logprobs"]` returns a dictionary containing:
- `tokens`: List of string tokens generated.
- `text_offset`: List of character start offsets in the completion text.
- `token_logprobs`: List of log probabilities for the chosen tokens ($\le 0.0$).
- `top_logprobs`: List of dictionaries mapping token pieces to their corresponding log probabilities.

```python
from llamojo import Llama

llm = Llama("data/SmolLM2-135M-Instruct-Q4_K_M.gguf")

res = llm(
    "The sun rises in the",
    max_tokens=3,
    temperature=0.0,
    logprobs=5,
)

lp = res["choices"][0]["logprobs"]
print("Tokens:", lp["tokens"])
print("Token Logprobs:", lp["token_logprobs"])
print("Top Candidates for first token:", lp["top_logprobs"][0])
```

Output:
```python
Tokens: [' east', ',', ' casting']
Token Logprobs: [-0.7477, -0.9470, -1.2522]
Top Candidates for first token: {
    ' east': -0.7477,
    ' west': -1.5872,
    ' sky': -2.7127,
    ' morning': -3.0870,
    ' East': -3.9077
}
```

---

## 2. Streaming Output Schema (`stream=True`)
When streaming with `logprobs=k`, each emitted chunk contains the per-token `logprobs` object:

```python
stream = llm(
    "The sun rises in the",
    max_tokens=3,
    temperature=0.0,
    logprobs=3,
    stream=True,
)

for chunk in stream:
    lp = chunk["choices"][0]["logprobs"]
    if lp is not None:
        print("Chunk token:", lp["tokens"][0], "Logprob:", lp["token_logprobs"][0])
```

---

## 3. High-Performance Numerically Stable Softmax
- Utilizes `np.argpartition` and in-place vector arithmetic to extract top-$k$ candidate tokens in $O(V)$ time without sorting the entire vocabulary ($V \approx 49,152$ tokens).
- Prevents numerical underflow/overflow by shifting logits by the max logit ($\vec{z} - \max(\vec{z})$).

