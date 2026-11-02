# Hugging Face / PyTorch-Style Facade for llamojo (`hf_compat`)

## 1. Purpose

llamojo's Python package (`llamojo.Llama`) mirrors `llama-cpp-python`. This facade adds a thin layer that follows the **`transformers` causal-LM idioms** (`AutoModelForCausalLM`, `generate()`, `model(input_ids).logits`, tokenizer `encode`/`decode`) so PyTorch-style code can run against a GGUF model.

It is **pure Python, lives only in llamojo**, and does not touch ggojo or the Mojo core.

- Module: `llamojo/python/llamojo/hf_compat.py`
- Tests: `llamojo/tests/test_python_hf_compat.py`
- Exports (from `llamojo`): `LlamojoForCausalLM`, `LlamojoTokenizer`, `CausalLMOutput`

## 2. Quick start

```python
from llamojo import LlamojoForCausalLM

model = LlamojoForCausalLM.from_pretrained("data/SmolLM2-135M-Instruct-Q4_K_M.gguf", n_ctx=2048)
tok = model.tokenizer

ids = tok("Once upon a time")["input_ids"]          # list[list[int]]
out = model.generate(ids, max_new_tokens=32, do_sample=False)
print(tok.batch_decode(out)[0])

logits = model(ids).logits                          # [batch, vocab], last token
```

With PyTorch installed, tensors in produce tensors out:

```python
import torch
out = model.generate(torch.tensor(ids), max_new_tokens=16, temperature=0.7, top_p=0.9)
# out is a torch.Tensor [batch, prompt+new]
```

## 3. API reference

### `LlamojoForCausalLM(llm)`
| Member | Behaviour |
|---|---|
| `from_pretrained(model_path, n_ctx=2048, **kw)` | Loads a **local GGUF file** via `Llama(...)`. Hub ids: use `Llama.from_pretrained`, then wrap with `LlamojoForCausalLM(llm)`. |
| `__call__(input_ids)` / `forward` | Returns `CausalLMOutput(logits=[batch, vocab])`. Each row is `reset()` and evaluated; the **last token's** logits are returned. |
| `generate(input_ids, max_new_tokens=20, do_sample=True, temperature=1.0, top_k=50, top_p=1.0, eos_token_id=None, seed=None)` | Returns prompt + new tokens. `do_sample=False` is greedy (temperature 0). Stops after emitting EOS (EOS is included). |
| `eval()`, `to(...)` | No-ops that return `self` (CPU only). |
| `config` | `dict` with `vocab_size`, `eos_token_id`, `bos_token_id`. |
| `close()` | Frees the underlying context. |

`generate` also accepts `pad_token_id` and `attention_mask`, which are ignored because there is no padding. Any other kwarg, and `num_beams != 1`, raises `NotImplementedError`. `max_new_tokens < 0` or empty `input_ids` raises `ValueError`.

### `LlamojoTokenizer(llm)`
`encode(text, add_special_tokens=True)`, `decode(ids, skip_special_tokens=False)`, `batch_decode(rows)`, `__call__(text | list[str])` → `{"input_ids", "attention_mask"}`, plus `eos_token_id`, `bos_token_id`, `vocab_size`. There is no padding or truncation.

### Input and output types
| Input | `generate` output |
|---|---|
| `list[int]` | `list[int]` |
| `list[list[int]]` | `list[list[int]]` |
| `torch.Tensor` | `torch.Tensor` (rows right-padded with their last token if lengths differ) |
| NumPy array | `list` (via `.tolist()`) |

`logits` is a NumPy array when NumPy is installed, else lists. If the input was a torch tensor, it is a torch tensor.

## 4. Compatibility matrix with PyTorch / `transformers`

| Idiom | Supported | Notes |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained(path)` | Partial | Local GGUF only. No safetensors or config.json. |
| `model.generate(input_ids, max_new_tokens, do_sample, temperature, top_k, top_p, eos_token_id)` | Yes | |
| `model(input_ids).logits` | Partial | Shape `[batch, vocab]`, not `[batch, seq, vocab]`. |
| `tokenizer(text)`, `encode`, `decode`, `batch_decode` | Yes | No padding, truncation or `return_tensors`. |
| `model.eval()`, `.to(device)` | No-op | |
| Beam search, `repetition_penalty`, `num_return_sequences`, `past_key_values` | No | Raises `NotImplementedError` for unknown kwargs. |
| `labels=` / loss, `backward()`, `state_dict()` | No | Inference only. Use ggojo for training. |
| Batched forward pass | No | Rows run sequentially. |
| `[batch, seq, vocab]` logits for perplexity / scoring | No | Would need a Mojo-side change to expose per-position logits. |

## 5. Design decisions

1. **Duck-typed backend.** The facade only calls `reset`, `eval`, `sample`, `scores`, `tokenize`, `detokenize`, `token_eos`, `token_bos`, `n_vocab`, `set_seed` and `close`. This keeps it testable with a fake model and lets other `Llama`-like backends be wrapped.
2. **Stateful engine, stateless API.** llamojo keeps one KV cache. Every row triggers `reset()`, so `generate` calls are independent, and there is no prompt-prefix reuse across calls.
3. **No silent ignoring.** Unsupported generation arguments raise instead of producing output that quietly differs from PyTorch.
4. **Optional dependencies.** NumPy and torch are imported lazily and only change the return types.
5. **llamojo only.** A training-side `nn.Module`-style API belongs in ggojo, and a future shared protocol would be defined without either package importing the other.

## 6. Limitations and follow-ups
- **Last-token logits only.** Full-sequence logits need a new Mojo/C export, which the Phase 2 `context_logits_view` could support.
- **`eval` decodes one token at a time.** `Llama.eval` runs a one-token batch per token, so long prompts are slower than a true batched prefill.
- **Sampling.** `top_k`, `top_p` and `temperature` go through llamojo's sampler chain. The sampler is seeded from the wall clock inside `_build_sampler`, so `seed=` does not currently make `do_sample=True` runs reproducible. Greedy decoding is deterministic.
- **Not yet verified against a real GGUF model.** The tests use a fake backend. Run `tests/test_python_bindings.py`-style checks with `data/SmolLM2-135M-Instruct-Q4_K_M.gguf` for an end-to-end test.
- **Possible extensions:** safetensors and key-mapping import or export, DLPack export of logits, a `pad`/`attention_mask`-aware batched prefill.

## 7. Running the tests
```bash
source /home/tasahi/miniconda/bin/activate moj
cd /mnt/c/Documents/Mojo/llamojo
PYTHONPATH=python python -m unittest tests.test_python_hf_compat -v
```
Result at the time of writing: 7 tests, all passing. The torch round-trip test runs only if torch is installed.

