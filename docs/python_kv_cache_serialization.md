# Python KV Cache & Context State Serialization (`save_state` & `load_state`)

## 1. Overview

This document describes the implementation of context state and KV-cache serialization in `llamojo`'s Python bindings, closing the parity gap with `llama-cpp-python`'s `save_state()` and `load_state()`.

This capability allows applications to:
- Snapshot intermediate multi-turn conversation states without re-evaluating prompt prefixes.
- Persist engine context to disk or cache storage between process restarts.
- Branch generation trajectories deterministically from any checkpointed state.

---

## 2. Binary State Serialization Specification (`LLAMOJOS`)

The serialized binary payload produced by `llama_state_get_data` and read by `llama_state_set_data` follows a deterministic layout:

```
+-------------------------------------------------------------------------------+
|  Header (32 Bytes)                                                            |
|  - uint64: Magic Identifier 0x4C4C414D4F4A4F53 ("LLAMOJOS")                   |
|  - uint64: Current n_tokens                                                   |
|  - uint64: Context max_ctx                                                    |
|  - uint64: Total payload size in bytes                                        |
+-------------------------------------------------------------------------------+
|  Cell Metadata Table (max_ctx * 8 Bytes)                                      |
|  - Array of LlamaKVCell { pos: int32, seq_id: int32 }                          |
+-------------------------------------------------------------------------------+
|  K Cache F32 Tensor (n_layer_kv * max_ctx * kv_dim * 4 Bytes)                 |
+-------------------------------------------------------------------------------+
|  V Cache F32 Tensor (n_layer_kv * max_ctx * kv_dim * 4 Bytes)                 |
+-------------------------------------------------------------------------------+
|  Optional Quantized Q8 Buffers (if active)                                    |
|  - K Cache Q8_0 Buffer                                                        |
|  - V Cache Q8_0 Buffer                                                        |
+-------------------------------------------------------------------------------+
```

Total state size is computed dynamically:
$$\text{StateSize} = 32 + (8 \times \text{max\_ctx}) + 2 \times (\text{n\_layer\_kv} \times \text{max\_ctx} \times \text{kv\_dim} \times 4) + [Q8\_buffers]$$

---

## 3. C-ABI Interface

The following C functions are exported by `libllamojo.so`:

```mojo
@export
def llama_state_get_size(ctx: llama_context_p) abi("c") -> UInt64

@export
def llama_state_get_data(ctx: llama_context_p, dst: BytePtr, size: UInt64) abi("c") -> UInt64

@export
def llama_state_set_data(ctx: llama_context_p, src: BytePtr, size: UInt64) abi("c") -> UInt64
```

---

## 4. Python API (`llamojo.LlamaState`)

### `LlamaState` Container

```python
class LlamaState:
    input_ids: Sequence[int]
    scores: Any
    n_tokens: int
    llama_state: bytes
    llama_state_size: int
    seed: int
```

### Python Methods on `Llama`

- `llm.save_state() -> LlamaState`: Serializes the current context memory and returns a `LlamaState` instance.
- `llm.load_state(state: LlamaState) -> None`: Restores the context memory, active token history, and seed.

### Usage Example

```python
from llamojo import Llama

llm = Llama(model_path="data/smollm2-135m-q4_k_m.gguf", n_ctx=512)

# 1. Evaluate prompt
prompt = "The capital of France is"
tokens = llm.tokenize(prompt, add_bos=True)
llm.eval(tokens)

# 2. Checkpoint state
state = llm.save_state()

# 3. Sample continuation
tok1 = llm.sample(temp=0.0)
print("Continuation 1:", llm.token_to_piece(tok1))  # ' Paris'

# 4. Wipe context
llm.reset()

# 5. Restore state
llm.load_state(state)

# 6. Sample continuation from restored state
tok2 = llm.sample(temp=0.0)
print("Continuation 2:", llm.token_to_piece(tok2))  # ' Paris'
assert tok1 == tok2
```

