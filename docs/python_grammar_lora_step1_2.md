# llamojo Python Integration: GBNF Grammar & LoRA C-ABI Exposition (Steps 1 & 2)

## Executive Summary
This document records the implementation of **Steps 1 & 2** from the [GBNF Grammar and LoRA Exposition Plan](./python_grammar_lora_plan.md).

We added native C-ABI export functions to [`src/llamojo/c_api.mojo`](../src/llamojo/c_api.mojo) and recompiled the shared dynamic library [`build/libllamojo.so`](../build/libllamojo.so). This bridges the pure-Mojo GBNF parser ([`grammar.mojo`](../src/llamojo/grammar.mojo)) and online LoRA weight merger ([`lora.mojo`](../src/llamojo/lora.mojo)) directly to foreign interfaces such as Python `ctypes`.

---

## 1. Exported C-ABI Functions

The following C symbols have been added to the public API with `@export ... abi("c")`:

### A. GBNF Grammar & Pushdown Automaton
1. **`llama_grammar_p llama_grammar_init(const char * grammar_cstr)`**
   - Parses a GBNF grammar specification string into an internal `GbnfGrammar` state machine.
   - Supports productions, alternations (`|`), character classes (`[0-9]`), and literals.
   - Returns a heap-allocated opaque handle `llama_grammar_p`.

2. **`void llama_grammar_free(llama_grammar_p grammar)`**
   - Deallocates the `GbnfGrammar` instance safely.

3. **`llama_sampler_p llama_sampler_init_grammar(llama_vocab_p vocab, const char * grammar_pattern_cstr)`**
   - Instantiates a grammar sampler stage attached to the vocabulary token strings.
   - Evaluates prefix validity against candidates and sets invalid token logits to $-10^{30}$ ($-\infty$).

4. **`void llama_sampler_accept_token(llama_sampler_p smpl, llama_token token)`**
   - Advances the state machine buffer when a token is accepted by the generation loop.

### B. Dynamic LoRA Adapters
1. **`llama_lora_adapter_p llama_lora_adapter_init(const char * path_cstr)`**
   - Loads and parses an adapter GGUF file containing low-rank tensor pairs $(A, B)$ and hyperparameters ($\alpha, r$).

2. **`void llama_lora_adapter_free(llama_lora_adapter_p adapter)`**
   - Releases adapter memory and mapped buffers.

3. **`int32_t llama_lora_adapter_apply(llama_model_p model, const char * adapter_path_cstr, float scale)`**
   - Loads a LoRA adapter from `adapter_path_cstr` and computes in-place additive low-rank updates:
     $$\Delta W = \text{scale} \cdot \left(\frac{\alpha}{r}\right) (B \times A)$$
   - Directly applies the delta to base model weights in memory.

---

## 2. Dynamic Library Verification (`build/libllamojo.so`)

The shared library was compiled using [`tools/build_shared_lib.sh`](../tools/build_shared_lib.sh):
```bash
./tools/build_shared_lib.sh
```

Inspecting the dynamic symbol table:
```bash
nm -D build/libllamojo.so | grep -E "llama_grammar|llama_lora|llama_sampler_accept|llama_sampler_init_grammar"
```
**Output**:
```
00000000000935c0 T llama_grammar_free
000000000007d3d0 T llama_grammar_init
0000000000033ec0 T llama_lora_adapter_apply
00000000000928b0 T llama_lora_adapter_free
000000000007caf0 T llama_lora_adapter_init
0000000000091050 T llama_sampler_accept_token
0000000000090580 T llama_sampler_init_grammar
```

All 7 symbols are properly exported as un-mangled C functions.

---

## 3. Python Verification Script

We verified that Python `ctypes` can call the newly exported endpoints:

```python
import ctypes

lib = ctypes.CDLL("build/libllamojo.so")

lib.llama_grammar_init.argtypes = [ctypes.c_char_p]
lib.llama_grammar_init.restype = ctypes.c_void_p
lib.llama_grammar_free.argtypes = [ctypes.c_void_p]
lib.llama_grammar_free.restype = None

# Test GBNF parsing
grammar_handle = lib.llama_grammar_init(b"root ::= [0-9]+")
assert grammar_handle is not None
lib.llama_grammar_free(grammar_handle)

# Test LoRA guard
lib.llama_lora_adapter_apply.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
lib.llama_lora_adapter_apply.restype = ctypes.c_int32

ret = lib.llama_lora_adapter_apply(None, b"invalid.gguf", 1.0)
assert ret == -1
```

Both tests passed cleanly without crashes or memory corruption.

