---
type: implementation-report
title: "Implementation Report: Phase 1 - Ministral 3 3B GGUF Ingestion & Architecture Configuration"
description: "Detailed documentation of Phase 1 implementation for Ministral 3 3B support in llamojo, including ModelConfig fields, GGUF metadata parsing, YaRN RoPE hyperparameter bindings, and vocabulary score decoding."
status: completed
tags: [ministral-3b, phase1, gguf, yarn-rope, modelconfig, vocab, llamojo]
timestamp: 2026-11-10T20:07:00Z
---

# Implementation Report: Phase 1 — Ministral 3 3B GGUF Ingestion & Architecture Configuration

## 1. Overview & Objectives

In **Phase 1**, we implemented the core metadata ingestion and structural model configuration required to load and parse **Ministral 3 3B** GGUF model files (such as `Ministral-3-3B-Instruct-2512` and community quants) within `llamojo`.

The primary goals of Phase 1 were:
1. Extend `ModelConfig` with dynamic configuration fields for **YaRN RoPE** (Yet another RoPE extensioN) and general sliding window attention (SWA).
2. Update the GGUF loader in `src/llamojo/gguf.mojo` to parse model hyper-parameters from the `{arch}.*` metadata namespace (`mistral3`, `ministral3`, `llama`, etc.).
3. Enhance the vocabulary loader in `src/llamojo/vocab.mojo` to safely ingest token scores represented as integer or float arrays (preventing parser failures observed in some GGUF converters).
4. Verify end-to-end GGUF parsing, tied embedding detection, and parameter population via a rigorous synthetic unit test suite.

---

## 2. Changes Implemented

### 2.1 Extension of `ModelConfig` (`src/llamojo/types.mojo`)
To represent Ministral 3's 256k context window and rotary scaling parameters accurately, `ModelConfig` was augmented with dedicated RoPE configuration attributes:

```mojo
@fieldwise_init
struct ModelConfig(ImplicitlyCopyable, Copyable, Movable):
    ...
    var quant_type: UInt32       # Primary weight GGMLType

    # Rotary Position Embedding & YaRN configuration
    var rope_scaling_type: Int32 # llama_rope_scaling_type (0: None, 1: Linear, 2: YaRN)
    var rope_orig_ctx_len: Int   # {arch}.rope.scaling.original_context_length (e.g. 16384)
    var yarn_beta_fast: Float32  # {arch}.rope.scaling.yarn_beta_fast (default 32.0)
    var yarn_beta_slow: Float32  # {arch}.rope.scaling.yarn_beta_slow (default 1.0)
    var yarn_attn_factor: Float32# {arch}.rope.scaling.yarn_attn_factor (default 1.0)
    ...
```

The initializer sets default parameters matching the YaRN baseline specification:
- `self.rope_scaling_type = 0` (`LLAMA_ROPE_SCALING_TYPE_NONE`)
- `self.rope_orig_ctx_len = 0`
- `self.yarn_beta_fast = 32.0`
- `self.yarn_beta_slow = 1.0`
- `self.yarn_attn_factor = 1.0`

### 2.2 GGUF Metadata Parsing & Auto-Detection (`src/llamojo/gguf.mojo`)
In `GGUFContext.build_model_config()`, dynamic parameter resolution was expanded:
1. **Sliding Window Attention**:
   Instead of hardcoding SWA exclusively for `gemma4`, `gguf.mojo` now extracts:
   ```mojo
   var sw_val = self.get_int(arch + ".attention.sliding_window", 0)
   if sw_val > 0:
       cfg.sliding_window = sw_val
   ```
2. **RoPE Scaling Algorithm Detection**:
   Parses `[arch].rope.scaling.type`:
   - `"yarn"` maps to `llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN` (value `2`)
   - `"linear"` maps to `llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_LINEAR` (value `1`)
   - `"none"` maps to `llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_NONE` (value `0`)
3. **YaRN Hyperparameters**:
   Parses the original base context length and frequency correction factors:
   ```mojo
   cfg.rope_orig_ctx_len = self.get_int(arch + ".rope.scaling.original_context_length", 0)
   cfg.yarn_beta_fast = self.get_float(arch + ".rope.scaling.yarn_beta_fast", 32.0)
   cfg.yarn_beta_slow = self.get_float(arch + ".rope.scaling.yarn_beta_slow", 1.0)
   cfg.yarn_attn_factor = self.get_float(arch + ".rope.scaling.yarn_attn_factor", 1.0)
   ```
4. **Tied Embedding Detection**:
   Verifies whether `output.weight` is present in the GGUF tensor directory; if absent, `cfg.is_tied_embeddings` is set to `True`, allowing the causal language head to automatically reuse `token_embd.weight`.

### 2.3 Robust Tokenizer Scores Ingestion (`src/llamojo/vocab.mojo`)
Certain GGUF converters for Ministral 3 encode `tokenizer.ggml.scores` as an array of `Int32` or `Int64` integers instead of floats. In older loaders, this resulted in empty scores or deserialization errors.

We updated `llama_vocab.__init__` in `src/llamojo/vocab.mojo`:
```mojo
var score_opt = gguf.metadata.find("tokenizer.ggml.scores")
if score_opt:
    if score_opt.value().vtype == GGUFValueType.ARRAY and len(score_opt.value().float_arr) > 0:
        self.scores = score_opt.value().float_arr.copy()
    elif score_opt.value().vtype == GGUFValueType.ARRAY and len(score_opt.value().int_arr) > 0:
        # Some GGUF exporters encode scores as Int32/Int64 integer arrays
        var i_arr = score_opt.value().int_arr.copy()
        for i in range(len(i_arr)):
            self.scores.append(Float32(i_arr[i]))
```

---

## 3. Verification & Testing

A new test suite [`tests/test_ministral3_phase1.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_ministral3_phase1.mojo) was constructed. It generates a spec-compliant synthetic GGUF v3 file modelling the complete parameter set of **Ministral 3 3B**:
- Architecture: `mistral3`
- Dimensions: $d_{\text{model}} = 3072$, $d_{\text{ffn}} = 9216$, $N_{\text{layers}} = 26$
- Attention heads: 32 query heads, 8 KV heads ($d_{\text{head}} = 96$)
- Max context length: 262,144 tokens
- RoPE theta: $1,000,000.0$
- YaRN configuration: `type = "yarn"`, `factor = 16.0`, `orig_ctx = 16384`, `beta_fast = 32.0`, `beta_slow = 1.0`, `attn_factor = 1.0`
- Tied embeddings: `output.weight` omitted
- Vocab: Integer-encoded token score array

### Test Execution Results
```bash
$ mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_ministral3_phase1.mojo
============================================================
Running Ministral 3 3B Phase 1 Tests
============================================================
Testing Ministral 3 3B GGUF metadata parsing and ModelConfig population...
  ✓ Ministral 3 metadata & YaRN parameters verified
All Phase 1 tests passed! ✓
```

In addition, regression testing across existing GGUF and llama API test suites verified zero regressions:
```bash
$ mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_gguf.mojo
All GGUF tests passed! ✓

$ mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_llama_api.mojo
All Phase 1 API tests passed! ✓
```

---

## 4. Next Steps (Phase 2 & Phase 3)

With Phase 1 complete, `llamojo` is ready for kernel-level Phase 2 work:
1. **Phase 2 (YaRN RoPE Implementation)**: Implement `apply_rope_yarn` in `src/llamojo/backend/kernels_norm_rope.mojo` to evaluate interpolated and extrapolated frequencies with the parsed $\beta$ boundaries and attention scaling factors.
2. **Phase 3 (Attention & Execution Integration)**: Pass `cfg.sliding_window` and dispatch YaRN RoPE inside `src/llamojo/backend/models.mojo:forward_layer_dense`.
