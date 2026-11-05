---
type: reference
title: "Gemma 4 Architecture Support: Phase 1 Implementation Documentation"
description: "Verification and integration of Gemma 4 GGUF tensor layout mapping and config schema extensions in llamojo."
status: completed
tags: [gemma4, phase1, gguf-parsing, config-schema]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Architecture Support: Phase 1 Implementation Documentation

This document records the design, implementation, and verification of **Phase 1: GGUF Metadata Parser & Model Config Upgrades** for the Gemma 4 architecture (`gemma4`) in **llamojo**.

---

## 1. Context & Motivation

During early compatibility testing of the Gemma 4 checkpoint (`gemma-4-E2B-it.Q4_K_M.gguf`), `llamojo`'s inference loop emitted repeating `<unused33>` tokens. 

An architectural investigation revealed that Gemma 4 introduces significant departures from standard LLaMA-family models:
1. **Array Metadata**: `gemma4.feed_forward_length` is encoded as a 35-element `ARRAY[INT32]` rather than a scalar integer. The previous parser attempted to read a scalar integer, resulting in `intermediate_dim = 0` and unallocated FFN buffers.
2. **Tied LM Head**: Gemma 4 omits a distinct `output.weight` tensor, reusing `token_embd.weight` for the output projection. In the absence of `output.weight`, `w_cls` was unbound (null pointer).
3. **Specialized Hybrid Hyperparameters**: The architecture introduces sliding-window attention (SWA) key/value dimensions, shared KV layer reuse (20 layers), dual-stream per-layer embedding input dimensions (PLE dim 256), and logit softcapping (30.0).

Phase 1 provides the foundational model metadata extraction and weight binding needed to support Gemma 4 cleanly in pure Mojo without regressing existing models.

---

## 2. Implemented Changes

### A. Extended Model Configuration (`src/llamojo/types.mojo`)

The `ModelConfig` struct was enhanced to support Gemma 4 specific architectural parameters:

```mojo
@fieldwise_init
struct ModelConfig(ImplicitlyCopyable, Copyable, Movable):
    var arch: String             # "llama", "qwen2", "gemma2", "gemma4", etc.
    var name: String             # general.name
    var hidden_dim: Int          # {arch}.embedding_length
    var intermediate_dim: Int    # {arch}.feed_forward_length (max across layers)
    var n_heads: Int             # {arch}.attention.head_count
    var n_kv_heads: Int          # {arch}.attention.head_count_kv
    var head_dim: Int            # Global attention key length (e.g. 512)
    var n_layers: Int            # {arch}.block_count
    var vocab_size: Int          # 262,144 for Gemma 4
    ...
    # Gemma 4 specific architectural metadata
    var shared_kv_layers: Int    # Number of upper layers that reuse KV cache (20)
    var sliding_window: Int      # Sliding window attention size (512)
    var ple_dim: Int             # Per-layer input embedding dim (256)
    var logit_softcap: Float32   # Final logit softcapping factor (30.0)
    var head_dim_swa: Int        # SWA head dim (256)
    var rope_theta_swa: Float32  # SWA rope frequency base (10,000.0)
    var is_tied_embeddings: Bool # True when output.weight is tied to token_embd.weight
```

### B. Array Handling & Metadata Extraction (`src/llamojo/gguf.mojo`)

1. **Array-Aware `get_int` Reader**:
   Updated `GGUFContext.get_int` to handle both scalar integer types (`INT8`, `INT16`, `INT32`, `INT64`, `UINT*`) and `GGUFValueType.ARRAY` containing integers, extracting `arr[0]` as the default scalar representation.

2. **Gemma 4 Configuration Populator**:
   Updated `GGUFContext.build_model_config` to parse:
   - `gemma4.attention.shared_kv_layers`
   - `gemma4.attention.sliding_window`
   - `gemma4.embedding_length_per_layer_input`
   - `gemma4.final_logit_softcapping`
   - `gemma4.attention.key_length_swa`
   - `gemma4.rope.freq_base_swa`
   - Maximum FFN dimension across all elements in `gemma4.feed_forward_length` (`12,288`) to guarantee adequate buffer allocation during context initialization.
   - Tied embedding detection (`cfg.is_tied_embeddings = not self.has_tensor("output.weight")`).

### C. Tied LM-Head Weight Binding (`src/llamojo/llama.mojo`)

In `llama_model.__init__`, when `output.weight` is absent from the GGUF file:
```mojo
self.output_weight = gguf.get_tensor_ptr("output.weight")
self.output_weight_type = GGMLType.F32
if gguf.has_tensor("output.weight"):
    try:
        self.output_weight_type = gguf.get_tensor("output.weight").dtype
    except:
        pass
elif Int(self.output_weight) == 0:
    # Gemma 4 / tied embedding fallback
    self.output_weight = self.token_embd
    self.output_weight_type = self.token_embd_type
```

---

## 3. Verification & Validation

### A. Dedicated Gemma 4 Phase 1 Test

A standalone verification test was executed against `gemma-4-E2B-it.Q4_K_M.gguf`:

```text
Testing Phase 1 GGUF & ModelConfig parser on Gemma 4...
  Architecture        : gemma4
  Name                : Gemma 4 E2B It
  Layers (n_layers)   : 35
  Hidden dim          : 1536
  Intermediate dim    : 12288
  Heads (Q / KV)      : 8 / 1
  Head dim (Global)   : 512
  Head dim (SWA)      : 256
  Sliding window      : 512
  Shared KV layers    : 20
  PLE dim             : 256
  Logit softcap       : 30.0
  RoPE freq base SWA  : 10000.0
  Tied embeddings     : True
  Vocab size          : 262144
Phase 1 GGUF & ModelConfig verification passed! ✓

Testing full model load with tied lm-head...
  Model loaded: total params = 4647450147 total bytes = 3412060300
  Layers allocated: 35
  token_embd ptr != 0: True
  output_weight ptr != 0: True
  output_weight == token_embd (tied): True
Phase 1 Model Load & Tied LM Head verification passed! ✓
```

### B. Full Regression Suite

Executed `./run-tests.sh` to ensure backwards compatibility across all existing model architectures and test suites:
- **Phase 1**: GGUF & Type Tests, llama.h API & Worker Pool Tests — **Passed ✓**
- **Phase 2**: Fused Kernels, RoPE, Q8_0, Q4_K, MXFP4, Fused GEMV — **Passed ✓**
- **Phase 3**: Context & Decode Pipeline, PagedAttention, KV Cache — **Passed ✓**
- **Phases 4, 5 & 7**: Vocab, SPM Unigram Viterbi, Sampler, Model Quantization — **Passed ✓**
- **Phase 8**: LoRA Adapters & Online Merging — **Passed ✓**
- **Phase 9**: GBNF Grammar Engine — **Passed ✓**
- **Phase 10**: Sparse MoE (Mixture of Experts) — **Passed ✓**
- **Phase 11**: Micro-Architecture Optimizations (Cache-Tiled GEMV, VNNI, NUMA) — **Passed ✓**
- **C-ABI**: `test_c_abi.c` against `build/libllama.so` — **Passed ✓**

---

## 4. Current Status and Next Steps

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | **GGUF Metadata Parser & Model Config Upgrades** | **Complete ✓** |
| **Phase 2** | Math Kernels: GELU, Double RMSNorm & Softcapping | *Next* |
| **Phase 3** | Hybrid & Shared KV Cache Engine | Pending |
| **Phase 4** | Dual-Stream Per-Layer Input Embedding (PLE) | Pending |
| **Phase 5** | Dedicated Gemma 4 Forward Pass Graph (`gemma4.mojo`) | Pending |
| **Phase 6** | Integration, Verification & CLI Benchmarking | Pending |

