---
type: howto
title: "Implementation Plan: ModernBERT & Laya Inference Engine in llamojo"
description: "Engineering plan for bidirectional encoder inference, alternating sliding window attention, and option decision scoring."
status: completed
tags: [modernbert, laya, inference-engine, encoder, decision-head]
timestamp: 2026-11-05T00:00:00Z
---

# Implementation Plan: ModernBERT & Laya Inference Engine in llamojo

## 1. Executive Summary

This document outlines the detailed architectural design and implementation subphases for adding **ModernBERT bidirectional encoder inference** and the **Laya System 1 decision engine** to the **`llamojo`** project (`/mnt/c/Documents/Mojo/llamojo`).

### Dual-Library Division of Responsibilities
* **`ggojo`** (`libgojo.so`): The **low-level tensor, kernel, and autodiff library** (the GGML equivalent in pure Mojo). It houses the mathematical operators, SIMD GEMM, backward backpropagation, activation checkpointing (`ActivationTape`), and gradient optimizers (AdamW).
* **`llamojo`** (`libllamojo.so`): The **high-level inference and serving runtime** (the `llama.cpp` equivalent in pure Mojo). It handles GGUF model lifecycle management, tokenization, sequence batching, C-ABI export, Python bindings, and end-user application execution.

---

## 2. Architectural Analysis: Code Distribution & Phase 8A Placement

### Question 1: Would an implementation in `llamojo` help to have less code in `ggojo`?
**No, moving compute logic to `llamojo` would break the dual-library architecture.**

`ggojo` is designed as the self-contained tensor and compute engine. If mathematical operators or backward passes were placed inside `llamojo`:
1. `ggojo` could no longer be used as an independent training library or GGML drop-in replacement.
2. `llamojo` would become burdened with training-specific data structures (gradient accumulators, loss derivatives, optimizer moments).
3. The clean dependency DAG (`llamojo` depends on `ggojo`, never the reverse) would be violated.

**The optimal design**:
- `ggojo` provides the **`ModernBERTEncoderNode`** (both forward evaluation and analytic backward propagation).
- `llamojo` provides the **high-level orchestration, GGUF parameter loading, tokenization, decision extraction, and C-API bindings**.
- `llamojo` consumes `ggojo`'s forward pass through zero-copy buffer views (`wrap_llamojo_buffer_*`), adding minimal new code to `llamojo` while keeping `ggojo` fully autonomous for training.

### Question 2: Does already implemented code from Phase 8A need to be moved?
**No.** 
The code in `src/ggojo/graph/blocks/modernbert.mojo` belongs strictly in `ggojo` because it contains:
1. The mathematical definition of bidirectional multi-head attention.
2. The alternating sliding window attention kernel.
3. The analytic backward pass and parameter gradient accumulation ($dW_q, dW_k, dW_v, dW_o, d\gamma_{\text{attn}}, d\gamma_{\text{ffn}}$).

Moving this into `llamojo` would break `ggojo`'s training capability. Instead, `llamojo` will import and invoke `ModernBERTEncoderNode` (or wrap its forward pass) for high-speed inference.

---

## 3. Detailed Subphases for `llamojo`

```
┌────────────────────────────────────────────────────────────────────────┐
│                   LLAMOJO LAYA INFERENCE SUBPHASES                     │
├────────────────────────────────┬───────────────────────────────────────┤
│ Subphase 1: GGUF Model Parser  │ Support 'modernbert' Architecture     │
│ Subphase 2: Encoder Backend    │ llamojo_encode() with Zero-Copy ggojo │
│ Subphase 3: Decision Head Run  │ Option-Marker Scorer & Act/Escalate   │
│ Subphase 4: High-Level C-API   │ C-ABI Export for Python & External Apps│
│ Subphase 5: E2E Verification   │ Unit Tests & Sub-35ms Latency Bench   │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

### Subphase 1: GGUF Model Ingestion & Hyperparameters
* **Target File**: `src/llamojo/gguf.mojo`, `src/llamojo/types.mojo`
* **Objective**: Enable `llamojo` to parse ModernBERT and Laya model files serialized in GGUF format.
* **Deliverables**:
  1. Add `LLAMA_ARCH_MODERNBERT` and `LLAMA_ARCH_BERT` to architecture enum.
  2. Parse ModernBERT metadata:
     - `general.architecture = "modernbert"`
     - `modernbert.attention.sliding_window` (e.g. 128)
     - `modernbert.attention.layer_types` (alternating global vs local schedule)
     - `modernbert.classifier_pooling` (first token `[CLS]` vs mean pooling vs option markers)
  3. Map GGUF tensor names into `llama_model`:
     - Embeddings: `token_embd.weight`
     - Attention: `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`, `blk.{i}.attn_output.weight`
     - Norms: `blk.{i}.attn_norm.weight`, `blk.{i}.ffn_norm.weight`
     - GeGLU FFN: `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`
     - Laya Decision Head: `laya.head.option_scorer.weight`, `laya.head.act_escalate.weight`

---

### Subphase 2: Bidirectional Encoder Forward Backend
* **Target File**: `src/llamojo/backend/encoder.mojo` (or `src/llamojo/llama.mojo`)
* **Objective**: Implement bidirectional non-causal sequence encoding without autoregressive KV-cache allocation.
* **Deliverables**:
  1. **Non-Causal Forward Engine**:
     - Unlike causal models that loop token-by-token, ModernBERT processes the entire sequence $[T, D]$ in parallel.
     - Leverage `ggojo.graph.blocks.modernbert.ModernBERTEncoderNode` directly via `ggojo` zero-copy wrappers:
       ```mojo
       # Wrap llamojo context buffers into ggojo tensors without copying
       var in_tensor = wrap_llamojo_buffer_1d(ctx[].embd, T * dim)
       var out_tensor = layer.forward(in_tensor, ggojo_ctx, dummy_tape)
       ```
  2. **Threadpool Partitioning**:
     - Wire `parallelize` across attention heads and sequence chunks for linear multi-core CPU scaling.

---

### Subphase 3: Laya System 1 Decision Head Runtime
* **Target File**: `src/llamojo/laya.mojo`
* **Objective**: Evaluate structured questions, extract option tokens, and output mathematically calibrated probabilities.
* **Deliverables**:
  1. **Option Token Identifier**:
     - Identifies positions of option markers `[OPT_0]`, `[OPT_1]`, `[OPT_2]`, etc. in the prompt sequence.
  2. **Option-Marker Scorer**:
     - Gathers pooled vectors: $h_k = X[\text{pos}(\text{opt}_k)]$.
     - Evaluates 2-layer MLP projection: $s_k = W_2 \cdot \text{GELU}(W_1 h_k)$.
     - Normalizes into categorical probability distribution: $p = \text{Softmax}(s)$.
  3. **Act / Escalate Calibrated Binary Head**:
     - Computes autonomous action probability: $p_{\text{act}} = \sigma(w_{\text{act}}^T h_{\text{pool}} + b)$.
     - Returns decision flags: `should_act: Bool`, `should_escalate: Bool` based on calibration threshold $\tau$.

---

### Subphase 4: C-API Export & Python Bindings
* **Target File**: `src/llamojo/c_api.mojo`, `python/llamojo/laya.py`
* **Objective**: Allow Python, Go, C++, or Rust applications to call the high-speed Laya engine.
* **Deliverables**:
  1. Export C-ABI functions:
     ```c
     // Load a trained Laya model (GGUF)
     llama_model_p llamojo_laya_load(const char* path, llama_model_params params);

     // Single-pass decision inference
     int32_t llamojo_laya_evaluate(
         llama_context_p ctx,
         const int32_t* tokens,
         int32_t n_tokens,
         const int32_t* opt_indices,
         int32_t n_options,
         float* out_option_probs,
         float* out_act_prob
     );
     ```
  2. Python `ctypes` wrapper class `LayaDecisionEngine` in `python/llamojo/laya.py`.

---

### Subphase 5: End-to-End Verification & Benchmarking
* **Target File**: `tests/test_laya_inference.mojo`, `benchmarks/bench_laya.mojo`
* **Deliverables**:
  1. Unit test verifying exact option score probability recovery on synthetic test tensors.
  2. Benchmark measuring single forward pass latency on CPU (verifying the <35ms target on 8-core CPUs for 512 tokens).

---

## 4. Summary: Project Code Allocation

| Component | Repository | Role | Status |
|---|---|---|---|
| **SIMD Kernels (GEMM, RoPE, RMSNorm)** | `ggojo` | Vectorized math primitives | Completed |
| **`ModernBERTEncoderNode` (Forward & Backward)** | `ggojo` | Encoder layer with backward autodiff | Completed (Phase 8A) |
| **AdamW & Checkpoint Autodiff** | `ggojo` | Full & LoRA training engine | Completed (Phase 7A-7C) |
| **ModernBERT GGUF Parser** | `llamojo` | Ingestion of `.gguf` weights | Subphase 1 (Planned) |
| **Bidirectional Inference Backend** | `llamojo` | Sequence forward evaluation | Subphase 2 (Planned) |
| **Laya Option & Act/Escalate Head** | `llamojo` | Decision scoring & calibration | Subphase 3 (Planned) |
| **C-ABI & Python Engine** | `llamojo` | Application serving API | Subphase 4 (Planned) |

