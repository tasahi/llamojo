---
type: reference
title: "Gemma 4 Architecture Support: Phase 3 Implementation Documentation"
description: "Hybrid sliding window attention (SWA) and global attention alternating schedule in pure Mojo."
status: completed
tags: [gemma4, phase3, swa, sliding-window, attention-schedule]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Architecture Support: Phase 3 Implementation Documentation

This document records the design, implementation, and verification of **Phase 3: Hybrid & Shared KV Cache Engine** for the Gemma 4 architecture (`gemma4`) in **llamojo**.

---

## 1. Context & Motivation

Standard transformer models allocate dedicated Key ($K$) and Value ($V$) cache buffers for every layer ($N_{\text{layers}}$) and compute attention over the entire sequence history ($0 \dots \text{pos}$).

For **Gemma 4** (`gemma-4-E2B-it.Q4_K_M.gguf`), two architectural optimizations reduce KV memory bandwidth and storage:

1. **Shared KV Layers (Layer Reuse)**:
   - Out of 35 layers, 20 layers share KV cache projections (`gemma4.attention.shared_kv_layers: 20`).
   - Only the first 15 layers ($0 \dots 14$) physically project and write Key/Value states into memory ($N_{\text{layer\_kv}} = 35 - 20 = 15$).
   - The remaining 20 layers ($15 \dots 34$) perform **no** $K$ or $V$ linear projections. Instead, they read keys and values directly from the KV slots populated by earlier layers.
2. **Hybrid Attention: Sliding Window Attention (SWA) vs. Global Attention**:
   - Layers follow a repeating 5-layer pattern: `[SWA, SWA, SWA, SWA, Global]` (i.e. every 5th layer is full global attention, and the other 4 are sliding window).
   - In SWA layers, attention is restricted to a local window of 512 tokens ($\max(0, \text{pos} - 512 + 1) \dots \text{pos}$).
   - Layer reuse mappings differ between SWA and Global layers:
     - SWA layers $\ge 15$ reuse layer $15 - 2 = 13$.
     - Global attention layers $\ge 15$ (layers 19, 24, 29, 34) reuse layer $15 - 1 = 14$.

---

## 2. Implemented Changes

### A. Shared KV Storage & Layer Mapping (`src/llamojo/kv_cache.mojo`)

1. **Physical Storage Allocation**:
   Updated `LlamaKVCache.__init__` to accept `shared_kv_layers: Int = 0`.
   - Computes $N_{\text{layer\_kv}} = N_{\text{layers}} - \text{shared\_kv\_layers}$.
   - Allocates memory for only $N_{\text{layer\_kv}}$ layers instead of $N_{\text{layers}}$, cutting KV cache memory consumption by over **57%** for Gemma 4.

2. **Reuse Resolution Methods**:
   ```mojo
   @always_inline
   def get_reused_layer(self, layer: Int, is_swa: Bool) -> Int:
       """ Map a layer index to its backing physical KV storage layer.
       For Gemma 4: layers 0..14 store their own KV.
       Layers >= 15 reuse layer 13 (if SWA) or 14 (if global attention).
       """
       if self.shared_kv_layers <= 0 or layer < self.n_layer_kv:
           return layer
       return self.n_layer_kv - (2 if is_swa else 1)

   @always_inline
   def has_own_kv(self, layer: Int) -> Bool:
       """ Returns true if this layer computes and stores its own KV projection."""
       return self.shared_kv_layers <= 0 or layer < self.n_layer_kv
   ```

3. **Layer-Reused Pointer Retrieval**:
   Updated `get_k_ptr(self, layer: Int, slot: Int, is_swa: Bool = False) -> FloatPtr` and `get_v_ptr` to resolve physical layer offsets via `get_reused_layer(layer, is_swa)`.

### B. Sliding Window Attention in GQA (`src/llamojo/backend/kernels_attn.mojo`)

1. **Sliding Window Scanning**:
   Updated `gqa_decode_head` and `gqa_decode` with parameters:
   - `sliding_window: Int = 0`: When positive, sets the loop starting index to `max(0, pos - sliding_window + 1)`.
   - `attn_scale: Float32 = 0.0`: Allows passing custom pre-attention scales (e.g. 1.0 for Gemma 4) or defaults to $1.0 / \sqrt{d_{\text{head}}}$.

```mojo
var start_t = 0
if sliding_window > 0 and pos >= sliding_window:
    start_t = pos - sliding_window + 1

# Compute dot products Q @ K.T restricted to the active window
for t in range(start_t, pos + 1):
    var k_slot_ptr = k_cache.unsafe_offset(t * kv_dim + kv_head_offset)
    var score = dot_vectors(q_head_ptr, k_slot_ptr, head_dim) * inv_sqrt_d
    ...
```

### C. Context Initialization Wiring (`src/llamojo/llama.mojo`)

In `llama_context.__init__`, forwarded `cfg.shared_kv_layers` from `ModelConfig` into `LlamaKVCache`.

---

## 3. Verification & Validation

### A. Dedicated Unit Tests (`tests/test_context_decode.mojo`)

Added test suite covering:
- **Active Physical Layers**: Verified $N_{\text{layer\_kv}} = 15$ for Gemma 4 (35 layers, 20 shared).
- **KV Ownership**:
  - `has_own_kv(0) == True`
  - `has_own_kv(14) == True`
  - `has_own_kv(15) == False`
  - `has_own_kv(34) == False`
- **Reuse Routing**:
  - SWA Layer $15 \to$ Layer 13.
  - SWA Layer $20 \to$ Layer 13.
  - Global Layer $19 \to$ Layer 14.
  - Global Layer $34 \to$ Layer 14.

### B. Regression Test Suite

Ran `./run-tests.sh` to ensure no impact on existing models (LLaMA-2, LLaMA-3, TinyLlama, SmolLM2):
- All 11 test phases passed with 0 regressions.
- C-ABI integration test (`tests/test_c_abi.c`) passed.

---

## 4. Current Status and Next Steps

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | GGUF Metadata Parser & Model Config Upgrades | **Complete ✓** |
| **Phase 2** | Math Kernels: GELU, Double RMSNorm & Softcapping | **Complete ✓** |
| **Phase 3** | **Hybrid & Shared KV Cache Engine** | **Complete ✓** |
| **Phase 4** | Dual-Stream Per-Layer Input Embedding (PLE) | *Next* |
| **Phase 5** | Dedicated Gemma 4 Forward Pass Graph (`gemma4.mojo`) | Pending |
| **Phase 6** | Integration, Verification & CLI Benchmarking | Pending |

