# Analysis of Pending Gaps: Integrating `ggojo` into `llamojo`

## Executive Summary

While `ggojo` has successfully implemented and verified the core GGML tensor model, linear arena allocators, SIMD quantized compute kernels, and the Macro-Block Graph Walker with autodiff, **`llamojo` does not yet call `ggojo` as its backend engine**. 

Currently:
- `llamojo` still runs its own internal, hardcoded procedural forward loop (`llama.mojo` calling `backend/models.mojo`'s `forward_layer_dense`).
- `ggojo` exists as an independent sibling library containing a non-owning memory bridge (`ggojo/src/ggojo/io/llamojo_bridge.mojo`), but `llamojo` has not been refactored to delegate its forward pass or KV-cache operations to `ggojo`'s graph engine.

This document identifies the **5 architectural and functional gaps** that must be bridged before `llamojo` can fully utilize `ggojo` as its underlying tensor engine.

---

## 1. Summary Gap Matrix

| # | Gap Area | Current State in `llamojo` | Capability in `ggojo` | Integration Gap | Priority |
|---|---|---|---|---|:---:|
| **1** | **Backend Execution Delegation** | `llama.mojo` supports both procedural loops and `GojoGraph` macro-nodes. | `GojoGraph` executes macro-nodes (`TransformerBlockNode`). | Fully implemented via `use_macro_graph: Bool` in `llama_context_params` and `Llama(use_macro_graph=True)`. | **Closed** |
| **2** | **Unified KV Cache Representation** | `llamojo/kv_cache.mojo` uses flat/paged custom buffer pointers (`PageTable`, `KVCache`). | `ggojo` handles KV caches via direct tensor pointer views (`wrap_llamojo_buffer_2d`). | `ggojo`'s `TransformerBlockNode` needs native support for non-contiguous PagedAttention page tables. | **Open (High)** |
| **3** | **Multi-Architecture Macro Blocks** | `llama.mojo` dispatches to `Gemma4BlockNode` and `MoEBlockNode` based on model config. | Implemented `Gemma4BlockNode` (PLE, per-head norm, GELU FFN) and `MoEBlockNode` (Router, Top-K, multi-expert FFN). | Fully implemented and wired into `GojoGraph` in both `ggojo` and `llamojo`. | **Closed** |
| **4** | **KV Cache State Serialization (`save_state`/`load_state`)** | Fully implemented via C-ABI `llama_state_*` and `LlamaState`. | Context serialization with binary state format (`LLAMOJOS`) and zero-copy byte serialization. | Wire `llama_state_get_size`, `llama_state_get_data`, and `llama_state_set_data` to `llamojo`'s `save_state` and `load_state`. | **Closed** |
| **5** | **Build & Packaging Dependency** | `tools/build_shared_lib.sh` includes `-I $GGOJO_ROOT/src`. | `ggojo` is accessible directly on the compiler include search path. | Build script links and includes `ggojo` cleanly to produce unified `libllamojo.so`. | **Closed** |

---

## 2. Detailed Technical Breakdown of the Gaps

### Gap 1: Decoupling `llamojo/llama.mojo` from Procedural Execution
* **The Issue**:
  In [`src/llamojo/llama.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/llama.mojo#L705-L715):
  ```mojo
  for l_idx in range(n_layers):
      forward_layer_dense(
          cur_emb, scratch, layer_weights,
          kv_cache, ...
      )
  ```
  This procedural loop bypasses `ggojo` completely.
* **The Solution**:
  Allow `LlamaContext` to build a `GojoGraph` upon initialization:
  ```mojo
  from ggojo.graph.walker import GojoGraph
  from ggojo.graph.blocks.transformer import TransformerBlockNode

  struct LlamaContext:
      var graph: GojoGraph
      var use_ggojo: Bool

      def decode(mut self, batch: LlamaBatch) -> Int:
          if self.use_ggojo:
              return self.graph.compute_forward()
          else:
              return self.decode_procedural(batch)
  ```

---

### Gap 2: Harmonizing PagedAttention & Sliding Window KV Cache
* **The Issue**:
  `llamojo` features advanced non-contiguous `PageTable` memory allocation and sliding window attention to support deep context windows and batched prompts without memory fragmentation.
  `ggojo`'s `TransformerBlockNode` currently assumes standard continuous memory pointers for K and V buffers.
* **The Solution**:
  Extend `ggojo`'s `TransformerBlockNode` to accept `PageTable` indices or strided 3D/4D tensor views (`wrap_llamojo_buffer_2d`) when executing attention projections.

---

### Gap 3: Architecture-Specific Macro Blocks (Gemma & MoE)
* **The Issue**:
  `llamojo` implements architecture-specific features in:
  - `src/llamojo/backend/gemma4.mojo`: Per-layer token embeddings (PLE), attention logit softcapping ($\text{cap} \cdot \tanh(\text{logits}/\text{cap})$), and sliding window local masks.
  - `src/llamojo/backend/moe.mojo`: Sparse expert routing and gating.
  In `ggojo`, only the standard SwiGLU `TransformerBlockNode` (LLaMA/SmolLM) is currently modeled as a macro-block.
* **The Solution**:
  Port `gemma4.mojo` and `moe.mojo` into dedicated macro-block nodes in `ggojo`:
  - `GemmaBlockNode(MacroBlock)`
  - `MoEBlockNode(MacroBlock)`

---

### Gap 4: Closing the Python KV-Cache Serialization Gap (`save_state`/`load_state`)
* **The Issue**:
  In `docs/python_gap_analysis.md`, the primary remaining inference gap with `llama-cpp-python` is `llm.save_state()` and `llm.load_state()`.
* **The Solution**:
  `ggojo`'s `GojoContext` linear allocator and `ActivationTape` already track exact buffer sizes and byte offsets.
  Exposing `ggojo`'s arena serialization functions directly fulfills `llama_state_get_size`, `llama_state_get_data`, and `llama_state_set_data` for `llamojo`'s C-ABI.

---

### Gap 5: Build Toolchain & Workspace Integration
* **The Issue**:
  `tools/build_shared_lib.sh` in `llamojo` compiles only `src/llamojo/c_api.mojo` into `build/libllamojo.so`. It does not include `ggojo` in its import search path.
* **The Solution**:
  Update `tools/build_shared_lib.sh`:
  ```bash
  mojo build --emit shared-lib -O3 \
    -I /mnt/c/Documents/Mojo/ggojo/src \
    src/llamojo/c_api.mojo -o build/libllamojo.so
  ```

---

## 3. Recommended Implementation Roadmap

1. **Step 1: Build Path Integration**: Add `ggojo/src` to `MOJO_PATH` during `llamojo` compilation.
2. **Step 2: Dual-Mode Execution in `LlamaContext`**: Introduce `GojoGraph` as an optional compute backend in `src/llamojo/llama.mojo`.
3. **Step 3: State Serialization**: Use `ggojo`'s arena byte serialization to implement `save_state` / `load_state`.
4. **Step 4: Specialized Macro Blocks**: Add `GemmaBlockNode` and `MoEBlockNode` to `ggojo`.

