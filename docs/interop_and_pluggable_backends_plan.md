# Pluggable Compute & Library Interoperability Architecture: `llamojo` & `ggojo`

## 1. Overview & Strategic Goals

This document specifies the architectural roadmap for enabling **swappable compute backends and foreign library interoperability** within **`llamojo`** (`/mnt/c/Documents/Mojo/llamojo`) and **`ggojo`** (`/mnt/c/Documents/Mojo/ggojo`).

The goal is to allow developers to swap or augment specific subsystems—such as **GEMM / GEMV engines**, **matrix decompositions (SVD / QR)**, **random augmentations**, and **graph execution planners**—using existing Mojo libraries (**`NuMojo`**, **`tensor_graph` / `TenMo`**, and **`MAX Graph`**) with minimal friction and **zero memory duplication**.

---

## 2. Shared Principles & Boundaries

1. **Zero-Copy First**:
   - `llamojo` and `ggojo` memory representations rely on contiguous pointers (`Pointer[Float32, MutUntrackedOrigin]`, `Pointer[UInt8, MutUntrackedOrigin]`).
   - Any external integration must wrap existing allocations as non-owning views rather than allocating duplicate buffers.
2. **Zero Mandatory External Dependencies**:
   - Both libraries must compile and run as pure, self-contained Mojo applications by default.
   - Foreign library bridges live in `interop/` submodules and are activated conditionally.
3. **Strict Library Separation**:
   - `ggojo` remains the mathematical operator, tensor, and autodiff library (`ggml` equivalent).
   - `llamojo` remains the inference runtime, GGUF model loader, and tokenizer manager (`llama.cpp` equivalent).

---

## 3. Four-Phase Modular Implementation

```
Phase 1: Memory Abstraction & Zero-Copy Adapters [COMPLETED]
 ├── Subphase 1.1: Tensor Memory View Protocol (as_view, raw_ptr, is_contiguous) [DONE]
 └── Subphase 1.2: NuMojo NDArray Zero-Copy Bridge (GojoTensor / Matrix <-> NuMojoNDArrayView) [DONE]

Phase 2: Swappable Kernel Dispatch Interface [IMPLEMENTED]
 ├── Subphase 2.1: ComputeBackend Trait & Global Configuration
 ├── Subphase 2.2: GEMM Router Implementation (Native SIMD vs NuMojo)
 └── Subphase 2.3: llamojo Context / Batch Buffer View Exporters

Phase 3: Extended Mathematical Capabilities (NuMojo Integration) [IMPLEMENTED in ggojo, pure Mojo]
 ├── Subphase 3.1: Linear Algebra Decompositions (SVD / QR for LoRA & Calibration)
 ├── Subphase 3.2: PRNG Sampling & Dynamic Masked Augmentation
 └── Subphase 3.3: Statistical Calibration & Metrics (ECE / Brier Scores)

Phase 4: Advanced Graph & Compiler Backends [PARTIAL: planner done, MAX stub only; see ggojo plan]
 ├── Subphase 4.1: tensor_graph Activation Buffer Recycling (Memory Planner)
 └── Subphase 4.2: MAX Graph Engine Lowering for Dense Forward Passes
```

---

## 4. Phase 1 Implementation Status & Verification

Phase 1 has been fully implemented and verified across both `ggojo` and `llamojo`:

### 1. Unified Tensor Memory View Protocol (`TensorMemoryView`)
- Implemented in `ggojo/src/ggojo/interop/view_protocol.mojo` and `llamojo/src/llamojo/interop/view_protocol.mojo`.
- Exposes non-owning, zero-overhead memory descriptors containing raw untracked byte pointers, GGML dtype enum, rank/ndim, dimension shape static tuples, byte strides, and contiguity flags.
- Conversion functions:
  - `tensor_to_memory_view(tensor: GojoTensor) -> TensorMemoryView`
  - `memory_view_to_tensor(view: TensorMemoryView) -> GojoTensor`
  - `matrix_to_memory_view(m: Matrix, dtype: UInt32 = GGMLType.F32) -> TensorMemoryView`
  - `memory_view_to_matrix(view: TensorMemoryView) -> Matrix`

### 2. NuMojo Zero-Copy View Bridge (`NuMojoNDArrayView`)
- Implemented in `ggojo/src/ggojo/interop/numojo_bridge.mojo` and `llamojo/src/llamojo/interop/numojo_bridge.mojo`.
- Standardized layout adapter matching NuMojo's 2D `NDArray` and `DataContainer` stride conventions.
- Supports instant zero-copy wrapping and mutation reflecting directly in underlying buffers without allocating heap buffers.
- Re-exported at top-level `llamojo.interop` and `ggojo.interop`.

### 3. Verification & Test Coverage
- `ggojo/tests/test_interop_views.mojo`: Verifies `GojoTensor` round-trip, `llama2.Matrix` view reconstruction, and `NuMojoNDArrayView` mutation. All tests pass on Mojo 1.1.0.
- `llamojo/tests/test_interop_views.mojo`: Verifies `llama2.Matrix` <-> `TensorMemoryView` and `NuMojoNDArrayView` zero-copy bidirectional mapping. All tests pass on Mojo 1.1.0.
- Integrated into `ggojo/scripts/run_tests.sh`.

---

## 5. Key Target Files by Project

### In `ggojo` (`/mnt/c/Documents/Mojo/ggojo`):
* `src/ggojo/core/tensor.mojo`: Zero-copy view accessors (`as_view`, pointer exports).
* `src/ggojo/interop/numojo_bridge.mojo`: Zero-copy bi-directional conversion with NuMojo `NDArray`.
* `src/ggojo/kernels/dispatch.mojo`: Backend router for GEMM, norm, and activation functions.
* `src/ggojo/train/lora_init.mojo`: SVD/PiSSA adaptor initialization via NuMojo linalg.
* `src/ggojo/graph/memory_planner.mojo`: Activation buffer coloring inspired by `tensor_graph`.

### In `llamojo` (`/mnt/c/Documents/Mojo/llamojo`):
* `src/llamojo/types.mojo`: Matrix view export utilities.
* `src/llamojo/llama.mojo`: Non-owning view exposure for inference context activations (`ctx[].embd`).
* `src/llamojo/laya.mojo`: Integration of calibrated decision statistics (ECE, quantile thresholds).

---

## 5. Master Roadmap Reference

For the comprehensive specification, code templates, and testing matrix, refer to the master plan document in:
- [`/mnt/c/Documents/Mojo/ggojo/docs/interop_and_pluggable_backends_plan.md`](../../ggojo/docs/interop_and_pluggable_backends_plan.md)


## Phase 2 Implementation Status [IMPLEMENTED]
- `src/llamojo/interop/context_views.mojo`: zero-copy `TensorMemoryView` exporters: `f32_buffer_view`, `context_hidden_view`, `context_logits_view`, `context_embeddings_view`, `context_kv_k_view`, `context_kv_v_view`, `batch_embd_view`, `batch_tokens_view` (re-exported from `llamojo.interop`).
- `tests/test_context_views.mojo` (added to `run-tests.sh`): runs a real synthetic-model decode and checks pointer equality, shape and contiguity. Passes on Mojo 1.1.0.

## Phase 3 Implementation Status
Implemented in ggojo (`train/lora_init.mojo`, `data/sampler.mojo`, `train/calibration_metrics.mojo`; see the ggojo plan). llamojo does not duplicate them. `laya.mojo` can import the metrics when needed.

## Phase 4 Implementation Status
Implemented in ggojo (`graph/memory_planner.mojo`, `interop/max_graph.mojo`; see the ggojo plan). llamojo is unchanged.

## Remaining Work
Adapters for external backends, graph-planner integration and real MAX lowering are tracked in the ggojo plan. In llamojo, the exporters in `context_views.mojo` are the only integration point needed.
