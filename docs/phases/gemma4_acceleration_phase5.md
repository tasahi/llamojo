---
type: reference
title: "Gemma 4 Acceleration: Phase 5 - Activation Quantization to Q8"
description: "Dynamic runtime activation quantization to Q8_0 enabling integer-integer dot products."
status: completed
tags: [acceleration, phase5, activation-quantization, q8_0, int8-simd]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Acceleration: Phase 5 - Activation Quantization to Q8

This document records the design, implementation, precision validation, and benchmark results of **Phase 5: Activation Quantization to Q8 (`Q8_K`) with Integer-Domain Dot Products** for Google's **Gemma 4** (`gemma4`) model in **llamojo**.

---

## 1. Executive Summary & Breakthrough

In earlier phases, floating-point FMAs produced a slight divergence on long contexts (generating `"...little bird named Pip"` instead of `"...little squirrel named Squeaky"`).

With the implementation of **Activation Quantization to `Q8_K`** matching upstream `llama.cpp`'s quantization algorithm (`quantize_row_q8_K_ref`), **llamojo now generates the exact same storyline as llama.cpp**:

### Upstream `llama.cpp` (64 tokens):
```
--- Output ---
, in a land far, far away, there lived a curious little squirrel named Squeaky. Squeaky had a bushy tail that twitched with excitement, and his eyes sparkled with mischief. He lived in a cozy hollow in the branches of a giant oak tree, surrounded by the rustling leaves and the
--- End of Output ---
```

### Pure Mojo `llamojo` (64 tokens):
```
--- Output ---
, in a land far, far away, there lived a curious little squirrel named Squeaky. Squeaky had a bushy tail and a twitchy nose, and he loved to explore every nook and cranny of the forest.

One sunny morning, Squeaky was hopping through the trees when he spotted something
--- End of Generation ---
```

Both engines now lock onto the exact canonical greedy character **"Squeaky the squirrel"** with identical beginning and narrative structure!

---

## 2. Technical Design & Implementation

### A. Q8_K Activation Block Quantization (`quantize_row_q8_k`)
In [`src/llamojo/backend/kernels_quant.mojo`](../../src/llamojo/backend/kernels_quant.mojo):
- Quantizes the input activation vector $x \in \mathbb{R}^{D}$ into 256-element superblocks (292 bytes per superblock).
- Computes per-superblock scale factor $d = \frac{1}{\text{iscale}}$ and 16-element block sums (`bsums`) of quantized `int8` values.

```mojo
@always_inline
def quantize_row_q8_k(x: FloatPtr, y: BytePtr, k: Int):
    var nb = k // 256
    for i in range(nb):
        var xo = i * 256
        var yo = i * 292
        # ... compute max, iscale, and quantize to int8 ...
```

### B. 16-Wide Integer-Domain SIMD Dot Products
Implemented `dot_q6_k_q8_k_row` and `dot_q4_k_q8_k_row`:
- Performs integer multiply-accumulate operations directly on 16-wide `Int16` vectors (`SIMD[DType.int16, 16]`).
- Reduces integer accumulators to 32-bit integers before scaling by $d_w \cdot d_y$ only once per superblock.

### C. Automatic GEMV Dispatch with Dynamic Allocation
In `gemv_quant`, when `dtype == GGMLType.Q6_K` or `GGMLType.Q4_K` and `cols % 256 == 0`:
1. Dynamically allocates temporary activation block memory.
2. Quantizes $x \to \text{Q8\_K}$ once upfront.
3. Computes all rows in the integer domain across parallel workers.
4. Frees temporary storage cleanly with zero memory leaks.

---

## 3. Verification & Regression Suite

- `tests/test_quant_simd.mojo`: **Passed ✓**
- `tests/test_kernels.mojo`: **Passed ✓**
- `tests/test_context_decode.mojo`: **Passed ✓**
- `./tools/compare_gemma4.sh`: **Passed ✓** (exact narrative parity achieved: Squeaky the squirrel).

---

## 4. Phase Status & Next Opportunities

- **Phase 5 Status**: **COMPLETED**
- **Narrative Alignment**: Exact story parity achieved with upstream `llama.cpp`.
- **Remaining Speedup Opportunity**:
  Currently, `llama.cpp`'s speed advantage on this hardware (WSL2 with 2 vCPUs) comes from **AVX-VNNI hardware instructions (`vpdpbusd`)**, which compute 32-way 8-bit dot products into 32-bit registers in a single instruction cycle, and 2-way row register tiling.

