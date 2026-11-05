---
type: reference
title: "Whisper ASR Support: Phase 1 Implementation Documentation"
description: "Implementation and verification of Phase 1 core kernels for Whisper ASR in llamojo: LayerNorm with bias, biased quantized matmuls, 1D Convolutional Stem (F32/F16, stride 1/2), and positional embedding transposition."
status: completed
tags: [whisper, phase1, layer-norm, conv1d, biased-matmul, simd-kernels]
timestamp: 2026-11-05T00:00:00Z
---

# Whisper ASR Support: Phase 1 Implementation Documentation

## 1. Overview

Phase 1 of the Whisper ASR integration into `llamojo` establishes all foundational compute primitives required by the Whisper Encoder-Decoder architecture that differ from standard LLaMA/Gemma decoder-only transformers:

1. **Standard LayerNorm with Weight & Bias (`layer_norm` & `fused_layer_norm_residual`)**:
   - Unlike LLaMA/Gemma (which use zero-mean-assumed `RMSNorm`), Whisper's encoder and decoder blocks (`attn_ln`, `cross_attn_ln`, `mlp_ln`, `ln_post`, `decoder.ln`) require full mean and variance normalization with both scale ($\gamma$) and bias ($\beta$).
2. **Biased Quantized Linear Projections (`matmul_quant_fused_biased`, `gemv_quant_biased`, `gemv_gelu_biased`)**:
   - In Whisper, `Q`, `V`, `out`, `mlp.0`, and `mlp.2` projections include 1D bias vectors, whereas `K` (`attn.key.weight` and `cross_attn.key.weight`) has **no bias**.
3. **1D Convolutional Audio Stem (`conv1d_f32`, `conv1d_k3_gelu`)**:
   - Implements the two-layer convolutional stem (`encoder.conv1` with $K=3, S=1, P=1$ and `encoder.conv2` with $K=3, S=2, P=1$) supporting both `GGMLType.F16` and `GGMLType.F32` weights with fused `GELU` activation.
4. **Channel-to-Sequence Transposition + Positional Embedding (`transpose_add_pos_embd`)**:
   - Transposes the channel-first convolutional output $[n_{\text{audio\_state}}, n_{\text{audio\_ctx}}]$ into sequence-first $[n_{\text{audio\_ctx}}, n_{\text{audio\_state}}]$ while fusing the addition of `encoder.positional_embedding` in a single pass.

---

## 2. Implemented Kernels & File Mapping

### 2.1 [`src/llamojo/backend/kernels_norm_rope.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_norm_rope.mojo)
- **`layer_norm(out_ptr, x, weight, bias, n, eps=1e-5)`**:
  - Computes $\mu = \frac{1}{n}\sum x_i$ and $\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2$ using `SIMD_WIDTH` horizontal reductions.
  - Applies $(x_i - \mu) \cdot (\sigma^2 + \epsilon)^{-1/2} \cdot \gamma_i + \beta_i$, gracefully handling nullable `weight` or `bias` pointers.
- **`fused_layer_norm_residual(x, residual, weight, bias, out_normed, out_residual, n, eps=1e-5)`**:
  - Fuses residual addition `out_residual = x + residual` with mean computation in the first SIMD pass, eliminating redundant memory round-trips.

### 2.2 [`src/llamojo/backend/kernels_quant.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_quant.mojo)
- **`matmul_quant_fused_biased[n](dtype, C, x, W, B, row_start, row_end, nc)`**:
  - Extends `matmul_quant_fused` with a `StaticTuple[FloatPtr, n]` of bias pointers `B`.
  - When `Int(B[k]) != 0`, adds `B[k][r]` in-register before writing to `C[k][r]`. Directly models Whisper QKV where `B = (bq, null_ptr, bv)`.
- **`gemv_quant_biased(dst, w, bias, dtype, x, rows, cols, workers)`**:
  - Multi-threaded quantized GEMV (leveraging fast `Q4_K`/`Q6_K` 2-row register-tiled Q8 paths) followed by vectorized bias addition.
- **`gemv_gelu_biased(dst, w, bias, dtype, x, rows, cols, workers)`**:
  - Fuses `dot_quant_row` + bias addition + `gelu()` in a single register pass for Whisper's `mlp.0` expansion layer ($d \to 4d$).

### 2.3 [`src/llamojo/backend/kernels_conv.mojo`](file:///mnt/c/Documents/Mojo/llamojo/src/llamojo/backend/kernels_conv.mojo)
- **`conv1d_f32(...)`**:
  - Reference general 1D convolution supporting arbitrary `kernel_size`, `stride`, `padding`, `F32`/`F16` weights, bias, and optional `GELU`.
- **`conv1d_k3_gelu(...)`**:
  - High-throughput SIMD 3-tap convolution specialized for Whisper's `conv1` ($S=1$) and `conv2` ($S=2$), vectorizing across time frames with `SIMD_WIDTH` FMA operations.
- **`transpose_add_pos_embd(out_seq_major, conv_chan_major, pos_embd, n_state, n_ctx)`**:
  - Converts $[C, T] \to [T, C]$ and adds `encoder.positional_embedding` ($[T, C]$) so each audio frame is contiguous for subsequent Transformer Encoder self-attention blocks.

---

## 3. Verification

Unit tests in [`tests/test_whisper_phase1.mojo`](file:///mnt/c/Documents/Mojo/llamojo/tests/test_whisper_phase1.mojo) verify:
- Numerical accuracy of `layer_norm` and `fused_layer_norm_residual` against analytical ground truth.
- `matmul_quant_fused_biased` with mixed biased (`Q`, `V`) and unbiased (`K`) projections.
- `gemv_quant_biased` and `gemv_gelu_biased` accuracy.
- Exact agreement (`atol=1e-5`) between general `conv1d_f32` and SIMD `conv1d_k3_gelu` across both `stride=1` and `stride=2` for both `GGMLType.F32` and `GGMLType.F16` weights, plus `transpose_add_pos_embd`.

