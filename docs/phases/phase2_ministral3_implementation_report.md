---
type: implementation-report
title: "Implementation Report: Phase 2 - YaRN RoPE Implementation & Execution Dispatch"
description: "Detailed documentation of Phase 2 implementation for YaRN RoPE (Yet another RoPE extensioN) in llamojo, including frequency partitioning, correction dimension calculations, smooth ramp mixing, magnitude scaling (mscale), and dispatch in dense transformer layers."
status: completed
tags: [ministral-3b, phase2, yarn-rope, kernels, attention, llamojo]
timestamp: 2026-11-10T20:24:00Z
---

# Implementation Report: Phase 2 — YaRN RoPE Implementation & Execution Dispatch

## 1. Overview & Objectives

In **Phase 2**, we implemented the mathematical and execution kernels for **YaRN (Yet another RoPE extensioN)** within `llamojo`'s normalization and rotary embedding engine (`src/llamojo/backend/kernels_norm_rope.mojo`) and wired them into the dense layer execution pipeline (`src/llamojo/backend/models.mojo:forward_layer_dense`).

Ministral 3 3B scales its context window up to **262,144 tokens** ($16\times$ extension over its native $16,384$ training limit) via YaRN. Unlike naive linear position interpolation which uniformly scales all frequency dimensions—causing severe performance degradation and perplexity explosion—YaRN partitions the frequency spectrum using an NTK-by-parts approach:
1. **High-frequency dimensions (local dependencies)**: Kept uncompressed / extrapolated to maintain acute short-range positional resolution.
2. **Low-frequency dimensions (global context)**: Scaled via position interpolation by factor $s = 16.0$ ($\text{scale} = 1 / s$).
3. **Mid-frequency transition band**: Smoothly blended between interpolation and extrapolation via linear ramp boundaries parameterized by $\beta_{\text{fast}}$ and $\beta_{\text{slow}}$.
4. **Attention magnitude correction (`mscale`)**: Multiplies query and key rotary components by an energy preservation factor $t = 1.0 + 0.1 \ln(s)$ to prevent dot-product entropy collapse.

---

## 2. Mathematical Formulation & Implementation Details

### 2.1 Correction Factor & Boundary Calculation (`rope_yarn_corr_dims`)
The boundary dimensions where the transition between interpolation and extrapolation occurs are defined by:

$$\text{corr\_fac}(n_{\text{rot}}) = \frac{d_{\text{head}} \cdot \ln\left(\frac{L_0}{2\pi \cdot n_{\text{rot}}}\right)}{2 \cdot \ln(\theta)}$$

Where:
- $d_{\text{head}}$: Head dimension (`head_dim`, e.g. 128 or 96).
- $L_0$: Original training context length (`n_ctx_orig`, e.g. 16,384).
- $\theta$: Base rotary frequency (`rope_theta`, e.g. $1,000,000.0$).
- $n_{\text{rot}}$: Boundary parameters ($\beta_{\text{fast}} = 32.0$ for lower boundary, $\beta_{\text{slow}} = 1.0$ for upper boundary).

In `src/llamojo/backend/kernels_norm_rope.mojo`:
```mojo
@always_inline
def rope_yarn_corr_factor(n_dims: Int, n_ctx_orig: Int, n_rot: Float32, base: Float32) -> Float32:
    var pi_2 = Float32(6.283185307179586)
    var num = log(Float32(n_ctx_orig) / (n_rot * pi_2))
    var den = Float32(2.0) * log(base)
    return Float32(n_dims) * (num / den)

@always_inline
def rope_yarn_corr_dims(
    n_dims: Int,
    n_ctx_orig: Int,
    freq_base: Float32,
    beta_fast: Float32,
    beta_slow: Float32,
    mut dims_low: Float32,
    mut dims_high: Float32,
):
    var f_low = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)
    var f_high = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)
    dims_low = max(Float32(0.0), floor(f_low))
    dims_high = min(Float32(n_dims - 1), ceil(f_high))
```

### 2.2 Ramp Blending Function (`rope_yarn_ramp`)
The ramp mixing factor transitions linearly between $\text{low}$ and $\text{high}$:

$$\text{ramp\_mix}(i) = 1.0 - \text{clamp}\left(\frac{i - \text{low}}{\max(0.001, \text{high} - \text{low})}, 0.0, 1.0\right)$$

- When $i \le \text{low}$: $\text{ramp\_mix} = 1.0 \implies$ full extrapolation.
- When $i \ge \text{high}$: $\text{ramp\_mix} = 0.0 \implies$ full interpolation.
- Between $\text{low}$ and $\text{high}$: smooth continuous weighting.

### 2.3 Magnitude Scale (`mscale`)
To stabilize attention entropy at large context lengths, `mscale` is computed as:

$$\text{mscale} = \gamma \cdot \left(1.0 + 0.1 \cdot \ln(s)\right)$$

Where $s = 1.0 / \text{rope\_freq\_scale} = 16.0$, and $\gamma$ is `attn_factor` (default `1.0`).

### 2.4 Vectorized YaRN RoPE Kernel (`apply_rope_yarn`)
The complete kernel computes pre-tabulated sine and cosine tables for all $(d_{\text{head}} / 2)$ rotary pairs and applies in-place rotations across all query and key heads. It supports both **Interleaved/NORM** ($2i, 2i+1$) and **NeoX/Split** ($i, i + d/2$) coordinate layouts:

```mojo
@always_inline
def apply_rope_yarn(
    q: FloatPtr,
    k: FloatPtr,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    pos: Int,
    rope_theta: Float32 = 10000.0,
    rope_freq_scale: Float32 = 1.0,
    n_ctx_orig: Int = 16384,
    beta_fast: Float32 = 32.0,
    beta_slow: Float32 = 1.0,
    ext_factor: Float32 = 1.0,
    attn_factor: Float32 = 1.0,
    is_neox: Bool = False,
):
...
```

### 2.5 Forward Layer Execution Dispatch (`src/llamojo/backend/models.mojo`)
In `forward_layer_dense`, Step 3 (RoPE application) was upgraded to inspect `cfg.rope_scaling_type`:
```mojo
# 3. RoPE: apply in-place to worker's heads (worker 0 applies for all if 1 worker)
if wid == 0:
    if cfg.rope_scaling_type == 2: # YaRN
        apply_rope_yarn(
            q,
            k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            pos,
            cfg.rope_theta,
            cfg.rope_freq_scale,
            cfg.rope_orig_ctx_len,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
            Float32(1.0),
            cfg.yarn_attn_factor,
            cfg.rope_type != 0,
        )
    elif cfg.rope_type == 0:
        apply_rope_norm(q, k_buf, n_heads, n_kv_heads, head_dim, pos, cfg.rope_theta, cfg.rope_freq_scale)
    else:
        apply_rope_neox(q, k_buf, n_heads, n_kv_heads, head_dim, pos, cfg.rope_theta, cfg.rope_freq_scale)
llama_barrier(sh)
```

---

## 3. Verification & Testing

A dedicated test suite [`tests/test_ministral3_phase2.mojo`](../../tests/test_ministral3_phase2.mojo) was constructed to validate:
1. **Boundary & Ramp Correctness**:
   - $\text{corr\_low} \ge 0$, $\text{corr\_high} < d_{\text{head}}$, $\text{corr\_low} \le \text{corr\_high}$.
   - $\text{ramp\_mix} = 1.0$ at boundary low, $0.0$ at boundary high, and strictly bounded $y \in [0, 1]$ in the interior.
2. **Kernel Execution & Numerical Stability**:
   - Multi-head query and key transformations across 128 sequence positions.
   - Verified that rotated values are finite, numerically stable, and free of NaNs/Infs for both NORM and NeoX layouts.

### Test Execution Output
```bash
$ mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_ministral3_phase2.mojo
============================================================
Running Ministral 3 3B Phase 2 (YaRN RoPE) Tests
============================================================
Testing YaRN correction dimension bounds and ramp calculations...
  ✓ YaRN corr dims and ramp verified
Testing apply_rope_yarn execution and rotation accuracy...
  ✓ apply_rope_yarn execution and numerical stability verified
All Phase 2 tests passed! ✓
```

### Regression Verification
All existing test suites were executed to confirm zero regressions:
```bash
$ mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_ministral3_phase1.mojo
All Phase 1 tests passed! ✓

$ mojo -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_kernels.mojo
All Phase 2 Kernel tests passed! ✓
```

---

## 4. Next Steps (Phase 3)

With Phase 1 (GGUF Ingestion) and Phase 2 (YaRN RoPE Kernel) completed:
1. **Phase 3 (Sliding Window & End-to-End Execution)**:
   - Wire `cfg.sliding_window` directly into the `gqa_decode` call in `models.mojo:forward_layer_dense`.
   - Validate full multi-token sequence generation on CPU using Ministral 3 3B weights.
