# Timestral Implementation: Phase 3 Specification & Implementation Report

## Overview
Phase 3 builds the core transformer computation and prediction backend of the Mistral Timestral time-series foundation model in pure Mojo 1.1 (`/home/tasahi/miniconda/envs/moj/bin/mojo`), eliminating any Python or C++ runtime dependency.

It implements:
1. **Partial Rotary Positional Embedding (RoPE):** `apply_rope_partial` in [`src/llamojo/backend/kernels_norm_rope.mojo`](../../src/llamojo/backend/kernels_norm_rope.mojo).
2. **Time-Axis Attention with DHF Causal Mask & Sigmoid Gating:** `forward_time_attention` in [`src/llamojo/timestral/model.mojo`](../../src/llamojo/timestral/model.mojo).
3. **Cross-Variate Multi-Head Attention:** `forward_variate_attention` across rows for layers scheduled as `"time_variate"`.
4. **Fused SwiGLU Feed-Forward Blocks:** `forward_swiglu_mlp` with $\text{gate} \cdot \text{SiLU}(\text{up})$ and down-projection.
5. **Pre-allocated Scratchpad Arena (`TimestralState`):** Zero-allocation activation workspace for multi-layer evaluation.
6. **Quantile Output Head & Monotonic Fan-Out:** `monotone_quantiles` and `forward_quantile_head` yielding non-crossing quantiles across all 21 knots in physical units via inverted `asinh` scaling.
7. **End-to-End Forward Pass:** `forward_timestral` connecting patchified representations to denormalized forecast predictions.

---

## 1. Architectural Details

### 1.1 Partial RoPE Kernel (`apply_rope_partial`)
In Mistral Timestral (`layers.py:SinCosRoPE`), RoPE is applied with `rope_partial_factor = 0.5`. Only the first $\text{rope\_dim} = \text{head\_dim} \cdot 0.5$ dimensions rotate; the remaining dimensions undergo an identity transformation ($\cos = 1, \sin = 0$).

The rotation operates using NeoX/half-split pairs:
$$\tilde{q}_i = q_i \cos(\theta_i) - q_{i + \text{half\_head}} \sin(\theta_i)$$
$$\tilde{q}_{i + \text{half\_head}} = q_i \sin(\theta_i) + q_{i + \text{half\_head}} \cos(\theta_i)$$
for $i \in [0, \frac{\text{rope\_dim}}{2})$.

### 1.2 Dual-Axis Attention Mechanics
Timestral uses a dynamic `block_schedule` alternating between `"time"` and `"time_variate"` layers:
- **Time Attention:** Operates independently per row $r \in [0, R)$ along the patch sequence $N$.
  - Uses fused projection $W_{QKV} \in \mathbb{R}^{3 d_{\text{model}} \times d_{\text{model}}}$.
  - Applies `apply_rope_partial` anchored by patch index.
  - Implements **Decoupled Horizon Forecasting (DHF) TwoStreamMask**:
    - Context queries ($p < n_{\text{hist}}$) attend to historical keys $k \le p$ (causal).
    - Horizon queries ($p \ge n_{\text{hist}}$) attend to all valid context keys $0..n_{\text{hist}}-1$ and causally to horizon queries $n_{\text{hist}}..p$.
  - Applies `sdpa_sigmoid` gating: $\text{attn\_out} = \text{attn\_out} \odot \sigma(W_{\text{gate}} x_{\text{normed}})$ before projection by $W_O$.
- **Variate Attention:** Operates independently per temporal patch $p \in [0, N)$ across rows $R$ (variates and covariates of the same series).
  - Unmasked multi-head self-attention without positional encoding.
  - Captures inter-series interactions and exogenous covariate dynamics.
  - Applies `sdpa_sigmoid` gating before output projection $W_O$.

### 1.3 Quantile Output Head & Monotonic Cumsum
The quantile output head projects normalized final representations ($d_{\text{model}}$) through a ResidualBlock to produce $32 \times 21 = 672$ outputs per horizon patch:
$$\text{raw} = W_{\text{output}} \text{SiLU}(W_{\text{hidden}} x + b_{\text{hidden}}) + b_{\text{output}} + W_{\text{residual}} x + b_{\text{residual}}$$

To enforce strict non-crossing quantile guarantees ($q_1 \le q_2 \le \dots \le q_{21}$), `monotone_quantiles` evaluates:
1. The median channel (knot index 10, $q = 0.50$) is free.
2. The increments fan out symmetrically with probability gap weights $\Delta p$:
   $$\text{inc}_i = \text{width\_factor} \cdot \Delta p_i \cdot \text{softplus}(\text{raw}_i)$$
3. Upper quantiles ($i > 10$):
   $$\hat{q}_i = \hat{q}_{i-1} + \text{inc}_i$$
4. Lower quantiles ($i < 10$):
   $$\hat{q}_i = \hat{q}_{i+1} - \text{inc}_i$$
5. Finally, `invert_affine_asinh` denormalizes normalized quantiles using the historical anchor statistics $(\mu_{\text{anchor}}, \sigma_{\text{anchor}})$:
   $$\hat{y} = \mu_{\text{anchor}} + \sigma_{\text{anchor}} \cdot \sinh(\hat{q})$$

---

## 2. Implemented Source Files

| File | Description |
|---|---|
| [`src/llamojo/backend/kernels_norm_rope.mojo`](../../src/llamojo/backend/kernels_norm_rope.mojo) | Added `apply_rope_partial` kernel supporting proportional RoPE rotation. |
| [`src/llamojo/timestral/model.mojo`](../../src/llamojo/timestral/model.mojo) | Full implementation of `TimestralState`, `forward_time_attention`, `forward_variate_attention`, `forward_swiglu_mlp`, `forward_timestral_block`, `monotone_quantiles`, `forward_quantile_head`, and `forward_timestral`. |
| [`src/llamojo/timestral/config.mojo`](../../src/llamojo/timestral/config.mojo) | Added `.rope_dim()` calculation. |
| [`src/llamojo/timestral/__init__.mojo`](../../src/llamojo/timestral/__init__.mojo) | Exported Phase 3 state structs, attention functions, and forward runners. |
| [`src/llamojo/__init__.mojo`](../../src/llamojo/__init__.mojo) | Re-exported Phase 3 APIs at the top-level package namespace. |
| [`tests/test_timestral_phase3.mojo`](../../tests/test_timestral_phase3.mojo) | Unit verification suite covering partial RoPE, monotonic quantiles, and end-to-end forward prediction. |

---

## 3. Verification & Test Results

Run tests via:
```bash
/home/tasahi/miniconda/envs/moj/bin/mojo run -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_timestral_phase3.mojo
```

### Test Output:
```text
=== Running Phase 3 Timestral Transformer & Attention Tests ===
  Testing apply_rope_partial kernel...
    ✓ apply_rope_partial verified.
  Testing monotone_quantiles...
    ✓ monotone_quantiles non-crossing monotonicity verified.
  Testing end-to-end forward_timestral...
Writing 29 tensors with precision 'f32'...
Successfully generated GGUF model: /tmp/timestral_phase3_test.gguf
    Target 0 sample median forecast: 14.202149
    Target 1 sample median forecast: 53.32058
    ✓ End-to-end forward_timestral execution and non-crossing quantiles verified.
=== All Phase 3 Timestral Tests Passed! ===
```

All previous phases (Phase 1 and Phase 2) also continue to pass with zero regressions.
