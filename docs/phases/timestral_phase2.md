---
type: documentation
title: "Phase 2 Documentation: Timestral Time-Series Preprocessing, Causal Normalization & Patch Embedding"
description: "Specification and pure Mojo implementation of TimeSeriesBatch, causal Welford patch statistics, reversible asinh scaling, missingness indicator channels, and ResidualBlock patch projection."
status: completed
tags: [timestral, phase-2, time-series, preprocessing, asinh, welford, patch-projection, mojo]
timestamp: 2026-11-16T11:11:00Z
---

# Phase 2: Timestral Preprocessing, Causal Normalization & Patch Embedding

## 1. Overview

Phase 2 implements the complete input preprocessing and token representation pipeline for Mistral's **Timestral** model family directly in pure Mojo (`src/llamojo/timestral/patch.mojo`), running natively with Mojo 1.1 in the anaconda `moj` environment.

Unlike text language models that map integer token IDs to embedding vectors via table lookup, Timestral accepts continuous multi-variate sequences of shape `(Batch, Variates, Time)` containing missing values (`NaN`), accompanied by past-only covariates and known-future operational schedules.

Phase 2 delivers:
1. **`TimeSeriesBatch` Container:** Multi-variate input abstraction supporting continuous observations, missing values (`NaN`/`Inf`), and structural roles (`ROLE_TARGET`, `ROLE_PAST_COV`, `ROLE_KNOWN_COV`).
2. **Causal Welford Statistics (`CausalPatchStats`):** Strictly causal running mean and standard deviation along the historical timeline without lookahead leakage.
3. **Reversible `asinh` Transformation:** Stable, non-linear squashing that absorbs extreme spikes and non-stationarity while supporting exact numerical inversion.
4. **Length-32 Patchification & Observation Masks:** Left-padding to a multiple of 32 steps, concatenating normalized values and missingness flags into length-64 vectors.
5. **Patch Projection ResidualBlock:** Evaluates $W_{\text{out}} \text{SiLU}(W_{\text{hidden}} x + b_{\text{hidden}}) + b_{\text{out}} + W_{\text{res}} x + b_{\text{res}}$ to project 64-dimensional patches into `d_model` tokens.
6. **Decoupled Horizon Query Injection:** Injects the learned `timestral.forecast_query` token across horizon patches for targets while projecting future known covariates.

---

## 2. Mathematical & Algorithmic Foundation

### 2.1 Causal Welford Running Statistics

For each target series and past covariate, statistics must be calculated causally so past patches never observe future statistics.

For each time step $t \in [0, T_{\text{padded}}-1]$:
If $x_t$ is observed (finite, not padding, not `NaN`):
$$\begin{aligned}
w &\leftarrow w + 1 \\
\delta &\leftarrow x_t - \mu \\
\mu &\leftarrow \mu + \frac{\delta}{w} \\
m_2 &\leftarrow m_2 + \delta \cdot (x_t - \mu)
\end{aligned}$$

At the boundary of each patch $p$ ($t = (p + 1) \times 32 - 1$):
$$\begin{aligned}
\text{loc}[p] &= \mu \\
\text{var} &= \begin{cases} \frac{m_2}{w - 1} & \text{if } w > 1 \\ 0.0 & \text{otherwise} \end{cases} \\
\text{scale}[p] &= \max(\sqrt{\text{var}}, \text{minimum\_scale})
\end{aligned}$$

Every time step within patch $p$ is scaled using the patch statistics $(\text{loc}[p], \text{scale}[p])$ evaluated at the end of that patch.

### 2.2 Reversible `asinh` Instance Normalization

Observed points are transformed into normalized space:
$$\tilde{x}_t = \text{asinh}\left(\frac{x_t - \text{loc}[p]}{\text{scale}[p]}\right)$$

Unobserved / missing points (`NaN` or padding) are assigned $\tilde{x}_t = 0.0$.

When the model outputs predictions $\hat{y}$ in normalized space, they are inverted back to the physical domain via:
$$x_t = \text{loc}[p] + \text{scale}[p] \cdot \sinh(\hat{y}_t)$$

### 2.3 64-Dimensional Patch Packing

Each length-32 patch is mapped into a 64-dimensional input vector $x \in \mathbb{R}^{64}$:
- **Channels `0..31`:** Normalized values $\tilde{x}_k$ (or `0.0` if unobserved/padded).
- **Channels `32..63`:** Binary missingness indicator $m_k$ (`0.0` if observed, `1.0` if unobserved/padded/NaN).

$$x = \Big[ \tilde{x}_0, \dots, \tilde{x}_{31}, \; m_0, \dots, m_{31} \Big]^T$$

### 2.4 ResidualBlock Patch Projection

The packed 64-dimensional patch vector is projected into the transformer embedding space $\mathbb{R}^{d_{\text{model}}}$ using the model's `ResidualBlock`:

$$\begin{aligned}
h &= \text{SiLU}(W_{\text{hidden}} x + b_{\text{hidden}}) \in \mathbb{R}^{4 \cdot d_{\text{model}}} \\
\text{out} &= W_{\text{output}} h + b_{\text{output}} \in \mathbb{R}^{d_{\text{model}}} \\
\text{res} &= W_{\text{residual}} x + b_{\text{residual}} \in \mathbb{R}^{d_{\text{model}}} \\
\text{token} &= \text{out} + \text{res} \in \mathbb{R}^{d_{\text{model}}}
\end{aligned}$$

### 2.5 Horizon Token Decoupling

For the forecasting window of length $H$:
- **Historical context patches ($p < n_{\text{hist}}$):** Tokens are produced by projecting historical patches through `forward_patch_proj`.
- **Horizon patches ($p \ge n_{\text{hist}}$):**
  - **Targets:** Populated with the model's learned `timestral.forecast_query` parameter vector $\in \mathbb{R}^{d_{\text{model}}}$.
  - **Known-future covariates:** Normalized using global instance statistics and projected through `forward_patch_proj`.
  - **Past-only covariates:** Padded with all-missing indicator vectors ($m_k = 1.0$) and projected.

---

## 3. Pure Mojo Architecture (`src/llamojo/timestral/patch.mojo`)

### 3.1 Data Structures

```mojo
# Structural roles
comptime ROLE_TARGET: Int = 0
comptime ROLE_PAST_COV: Int = 1
comptime ROLE_KNOWN_COV: Int = 2

struct TimeSeriesBatch(Movable):
    var batch_size: Int
    var n_targets: Int
    var t_context: Int
    var t_horizon: Int
    var n_past_cov: Int
    var n_known_cov: Int
    var context_ptr: FloatPtr
    var past_cov_ptr: FloatPtr
    var known_cov_ptr: FloatPtr
    ...

struct CausalPatchStats(Movable):
    var n_patches: Int
    var loc: FloatPtr
    var scale: FloatPtr
    ...

struct PatchBuffer(Movable):
    var n_rows: Int
    var n_patches: Int
    var patch_size: Int
    var in_dim: Int # 64
    var packed_data: FloatPtr
    var loc_data: FloatPtr
    var scale_data: FloatPtr
    ...

struct EmbeddedTokens(Movable):
    var n_rows: Int
    var n_patches: Int
    var d_model: Int
    var data: FloatPtr
    ...

struct PatchifyResult(Movable):
    var tokens: EmbeddedTokens
    var patch_buf: PatchBuffer
    ...
```

### 3.2 Key Functions

- `apply_affine_asinh(x, loc, scale, is_obs, use_asinh) -> Float32`: Applies non-linear squashing.
- `invert_affine_asinh(y, loc, scale, use_asinh) -> Float32`: Inverts predictions to physical units.
- `forward_patch_proj(weights, in_vec, out_vec, scratch_hidden, d_model, in_dim)`: SIMD matrix-vector ResidualBlock evaluation.
- `patchify_and_embed(batch, config, patch_proj, forecast_query) -> PatchifyResult`: High-level entry point coordinating the complete preprocessing workflow.

---

## 4. Usage Example

```mojo
from llamojo import (
    gguf_load_from_file,
    TimestralConfig,
    load_timestral_weights,
    TimeSeriesBatch,
    patchify_and_embed,
)

# 1. Load weights
var ctx = gguf_load_from_file("models/timestral-28m-f32.gguf")
var cfg = TimestralConfig.from_gguf(ctx)
var weights = load_timestral_weights(ctx, cfg)

# 2. Construct batch (1 series, 2 targets, 96 context steps, 16 prediction steps, 1 known covariate)
var batch = TimeSeriesBatch(1, 2, 96, 16, n_past_cov=0, n_known_cov=1)
# Populate sensor telemetry and known production schedule...

# 3. Patchify and embed
var res = patchify_and_embed(batch, cfg, weights.patch_proj, weights.forecast_query)

# Access embedded tokens [n_rows, n_patches, d_model]
ref tokens = res.tokens
print("Total rows:", tokens.n_rows)
print("Total patches (history + horizon):", tokens.n_patches)
print("Embedding dimension:", tokens.d_model)

# Clean up
res.free()
batch.free()
ctx.free()
```

---

## 5. Verification & Test Suite

The Phase 2 implementation is verified via [`tests/test_timestral_phase2.mojo`](../../tests/test_timestral_phase2.mojo):
1. **`test_reversible_asinh`:** Validates round-trip numerical inversion $\text{invert}(\text{apply}(x)) \equiv x$ for positive, negative, and missing values.
2. **`test_causal_patch_stats`:** Verifies strictly causal running mean/scale accumulation across multiple patches.
3. **`test_time_series_batch_container`:** Tests multi-variate indexing, past/known covariates, and `NaN` missingness detection.
4. **`test_patchify_and_embed_pipeline`:** Tests end-to-end left-padding, missingness bit-packing, ResidualBlock projection, and horizon `forecast_query` token injection.

Run the test suite:
```bash
mojo run -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_timestral_phase2.mojo
```
Output:
```
=== Running Phase 2 Timestral Preprocessing Tests ===
  Testing reversible asinh transformation...
    ✓ Reversible asinh scaling and inversion verified.
  Testing causal patch statistics...
    ✓ Causal patch statistics verified.
  Testing TimeSeriesBatch container...
    ✓ TimeSeriesBatch container and NaN checks verified.
  Testing end-to-end patchify_and_embed pipeline...
    ✓ End-to-end patchify_and_embed pipeline verified successfully.
=== All Phase 2 Timestral Tests Passed! ===
```
