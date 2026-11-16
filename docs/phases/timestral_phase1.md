---
type: documentation
title: "Phase 1 Documentation: Timestral GGUF v3 Specification & Weight Conversion Engine"
description: "Specification of the Timestral GGUF v3 metadata schema, canonical tensor mapping from Hugging Face SafeTensors, conversion tooling, and pure Mojo config and weight loader."
status: completed
tags: [timestral, phase-1, gguf, safetensors, mistral, loader, mojo]
timestamp: 2026-11-16T10:41:00Z
---

# Phase 1: Timestral GGUF v3 Specification & Loader

## 1. Overview

Phase 1 establishes the canonical **GGUF v3 binary format** for Mistral AI's **Timestral** time-series foundation models (`5M`, `28M`, `123M`, `209M`), enabling zero-copy `mmap64` ingestion and memory-efficient quantized serving directly in `llamojo`.

Phase 1 delivers three core components:
1. **GGUF v3 Metadata & Tensor Specification:** Canonical keys and tensor naming for Timestral architectures.
2. **Hugging Face to GGUF Converter Tooling:** [`tools/convert_timestral_to_gguf.py`](../tools/convert_timestral_to_gguf.py) supporting online HF download (`mistralai/Timestral-28M`), local SafeTensors directories, synthetic miniature model generation, and precision options (`f32`, `f16`, `q8_0`).
3. **Pure Mojo Metadata Parser & Weight Loader:** [`src/llamojo/timestral/config.mojo`](../src/llamojo/timestral/config.mojo) and [`src/llamojo/timestral/loader.mojo`](../src/llamojo/timestral/loader.mojo).

---

## 2. GGUF v3 Metadata Schema

Timestral models store configuration parameters in standard GGUF key-value metadata under the architecture namespace `timestral.*`:

| GGUF Key | Type | Description | Reference Value (28M) |
| :--- | :--- | :--- | :--- |
| `general.architecture` | `STRING` | Architecture identifier (`"timestral"`) | `"timestral"` |
| `general.name` | `STRING` | Model identifier | `"timestral-28m"` |
| `timestral.patch_size` | `UINT32` | Number of continuous time steps per patch | `32` |
| `timestral.d_model` | `UINT32` | Hidden state dimension size | `384` |
| `timestral.num_heads` | `UINT32` | Number of multi-head attention heads | `6` |
| `timestral.mlp_hidden_dim` | `UINT32` | SwiGLU FFN intermediate dimension | `1024` |
| `timestral.num_layers` | `UINT32` | Total number of transformer layers | `12` |
| `timestral.context_length` | `UINT32` | Maximum historical time-step context | `32768` |
| `timestral.block_schedule` | `ARRAY[STRING]` | Layer-by-layer attention axis schedule | `["time", "time", "time", "time_variate", ...]` |
| `timestral.rope_theta` | `FLOAT32` | RoPE base frequency | `10000.0` |
| `timestral.rope_partial_factor` | `FLOAT32` | Fraction of head dimension rotated by RoPE | `0.5` |
| `timestral.norm_eps` | `FLOAT32` | RMSNorm epsilon | `1e-6` |
| `timestral.quantile_knots` | `ARRAY[FLOAT32]` | Target quantile prediction knots (21 values) | `[0.01, 0.05, ..., 0.95, 0.99]` |
| `timestral.use_asinh` | `BOOL` | Whether reversible `asinh` scaling is enabled | `true` |
| `timestral.minimum_scale` | `FLOAT32` | Lower bound epsilon for series variance | `1e-6` |
| `timestral.quantile_width_factor` | `FLOAT32` | Width multiplier for quantile head loss | `3.0` |
| `timestral.split_context_horizon_attn` | `BOOL` | Decoupled Horizon Forecasting (DHF) flag | `true` |
| `timestral.attend_missing_patches` | `BOOL` | Whether attention processes missing patches | `true` |
| `timestral.covariate_enabled` | `BOOL` | Native past/known covariate support | `true` |

---

## 3. Canonical Tensor Mapping

Hugging Face SafeTensors (`model.safetensors`) tensors map directly into the GGUF tensor directory:

### 3.1 Input Patch Projection & Horizon Query

| Upstream Hugging Face Tensor | Canonical GGUF Tensor Name | Shape | Function |
| :--- | :--- | :--- | :--- |
| `representation.patch_proj.hidden_layer.weight` | `timestral.patch_proj.hidden.weight` | `[proj_hidden, 2 * patch_size]` | First linear layer of patch projection ($64 \to 1536$) |
| `representation.patch_proj.hidden_layer.bias` | `timestral.patch_proj.hidden.bias` | `[proj_hidden]` | Bias for hidden projection |
| `representation.patch_proj.output_layer.weight` | `timestral.patch_proj.output.weight` | `[d_model, proj_hidden]` | Output projection back to `d_model` ($1536 \to 384$) |
| `representation.patch_proj.output_layer.bias` | `timestral.patch_proj.output.bias` | `[d_model]` | Output projection bias |
| `representation.patch_proj.residual_layer.weight` | `timestral.patch_proj.residual.weight` | `[d_model, 2 * patch_size]` | Skip connection ($64 \to 384$) |
| `representation.patch_proj.residual_layer.bias` | `timestral.patch_proj.residual.bias` | `[d_model]` | Skip connection bias |
| `representation.forecast_query` | `timestral.forecast_query` | `[d_model]` | Learned horizon query token ($384$) |

> [!NOTE]
> The input dimension to `patch_proj` is $2 \times \text{patch\_size} = 64$. This is because each time step consists of both its normalized continuous value and a binary observation indicator (`1.0` if observed, `0.0` if `NaN`/missing).

### 3.2 Transformer Layers (`blk.{i}.*`)

For each layer $i \in [0, \text{num\_layers} - 1]$:

| Upstream Hugging Face Tensor | Canonical GGUF Tensor Name | Shape | Function |
| :--- | :--- | :--- | :--- |
| `transformer.layers.{i}.attn_norms.0.scale` | `blk.{i}.time_attn_norm.weight` | `[d_model]` | Pre-attention RMSNorm scale for Time attention |
| `transformer.layers.{i}.attns.0._gate.weight` | `blk.{i}.time_attn_gate.weight` | `[d_model, d_model]` | SDPA sigmoid gating projection |
| `transformer.layers.{i}.attns.0.wQKV.weight` | `blk.{i}.time_attn_qkv.weight` | `[3 * d_model, d_model]` | Fused Q, K, V projections ($1152 \times 384$) |
| `transformer.layers.{i}.attns.0.wO.weight` | `blk.{i}.time_attn_o.weight` | `[d_model, d_model]` | Attention output projection ($384 \times 384$) |
| `transformer.layers.{i}.attn_norms.1.scale` | `blk.{i}.variate_attn_norm.weight` | `[d_model]` | Pre-attention RMSNorm scale for Variate attention (*only if `time_variate`*) |
| `transformer.layers.{i}.attns.1._gate.weight` | `blk.{i}.variate_attn_gate.weight` | `[d_model, d_model]` | Variate SDPA sigmoid gating projection |
| `transformer.layers.{i}.attns.1.wQKV.weight` | `blk.{i}.variate_attn_qkv.weight` | `[3 * d_model, d_model]` | Variate fused Q, K, V projections |
| `transformer.layers.{i}.attns.1.wO.weight` | `blk.{i}.variate_attn_o.weight` | `[d_model, d_model]` | Variate attention output projection |
| `transformer.layers.{i}.mlp_norm.scale` | `blk.{i}.ffn_norm.weight` | `[d_model]` | Pre-FFN RMSNorm scale ($384$) |
| `transformer.layers.{i}.mlp.w12.weight` | `blk.{i}.ffn_gate_up.weight` | `[2 * mlp_hidden, d_model]` | Fused Gate & Up projection ($2048 \times 384$) |
| `transformer.layers.{i}.mlp.w3.weight` | `blk.{i}.ffn_down.weight` | `[d_model, mlp_hidden]` | Down projection ($384 \times 1024$) |

### 3.3 Output Norm & Quantile Projection Head

| Upstream Hugging Face Tensor | Canonical GGUF Tensor Name | Shape | Function |
| :--- | :--- | :--- | :--- |
| `transformer.out_norm.scale` | `timestral.out_norm.weight` | `[d_model]` | Final RMSNorm scale ($384$) |
| `head.projection.hidden_layer.weight` | `timestral.head.hidden.weight` | `[head_hidden, d_model]` | Hidden layer of quantile head ($1536 \times 384$) |
| `head.projection.output_layer.weight` | `timestral.head.output.weight` | `[patch_size * 21, head_hidden]` | Output layer projecting to 21 quantiles ($672 \times 1536$) |
| `head.projection.residual_layer.weight` | `timestral.head.residual.weight` | `[patch_size * 21, d_model]` | Residual skip layer ($672 \times 384$) |

---

## 4. Weight Conversion Tooling

The CLI tool [`tools/convert_timestral_to_gguf.py`](../tools/convert_timestral_to_gguf.py) converts models from Hugging Face or creates synthetic verification checkpoints:

```bash
# 1. Convert an official Hugging Face repository directly:
python tools/convert_timestral_to_gguf.py \
  --model-id mistralai/Timestral-28M \
  --out models/timestral-28m-f32.gguf \
  --precision f32

# 2. Convert from a local directory:
python tools/convert_timestral_to_gguf.py \
  --model-dir /path/to/local/timestral-209m \
  --out models/timestral-209m-q8_0.gguf \
  --precision q8_0

# 3. Generate a lightweight synthetic model for testing without downloading:
python tools/convert_timestral_to_gguf.py \
  --synthetic \
  --out /tmp/timestral_synthetic.gguf
```

---

## 5. Pure Mojo API Usage

### Reading Configuration
```mojo
from llamojo import gguf_load_from_file, TimestralConfig, load_timestral_weights

# Load GGUF file with zero-copy mmap
var ctx = gguf_load_from_file("models/timestral-28m-f32.gguf", use_mmap=True)

# Parse hyperparameters and block schedule
var cfg = TimestralConfig.from_gguf(ctx)
print("Layers:", cfg.num_layers)
print("Quantiles:", cfg.num_quantiles())
print("Has variate attention at layer 3:", cfg.has_variate_attn(3))
```

### Binding Tensor Pointers
```mojo
# Bind all tensors to structured zero-allocation accessors
var weights = load_timestral_weights(ctx, cfg)

# Inspect bound pointers
print("Total model bytes:", weights.total_bytes)
print("Bound tensors:", weights.n_tensors)
```

---

## 6. Verification & Test Suite

Phase 1 verification is automated via [`tests/test_timestral_phase1.mojo`](../tests/test_timestral_phase1.mojo):
- Verifies `TimestralConfig` default values and GGUF-parsed values.
- Checks `block_schedule` parity (`has_variate_attn(i)`).
- Validates 29 tensor bindings in synthetic models including dual-axis attention structures.

Run the test suite:
```bash
mojo run -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src tests/test_timestral_phase1.mojo
```
Output:
```
=== Running Phase 1 Timestral GGUF Tests ===
  Testing TimestralConfig default initialization...
    ✓ Default TimestralConfig values verified.
  Ensuring synthetic GGUF model exists at: /tmp/timestral_synthetic.gguf...
  Testing GGUF loading from: /tmp/timestral_synthetic.gguf...
    ✓ GGUF v3 Timestral config and weights verified successfully.
=== All Phase 1 Timestral Tests Passed! ===
```
