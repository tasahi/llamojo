# Timestral Quantized Weights (Q8_0): Implementation Report (Gap B Remedy)

This document records the design, implementation, file size reductions, and verification of **Gap B: Quantized Matrix Multiplication and GGUF v3 Q8_0 Serialization** for the Timestral engine in `llamojo`.

---

## 1. Background & Motivation

While FP32 weights provide pristine numerical parity during reference validation, edge and industrial deployments require minimal memory footprint, reduced cache eviction, and high-throughput execution:
- **FP32 vs Q8_0:** A 32-element row requires $32 \times 4 = 128$ bytes in FP32, whereas a standard `GGMLType.Q8_0` block requires only 34 bytes (2-byte FP16 scale $d$ + 32 $\times$ 1-byte signed `int8` quants $q_i$).
- **File Size Reduction:** Achieves nearly **$3.75\times$ file compression** ($128 / 34 \approx 3.76$), shrinking `Timestral-28M` from ~111 MB to **~30 MB**, and `Timestral-209M` from ~837 MB to **~225 MB**.
- **Kernel Efficiency:** Evaluated via SIMD `dot_quant_row` / `dot_q8_0_row` using fast integer vector multiply-adds.

---

## 2. Architecture & Implementation

### 2.1 Pure Mojo Engine Adaptations (`src/llamojo/timestral/patch.mojo`)

In `src/llamojo/timestral/patch.mojo`, the patch projection residual block previously used fixed row indexing:
```mojo
var row_ptr = weights.hidden_w.unsafe_offset(j * in_dim * 4)  # Assumed FP32
```
When loading a `Q8_0` quantized GGUF model, `weights.patch_proj.dtype_hidden` is `GGMLType.Q8_0`, so row offsets must be computed dynamically using the block size ($32$) and type size ($34$ bytes) of the tensor:

```mojo
@always_inline
def _get_patch_row_ptr(weight_base: BytePtr, dtype: UInt32, row_idx: Int, n_cols: Int) -> BytePtr:
    """ Computes the byte offset for a specific matrix row based on quantization block size."""
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var n_blocks = (n_cols + blk_sz - 1) // blk_sz
    var row_bytes = n_blocks * type_sz
    return weight_base.unsafe_offset(row_idx * row_bytes)
```
This helper was added to `patch.mojo` and applied across:
1. `weights.patch_proj.hidden_w` (size $4 \cdot d_{\text{model}} \times \text{in\_dim}$)
2. `weights.patch_proj.output_w` (size $d_{\text{model}} \times 4 \cdot d_{\text{model}}$)
3. `weights.patch_proj.residual_w` (size $d_{\text{model}} \times \text{in\_dim}$)

Combined with the existing `_get_row_ptr` in `src/llamojo/timestral/model.mojo` for attention QKV, gate, output, SwiGLU FFN gate/up, FFN down, and quantile projection heads, all linear layers seamlessly evaluate arbitrary quantization types (`F32`, `F16`, `Q8_0`, `Q4_K`).

### 2.2 GGUF Conversion Tool (`tools/convert_timestral_to_gguf.py`)

In `tools/convert_timestral_to_gguf.py`:
- Updated `write_timestral_gguf` to selectively quantize all 2D linear weight matrices to `GGMLQuantizationType.Q8_0` with `raw_dtype=gguf.GGMLQuantizationType.Q8_0` when `--precision q8_0` is selected.
- Layer norms and 1D bias vectors (`.bias`, `.norm.`, `_norm.`, `forecast_query`) remain in `float32` for maximal numerical precision and numerical stability.
- Verified that model files generated with `--precision q8_0` show an immediate reduction from $1.5\text{ MB} \to 400\text{ KB}$ for the test model.

---

## 3. Verification & Benchmark

### 3.1 Pure Mojo Suite (`tests/test_timestral_quant.mojo`)

We added an automated test suite verifying end-to-end execution on a `Q8_0` quantized model:
1. **Weight Inspection:** Verified that every linear projection tensor reports `dtype = GGMLType.Q8_0 (8)`.
2. **Probabilistic Forecasting:** Evaluated zero-shot forecasting on continuous telemetry ($T=64, H=32$). Verified that all 21 quantiles are finite, monotonic, and non-crossing:
   $$q_{0.10} \le q_{0.50} \le q_{0.90}$$
3. **Causal Imputation:** Tested causal iterative missing-data filling on a sequence with a 10-step `NaN` gap. Verified that all missing steps are filled with smooth, finite predicted values.

**Test Execution:**
```
=== Running Timestral Quantized Weights (Q8_0) Engine Test ===
Writing 29 tensors with precision 'q8_0'...
Successfully generated GGUF model: /tmp/synth_q8_cli.gguf
  Loading TimestralForecaster from Q8_0 GGUF...
  ✓ All linear projections verified as GGMLType.Q8_0 (dtype=8).
  Testing zero-shot forecast with Q8_0 model...
  ✓ Q8_0 forecast non-crossing quantiles verified. Median at h=0: 24.224268
  Testing causal missing data imputation with Q8_0 model...
  ✓ Q8_0 imputation successfully filled missing gap. Sample imputed val: 50.933853
=== Timestral Quantized Weights (Q8_0) Tests Passed! ===
```

### 3.2 Python Interoperability

Tested `llamojo.timestral.Timestral` loading `/tmp/synth_q8_cli.gguf` through `libllamojo.so`:
- Successfully generated probabilistic forecasts with 21 non-crossing quantiles.
- Verified `.median` and `.quantile(0.10)` / `.quantile(0.90)` consistency.
- Verified in-place array imputation (`.impute(...)`).
