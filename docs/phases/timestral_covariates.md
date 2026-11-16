# Timestral Covariate Encoding: Implementation Report (Gap A Remedy)

This document records the design, implementation, and parity verification of **Gap A: Categorical Covariate Target Encoding** (`timestral.covariate_encoding.target_encode`) for the Timestral engine in pure Mojo and high-level Python.

---

## 1. Background & Formulation

Upstream Mistral Timestral models were trained on numeric, target-scaled continuous covariates. When categorical covariates are provided (e.g. equipment IDs, operational mode flags, shift identifiers, holiday markers), they must undergo smoothed mean-target encoding per time-series before entering patch projection.

The upstream formula implemented in `timestral.covariate_encoding.target_encode` is:

$$\text{encoded}(c) = \frac{\text{smooth} \cdot \mu_{\text{item}} + \sum_{(t, c)} y_t}{\text{smooth} + N_{(t, c)}}$$

where:
1. **Valid Item Mean ($\mu_{\text{item}}$):** Computed across all finite target entries $y_t \in \mathbb{R}$ in the historical context window ($T$). Non-finite values (`NaN`, `Inf`) are omitted.
2. **Missing Covariate Values (`NaN`):** `NaN` values in `past_codes` form their own distinct category slot (`nan_slot`).
3. **Unseen Future Categories:** Any category occurring in the prediction horizon ($H$) that was never observed in context ($T$) resolves directly to the series context mean $\mu_{\text{item}}$.
4. **Default Smoothing:** `smooth = 1.0`.

---

## 2. Architecture & Modules

### 2.1 Pure Mojo Engine (`src/llamojo/timestral/covariates.mojo`)

We added `src/llamojo/timestral/covariates.mojo` providing zero external dependency target encoding:

```mojo
struct TargetEncodingResult(Movable):
    var past_encoded: FloatPtr
    var t_len: Int
    var future_encoded: FloatPtr
    var h_len: Int
    var has_future: Bool

    def free(mut self):
        ...

def target_encode(
    target: FloatPtr,
    past_codes: FloatPtr,
    t_len: Int,
    future_codes: FloatPtr = null_ptr[Float32](),
    h_len: Int = 0,
    smooth: Float32 = 1.0,
) -> TargetEncodingResult:
    ...
```

#### Key Implementation Steps:
1. **Item Mean Accumulation:** Computes $\mu_{\text{item}}$ excluding non-finite targets.
2. **Category Extraction & Sorting:** Uniquely extracts all finite categories present in `past_codes` and sorts them in ascending order.
3. **Slot Mapping:**
   - Indices `0 .. n_cat - 1`: Finite categories.
   - Index `n_cat`: `nan_slot` for missing codes.
   - Index `n_cat + 1`: `unseen_slot` resolving to $\mu_{\text{item}}$.
4. **Weighted Histogram:** Computes `sums` and `counts` over valid targets for each slot.
5. **Lookup & Encoding:** Generates lookup tables and projects both `past_codes` and `future_codes` into continuous Float32 arrays.

### 2.2 Public Export Updates

- **`src/llamojo/timestral/__init__.mojo`:** Exports `TargetEncodingResult` and `target_encode`.
- **`src/llamojo/__init__.mojo`:** Re-exports `TargetEncodingResult` and `target_encode` at the top level of `llamojo`.

### 2.3 Python Bindings (`python/llamojo/timestral.py`)

1. **`target_encode(...)`:** Standalone function replicating the exact interface of upstream `timestral.covariate_encoding.target_encode`.
2. **`Timestral.forecast(...)`:** Extended to accept:
   - `categorical_past_covariates`: Array of shape `(batch, n_cat_past, t_context)` or `(n_cat_past, t_context)`.
   - `categorical_known_covariates`: Array of shape `(batch, n_cat_known, t_context + prediction_length)` or `(n_cat_known, ...)`.
   - Automatically executes smoothed target encoding against the target context and concatenates the resulting numeric covariates with any existing numeric covariates before invoking the C ABI engine.

---

## 3. Verification & Numerical Parity

### 3.1 Pure Mojo Unit Test (`tests/test_timestral_categorical.mojo`)

We constructed a test case containing missing values (`NaN`) in both the target series and covariate codes, as well as an unseen future category code (`99.0`):

- **Target:** `[10.0, 20.0, NaN, 40.0, 50.0]` $\implies \mu_{\text{item}} = 30.0$
- **Past Codes:** `[1.0, 2.0, 1.0, NaN, 2.0]`
- **Future Codes:** `[1.0, NaN, 99.0, 2.0]`

#### Analytical & Expected Results:
- Category `1.0`: `(1.0 * 30 + 10) / (1.0 + 1) = 20.0`
- Category `2.0`: `(1.0 * 30 + 70) / (1.0 + 2) = 33.333332`
- `NaN` Slot: `(1.0 * 30 + 40) / (1.0 + 1) = 35.0`
- Unseen Slot (`99.0`): `30.0`

**Result in Mojo:**
```
=== Running Timestral Categorical Covariate Target Encoding Test ===
Checking past encoding parity...
  past[ 0 ] actual= 20.0  expected= 20.0  diff= 0.0
  past[ 1 ] actual= 33.333332  expected= 33.333332  diff= 0.0
  past[ 2 ] actual= 20.0  expected= 20.0  diff= 0.0
  past[ 3 ] actual= 35.0  expected= 35.0  diff= 0.0
  past[ 4 ] actual= 33.333332  expected= 33.333332  diff= 0.0
Checking future encoding parity...
  fut[ 0 ] actual= 20.0  expected= 20.0  diff= 0.0
  fut[ 1 ] actual= 35.0  expected= 35.0  diff= 0.0
  fut[ 2 ] actual= 30.0  expected= 30.0  diff= 0.0
  fut[ 3 ] actual= 33.333332  expected= 33.333332  diff= 0.0
Categorical Covariate Target Encoding test passed successfully!
```

### 3.2 Upstream PyTorch / Timestral Library Parity

Verified against `timestral.covariate_encoding.target_encode` in Python:
- `np.allclose(llamojo_te(tgt, past, fut), upstream_te(tgt, past, fut))` $\implies$ **Exact Match**.

### 3.3 Regression & Shared Library Verification

- Built `build/libllamojo.so` cleanly with zero errors.
- Ran `tests/test_timestral_phase5.mojo`: All numerical parity tests ($\text{MAE} < 10^{-4}$), industrial predictive maintenance multi-sensor tests, and CPU latency benchmarks passed without regressions.
