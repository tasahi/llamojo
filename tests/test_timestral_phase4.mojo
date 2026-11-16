""" Phase 4 Unit Tests: High-Level TimestralForecaster API, Probabilistic Quantiles, and Imputation.

Verifies:
1. `TimestralForecaster` loading directly from GGUF file (`from_gguf`).
2. `.forecast(...)` zero-shot multi-step probabilistic forecasting on continuous multi-variate telemetry.
3. `ForecastResult` quantile inspection: `.median()`, `.quantile(q)`, non-crossing properties across horizon steps.
4. Multimodal input handling with past covariates and known-future covariates.
5. In-place missing data imputation (`.impute(...)`) filling NaN values using causal forecasting.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.memory import Pointer, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.math import nan, isnan
from std.ffi import external_call

from llamojo import (
    TimestralConfig,
    ForecastResult,
    TimestralForecaster,
)
from llamojo.gguf import _cstr_alloc


def test_forecaster_basic_and_quantiles() raises:
    """ Verify TimestralForecaster initialization, forecast, and quantile lookups."""
    print("  Testing TimestralForecaster zero-shot forecast and quantile access...")

    var synthetic_path = "/tmp/timestral_phase4_test.gguf"
    var py_cmd = _cstr_alloc(
        "/home/tasahi/miniconda/envs/moj/bin/python tools/convert_timestral_to_gguf.py"
        + " --synthetic --out " + synthetic_path
    )
    _ = external_call["system", Int32](py_cmd)
    py_cmd.unsafe_free()

    # 1. Initialize Forecaster from GGUF
    var forecaster = TimestralForecaster.from_gguf(synthetic_path)
    assert_equal(forecaster.config.patch_size, 32)
    assert_equal(forecaster.config.num_quantiles(), 21)

    # 2. Prepare 100 historical steps of telemetry for 2 targets
    var t_hist = 100
    var h_pred = 32
    var n_targets = 2
    var context = unsafe_alloc[Float32](n_targets * t_hist)

    # Target 0: upward trend around 25.0
    for t in range(t_hist):
        context[unsafe_offset=t] = 25.0 + Float32(t) * 0.1

    # Target 1: cyclic around 70.0
    for t in range(t_hist):
        context[unsafe_offset=t_hist + t] = 70.0 + Float32(t % 10)

    # 3. Execute forecast
    var forecast = forecaster.forecast(
        context=context,
        t_context=t_hist,
        prediction_length=h_pred,
        batch_size=1,
        n_targets=n_targets,
    )

    assert_equal(forecast.batch_size, 1)
    assert_equal(forecast.n_targets, 2)
    assert_equal(forecast.horizon, 32)

    # 4. Check quantile monotonicity and bounds
    for v in range(n_targets):
        for h in range(h_pred):
            var med = forecast.median(0, v, h)
            var p10 = forecast.quantile(0, v, h, 0.10)
            var p50 = forecast.quantile(0, v, h, 0.50)
            var p90 = forecast.quantile(0, v, h, 0.90)

            # Monotonicity checks
            assert_almost_equal(med, p50, atol=1e-5)
            assert_true(p10 <= p50)
            assert_true(p50 <= p90)

    # Target 0 predictions should be near 25..45
    var t0_med = forecast.median(0, 0, 0)
    print("    Target 0 first step median:", t0_med)
    assert_true(t0_med > 10.0 and t0_med < 60.0)

    # Target 1 predictions should be near 60..90
    var t1_med = forecast.median(0, 1, 0)
    print("    Target 1 first step median:", t1_med)
    assert_true(t1_med > 40.0 and t1_med < 110.0)

    context.unsafe_free()
    forecast.free()
    forecaster.free()
    print("    ✓ TimestralForecaster zero-shot forecast and quantiles verified.")


def test_forecast_with_covariates() raises:
    """ Verify forecasting with past-only and known-future covariates."""
    print("  Testing TimestralForecaster with past and known covariates...")

    var synthetic_path = "/tmp/timestral_phase4_test.gguf"
    var forecaster = TimestralForecaster.from_gguf(synthetic_path)

    var t_hist = 64
    var h_pred = 32
    var n_targets = 1
    var n_past = 1
    var n_known = 1

    var context = unsafe_alloc[Float32](t_hist)
    var past = unsafe_alloc[Float32](t_hist)
    var known = unsafe_alloc[Float32](t_hist + h_pred)

    for t in range(t_hist):
        context[unsafe_offset=t] = 100.0 + Float32(t) * 0.2
        past[unsafe_offset=t] = Float32(t % 5)

    for t in range(t_hist + h_pred):
        known[unsafe_offset=t] = 10.0 if t < t_hist else 20.0

    var forecast = forecaster.forecast(
        context=context,
        t_context=t_hist,
        prediction_length=h_pred,
        batch_size=1,
        n_targets=n_targets,
        past_covariates=past,
        n_past_cov=n_past,
        known_covariates=known,
        n_known_cov=n_known,
    )

    assert_equal(forecast.horizon, 32)
    var med = forecast.median(0, 0, 0)
    print("    Covariate-informed forecast median:", med)
    assert_true(med > 50.0)

    context.unsafe_free()
    past.unsafe_free()
    known.unsafe_free()
    forecast.free()
    forecaster.free()
    print("    ✓ Forecast with past & known covariates verified.")


def test_causal_imputation() raises:
    """ Verify in-place missing value imputation."""
    print("  Testing TimestralForecaster causal imputation...")

    var synthetic_path = "/tmp/timestral_phase4_test.gguf"
    var forecaster = TimestralForecaster.from_gguf(synthetic_path)

    var t_total = 80
    var data = unsafe_alloc[Float32](t_total)

    # Create continuous series around 50.0
    for t in range(t_total):
        data[unsafe_offset=t] = 50.0 + Float32(t) * 0.1

    # Inject a missing gap of 10 steps from index 35 to 44
    var missing_val = nan[DType.float32]()
    for t in range(35, 45):
        data[unsafe_offset=t] = missing_val

    # Verify gap exists
    assert_true(isnan(data[unsafe_offset=35]))
    assert_true(isnan(data[unsafe_offset=44]))

    # Run imputation
    forecaster.impute(data, t_total=t_total, batch_size=1, n_targets=1)

    # Verify all NaNs are filled with valid numeric values
    for t in range(t_total):
        var v = data[unsafe_offset=t]
        assert_true(not isnan(v))

    var filled_sample = data[unsafe_offset=40]
    print("    Sample imputed value at t=40:", filled_sample)
    assert_true(filled_sample > 30.0 and filled_sample < 80.0)

    data.unsafe_free()
    forecaster.free()
    print("    ✓ Causal imputation verified successfully.")


def main() raises:
    print("=== Running Phase 4 Timestral Forecaster & API Tests ===")
    test_forecaster_basic_and_quantiles()
    test_forecast_with_covariates()
    test_causal_imputation()
    print("=== All Phase 4 Timestral Tests Passed! ===")
