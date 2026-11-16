""" Unit and benchmark test for Multithreaded Parallelization Across Variates and Tokens (`tests/test_timestral_parallel.mojo`)."""

from std.testing import assert_true, assert_equal
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memset_zero
from std.math import isnan, isinf
from std.time import perf_counter_ns

from llamojo.timestral import TimestralForecaster


def main() raises:
    print("=== Running Timestral Multithreaded Parallelization Test (Gap C) ===")

    var model_path = "/tmp/synth_f32.gguf"

    # 1. Load Forecaster with multi-variate capacity
    var forecaster = TimestralForecaster.from_gguf(model_path, max_rows=32, max_patches=128)
    print("  Model loaded successfully with threadpool worker count:", forecaster.state.n_workers)
    assert_true(forecaster.state.n_workers >= 1)

    # 2. Prepare multi-variate telemetry: 16 variates (sensors) across 128 context steps
    var n_variates = 16
    var t_hist = 128
    var h_pred = 32

    var context = unsafe_alloc[Float32](n_variates * t_hist)
    for v in range(n_variates):
        var base_val = 10.0 * Float32(v + 1)
        for t in range(t_hist):
            context[unsafe_offset=v * t_hist + t] = base_val + Float32(t) * 0.05

    print("  Evaluating multi-variate forecast across", n_variates, "variates in parallel...")
    var t0 = perf_counter_ns()
    var res = forecaster.forecast(
        context=context,
        t_context=t_hist,
        prediction_length=h_pred,
        batch_size=1,
        n_targets=n_variates,
    )
    var t1 = perf_counter_ns()
    var elapsed_ms = Float64(t1 - t0) / 1_000_000.0

    print("  ✓ Multi-variate forecast finished in", elapsed_ms, "ms.")
    assert_equal(res.batch_size, 1)
    assert_equal(res.n_targets, n_variates)
    assert_equal(res.horizon, h_pred)

    # Verify predictions across all 16 variates
    for v in range(n_variates):
        for h in range(h_pred):
            var med = res.median(0, v, h)
            if isnan(med) or isinf(med):
                raise Error("Non-finite median detected in variate output")
            var p10 = res.quantile(0, v, h, 0.10)
            var p90 = res.quantile(0, v, h, 0.90)
            if isnan(p10) or isnan(p90):
                raise Error("Non-finite quantiles detected in variate output")
            if p10 > med or med > p90:
                raise Error("Quantile monotonicity violated in parallel variate execution")

    print("  ✓ Non-crossing quantile monotonicity verified for all 16 variates.")

    # 3. Test multi-variate causal imputation
    print("  Testing parallelized causal imputation across 16 variates...")
    # Inject missing gaps in variates 0..3
    for v in range(4):
        var v_ptr = context.unsafe_offset(v * t_hist)
        for t in range(50, 65):
            v_ptr[unsafe_offset=t] = Float32(0.0) / Float32(0.0)

    forecaster.impute(context, t_hist, 1, n_variates)

    for v in range(4):
        var v_ptr = context.unsafe_offset(v * t_hist)
        for t in range(50, 65):
            var val = v_ptr[unsafe_offset=t]
            if isnan(val) or isinf(val):
                raise Error("Imputation failed in multi-variate series")

    print("  ✓ Parallel multi-variate causal imputation verified.")

    res.free()
    context.unsafe_free()
    print("=== Timestral Multithreaded Parallelization Tests Passed! ===")
