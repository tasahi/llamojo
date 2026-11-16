""" Unit and validation test for Timestral Quantized Weights (Q8_0) in GGUF (`tests/test_timestral_quant.mojo`)."""

from std.testing import assert_true, assert_equal
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memset_zero
from std.math import isnan, isinf
from std.ffi import external_call

from llamojo.types import GGMLType
from llamojo.gguf import _cstr_alloc
from llamojo.timestral import TimestralForecaster


def main() raises:
    print("=== Running Timestral Quantized Weights (Q8_0) Engine Test ===")

    var q8_model_path = "/tmp/synth_q8_cli.gguf"

    # 1. Generate Q8_0 synthetic model via convert_timestral_to_gguf.py if needed
    var py_cmd = _cstr_alloc(
        "/home/tasahi/miniconda/envs/moj/bin/python tools/convert_timestral_to_gguf.py"
        + " --synthetic --out " + q8_model_path + " --precision q8_0"
    )
    var ret = external_call["system", Int32](py_cmd)
    py_cmd.unsafe_free()
    if ret != 0:
        raise Error("Failed to execute python converter for Q8_0 model")

    # 2. Load Forecaster from Q8_0 GGUF
    print("  Loading TimestralForecaster from Q8_0 GGUF...")
    var forecaster = TimestralForecaster.from_gguf(q8_model_path)
    assert_equal(forecaster.config.patch_size, 32)
    assert_equal(forecaster.config.d_model, 64)
    assert_equal(forecaster.config.num_quantiles(), 21)

    # Validate that linear projections are indeed Q8_0
    var pp = forecaster.weights.patch_proj
    assert_equal(pp.dtype_hidden, GGMLType.Q8_0)
    assert_equal(pp.dtype_output, GGMLType.Q8_0)
    assert_equal(pp.dtype_residual, GGMLType.Q8_0)

    var l0 = forecaster.weights.layers[0]
    assert_equal(l0.time_attn.dtype_qkv, GGMLType.Q8_0)
    assert_equal(l0.time_attn.dtype_gate, GGMLType.Q8_0)
    assert_equal(l0.time_attn.dtype_o, GGMLType.Q8_0)
    assert_equal(l0.dtype_gate_up, GGMLType.Q8_0)
    assert_equal(l0.dtype_down, GGMLType.Q8_0)

    var head = forecaster.weights.head
    assert_equal(head.dtype_hidden, GGMLType.Q8_0)
    assert_equal(head.dtype_output, GGMLType.Q8_0)
    assert_equal(head.dtype_residual, GGMLType.Q8_0)
    print("  ✓ All linear projections verified as GGMLType.Q8_0 (dtype=8).")

    # 3. Test zero-shot forecast on continuous sequence
    print("  Testing zero-shot forecast with Q8_0 model...")
    var t_hist = 64
    var h_pred = 32
    var context = unsafe_alloc[Float32](t_hist)
    for t in range(t_hist):
        context[unsafe_offset=t] = 20.0 + Float32(t) * 0.15

    var res = forecaster.forecast(context, t_hist, h_pred)
    assert_equal(res.batch_size, 1)
    assert_equal(res.n_targets, 1)
    assert_equal(res.horizon, h_pred)

    for h in range(h_pred):
        var med = res.median(0, 0, h)
        if isnan(med) or isinf(med):
            raise Error("Forecast median is non-finite at horizon step")
        var p10 = res.quantile(0, 0, h, 0.10)
        var p90 = res.quantile(0, 0, h, 0.90)
        if isnan(p10) or isnan(p90):
            raise Error("Quantiles are non-finite at horizon step")
        # Ensure monotonic order
        if p10 > med or med > p90:
            raise Error("Monotonic quantile property violated in Q8_0 forecast")

    print("  ✓ Q8_0 forecast non-crossing quantiles verified. Median at h=0:", res.median(0, 0, 0))
    res.free()
    context.unsafe_free()

    # 4. Test missing data causal imputation with Q8_0
    print("  Testing causal missing data imputation with Q8_0 model...")
    var t_total = 100
    var imp_data = unsafe_alloc[Float32](t_total)
    for t in range(t_total):
        imp_data[unsafe_offset=t] = 50.0 + Float32(t) * 0.05

    # Introduce gap of 10 NaNs
    var nan_val = Float32(0.0) / Float32(0.0)
    for t in range(40, 50):
        imp_data[unsafe_offset=t] = nan_val

    forecaster.impute(imp_data, t_total, 1, 1)

    for t in range(40, 50):
        var val = imp_data[unsafe_offset=t]
        if isnan(val) or isinf(val):
            raise Error("Imputation failed to fill NaN gap with finite number")

    print("  ✓ Q8_0 imputation successfully filled missing gap. Sample imputed val:", imp_data[unsafe_offset=45])
    imp_data.unsafe_free()

    print("=== Timestral Quantized Weights (Q8_0) Tests Passed! ===")
