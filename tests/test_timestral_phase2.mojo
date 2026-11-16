""" Phase 2 Unit Tests: Timestral Time-Series Patchification, Causal Normalization, and Missingness Processing.

Verifies:
1. Reversible `asinh` scaling and inversion (`apply_affine_asinh` and `invert_affine_asinh`).
2. Causal Welford patch statistics computation (`CausalPatchStats`).
3. Multi-variate input batch handling (`TimeSeriesBatch`) with missing values (NaN).
4. ResidualBlock patch projection evaluation (`forward_patch_proj`).
5. End-to-end `patchify_and_embed` pipeline including horizon query token injection.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.math import nan
from std.ffi import external_call

from llamojo import (
    gguf_load_from_file,
    GGUFContext,
    TimestralConfig,
    TimestralModelWeights,
    load_timestral_weights,
    TimeSeriesBatch,
    CausalPatchStats,
    PatchBuffer,
    EmbeddedTokens,
    PatchifyResult,
    apply_affine_asinh,
    invert_affine_asinh,
    forward_patch_proj,
    patchify_and_embed,
)
from llamojo.gguf import _cstr_alloc


def test_reversible_asinh() raises:
    """ Test reversible affine asinh scaling and inversion."""
    print("  Testing reversible asinh transformation...")
    var loc: Float32 = 10.0
    var scale: Float32 = 2.5

    # Test observed value round-trip
    var orig: Float32 = 15.0
    var scaled = apply_affine_asinh(orig, loc, scale, is_obs=True, use_asinh=True)
    var restored = invert_affine_asinh(scaled, loc, scale, use_asinh=True)
    assert_almost_equal(restored, orig, atol=1e-4)

    # Test negative value
    var orig_neg: Float32 = 3.0
    var scaled_neg = apply_affine_asinh(orig_neg, loc, scale, is_obs=True, use_asinh=True)
    var restored_neg = invert_affine_asinh(scaled_neg, loc, scale, use_asinh=True)
    assert_almost_equal(restored_neg, orig_neg, atol=1e-4)

    # Test missing / unobserved value
    var unobs = apply_affine_asinh(999.0, loc, scale, is_obs=False, use_asinh=True)
    assert_equal(unobs, 0.0)

    # Test linear mode (use_asinh = False)
    var scaled_lin = apply_affine_asinh(orig, loc, scale, is_obs=True, use_asinh=False)
    var restored_lin = invert_affine_asinh(scaled_lin, loc, scale, use_asinh=False)
    assert_almost_equal(restored_lin, orig, atol=1e-4)
    print("    ✓ Reversible asinh scaling and inversion verified.")


def test_causal_patch_stats() raises:
    """Test causal running Welford statistics computation along patches."""
    print("  Testing causal patch statistics...")
    var ps = 32
    var t_len = 64  # exactly 2 patches
    var stats = CausalPatchStats(2)

    var series = TimeSeriesBatch(1, 1, t_len, 0)
    # Patch 0: values from 1.0 to 32.0 (mean = 16.5)
    for i in range(32):
        series.set_target(0, 0, i, Float32(i + 1))
    # Patch 1: constant 100.0
    for i in range(32, 64):
        series.set_target(0, 0, i, 100.0)

    stats.compute(series.context_ptr, t_len, pad_len=0, patch_size=ps, minimum_scale=1e-6)

    # Patch 0 stats
    var loc0 = stats.loc[unsafe_offset=0]
    var scale0 = stats.scale[unsafe_offset=0]
    assert_almost_equal(loc0, 16.5, atol=1e-3)
    assert_true(scale0 > 0.0)

    # Patch 1 stats: causal accumulation of all 64 steps
    var loc1 = stats.loc[unsafe_offset=1]
    assert_almost_equal(loc1, 58.25, atol=1e-2)
    assert_true(loc1 > loc0)

    stats.free()
    series.free()
    print("    ✓ Causal patch statistics verified.")


def test_time_series_batch_container() raises:
    """ Test TimeSeriesBatch container with targets, covariates, and NaNs."""
    print("  Testing TimeSeriesBatch container...")
    var batch = TimeSeriesBatch(
        batch_size=1,
        n_targets=2,
        t_context=100,
        t_horizon=32,
        n_past_cov=1,
        n_known_cov=1,
    )

    assert_equal(batch.batch_size, 1)
    assert_equal(batch.n_targets, 2)
    assert_equal(batch.t_context, 100)
    assert_equal(batch.t_horizon, 32)
    assert_equal(batch.n_past_cov, 1)
    assert_equal(batch.n_known_cov, 1)
    assert_equal(batch.total_rows(), 4)

    # Set values
    batch.set_target(0, 0, 10, 42.0)
    assert_equal(batch.get_target(0, 0, 10), 42.0)
    assert_true(batch.is_observed(42.0))

    # Test NaN detection
    var missing_val = nan[DType.float32]()
    batch.set_target(0, 1, 15, missing_val)
    assert_true(not batch.is_observed(batch.get_target(0, 1, 15)))

    # Set past and known covariates
    batch.set_past_cov(0, 0, 5, 12.5)
    assert_equal(batch.get_past_cov(0, 0, 5), 12.5)

    batch.set_known_cov(0, 0, 120, 99.0)  # Future horizon step
    assert_equal(batch.get_known_cov(0, 0, 120), 99.0)

    batch.free()
    print("    ✓ TimeSeriesBatch container and NaN checks verified.")


def test_patchify_and_embed_pipeline() raises:
    """ Test end-to-end patchification, missingness encoding, and ResidualBlock projection."""
    print("  Testing end-to-end patchify_and_embed pipeline...")
    var path = "/tmp/timestral_synthetic.gguf"
    var cmd = _cstr_alloc("/home/tasahi/miniconda/envs/moj/bin/python tools/convert_timestral_to_gguf.py --synthetic --out " + path + " > /dev/null 2>&1")
    _ = external_call["system", Int32](cmd)

    var ctx = gguf_load_from_file(path, use_mmap=True)
    var cfg = TimestralConfig.from_gguf(ctx)
    var weights = load_timestral_weights(ctx, cfg)

    # Create batch: T=50 context (padded to 64 = 2 patches), H=32 horizon (1 patch) -> total 3 patches
    var batch = TimeSeriesBatch(
        batch_size=1,
        n_targets=1,
        t_context=50,
        t_horizon=32,
        n_past_cov=1,
        n_known_cov=1,
    )

    # Fill context with values and some NaNs
    for t in range(50):
        batch.set_target(0, 0, t, Float32(t + 10))
        batch.set_past_cov(0, 0, t, Float32(t * 2))
    # Inject missing values (NaN)
    var missing_val = nan[DType.float32]()
    batch.set_target(0, 0, 20, missing_val)
    batch.set_target(0, 0, 21, missing_val)

    # Fill known covariate spanning T+H = 82 steps
    for t in range(82):
        batch.set_known_cov(0, 0, t, Float32(t % 7))

    # Execute patchify and embed
    var res = patchify_and_embed(batch, cfg, weights.patch_proj, weights.forecast_query)
    ref tokens = res.tokens
    ref patch_buf = res.patch_buf

    # Verify dimensions: 3 rows (target + past + known), 3 total patches (2 hist + 1 horizon)
    assert_equal(tokens.n_rows, 3)
    assert_equal(tokens.n_patches, 3)
    assert_equal(tokens.d_model, cfg.d_model)

    assert_equal(patch_buf.n_rows, 3)
    assert_equal(patch_buf.n_patches, 3)
    assert_equal(patch_buf.in_dim, 64)

    # Verify missingness mask on patch 0:
    # pad_len = 64 - 50 = 14 leading padding positions
    var p0_vec = patch_buf.patch_ptr(0, 0)
    for k in range(14):
        # Leading padded steps must be marked as missing (channel 2 = 1.0)
        assert_equal(p0_vec[unsafe_offset=32 + k], 1.0)
        # Scaled value must be 0.0
        assert_equal(p0_vec[unsafe_offset=k], 0.0)

    # Verify that unpadded observed step is marked observed (channel 2 = 0.0)
    assert_equal(p0_vec[unsafe_offset=32 + 15], 0.0)

    # Verify horizon patch (patch 2) for target variate (row 0):
    # Should equal the learned forecast_query token!
    var horizon_tok = tokens.token_ptr(0, 2)
    for d in range(cfg.d_model):
        var expected = weights.forecast_query[unsafe_offset=d]
        assert_almost_equal(horizon_tok[unsafe_offset=d], expected, atol=1e-5)

    # Verify embedded context tokens are non-zero
    var ctx_tok = tokens.token_ptr(0, 0)
    var has_nonzero = False
    for d in range(cfg.d_model):
        if ctx_tok[unsafe_offset=d] != 0.0:
            has_nonzero = True
            break
    assert_true(has_nonzero)

    res.free()
    batch.free()
    ctx.free()
    print("    ✓ End-to-end patchify_and_embed pipeline verified successfully.")


def main() raises:
    print("=== Running Phase 2 Timestral Preprocessing Tests ===")
    test_reversible_asinh()
    test_causal_patch_stats()
    test_time_series_batch_container()
    test_patchify_and_embed_pipeline()
    print("=== All Phase 2 Timestral Tests Passed! ===")
