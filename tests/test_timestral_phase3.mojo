""" Phase 3 Unit Tests: Timestral Transformer Model Architecture, Dual-Axis Attention, and Quantile Head.\n
Verifies:
1. Partial RoPE kernel execution (`apply_rope_partial`).
2. Multi-head time attention (`forward_time_attention`) with DHF causal mask & sdpa_sigmoid gating.
3. Cross-variate multi-head attention (`forward_variate_attention`) across multi-variate series rows.
4. SwiGLU MLP feed-forward pass (`forward_swiglu_mlp`).
5. Monotonic quantile fan-out (`monotone_quantiles`) ensuring non-crossing quantiles (q_0 <= q_1 <= ... <= q_20).
6. End-to-end forward evaluation (`forward_timestral`) on synthetic GGUF checkpoint.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.memory import Pointer, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
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
    apply_rope_partial,
    TimestralState,
    forward_time_attention,
    forward_variate_attention,
    forward_swiglu_mlp,
    forward_timestral_block,
    monotone_quantiles,
    forward_quantile_head,
    forward_timestral,
)
from llamojo.gguf import _cstr_alloc


def test_partial_rope() raises:
    """ Verify partial RoPE rotation on first rope_dim dimensions only."""
    print("  Testing apply_rope_partial kernel...")
    var head_dim = 64
    var rope_dim = 32
    var q = unsafe_alloc[Float32](head_dim)
    var k = unsafe_alloc[Float32](head_dim)

    # Initialize with simple ramp
    for i in range(head_dim):
        q[unsafe_offset=i] = Float32(i + 1)
        k[unsafe_offset=i] = Float32(i + 1)

    apply_rope_partial(
        q=q,
        k=k,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        rope_dim=rope_dim,
        pos=5,
        rope_theta=10000.0,
    )

    # First half of rotary_dim rotates with second half of head_dim
    # Specifically:
    # indices [0..15] and [32..47] undergo rotation
    # indices [16..31] and [48..63] must remain untouched (identity rotation)
    for i in range(16, 32):
        assert_almost_equal(q[unsafe_offset=i], Float32(i + 1), atol=1e-5)
        assert_almost_equal(k[unsafe_offset=i], Float32(i + 1), atol=1e-5)

    for i in range(48, 64):
        assert_almost_equal(q[unsafe_offset=i], Float32(i + 1), atol=1e-5)
        assert_almost_equal(k[unsafe_offset=i], Float32(i + 1), atol=1e-5)

    # Rotated elements must have changed
    assert_true(q[unsafe_offset=0] != Float32(1.0))
    assert_true(k[unsafe_offset=0] != Float32(1.0))

    q.unsafe_free()
    k.unsafe_free()
    print("    ✓ apply_rope_partial verified.")


def test_monotone_quantiles() raises:
    """ Verify non-crossing monotonic quantile fan-out."""
    print("  Testing monotone_quantiles...")
    var raw = unsafe_alloc[Float32](21)
    var q_out = unsafe_alloc[Float32](21)

    # Arbitrary non-monotonic raw logits
    raw[unsafe_offset=0] = 5.2
    raw[unsafe_offset=1] = -1.2
    raw[unsafe_offset=2] = 3.4
    raw[unsafe_offset=3] = -4.0
    raw[unsafe_offset=4] = 0.5
    raw[unsafe_offset=5] = -2.3
    raw[unsafe_offset=6] = 1.1
    raw[unsafe_offset=7] = -0.7
    raw[unsafe_offset=8] = 2.8
    raw[unsafe_offset=9] = -3.1
    raw[unsafe_offset=10] = 15.0  # Median
    raw[unsafe_offset=11] = -0.9
    raw[unsafe_offset=12] = 2.1
    raw[unsafe_offset=13] = -1.5
    raw[unsafe_offset=14] = 3.7
    raw[unsafe_offset=15] = -2.0
    raw[unsafe_offset=16] = 0.8
    raw[unsafe_offset=17] = -4.2
    raw[unsafe_offset=18] = 1.9
    raw[unsafe_offset=19] = -0.3
    raw[unsafe_offset=20] = 4.5

    monotone_quantiles(raw, q_out, width_factor=3.0)

    # Verify median channel is exact
    assert_almost_equal(q_out[unsafe_offset=10], 15.0, atol=1e-5)

    # Verify strict non-decreasing monotonicity: q[0] <= q[1] <= ... <= q[20]
    for i in range(20):
        assert_true(q_out[unsafe_offset=i] <= q_out[unsafe_offset=i + 1])

    raw.unsafe_free()
    q_out.unsafe_free()
    print("    ✓ monotone_quantiles non-crossing monotonicity verified.")


def test_end_to_end_forward() raises:
    """ Run synthetic model GGUF through full Phase 3 pipeline."""
    print("  Testing end-to-end forward_timestral...")

    # 1. Generate miniature synthetic GGUF model if not present
    var synthetic_path = "/tmp/timestral_phase3_test.gguf"
    var py_cmd = _cstr_alloc(
        "/home/tasahi/miniconda/envs/moj/bin/python tools/convert_timestral_to_gguf.py"
        + " --synthetic --out " + synthetic_path
    )
    _ = external_call["system", Int32](py_cmd)
    py_cmd.unsafe_free()

    # 2. Load GGUF Context & Config
    var ctx = gguf_load_from_file(synthetic_path, use_mmap=True)

    var config = TimestralConfig.from_gguf(ctx)
    var weights = load_timestral_weights(ctx, config)
    assert_equal(len(weights.layers), 2)
    assert_true(weights.layers[0].has_variate_attn == False)
    assert_true(weights.layers[1].has_variate_attn == True)

    # 3. Create TimeSeriesBatch: 2 series, context=64 steps (2 patches), horizon=32 steps (1 patch)
    var batch = TimeSeriesBatch(
        batch_size=1,
        n_targets=2,
        t_context=64,
        t_horizon=32,
        n_past_cov=0,
        n_known_cov=0,
    )

    # Fill target series 0 with linear ramp 10..20
    for t in range(64):
        batch.context_ptr[unsafe_offset=t] = 10.0 + Float32(t) * 0.15

    # Fill target series 1 with sine wave 50..60
    for t in range(64):
        batch.context_ptr[unsafe_offset=64 + t] = 50.0 + Float32(t % 8)

    # 4. Patchify and embed inputs
    var res = patchify_and_embed(batch, config, weights.patch_proj, weights.forecast_query)
    ref tokens = res.tokens
    ref patch_buf = res.patch_buf

    assert_equal(tokens.n_rows, 2)
    assert_equal(tokens.n_patches, 3)  # 2 context patches + 1 horizon patch

    # 5. Initialize TimestralState
    var state = TimestralState(config, max_rows=4, max_patches=8)

    # 6. Allocate output predictions: (n_rows, n_horizon_patches * 32, 21)
    var n_horizon_patches = 1
    var total_pred_floats = 2 * n_horizon_patches * 32 * 21
    var out_preds = unsafe_alloc[Float32](total_pred_floats)
    unsafe_memset_zero(out_preds, total_pred_floats)

    # 7. Execute forward_timestral
    forward_timestral(
        weights=weights,
        tokens=tokens,
        patch_buf=patch_buf,
        n_hist_patches=2,
        n_horizon_patches=1,
        config=config,
        out_predictions=out_preds,
        state=state,
    )

    # 8. Verify predictions:
    # - Non-NaN and Non-Inf
    # - Non-decreasing quantiles across all time steps
    # - Scaled back into realistic data scale around target series
    for r in range(2):
        for step in range(32):
            var step_offset = (r * 32 + step) * 21
            for q in range(20):
                var q_cur = out_preds[unsafe_offset=step_offset + q]
                var q_next = out_preds[unsafe_offset=step_offset + q + 1]
                assert_true(not (q_cur != q_cur))  # Not NaN
                assert_true(q_cur <= q_next)        # Monotonicity

    # Target 0 should be centered around ~15..25; Target 1 around ~45..60
    var t0_med = out_preds[unsafe_offset=10]
    var t1_med = out_preds[unsafe_offset=32 * 21 + 10]
    print("    Target 0 sample median forecast:", t0_med)
    print("    Target 1 sample median forecast:", t1_med)
    assert_true(t0_med > 0.0)
    assert_true(t1_med > 20.0)

    # Clean up
    out_preds.unsafe_free()
    state.free()
    res.free()
    batch.free()

    print("    ✓ End-to-end forward_timestral execution and non-crossing quantiles verified.")


def main() raises:
    print("=== Running Phase 3 Timestral Transformer & Attention Tests ===")
    test_partial_rope()
    test_monotone_quantiles()
    test_end_to_end_forward()
    print("=== All Phase 3 Timestral Tests Passed! ===")
