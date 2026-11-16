""" Stage 1 Unit Tests: Timestral GGUF v3 Metadata Parsing and Tensor Weight Binding.

Verifies:
1. GGUF v3 metadata parsing into `TimestralConfig` (hyperparameters, block_schedule, 21 quantiles).
2. Tensor directory inspection and zero-copy pointer bindings via `load_timestral_weights`.
3. Validation of alternating time / time_variate attention layers.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.ffi import external_call
from llamojo import (
    gguf_load_from_file,
    GGUFContext,
    TimestralConfig,
    TimestralModelWeights,
    load_timestral_weights,
)
from llamojo.gguf import _cstr_alloc


def test_timestral_config_defaults() raises:
    """ Test standard default configuration parameters."""
    print("  Testing TimestralConfig default initialization...")
    var cfg = TimestralConfig()
    assert_equal(cfg.arch, "timestral")
    assert_equal(cfg.patch_size, 32)
    assert_equal(cfg.d_model, 384)
    assert_equal(cfg.num_heads, 6)
    assert_equal(cfg.head_dim, 64)
    assert_equal(cfg.mlp_hidden_dim, 1024)
    assert_equal(cfg.num_layers, 12)
    assert_equal(cfg.context_length, 32768)
    assert_almost_equal(cfg.rope_theta, 10000.0, atol=1e-4)
    assert_almost_equal(cfg.rope_partial_factor, 0.5, atol=1e-4)
    assert_true(cfg.use_asinh)
    assert_true(cfg.split_context_horizon_attn)
    assert_equal(cfg.num_quantiles(), 21)
    assert_almost_equal(cfg.quantile_knots[0], 0.01, atol=1e-4)
    assert_almost_equal(cfg.quantile_knots[10], 0.50, atol=1e-4)
    assert_almost_equal(cfg.quantile_knots[20], 0.99, atol=1e-4)
    print("    ✓ Default TimestralConfig values verified.")


def test_timestral_gguf_loading() raises:
    """ Test loading and parsing a synthetic Timestral GGUF model."""
    var path = "/tmp/timestral_synthetic.gguf"
    print("  Ensuring synthetic GGUF model exists at: " + path + "...")
    
    # Ensure fresh synthetic test model is generated via conversion tool
    var cmd = _cstr_alloc("/home/tasahi/miniconda/envs/moj/bin/python tools/convert_timestral_to_gguf.py --synthetic --out " + path + " > /dev/null 2>&1")
    _ = external_call["system", Int32](cmd)

    print("  Testing GGUF loading from: " + path + "...")
    var ctx = gguf_load_from_file(path, use_mmap=True)
    assert_true(ctx.version >= 2)
    assert_equal(ctx.get_string("general.architecture", ""), "timestral")

    # 1. Parse Config
    var cfg = TimestralConfig.from_gguf(ctx)
    assert_equal(cfg.arch, "timestral")
    assert_equal(cfg.patch_size, 32)
    assert_equal(cfg.d_model, 64)
    assert_equal(cfg.num_heads, 2)
    assert_equal(cfg.head_dim, 32)
    assert_equal(cfg.mlp_hidden_dim, 128)
    assert_equal(cfg.num_layers, 2)
    assert_equal(cfg.context_length, 512)
    assert_true(cfg.use_asinh)
    assert_true(cfg.split_context_horizon_attn)

    # 2. Verify Block Schedule
    assert_equal(len(cfg.block_schedule), 2)
    assert_equal(cfg.block_schedule[0], "time")
    assert_equal(cfg.block_schedule[1], "time_variate")
    assert_true(not cfg.has_variate_attn(0))
    assert_true(cfg.has_variate_attn(1))
    assert_equal(cfg.num_quantiles(), 21)

    # 3. Load & Verify Weights Binding
    var weights = load_timestral_weights(ctx, cfg)
    assert_equal(weights.n_tensors, 29)
    assert_true(weights.total_bytes > 0)

    # Patch projection tensors
    assert_true(Int(weights.patch_proj.hidden_w) != 0)
    assert_true(Int(weights.patch_proj.hidden_b) != 0)
    assert_true(Int(weights.patch_proj.output_w) != 0)
    assert_true(Int(weights.patch_proj.output_b) != 0)
    assert_true(Int(weights.patch_proj.residual_w) != 0)
    assert_true(Int(weights.patch_proj.residual_b) != 0)

    # Forecast query token
    assert_true(Int(weights.forecast_query) != 0)

    # Layers
    assert_equal(len(weights.layers), 2)

    # Layer 0 (time-only)
    assert_true(not weights.layers[0].has_variate_attn)
    assert_true(Int(weights.layers[0].time_attn.norm_w) != 0)
    assert_true(Int(weights.layers[0].time_attn.gate_w) != 0)
    assert_true(Int(weights.layers[0].time_attn.qkv_w) != 0)
    assert_true(Int(weights.layers[0].time_attn.o_w) != 0)
    assert_true(Int(weights.layers[0].ffn_norm_w) != 0)
    assert_true(Int(weights.layers[0].ffn_gate_up_w) != 0)
    assert_true(Int(weights.layers[0].ffn_down_w) != 0)

    # Layer 1 (time_variate: has both time and variate attention)
    assert_true(weights.layers[1].has_variate_attn)
    assert_true(Int(weights.layers[1].time_attn.norm_w) != 0)
    assert_true(Int(weights.layers[1].variate_attn.norm_w) != 0)
    assert_true(Int(weights.layers[1].variate_attn.gate_w) != 0)
    assert_true(Int(weights.layers[1].variate_attn.qkv_w) != 0)
    assert_true(Int(weights.layers[1].variate_attn.o_w) != 0)
    assert_true(Int(weights.layers[1].ffn_norm_w) != 0)
    assert_true(Int(weights.layers[1].ffn_gate_up_w) != 0)
    assert_true(Int(weights.layers[1].ffn_down_w) != 0)

    # Final norm & Quantile head
    assert_true(Int(weights.out_norm_w) != 0)
    assert_true(Int(weights.head.hidden_w) != 0)
    assert_true(Int(weights.head.output_w) != 0)
    assert_true(Int(weights.head.residual_w) != 0)

    ctx.free()
    print("    ✓ GGUF v3 Timestral config and weights verified successfully.")


def main() raises:
    print("=== Running Stage 1 Timestral GGUF Tests ===")
    test_timestral_config_defaults()
    test_timestral_gguf_loading()
    print("=== All Stage 1 Timestral Tests Passed! ===")
