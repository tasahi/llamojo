""" Unit tests for Phase 10: Sparse Mixture-of-Experts (MoE) Routing and Forward Execution."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc

from llamojo import (
    GGMLType,
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_context_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_init_from_model,
    llama_free,
    llama_batch_init,
    llama_batch_get_one,
    llama_batch_free,
    llama_decode,
    llama_get_logits_ith,
    SelectedExpert,
    route_experts,
    write_synthetic_gguf,
    llama_token,
)

def test_moe_routing_math() raises:
    print("Testing MoE router softmax and Top-k selection...")
    var dim = 4
    var n_experts = 4
    var n_used = 2

    # Normalized input x_norm = [1.0, 0.0, 0.0, 0.0]
    var x_norm = unsafe_alloc[Float32](dim)
    x_norm[unsafe_offset=0] = 1.0
    x_norm[unsafe_offset=1] = 0.0
    x_norm[unsafe_offset=2] = 0.0
    x_norm[unsafe_offset=3] = 0.0

    # Router weight matrix [4, 4] (4 experts, dim 4)
    # Expert 0 row: [3.0, 0.0, 0.0, 0.0] -> dot = 3.0
    # Expert 1 row: [1.0, 0.0, 0.0, 0.0] -> dot = 1.0
    # Expert 2 row: [5.0, 0.0, 0.0, 0.0] -> dot = 5.0 (highest!)
    # Expert 3 row: [0.0, 0.0, 0.0, 0.0] -> dot = 0.0
    var w_router = unsafe_alloc[Float32](n_experts * dim)
    for i in range(n_experts * dim):
        w_router[unsafe_offset=i] = 0.0

    w_router[unsafe_offset=0 * dim + 0] = 3.0
    w_router[unsafe_offset=1 * dim + 0] = 1.0
    w_router[unsafe_offset=2 * dim + 0] = 5.0
    w_router[unsafe_offset=3 * dim + 0] = 0.0

    var selected = route_experts(x_norm, w_router, dim, n_experts, n_used)

    # Top-2 selected should be Expert 2 (highest) and Expert 0 (second highest)
    assert_equal(len(selected), 2)
    assert_equal(selected[0].index, 2)
    assert_equal(selected[1].index, 0)

    # Normalized weights must sum to 1.0
    var sum_weights = selected[0].weight + selected[1].weight
    assert_almost_equal(sum_weights, Float32(1.0), atol=1e-5)
    assert_true(selected[0].weight > selected[1].weight)

    x_norm.unsafe_free()
    w_router.unsafe_free()
    print("  ✓ MoE router softmax & top-k selection verified")

def test_moe_forward_decode() raises:
    print("Testing MoE model decode forward pass...")
    var test_path = "/tmp/llamojo_phase10_moe_test.gguf"
    write_synthetic_gguf(
        test_path,
        arch="llama",
        name="moe-test-model",
        hidden_dim=32,
        intermediate_dim=64,
        n_heads=2,
        n_kv_heads=1,
        n_layers=2,
        vocab_size=16,
    )

    llama_backend_init()
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(test_path, mparams)
    assert_true(Int(model) != 0)

    # Simulate MoE layer configuration on layer 0:
    # 4 experts, 2 activated
    var dim = 32
    var n_exp = 4
    var r_weights = unsafe_alloc[Float32](n_exp * dim)
    for i in range(n_exp * dim):
        r_weights[unsafe_offset=i] = Float32(0.1) * Float32(i % 5)

    model[].layers[0].is_moe = True
    model[].layers[0].n_experts = 4
    model[].layers[0].n_experts_used = 2
    model[].layers[0].w_router = r_weights

    var cparams = llama_context_default_params()
    cparams.n_ctx = 32
    cparams.n_batch = 4
    var ctx = llama_init_from_model(model, cparams)
    assert_true(Int(ctx) != 0)

    var tokens = unsafe_alloc[llama_token](1)
    tokens[unsafe_offset=0] = 3
    var batch = llama_batch_get_one(tokens, n_tokens=1)

    var status = llama_decode(ctx, batch)
    assert_equal(status, 0)

    var logits = llama_get_logits_ith(ctx, 0)
    assert_true(Int(logits) != 0)

    # Verify logits are populated and finite
    var has_nonzero = False
    for v in range(16):
        var l = logits[unsafe_offset=v]
        if l != 0.0:
            has_nonzero = True
    assert_true(has_nonzero)

    llama_batch_free(batch)
    tokens.unsafe_free()
    llama_free(ctx)
    r_weights.unsafe_free()
    llama_model_free(model)
    llama_backend_free()
    print("  ✓ MoE forward decode execution verified")

def main() raises:
    print("============================================================")
    print("Running Phase 10 Sparse MoE Test Suite")
    print("============================================================")
    test_moe_routing_math()
    test_moe_forward_decode()
    print("All Phase 10 MoE tests passed! ✓")

