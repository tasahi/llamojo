""" Unit tests for Phase 8: Dynamic LoRA Adapter Ingestion and Online Merging."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc

from llamojo import (
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_model_load_from_file,
    llama_model_free,
    write_synthetic_gguf,
    write_synthetic_lora_gguf,
    llama_lora_adapter,
    llama_lora_adapter_init,
    llama_lora_adapter_set,
    llama_lora_adapter_free,
    compute_lora_delta,
    apply_lora_to_f32_tensor,
    llama_adapter_lora_apply,
    null_ptr,
)

def test_lora_adapter_loading_and_math() raises:
    print("Testing LoRA adapter parsing and Delta W computation...")
    var lora_path = "/tmp/test_adapter.gguf"
    # in_features = 4, out_features = 2, rank = 2, alpha = 4.0
    write_synthetic_lora_gguf(
        lora_path,
        base_name="blk.0.attn_q.weight",
        in_features=4,
        out_features=2,
        rank=2,
        alpha=4.0,
    )

    var adapter = llama_lora_adapter(lora_path)
    assert_almost_equal(adapter.alpha, Float32(4.0), atol=1e-5)
    assert_equal(adapter.rank, 2)
    assert_equal(len(adapter.pairs), 1)

    var pair = adapter.pairs[0]
    assert_equal(pair.name, "blk.0.attn_q.weight")
    assert_equal(pair.a_rows, 2)  # rank
    assert_equal(pair.a_cols, 4)  # in_features
    assert_equal(pair.b_rows, 2)  # out_features
    assert_equal(pair.b_cols, 2)  # rank

    # Test Delta computation:
    # A is filled with 1.0 (shape 2x4)
    # B is filled with 0.5 (shape 2x2)
    # Each element of B @ A is: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    # multiplier = (alpha / rank) * scale = (4.0 / 2) * 1.0 = 2.0
    # So every element in delta should be 1.0 * 2.0 = 2.0
    var delta = unsafe_alloc[Float32](8)
    compute_lora_delta(pair, delta, multiplier=2.0)
    for i in range(8):
        assert_almost_equal(delta[unsafe_offset=i], Float32(2.0), atol=1e-5)
    delta.unsafe_free()

    # Test in-place additive application to a dummy weight tensor
    var target = unsafe_alloc[Float32](8)
    for i in range(8):
        target[unsafe_offset=i] = Float32(10.0)

    apply_lora_to_f32_tensor(target, 8, pair, alpha=4.0, rank=2, scale=1.0)
    for i in range(8):
        assert_almost_equal(target[unsafe_offset=i], Float32(12.0), atol=1e-5)
    target.unsafe_free()

    adapter.free()
    print("  ✓ LoRA adapter parsing and math verified")

def test_lora_online_model_merging() raises:
    print("Testing online LoRA model merging into loaded llama_model...")
    var model_path = "/tmp/test_lora_base_model.gguf"
    var lora_path = "/tmp/test_lora_layer_adapter.gguf"

    # Base model with hidden_dim=32, intermediate_dim=64, 2 layers
    write_synthetic_gguf(
        model_path,
        arch="llama",
        name="lora-test-model",
        hidden_dim=32,
        intermediate_dim=64,
        n_heads=2,
        n_kv_heads=1,
        n_layers=2,
        vocab_size=16,
    )

    # Write LoRA adapter targeting token_embd.weight [32, 16] (out=16, in=32)
    write_synthetic_lora_gguf(
        lora_path,
        base_name="token_embd.weight",
        in_features=32,
        out_features=16,
        rank=4,
        alpha=8.0,
    )

    # Test with heap-backed model (use_mmap = False) to allow in-place weight mutation
    llama_backend_init()
    var mparams = llama_model_default_params()
    mparams.use_mmap = False
    var model = llama_model_load_from_file(model_path, mparams)
    assert_true(Int(model) != 0)

    # Read original weight before LoRA (Tensor 0 token_embd.weight is F32)
    var orig_val = model[].token_embd.unsafe_bitcast[Float32]()[unsafe_offset=0]

    # Apply LoRA with scale = 0.5
    # (alpha / rank) * scale = (8 / 4) * 0.5 = 1.0
    # B @ A = 4 * (0.5 * 1.0) = 2.0
    # delta = 1.0 * 2.0 = 2.0
    var status = llama_adapter_lora_apply(model, lora_path, scale=0.5)
    assert_equal(status, 0)

    var new_val = model[].token_embd.unsafe_bitcast[Float32]()[unsafe_offset=0]
    assert_almost_equal(new_val, orig_val + Float32(2.0), atol=1e-5)

    llama_model_free(model)
    llama_backend_free()
    print("  ✓ Online LoRA model merging verified")

def main() raises:
    print("============================================================")
    print("Running Phase 8 LoRA Test Suite")
    print("============================================================")
    test_lora_adapter_loading_and_math()
    test_lora_online_model_merging()
    print("All Phase 8 LoRA tests passed! ✓")
