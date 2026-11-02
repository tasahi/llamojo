"""Phase 2 tests: llamojo context / KV / batch buffer view exporters."""

from std.testing import assert_true, assert_equal
from std.memory.alloc import unsafe_alloc
from llamojo import (
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_context_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_init_from_model,
    llama_free,
    llama_batch_get_one,
    llama_batch_free,
    llama_decode,
    llama_get_logits,
    write_synthetic_gguf,
    llama_token,
)
from llamojo.interop import (
    context_logits_view,
    context_kv_k_view,
    batch_tokens_view,
)


def test_context_views() raises:
    var path = "/tmp/llamojo_phase2_views.gguf"
    write_synthetic_gguf(
        path, arch="llama", name="p2", hidden_dim=32, intermediate_dim=64,
        n_heads=2, n_kv_heads=1, n_layers=2, vocab_size=16,
    )
    llama_backend_init()
    var model = llama_model_load_from_file(path, llama_model_default_params())
    var cparams = llama_context_default_params()
    cparams.n_ctx = 64
    cparams.n_batch = 16
    var ctx = llama_init_from_model(model, cparams)
    var tokens = unsafe_alloc[llama_token](2)
    tokens[unsafe_offset=0] = 1
    tokens[unsafe_offset=1] = 2
    var batch = llama_batch_get_one(tokens, n_tokens=2)
    assert_equal(llama_decode(ctx, batch), 0)

    var lv = context_logits_view(ctx[])
    assert_equal(lv.total_elements, 16)
    assert_true(lv.is_contiguous_layout)
    assert_equal(lv.ptr_f32(), llama_get_logits(ctx))

    var kv = context_kv_k_view(ctx[])
    assert_equal(kv.ndim, 3)
    assert_equal(Int(kv.shape[1]), 64)
    assert_equal(Int(kv.shape[2]), 2)
    assert_equal(kv.ptr_f32(), ctx[].kv_cache.k_f32)

    var bv = batch_tokens_view(batch)
    assert_equal(bv.total_elements, 2)
    assert_equal(bv.ptr_u8(), batch.token.unsafe_bitcast[UInt8]())

    llama_batch_free(batch)
    tokens.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()


def main() raises:
    test_context_views()
    print("test_context_views: all passed")
