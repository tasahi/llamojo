"""Unit tests for Phase 1 public llama.h API, sublibrary imports, and spin-barrier worker pool."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from max.algorithm import parallelize

from llamojo import (
    Matrix,
    free_matrix,
    chunk_start,
    chunk_end,
    LlamaWorkerPool,
    llama_barrier,
    write_synthetic_gguf,
    llama_backend_init,
    llama_backend_free,
    llama_time_us,
    llama_supports_mmap,
    llama_supports_mlock,
    llama_print_system_info,
    llama_model_default_params,
    llama_context_default_params,
    llama_model_quantize_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_model_n_params,
    llama_model_size,
    llama_model_n_embd,
    llama_model_n_layer,
    llama_model_n_head,
    llama_model_n_head_kv,
    llama_model_n_ctx_train,
    llama_model_n_vocab,
    llama_model_desc,
    llama_model_meta_val_str,
    llama_model_meta_count,
    llama_model_has_decoder,
    llama_model_has_encoder,
)

# Also verify direct kernel import from the llama2.mojo sublibrary
from llama2 import rmsnorm, matmul_rows
from std.utils import StaticTuple

def test_llama_lifecycle_and_model_api() raises:
    print("Testing llama.h backend lifecycle & model load/free API...")
    llama_backend_init()
    var sys_info = llama_print_system_info()
    print("  System Info:", sys_info)
    assert_true("llamojo" in sys_info)
    assert_true(llama_supports_mmap())
    assert_true(llama_supports_mlock())

    var t0 = llama_time_us()
    assert_true(t0 > 0)

    var mparams = llama_model_default_params()
    assert_true(mparams.use_mmap)
    assert_true(not mparams.vocab_only)

    var cparams = llama_context_default_params()
    assert_equal(cparams.n_ctx, UInt32(512))
    assert_true(cparams.n_threads >= 1)

    var qparams = llama_model_quantize_default_params()
    assert_true(qparams.quantize_output_tensor)

    var test_path = "/tmp/llamojo_api_test.gguf"
    write_synthetic_gguf(
        test_path,
        arch="llama",
        name="llama-api-test",
        hidden_dim=64,
        intermediate_dim=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        vocab_size=32,
    )

    var model = llama_model_load_from_file(test_path, mparams)
    assert_true(Int(model) != 0)

    assert_equal(llama_model_n_embd(model), Int32(64))
    assert_equal(llama_model_n_layer(model), Int32(2))
    assert_equal(llama_model_n_head(model), Int32(4))
    assert_equal(llama_model_n_head_kv(model), Int32(2))
    assert_equal(llama_model_n_ctx_train(model), Int32(512))
    assert_equal(llama_model_n_vocab(model), Int32(32))
    assert_equal(llama_model_meta_count(model), Int32(10))
    assert_equal(llama_model_meta_val_str(model, "general.name"), "llama-api-test")
    assert_true(llama_model_has_decoder(model))
    assert_true(not llama_model_has_encoder(model))

    # Total params = 64*32 (embd) + 64 (norm) + 64*32 (output) = 4160
    assert_equal(llama_model_n_params(model), UInt64(4160))
    assert_true(llama_model_size(model) > UInt64(0))
    print("  Model Desc:", llama_model_desc(model))

    llama_model_free(model)
    llama_backend_free()
    print("  ✓ llama.h model load/inspect/free API verified")

def test_spin_barrier_pool_and_llama2_sublibrary() raises:
    print("Testing persistent spin-barrier worker pool & llama2.mojo sublibrary kernels...")
    var dim = 64
    var pool = LlamaWorkerPool(workers=2, dim=dim)
    var sh = pool.sh

    var x = Matrix(dim)
    var weight = Matrix(dim)
    var out = Matrix(dim)
    for i in range(dim):
        x[i] = Float32(1.0)
        weight[i] = Float32(2.0)
    out.zero()

    var x_ptr = x.data
    var w_ptr = weight.data
    var out_ptr = out.data
    var workers = pool.workers

    # Dispatch multi-stage parallel work synchronized via sense-reversing `llama_barrier`
    def stage_worker(wid: Int) {imm}:
        var scratch = sh[].xbp.unsafe_offset(wid * dim)
        # Stage 1: Each worker computes rmsnorm into its private scratch buffer
        rmsnorm(scratch, x_ptr, w_ptr, dim)
        llama_barrier(sh)

        # Stage 2: Each worker writes its assigned chunk of `out`
        var c0 = chunk_start(dim, wid, workers)
        var c1 = chunk_end(dim, wid, workers)
        for idx in range(c0, c1):
            out_ptr[unsafe_offset=idx] = scratch[unsafe_offset=idx] * Float32(3.0)
        llama_barrier(sh)

    parallelize(stage_worker, workers, workers)

    for i in range(dim):
        assert_almost_equal(out[i], Float32(6.0), atol=1e-3)

    free_matrix(x)
    free_matrix(weight)
    free_matrix(out)
    pool.free()
    print("  ✓ Spin-barrier worker pool and llama2 sublibrary kernels verified")

def main() raises:
    print("============================================================")
    print("Running Phase 1 llama.h API & Worker Pool Tests")
    print("============================================================")
    test_llama_lifecycle_and_model_api()
    test_spin_barrier_pool_and_llama2_sublibrary()
    print("All Phase 1 API tests passed! ✓")

