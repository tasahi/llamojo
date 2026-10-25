"""Unit tests for Phase 3: KV Cache, PageTable, llama_batch, context, and multi-layer llama_decode."""

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
    llama_get_logits,
    llama_get_logits_ith,
    llama_get_embeddings,
    llama_kv_cache_clear,
    llama_kv_cache_seq_rm,
    llama_kv_cache_seq_cp,
    llama_kv_cache_seq_add,
    llama_kv_cache_seq_keep,
    PageTable,
    LlamaKVCache,
    write_synthetic_gguf,
    llama_token,
)

def test_pagetable_allocation() raises:
    print("Testing PagedAttention PageTable block allocation...")
    var pt = PageTable(max_pages=8, page_size=16, n_layers=2, n_kv_heads=2, head_dim=16)
    assert_equal(pt.free_count, 8)

    var p0 = pt.alloc_page()
    var p1 = pt.alloc_page()
    assert_true(p0 >= 0 and p1 >= 0 and p0 != p1)
    assert_equal(pt.free_count, 6)

    pt.map_logical(logical_page=0, physical_page=p0)
    pt.map_logical(logical_page=1, physical_page=p1)
    assert_equal(Int(pt.logical_map[unsafe_offset=0]), p0)
    assert_equal(Int(pt.logical_map[unsafe_offset=1]), p1)

    pt.free_page(p0)
    assert_equal(pt.free_count, 7)
    pt.deinit_storage()
    print("  ✓ PageTable allocation passed")

def test_kv_cache_seq_ops() raises:
    print("Testing LlamaKVCache cell metadata and seq_* manipulation...")
    var kv = LlamaKVCache(max_ctx=32, n_layers=2, n_kv_heads=2, head_dim=16, type_k=GGMLType.F16, type_v=GGMLType.F16)
    var k_in = unsafe_alloc[Float32](32)
    var v_in = unsafe_alloc[Float32](32)
    for i in range(32):
        k_in[unsafe_offset=i] = 1.0
        v_in[unsafe_offset=i] = 2.0

    # Write tokens for sequence 0 (pos 0, 1, 2)
    kv.write_kv(layer=0, slot=0, k_in=k_in, v_in=v_in, pos=0, seq_id=0)
    kv.write_kv(layer=0, slot=1, k_in=k_in, v_in=v_in, pos=1, seq_id=0)
    kv.write_kv(layer=0, slot=2, k_in=k_in, v_in=v_in, pos=2, seq_id=0)

    # Write tokens for sequence 1 (pos 0, 1)
    kv.write_kv(layer=0, slot=3, k_in=k_in, v_in=v_in, pos=0, seq_id=1)
    kv.write_kv(layer=0, slot=4, k_in=k_in, v_in=v_in, pos=1, seq_id=1)

    assert_equal(kv.cells[unsafe_offset=0].pos, 0)
    assert_equal(kv.cells[unsafe_offset=0].seq_id, 0)
    assert_equal(kv.cells[unsafe_offset=3].seq_id, 1)

    # Shift positions in sequence 0 by +5: seq_add
    kv.seq_add(seq_id=0, p0=0, p1=3, delta=5)
    assert_equal(kv.cells[unsafe_offset=0].pos, 5)
    assert_equal(kv.cells[unsafe_offset=1].pos, 6)

    # Remove sequence 1: seq_rm
    var removed = kv.seq_rm(seq_id=1, p0=-1, p1=-1)
    assert_true(removed)
    assert_equal(kv.cells[unsafe_offset=3].seq_id, -1)

    # Clear all slots: clear
    kv.clear()
    assert_equal(kv.cells[unsafe_offset=0].seq_id, -1)

    k_in.unsafe_free()
    v_in.unsafe_free()
    kv.deinit_storage()
    print("  ✓ LlamaKVCache seq operations passed")

def test_batch_lifecycle() raises:
    print("Testing llama_batch allocation, views, and free...")
    var batch = llama_batch_init(n_tokens=4, embd=0, n_seq_max=1)
    assert_equal(batch.n_tokens, 4)
    batch.token[unsafe_offset=0] = 10
    batch.token[unsafe_offset=1] = 20
    assert_equal(batch.logits[unsafe_offset=3], 1)
    llama_batch_free(batch)
    assert_equal(batch.n_tokens, 0)

    # Test llama_batch_get_one
    var tokens = unsafe_alloc[llama_token](3)
    tokens[unsafe_offset=0] = 101
    tokens[unsafe_offset=1] = 102
    tokens[unsafe_offset=2] = 103
    var b_one = llama_batch_get_one(tokens, n_tokens=3)
    assert_equal(b_one.n_tokens, 3)
    assert_equal(b_one.token[unsafe_offset=0], 101)
    assert_equal(b_one.pos[unsafe_offset=1], 1)
    assert_equal(b_one.logits[unsafe_offset=2], 1)

    llama_batch_free(b_one)
    tokens.unsafe_free()
    print("  ✓ llama_batch lifecycle passed")

def test_full_decode_pipeline() raises:
    print("Testing full llama_init_from_model, llama_decode forward pass, and logits extraction...")
    var test_path = "/tmp/llamojo_phase3_test.gguf"
    write_synthetic_gguf(
        test_path,
        arch="llama",
        name="phase3-model",
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

    var cparams = llama_context_default_params()
    cparams.n_ctx = 64
    cparams.n_batch = 16
    cparams.embeddings = True

    var ctx = llama_init_from_model(model, cparams)
    assert_true(Int(ctx) != 0)

    # Create batch with 2 tokens: [tok_0, tok_1]
    var tokens = unsafe_alloc[llama_token](2)
    tokens[unsafe_offset=0] = 1
    tokens[unsafe_offset=1] = 2
    var batch = llama_batch_get_one(tokens, n_tokens=2)

    # Run transformer decode
    var status = llama_decode(ctx, batch)
    assert_equal(status, 0)

    # Verify logits buffer produced for the tokens
    var logits_0 = llama_get_logits_ith(ctx, 0)
    var logits_1 = llama_get_logits_ith(ctx, 1)
    assert_true(Int(logits_0) != 0)
    assert_true(Int(logits_1) != 0)

    # Check that final token produced non-zero logits across vocab
    var has_nonzero = False
    for v in range(16):
        if logits_1[unsafe_offset=v] != 0.0:
            has_nonzero = True
            break
    assert_true(has_nonzero)

    # Test embeddings output
    var embd = llama_get_embeddings(ctx)
    assert_true(Int(embd) != 0)

    # Test KV cache manipulation on live context
    llama_kv_cache_clear(ctx)
    assert_equal(ctx[].kv_cache.cells[unsafe_offset=0].seq_id, -1)

    llama_batch_free(batch)
    tokens.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
    print("  ✓ Full decode pipeline passed")

def main() raises:
    print("============================================================")
    print("Running Phase 3 Test Suite")
    print("============================================================")
    test_pagetable_allocation()
    test_kv_cache_seq_ops()
    test_batch_lifecycle()
    test_full_decode_pipeline()
    print("All Phase 3 tests passed! ✓")

