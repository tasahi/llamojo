"""Unit tests for Phase 1 GGUF v3 parser, quantization types, and mmap tensor binding."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from llamojo import (
    GGMLType,
    ggml_type_size,
    ggml_block_size,
    ggml_type_size_in_bytes,
    ggml_type_name,
    f16_to_f32,
    bf16_to_f32,
    gguf_load_from_file,
    write_synthetic_gguf,
    free_matrix,
)

def test_ggml_types_and_sizes() raises:
    print("Testing GGMLType block sizes and byte calculations...")
    assert_equal(ggml_type_size(GGMLType.F32), UInt64(4))
    assert_equal(ggml_block_size(GGMLType.F32), UInt64(1))
    assert_equal(ggml_type_size_in_bytes(GGMLType.F32, 1024), UInt64(4096))

    assert_equal(ggml_type_size(GGMLType.F16), UInt64(2))
    assert_equal(ggml_block_size(GGMLType.F16), UInt64(1))

    assert_equal(ggml_type_size(GGMLType.Q4_0), UInt64(18))
    assert_equal(ggml_block_size(GGMLType.Q4_0), UInt64(32))
    assert_equal(ggml_type_size_in_bytes(GGMLType.Q4_0, 320), UInt64(180))

    assert_equal(ggml_type_size(GGMLType.Q5_0), UInt64(22))
    assert_equal(ggml_block_size(GGMLType.Q5_0), UInt64(32))

    assert_equal(ggml_type_size(GGMLType.Q8_0), UInt64(34))
    assert_equal(ggml_block_size(GGMLType.Q8_0), UInt64(32))
    assert_equal(ggml_type_size_in_bytes(GGMLType.Q8_0, 64), UInt64(68))

    assert_equal(ggml_type_size(GGMLType.Q4_K), UInt64(144))
    assert_equal(ggml_block_size(GGMLType.Q4_K), UInt64(256))

    assert_equal(ggml_type_size(GGMLType.Q5_K), UInt64(176))
    assert_equal(ggml_block_size(GGMLType.Q5_K), UInt64(256))

    assert_equal(ggml_type_size(GGMLType.Q6_K), UInt64(210))
    assert_equal(ggml_block_size(GGMLType.Q6_K), UInt64(256))

    assert_equal(ggml_type_size(GGMLType.MXFP4), UInt64(17))
    assert_equal(ggml_block_size(GGMLType.MXFP4), UInt64(32))

    assert_equal(ggml_type_name(GGMLType.Q4_K), "Q4_K")
    assert_equal(ggml_type_name(GGMLType.Q8_0), "Q8_0")
    print("  ✓ GGMLType sizes verified")

def test_f16_and_bf16_conversions() raises:
    print("Testing F16 and BF16 -> F32 conversions...")
    # IEEE 754 float16: 0x3C00 = 1.0, 0xC000 = -2.0, 0x3800 = 0.5, 0x0000 = 0.0
    assert_almost_equal(f16_to_f32(UInt16(0x3C00)), Float32(1.0), atol=1e-5)
    assert_almost_equal(f16_to_f32(UInt16(0xC000)), Float32(-2.0), atol=1e-5)
    assert_almost_equal(f16_to_f32(UInt16(0x3800)), Float32(0.5), atol=1e-5)
    assert_almost_equal(f16_to_f32(UInt16(0x0000)), Float32(0.0), atol=1e-5)

    # BFloat16: 0x3F80 = 1.0, 0xC000 = -2.0
    assert_almost_equal(bf16_to_f32(UInt16(0x3F80)), Float32(1.0), atol=1e-5)
    assert_almost_equal(bf16_to_f32(UInt16(0xC000)), Float32(-2.0), atol=1e-5)
    print("  ✓ F16 and BF16 conversions verified")

def test_gguf_mmap_and_heap_loader() raises:
    print("Testing GGUF v3 writer, zero-copy mmap loader, and llama2.Matrix view...")
    var test_path = "/tmp/llamojo_phase1_test.gguf"
    write_synthetic_gguf(
        test_path,
        arch="llama",
        name="phase1-unit-test",
        hidden_dim=64,
        intermediate_dim=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        vocab_size=32,
    )

    # 1. Load via POSIX mmap (use_mmap = True)
    var ctx = gguf_load_from_file(test_path, use_mmap=True, check_tensors=True)
    assert_true(ctx.is_mmapped)
    assert_equal(ctx.version, UInt32(3))
    assert_equal(ctx.get_string("general.architecture"), "llama")
    assert_equal(ctx.get_string("general.name"), "phase1-unit-test")
    assert_equal(ctx.get_int("llama.embedding_length"), 64)
    assert_equal(ctx.get_int("llama.feed_forward_length"), 128)
    assert_equal(ctx.get_int("llama.attention.head_count"), 4)
    assert_equal(ctx.get_int("llama.attention.head_count_kv"), 2)
    assert_equal(ctx.get_int("llama.block_count"), 2)
    assert_almost_equal(ctx.get_float("llama.attention.layer_norm_rms_epsilon"), Float32(1e-5), atol=1e-7)

    var cfg = ctx.build_model_config()
    assert_equal(cfg.vocab_size, 32)
    assert_equal(cfg.head_dim, 16)
    assert_equal(len(ctx.tensors), 3)

    # Verify non-owning llama2.Matrix view over mmap-bound token_embd.weight
    var t_embd = ctx.get_tensor("token_embd.weight")
    assert_equal(t_embd.n_elements, UInt64(64 * 32))
    var mat = t_embd.as_matrix()
    assert_equal(mat.allocated, 0)  # Zero-copy non-owning view!
    assert_equal(mat.dim(0), 32)
    assert_equal(mat.dim(1), 64)
    assert_almost_equal(mat[0], Float32(0.25), atol=1e-5)
    assert_almost_equal(mat[1], Float32(0.50), atol=1e-5)
    assert_almost_equal(mat[0, 3], Float32(1.00), atol=1e-5)

    # Verify norm tensor
    var t_norm = ctx.get_tensor("blk.0.attn_norm.weight")
    var norm_mat = t_norm.as_matrix()
    assert_almost_equal(norm_mat[0], Float32(1.0), atol=1e-5)
    assert_almost_equal(norm_mat[63], Float32(1.0), atol=1e-5)

    ctx.free()

    # 2. Load via heap fallback (use_mmap = False)
    var ctx_heap = gguf_load_from_file(test_path, use_mmap=False, check_tensors=True)
    assert_true(not ctx_heap.is_mmapped)
    var t_embd_heap = ctx_heap.get_tensor("token_embd.weight")
    var mat_heap = t_embd_heap.as_matrix()
    assert_almost_equal(mat_heap[0, 3], Float32(1.00), atol=1e-5)
    ctx_heap.free()
    print("  ✓ GGUF v3 mmap & heap loader verified")

def main() raises:
    print("============================================================")
    print("Running Phase 1 GGUF & Type Tests")
    print("============================================================")
    test_ggml_types_and_sizes()
    test_f16_and_bf16_conversions()
    test_gguf_mmap_and_heap_loader()
    print("All GGUF tests passed! ✓")

