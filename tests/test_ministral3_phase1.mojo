""" Unit tests for Phase 1 Ministral 3B GGUF metadata parsing and ModelConfig population."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.memory import bitcast
from llamojo import (
    GGMLType,
    ggml_type_size_in_bytes,
    gguf_load_from_file,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType,
    llama_rope_scaling_type,
    ModelConfig,
    llama_vocab,
)
from llamojo.gguf import align_offset

@always_inline
def _append_u32(mut buf: List[UInt8], v: UInt32):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))

@always_inline
def _append_u64(mut buf: List[UInt8], v: UInt64):
    _append_u32(buf, UInt32(v & 0xFFFFFFFF))
    _append_u32(buf, UInt32((v >> 32) & 0xFFFFFFFF))

@always_inline
def _append_f32(mut buf: List[UInt8], v: Float32):
    _append_u32(buf, bitcast[DType.uint32](v))

@always_inline
def _append_string(mut buf: List[UInt8], s: String):
    var blen = s.byte_length()
    _append_u64(buf, UInt64(blen))
    var ptr = s.unsafe_ptr()
    for i in range(blen):
        buf.append(ptr[unsafe_offset=i])

def write_synthetic_ministral3_gguf(
    path: String,
    arch: String = "mistral3",
    name: String = "ministral-3-3b-synthetic",
    hidden_dim: Int = 3072,
    intermediate_dim: Int = 9216,
    n_heads: Int = 32,
    n_kv_heads: Int = 8,
    n_layers: Int = 26,
    vocab_size: Int = 16,
    sliding_window: Int = 0,
    rope_scaling_type: String = "yarn",
    rope_orig_ctx_len: Int = 16384,
    factor: Float32 = 16.0,
    beta_fast: Float32 = 32.0,
    beta_slow: Float32 = 1.0,
    tie_embeddings: Bool = True,
) raises:
    """Write a synthetic GGUF v3 file representing Ministral 3 3B architecture."""
    var buf = List[UInt8]()

    var n_kv: UInt64 = 19
    var n_tensors: UInt64 = 2 if tie_embeddings else 3

    # Header
    _append_u32(buf, GGUF_MAGIC)
    _append_u32(buf, GGUF_VERSION)
    _append_u64(buf, n_tensors)
    _append_u64(buf, n_kv)

    # Metadata KVs
    _append_string(buf, "general.architecture")
    _append_u32(buf, GGUFValueType.STRING)
    _append_string(buf, arch)

    _append_string(buf, "general.name")
    _append_u32(buf, GGUFValueType.STRING)
    _append_string(buf, name)

    _append_string(buf, arch + ".embedding_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(hidden_dim))

    _append_string(buf, arch + ".feed_forward_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(intermediate_dim))

    _append_string(buf, arch + ".attention.head_count")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(n_heads))

    _append_string(buf, arch + ".attention.head_count_kv")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(n_kv_heads))

    _append_string(buf, arch + ".block_count")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(n_layers))

    _append_string(buf, arch + ".context_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(262144))

    _append_string(buf, arch + ".rope.freq_base")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, 1000000.0)

    _append_string(buf, arch + ".attention.layer_norm_rms_epsilon")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, 1e-5)

    _append_string(buf, arch + ".rope.scaling.type")
    _append_u32(buf, GGUFValueType.STRING)
    _append_string(buf, rope_scaling_type)

    _append_string(buf, arch + ".rope.scaling.factor")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, factor)

    _append_string(buf, arch + ".rope.scaling.original_context_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(rope_orig_ctx_len))

    _append_string(buf, arch + ".rope.scaling.yarn_beta_fast")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, beta_fast)

    _append_string(buf, arch + ".rope.scaling.yarn_beta_slow")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, beta_slow)

    _append_string(buf, arch + ".rope.scaling.yarn_attn_factor")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, 1.0)

    _append_string(buf, arch + ".attention.sliding_window")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(sliding_window))

    # tokenizer.ggml.tokens
    _append_string(buf, "tokenizer.ggml.tokens")
    _append_u32(buf, GGUFValueType.ARRAY)
    _append_u32(buf, GGUFValueType.STRING)
    _append_u64(buf, UInt64(vocab_size))
    for tok_i in range(vocab_size):
        _append_string(buf, "tok_" + String(tok_i))

    # tokenizer.ggml.scores (encoded as INT32 array as seen in some Ministral conversions)
    _append_string(buf, "tokenizer.ggml.scores")
    _append_u32(buf, GGUFValueType.ARRAY)
    _append_u32(buf, GGUFValueType.INT32)
    _append_u64(buf, UInt64(vocab_size))
    for score_i in range(vocab_size):
        _append_u32(buf, UInt32(score_i))

    # Tensor Info
    # Tensor 0: token_embd.weight [hidden_dim, vocab_size] F32
    var t0_elems = UInt64(hidden_dim * vocab_size)
    var t0_bytes = ggml_type_size_in_bytes(GGMLType.F32, t0_elems)
    var t0_off: UInt64 = 0
    _append_string(buf, "token_embd.weight")
    _append_u32(buf, 2)
    _append_u64(buf, UInt64(hidden_dim))
    _append_u64(buf, UInt64(vocab_size))
    _append_u32(buf, GGMLType.F32)
    _append_u64(buf, t0_off)

    # Tensor 1: blk.0.attn_norm.weight [hidden_dim] F32
    var t1_elems = UInt64(hidden_dim)
    var t1_bytes = ggml_type_size_in_bytes(GGMLType.F32, t1_elems)
    var t1_off = align_offset(t0_off + t0_bytes, GGUF_DEFAULT_ALIGNMENT)
    _append_string(buf, "blk.0.attn_norm.weight")
    _append_u32(buf, 1)
    _append_u64(buf, UInt64(hidden_dim))
    _append_u32(buf, GGMLType.F32)
    _append_u64(buf, t1_off)

    var t2_off = t1_off
    if not tie_embeddings:
        var t2_elems = UInt64(hidden_dim * vocab_size)
        var _t2_bytes = ggml_type_size_in_bytes(GGMLType.F32, t2_elems)
        t2_off = align_offset(t1_off + t1_bytes, GGUF_DEFAULT_ALIGNMENT)
        _append_string(buf, "output.weight")
        _append_u32(buf, 2)
        _append_u64(buf, UInt64(hidden_dim))
        _append_u64(buf, UInt64(vocab_size))
        _append_u32(buf, GGMLType.F32)
        _append_u64(buf, t2_off)

    # Tensor data section
    var data_start = align_offset(UInt64(len(buf)), GGUF_DEFAULT_ALIGNMENT)
    while UInt64(len(buf)) < data_start:
        buf.append(UInt8(0))

    # Tensor 0 dummy data
    for _ in range(Int(t0_elems)):
        _append_f32(buf, 0.01)
    while UInt64(len(buf)) < data_start + t1_off:
        buf.append(UInt8(0))

    # Tensor 1 dummy data
    for _ in range(Int(t1_elems)):
        _append_f32(buf, 1.0)

    if not tie_embeddings:
        while UInt64(len(buf)) < data_start + t2_off:
            buf.append(UInt8(0))
        for _ in range(Int(t0_elems)):
            _append_f32(buf, 0.02)

    # Write out file
    with open(path, "w") as f:
        f.write_bytes(buf)

def test_ministral3_metadata_parsing() raises:
    print("Testing Ministral 3 3B GGUF metadata parsing and ModelConfig population...")
    var path = "/tmp/test_ministral3_synthetic.gguf"
    write_synthetic_ministral3_gguf(
        path,
        arch="mistral3",
        name="Ministral-3-3B-Instruct-2512",
        hidden_dim=3072,
        intermediate_dim=9216,
        n_heads=32,
        n_kv_heads=8,
        n_layers=26,
        vocab_size=16,
        sliding_window=0,
        rope_scaling_type="yarn",
        rope_orig_ctx_len=16384,
        factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
        tie_embeddings=True,
    )

    var ctx = gguf_load_from_file(path, use_mmap=True)
    assert_equal(ctx.get_string("general.architecture"), "mistral3")
    assert_equal(ctx.get_string("general.name"), "Ministral-3-3B-Instruct-2512")

    var cfg = ctx.build_model_config()
    assert_equal(cfg.arch, "mistral3")
    assert_equal(cfg.hidden_dim, 3072)
    assert_equal(cfg.intermediate_dim, 9216)
    assert_equal(cfg.n_heads, 32)
    assert_equal(cfg.n_kv_heads, 8)
    assert_equal(cfg.head_dim, 96) # 3072 // 32
    assert_equal(cfg.n_layers, 26)
    assert_equal(cfg.max_seq_len, 262144)
    assert_almost_equal(cfg.rope_theta, Float32(1000000.0), atol=1e-3)
    assert_almost_equal(cfg.norm_eps, Float32(1e-5), atol=1e-7)

    # Verify YaRN RoPE settings
    assert_equal(cfg.rope_scaling_type, llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN)
    assert_almost_equal(cfg.rope_freq_scale, Float32(16.0), atol=1e-5)
    assert_equal(cfg.rope_orig_ctx_len, 16384)
    assert_almost_equal(cfg.yarn_beta_fast, Float32(32.0), atol=1e-5)
    assert_almost_equal(cfg.yarn_beta_slow, Float32(1.0), atol=1e-5)
    assert_almost_equal(cfg.yarn_attn_factor, Float32(1.0), atol=1e-5)

    # Verify tied embeddings
    assert_true(cfg.is_tied_embeddings)

    # Verify vocab scores conversion from integer array
    var vocab = llama_vocab(ctx)
    assert_equal(vocab.vocab_size, 16)
    assert_equal(len(vocab.scores), 16)
    assert_almost_equal(vocab.scores[0], Float32(0.0), atol=1e-5)
    assert_almost_equal(vocab.scores[15], Float32(15.0), atol=1e-5)

    ctx.free()
    print("  ✓ Ministral 3 metadata & YaRN parameters verified")

def main() raises:
    print("============================================================")
    print("Running Ministral 3 3B Phase 1 Tests")
    print("============================================================")
    test_ministral3_metadata_parsing()
    print("All Phase 1 tests passed! ✓")
