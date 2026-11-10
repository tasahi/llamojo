""" Unit tests for Phase 3 Ministral 3B End-to-End Execution."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc

from llamojo import (
    GGMLType,
    ggml_type_size_in_bytes,
    gguf_load_from_file,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType,
    llama_rope_scaling_type,
    llama_backend_init,
    llama_backend_free,
    llama_model_load_from_file,
    llama_model_free,
    llama_model_default_params,
    llama_context_default_params,
    llama_init_from_model,
    llama_free,
    llama_batch_get_one,
    llama_batch_free,
    llama_decode,
    llama_get_logits_ith,
    llama_get_embeddings,
    llama_token,
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

def write_complete_ministral3_model_gguf(
    path: String,
    arch: String = "mistral3",
    name: String = "ministral-3-3b-phase3",
    hidden_dim: Int = 32,
    intermediate_dim: Int = 64,
    n_heads: Int = 4,
    n_kv_heads: Int = 2,
    n_layers: Int = 2,
    vocab_size: Int = 16,
    sliding_window: Int = 4,
    rope_scaling_type: String = "yarn",
    rope_orig_ctx_len: Int = 512,
    factor: Float32 = 4.0,
    beta_fast: Float32 = 32.0,
    beta_slow: Float32 = 1.0,
) raises:
    """Write a synthetic GGUF v3 file containing full transformer weights and metadata."""
    var buf = List[UInt8]()

    var n_kv: UInt64 = 18
    # 2 global tensors (token_embd.weight, output_norm.weight)
    # 7 per-layer tensors * n_layers (attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down) -> 9 * n_layers
    var n_tensors: UInt64 = UInt64(2 + 9 * n_layers)

    # 1. Header
    _append_u32(buf, GGUF_MAGIC)
    _append_u32(buf, GGUF_VERSION)
    _append_u64(buf, n_tensors)
    _append_u64(buf, n_kv)

    # 2. Metadata KVs
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
    _append_u32(buf, UInt32(2048))

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

    # tokenizer.ggml.tokens (ARRAY of STRING)
    _append_string(buf, "tokenizer.ggml.tokens")
    _append_u32(buf, GGUFValueType.ARRAY)
    _append_u32(buf, GGUFValueType.STRING)
    _append_u64(buf, UInt64(vocab_size))
    for tok_i in range(vocab_size):
        _append_string(buf, "tok_" + String(tok_i))

    # 3. Tensor Definitions
    var head_dim = hidden_dim // n_heads
    var kv_dim = n_kv_heads * head_dim

    var tensor_names = List[String]()
    var tensor_dims1 = List[UInt64]()
    var tensor_dims2 = List[UInt64]()
    var tensor_ndims = List[UInt32]()

    # Token embeddings (tied head)
    tensor_names.append("token_embd.weight")
    tensor_dims1.append(UInt64(hidden_dim))
    tensor_dims2.append(UInt64(vocab_size))
    tensor_ndims.append(2)

    # Output norm
    tensor_names.append("output_norm.weight")
    tensor_dims1.append(UInt64(hidden_dim))
    tensor_dims2.append(1)
    tensor_ndims.append(1)

    # Per layer weights
    for l in range(n_layers):
        var pfx = "blk." + String(l) + "."
        # attn_norm
        tensor_names.append(pfx + "attn_norm.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(1)
        tensor_ndims.append(1)

        # ffn_norm
        tensor_names.append(pfx + "ffn_norm.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(1)
        tensor_ndims.append(1)

        # attn_q
        tensor_names.append(pfx + "attn_q.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(UInt64(hidden_dim))
        tensor_ndims.append(2)

        # attn_k
        tensor_names.append(pfx + "attn_k.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(UInt64(kv_dim))
        tensor_ndims.append(2)

        # attn_v
        tensor_names.append(pfx + "attn_v.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(UInt64(kv_dim))
        tensor_ndims.append(2)

        # attn_output
        tensor_names.append(pfx + "attn_output.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(UInt64(hidden_dim))
        tensor_ndims.append(2)

        # ffn_gate
        tensor_names.append(pfx + "ffn_gate.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(UInt64(intermediate_dim))
        tensor_ndims.append(2)

        # ffn_up
        tensor_names.append(pfx + "ffn_up.weight")
        tensor_dims1.append(UInt64(hidden_dim))
        tensor_dims2.append(UInt64(intermediate_dim))
        tensor_ndims.append(2)

        # ffn_down
        tensor_names.append(pfx + "ffn_down.weight")
        tensor_dims1.append(UInt64(intermediate_dim))
        tensor_dims2.append(UInt64(hidden_dim))
        tensor_ndims.append(2)

    # Calculate offsets
    var cur_off: UInt64 = 0
    var tensor_offsets = List[UInt64]()
    var tensor_sizes = List[UInt64]()

    for i in range(len(tensor_names)):
        var elems = tensor_dims1[i] * tensor_dims2[i]
        var sz_bytes = ggml_type_size_in_bytes(GGMLType.F32, elems)
        var aligned_off = align_offset(cur_off, GGUF_DEFAULT_ALIGNMENT)
        tensor_offsets.append(aligned_off)
        tensor_sizes.append(sz_bytes)
        cur_off = aligned_off + sz_bytes

    # Append tensor directory entries
    for i in range(len(tensor_names)):
        _append_string(buf, tensor_names[i])
        _append_u32(buf, tensor_ndims[i])
        _append_u64(buf, tensor_dims1[i])
        if tensor_ndims[i] > 1:
            _append_u64(buf, tensor_dims2[i])
        _append_u32(buf, GGMLType.F32)
        _append_u64(buf, tensor_offsets[i])

    # Align to data section
    var data_start = align_offset(UInt64(len(buf)), GGUF_DEFAULT_ALIGNMENT)
    while UInt64(len(buf)) < data_start:
        buf.append(UInt8(0))

    # Populate dummy tensor weights
    for i in range(len(tensor_names)):
        while UInt64(len(buf)) < data_start + tensor_offsets[i]:
            buf.append(UInt8(0))
        var elems = Int(tensor_dims1[i] * tensor_dims2[i])
        var val: Float32 = 0.01 if tensor_ndims[i] == 2 else 1.0
        for _ in range(elems):
            _append_f32(buf, val)

    # Write file
    with open(path, "w") as f:
        f.write_bytes(buf)

def test_ministral3_end_to_end_pipeline() raises:
    print("Testing Ministral 3 3B Phase 3 complete inference pipeline...")
    var path = "/tmp/test_ministral3_phase3_full.gguf"
    write_complete_ministral3_model_gguf(
        path,
        arch="mistral3",
        name="Ministral-3-3B-Phase3-Test",
        hidden_dim=32,
        intermediate_dim=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        vocab_size=16,
        sliding_window=4,
        rope_scaling_type="yarn",
        rope_orig_ctx_len=512,
        factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
    )

    llama_backend_init()
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(path, mparams)
    assert_true(Int(model) != 0)

    # Verify model config populated correctly
    var cfg = model[].config
    assert_equal(cfg.arch, "mistral3")
    assert_equal(cfg.sliding_window, 4)
    assert_equal(cfg.rope_scaling_type, llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN)
    assert_true(cfg.is_tied_embeddings)

    var cparams = llama_context_default_params()
    cparams.n_ctx = 64
    cparams.n_batch = 16
    cparams.embeddings = True

    var ctx = llama_init_from_model(model, cparams)
    assert_true(Int(ctx) != 0)

    # Test multi-token batch decoding through sliding-window + YaRN RoPE
    var n_tokens = 6
    var tokens = unsafe_alloc[llama_token](n_tokens)
    for i in range(n_tokens):
        tokens[unsafe_offset=i] = Int32(i % 16)

    var batch = llama_batch_get_one(tokens, n_tokens=Int32(n_tokens))
    var ret = llama_decode(ctx, batch)
    assert_equal(ret, 0)

    # Verify output logits for the last decoded token
    var logits_last = llama_get_logits_ith(ctx, Int32(n_tokens - 1))
    assert_true(Int(logits_last) != 0)

    var has_nonzero = False
    for v in range(16):
        var val = logits_last[unsafe_offset=v]
        assert_true(val == val)
        assert_true(val > -1e6 and val < 1e6)
        if val != 0.0:
            has_nonzero = True

    assert_true(has_nonzero)

    # Verify embeddings output
    var embd = llama_get_embeddings(ctx)
    assert_true(Int(embd) != 0)

    # Cleanup
    llama_batch_free(batch)
    tokens.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

    print("  ✓ Full Ministral 3 end-to-end inference pass verified (SWA + YaRN + Tied Weights)")

def main() raises:
    print("============================================================")
    print("Running Ministral 3 3B Phase 3 Integration Tests")
    print("============================================================")
    test_ministral3_end_to_end_pipeline()
    print("All Phase 3 tests passed! ✓")
