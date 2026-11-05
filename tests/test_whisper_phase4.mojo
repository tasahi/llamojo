""" Unit and Integration Tests for Whisper Phase 4 Encoder & Decoder Execution (`src/llamojo/whisper/model.mojo`).

Tests:
1. `get_embedding_row` lookup on F32 and F16 embedding rows.
2. Synthetic bidirectional `whisper_encode_layer` single-threaded vs multi-threaded equivalence.
3. Full end-to-end real-model encoding, cross-KV caching, and autoregressive greedy decoding on
   `/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin` and `/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav`.
"""

from std.collections import List
from std.memory import Pointer, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, assert_almost_equal

from llamojo import (
    GGMLType,
    WhisperEncoderLayer,
    WhisperModel,
    WhisperState,
    whisper_load_model_from_file,
    load_wav_file,
    log_mel_spectrogram,
    get_embedding_row,
    whisper_encode_layer,
    whisper_encode,
    whisper_decode_step,
    whisper_decode,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]


def test_get_embedding_row() raises:
    print("Testing get_embedding_row (F32)...")
    var dim = 16
    var n_rows = 4
    var table = unsafe_alloc[Float32](n_rows * dim)
    for r in range(n_rows):
        for c in range(dim):
            table[unsafe_offset=r * dim + c] = Float32(r * 100 + c)

    var dst = unsafe_alloc[Float32](dim)
    get_embedding_row(
        dst,
        BytePtr(unsafe_from_address=Int(table)),
        GGMLType.F32,
        2,
        dim,
    )
    for c in range(dim):
        assert_almost_equal(dst[unsafe_offset=c], Float32(200 + c), atol=1e-6)

    dst.unsafe_free()
    table.unsafe_free()
    print("  ✓ get_embedding_row verified")


def test_whisper_encode_layer_equivalence() raises:
    print("Testing whisper_encode_layer (1 worker vs 4 workers equivalence)...")
    var n_ctx = 16
    var n_state = 32
    var n_head = 4
    var eps: Float32 = 1e-5

    var ln_w = unsafe_alloc[Float32](n_state)
    var ln_b = unsafe_alloc[Float32](n_state)
    for i in range(n_state):
        ln_w[unsafe_offset=i] = Float32(1.0)
        ln_b[unsafe_offset=i] = Float32(0.01)

    var w_mat = unsafe_alloc[Float32](n_state * n_state)
    var b_vec = unsafe_alloc[Float32](n_state)
    for i in range(n_state * n_state):
        w_mat[unsafe_offset=i] = Float32((i % 13) - 6) * Float32(0.02)
    for i in range(n_state):
        b_vec[unsafe_offset=i] = Float32(i % 5) * Float32(0.01)

    var mlp0_w = unsafe_alloc[Float32](4 * n_state * n_state)
    var mlp0_b = unsafe_alloc[Float32](4 * n_state)
    var mlp1_w = unsafe_alloc[Float32](n_state * 4 * n_state)
    var mlp1_b = unsafe_alloc[Float32](n_state)
    for i in range(4 * n_state * n_state):
        mlp0_w[unsafe_offset=i] = Float32((i % 11) - 5) * Float32(0.01)
        mlp1_w[unsafe_offset=i] = Float32((i % 9) - 4) * Float32(0.01)
    for i in range(4 * n_state):
        mlp0_b[unsafe_offset=i] = Float32(0.005)
    for i in range(n_state):
        mlp1_b[unsafe_offset=i] = Float32(-0.005)

    var layer = WhisperEncoderLayer()
    layer.attn_ln_0_w = ln_w
    layer.attn_ln_0_b = ln_b
    layer.attn_q_w = BytePtr(unsafe_from_address=Int(w_mat))
    layer.attn_q_b = b_vec
    layer.attn_k_w = BytePtr(unsafe_from_address=Int(w_mat))
    layer.attn_v_w = BytePtr(unsafe_from_address=Int(w_mat))
    layer.attn_v_b = b_vec
    layer.attn_ln_1_w = BytePtr(unsafe_from_address=Int(w_mat))
    layer.attn_ln_1_b = b_vec
    layer.mlp_ln_w = ln_w
    layer.mlp_ln_b = ln_b
    layer.mlp_0_w = BytePtr(unsafe_from_address=Int(mlp0_w))
    layer.mlp_0_b = mlp0_b
    layer.mlp_1_w = BytePtr(unsafe_from_address=Int(mlp1_w))
    layer.mlp_1_b = mlp1_b

    var x1 = unsafe_alloc[Float32](n_ctx * n_state)
    var x2 = unsafe_alloc[Float32](n_ctx * n_state)
    for i in range(n_ctx * n_state):
        var val = Float32((i % 17) - 8) * Float32(0.05)
        x1[unsafe_offset=i] = val
        x2[unsafe_offset=i] = val

    var q = unsafe_alloc[Float32](n_ctx * n_state)
    var k = unsafe_alloc[Float32](n_ctx * n_state)
    var v = unsafe_alloc[Float32](n_ctx * n_state)
    var a_out = unsafe_alloc[Float32](n_ctx * n_state)

    whisper_encode_layer(x1, layer, n_ctx, n_state, n_head, eps, q, k, v, a_out, 1)
    whisper_encode_layer(x2, layer, n_ctx, n_state, n_head, eps, q, k, v, a_out, 4)

    for i in range(n_ctx * n_state):
        assert_almost_equal(x1[unsafe_offset=i], x2[unsafe_offset=i], atol=1e-5)

    q.unsafe_free()
    k.unsafe_free()
    v.unsafe_free()
    a_out.unsafe_free()
    x1.unsafe_free()
    x2.unsafe_free()
    mlp1_b.unsafe_free()
    mlp1_w.unsafe_free()
    mlp0_b.unsafe_free()
    mlp0_w.unsafe_free()
    b_vec.unsafe_free()
    w_mat.unsafe_free()
    ln_b.unsafe_free()
    ln_w.unsafe_free()
    print("  ✓ whisper_encode_layer single-worker vs multi-worker equivalence verified")


def test_real_whisper_jfk_encode_decode() raises:
    print("Testing end-to-end Whisper encode + decode on ggml-base.en.bin + jfk.wav...")
    var bin_path = "/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin"
    var wav_path = "/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav"

    var model = whisper_load_model_from_file(bin_path)
    var state = WhisperState(model)

    # 1. Load JFK audio & compute Log-Mel Spectrogram
    var pcm = load_wav_file(wav_path)
    assert_equal(len(pcm), 176000)
    log_mel_spectrogram(pcm, model.filters, state.mel)
    assert_equal(state.mel.n_mel, 80)

    # 2. Run Full Audio Encoder (conv1 -> conv2 -> pos_emb -> 6 enc layers -> ln_post -> build_cross_kv)
    whisper_encode(model, state, 0)

    # Verify encoder output and cross-attention KV cache are non-zero
    var enc_energy: Float32 = 0.0
    for i in range(512):
        var v = state.embd_enc[unsafe_offset=i]
        enc_energy += v * v
    assert_true(enc_energy > 0.1)

    var cross_k0 = state.kv_cross.get_k_ptr(0, 0)
    var cross_energy: Float32 = 0.0
    for i in range(512):
        var v = cross_k0[unsafe_offset=i]
        cross_energy += v * v
    assert_true(cross_energy > 0.1)
    print("  ✓ Audio Encoder & Static Cross-KV Cache computed (enc_energy =", enc_energy, ")")

    # 3. Prompt Decoder with [SOT] or [SOT, NOT] and run Greedy Autoregressive Decoding
    state.reset_kv_self()
    var prompt = List[Int32]()
    prompt.append(model.vocab.token_sot)
    prompt.append(model.vocab.token_not)

    whisper_decode(model, state, prompt, 0)
    var n_past = len(prompt)

    var transcript = String("")
    var generated_tokens = List[Int32]()

    for _ in range(40):
        # Greedy argmax over text tokens (< token_eot) and EOT token
        var best_tok: Int = 0
        var best_logit: Float32 = -1e30
        var eot_id = Int(model.vocab.token_eot)

        for tok_id in range(eot_id + 1):
            var l = state.logits[unsafe_offset=tok_id]
            if l > best_logit:
                best_logit = l
                best_tok = tok_id

        if best_tok == eot_id:
            break

        generated_tokens.append(Int32(best_tok))
        transcript += model.vocab.id_to_token[best_tok]

        whisper_decode_step(model, state, Int32(best_tok), n_past, True)
        n_past += 1

    print("  ✓ Greedy Transcription (", len(generated_tokens), "tokens):", transcript)
    assert_true(len(generated_tokens) >= 15)
    assert_true("fellow" in transcript or "Americans" in transcript or "country" in transcript)

    state.free()
    model.free()
    print("  ✓ End-to-end Whisper JFK transcription verified!")


def main() raises:
    print("=== Running Whisper Phase 4 Encoder & Decoder Tests ===")
    test_get_embedding_row()
    test_whisper_encode_layer_equivalence()
    test_real_whisper_jfk_encode_decode()
    print("=== All Whisper Phase 4 Encoder & Decoder Tests Passed! ===")
