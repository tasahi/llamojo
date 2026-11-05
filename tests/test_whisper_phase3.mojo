""" Unit tests for Whisper Phase 3 Dual Model Loader (`src/llamojo/whisper/loader.mojo`).\n
Tests:
1. Real-model loading of `/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin` (`0x67676d6c`):
   - Hyperparameter verification (`n_vocab=51864`, `n_audio_ctx=1500`, `n_audio_state=512`,
     `n_audio_head=8`, `n_audio_layer=6`, `n_text_ctx=448`, `n_text_layer=6`, `n_mels=80`).
   - Mel filterbank zero-copy mmap binding (`[80 x 201]`).
   - Vocabulary & special/timestamp token generation (`token_eot=50256`, `token_sot=50257`,
     `token_beg=50363`, `[_TT_1]`..`[_TT_1500]`).
   - Complete 245-tensor binding across all 6 encoder layers and 6 decoder layers (including
     `Q`/`V`/`out`/`mlp` biases and unbiased `K`).
2. `WhisperVocab` multilingual offset calculation (`n_vocab >= 51865`).
"""

from std.testing import assert_true, assert_equal, assert_almost_equal

from llamojo import (
    GGMLType,
    WhisperVocab,
    WhisperModel,
    whisper_load_model_from_file,
)


def test_whisper_vocab_multilingual() raises:
    print("Testing WhisperVocab English vs Multilingual special token offsets...")
    var vocab_en = WhisperVocab()
    vocab_en.configure_special_tokens(51864)
    assert_true(not vocab_en.is_multilingual())
    assert_equal(Int(vocab_en.token_eot), 50256)
    assert_equal(Int(vocab_en.token_sot), 50257)
    assert_equal(Int(vocab_en.token_beg), 50363)
    assert_equal(len(vocab_en.id_to_token), 51864)
    assert_equal(vocab_en.id_to_token[50363], "[_BEG_]")
    assert_equal(vocab_en.id_to_token[50364], "[_TT_1]")

    # Multilingual Whisper (n_vocab = 51865, 99 languages -> dt = 1)
    var vocab_ml = WhisperVocab()
    vocab_ml.configure_special_tokens(51865)
    assert_true(vocab_ml.is_multilingual())
    assert_equal(vocab_ml.num_languages(), 99)
    assert_equal(Int(vocab_ml.token_eot), 50257)
    assert_equal(Int(vocab_ml.token_sot), 50258)
    assert_equal(Int(vocab_ml.token_translate), 50358)
    assert_equal(Int(vocab_ml.token_transcribe), 50359)
    assert_equal(Int(vocab_ml.token_beg), 50364)
    print("  ✓ WhisperVocab English & Multilingual special token offsets verified")


def test_load_real_whisper_base_en_bin() raises:
    print("Testing whisper_load_model_from_file on /mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin...")
    var bin_path = "/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin"
    var model = whisper_load_model_from_file(bin_path)

    # 1. Verify Hyperparameters
    assert_equal(model.hparams.n_vocab, 51864)
    assert_equal(model.hparams.n_audio_ctx, 1500)
    assert_equal(model.hparams.n_audio_state, 512)
    assert_equal(model.hparams.n_audio_head, 8)
    assert_equal(model.hparams.n_audio_layer, 6)
    assert_equal(model.hparams.n_text_ctx, 448)
    assert_equal(model.hparams.n_text_state, 512)
    assert_equal(model.hparams.n_text_head, 8)
    assert_equal(model.hparams.n_text_layer, 6)
    assert_equal(model.hparams.n_mels, 80)
    assert_equal(model.hparams.ftype, 1)

    # 2. Verify Mel Filters
    assert_equal(model.filters.n_mel, 80)
    assert_equal(model.filters.n_fft, 201)
    assert_true(Int(model.filters.data) != 0)

    # 3. Verify Vocabulary
    assert_equal(len(model.vocab.id_to_token), 51864)
    assert_equal(model.vocab.id_to_token[0], "!")
    assert_equal(model.vocab.id_to_token[50256], "<|endoftext|>")
    assert_equal(model.vocab.id_to_token[50257], "[_SOT_]")
    assert_equal(model.vocab.id_to_token[50363], "[_BEG_]")

    # 4. Verify All 245 Tensors Loaded & Bound
    assert_equal(model.n_loaded, 245)
    assert_true(Int(model.e_pe) != 0)
    assert_true(Int(model.e_conv_1_w) != 0)
    assert_true(Int(model.e_conv_1_b) != 0)
    assert_true(Int(model.e_conv_2_w) != 0)
    assert_true(Int(model.e_conv_2_b) != 0)
    assert_true(Int(model.e_ln_w) != 0)
    assert_true(Int(model.e_ln_b) != 0)
    assert_equal(model.dtype_conv_1, GGMLType.F16)
    assert_equal(model.dtype_conv_2, GGMLType.F16)

    assert_true(Int(model.d_pe) != 0)
    assert_true(Int(model.d_te) != 0)
    assert_true(Int(model.d_ln_w) != 0)
    assert_true(Int(model.d_ln_b) != 0)
    assert_equal(model.dtype_d_te, GGMLType.F16)

    assert_equal(len(model.layers_encoder), 6)
    for l in range(6):
        var el = model.layers_encoder[l]
        assert_true(Int(el.attn_ln_0_w) != 0)
        assert_true(Int(el.attn_ln_0_b) != 0)
        assert_true(Int(el.attn_q_w) != 0)
        assert_true(Int(el.attn_q_b) != 0)
        assert_true(Int(el.attn_k_w) != 0)
        assert_true(Int(el.attn_v_w) != 0)
        assert_true(Int(el.attn_v_b) != 0)
        assert_true(Int(el.attn_ln_1_w) != 0)
        assert_true(Int(el.attn_ln_1_b) != 0)
        assert_true(Int(el.mlp_ln_w) != 0)
        assert_true(Int(el.mlp_ln_b) != 0)
        assert_true(Int(el.mlp_0_w) != 0)
        assert_true(Int(el.mlp_0_b) != 0)
        assert_true(Int(el.mlp_1_w) != 0)
        assert_true(Int(el.mlp_1_b) != 0)

    assert_equal(len(model.layers_decoder), 6)
    for l in range(6):
        var dl = model.layers_decoder[l]
        assert_true(Int(dl.attn_ln_0_w) != 0)
        assert_true(Int(dl.attn_ln_0_b) != 0)
        assert_true(Int(dl.attn_q_w) != 0)
        assert_true(Int(dl.attn_q_b) != 0)
        assert_true(Int(dl.attn_k_w) != 0)
        assert_true(Int(dl.attn_v_w) != 0)
        assert_true(Int(dl.attn_v_b) != 0)
        assert_true(Int(dl.attn_ln_1_w) != 0)
        assert_true(Int(dl.attn_ln_1_b) != 0)

        assert_true(Int(dl.cross_attn_ln_0_w) != 0)
        assert_true(Int(dl.cross_attn_ln_0_b) != 0)
        assert_true(Int(dl.cross_attn_q_w) != 0)
        assert_true(Int(dl.cross_attn_q_b) != 0)
        assert_true(Int(dl.cross_attn_k_w) != 0)
        assert_true(Int(dl.cross_attn_v_w) != 0)
        assert_true(Int(dl.cross_attn_v_b) != 0)
        assert_true(Int(dl.cross_attn_ln_1_w) != 0)
        assert_true(Int(dl.cross_attn_ln_1_b) != 0)

        assert_true(Int(dl.mlp_ln_w) != 0)
        assert_true(Int(dl.mlp_ln_b) != 0)
        assert_true(Int(dl.mlp_0_w) != 0)
        assert_true(Int(dl.mlp_0_b) != 0)
        assert_true(Int(dl.mlp_1_w) != 0)
        assert_true(Int(dl.mlp_1_b) != 0)

    model.free()
    print("  ✓ Loaded all 245 tensors, 51,864 tokens, and [80 x 201] mel filters from ggml-base.en.bin zero-copy")


def main() raises:
    print("=== Running Whisper Phase 3 Dual Model Loader Tests ===")
    test_whisper_vocab_multilingual()
    test_load_real_whisper_base_en_bin()
    print("=== All Whisper Phase 3 Dual Model Loader Tests Passed! ===")
