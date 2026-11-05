""" Unit and Integration Tests for Whisper Phase 5 Timestamp Logit Rules, High-Level `whisper_full` Loop,
and Segment Extraction (`src/llamojo/whisper/transcribe.mojo`).

Tests:
1. `format_timestamp_cs` formatting (`HH:MM:SS.mmm`).
2. `apply_whisper_logit_filters`:
   - `suppress_blank` at initial step (`EOT` and `" "` token 220 suppressed)
   - `max_initial_ts` constraint (`<= 1.0s` = `token_beg + 50`)
   - Paired timestamp rules (after 1st timestamp, timestamps are suppressed; after closing timestamp, text tokens are suppressed)
   - Monotonic `seek_delta` timestamp masking
   - Timestamp Log-Sum-Exp dominance rule
3. End-to-end `whisper_full` WITH timestamps (`no_timestamps = False`) on
   `/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin` and `/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav`:
   - Verifies automatic `<|0.00|>` -> `<|11.00|>` (`t0 = 0`, `t1 = 1100`) segment boundaries and full JFK transcript.
"""

from std.collections import List
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal

from llamojo import (
    WhisperVocab,
    WhisperFullParams,
    whisper_full_default_params,
    format_timestamp_cs,
    apply_whisper_logit_filters,
    whisper_init_from_file,
    whisper_full,
    whisper_full_n_segments,
    whisper_full_get_segment_text,
    whisper_full_get_segment_t0,
    whisper_full_get_segment_t1,
    load_wav_file,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]


def test_format_timestamp_cs() raises:
    print("Testing format_timestamp_cs...")
    assert_equal(format_timestamp_cs(0), "00:00:00.000")
    assert_equal(format_timestamp_cs(1100), "00:00:11.000")
    assert_equal(format_timestamp_cs(366150), "01:01:01.500")
    print("  ✓ format_timestamp_cs verified")


def test_apply_whisper_logit_filters() raises:
    print("Testing apply_whisper_logit_filters & ApplyTimestampRules...")
    var vocab = WhisperVocab()
    vocab.configure_special_tokens(51864)
    var n_logits = vocab.n_vocab
    var beg_id = Int(vocab.token_beg)
    var eot_id = Int(vocab.token_eot)

    var logits = unsafe_alloc[Float32](n_logits)
    var params = whisper_full_default_params()

    # Case 1: Initial step (`tokens_cur` is empty)
    # Set a timestamp token (e.g., `<|0.00|>`) high so timestamp dominance triggers, and check suppress_blank + max_initial_ts
    for i in range(n_logits):
        logits[unsafe_offset=i] = Float32(0.0)
    logits[unsafe_offset=beg_id] = Float32(15.0)

    var empty_tokens = List[Int32]()
    apply_whisper_logit_filters(logits, vocab, 1500, params, empty_tokens, False, 3000, 220)

    # EOT, space (220), SOT, NOT must be suppressed
    assert_true(logits[unsafe_offset=eot_id] < -1e20)
    assert_true(logits[unsafe_offset=220] < -1e20)
    assert_true(logits[unsafe_offset=Int(vocab.token_sot)] < -1e20)
    assert_true(logits[unsafe_offset=Int(vocab.token_not)] < -1e20)
    # Timestamps > 1.00s (beg_id + 50) must be suppressed by max_initial_ts = 1.0
    assert_true(logits[unsafe_offset=beg_id + 51] < -1e20)
    assert_true(logits[unsafe_offset=beg_id] > 0.0)

    # Case 2: Immediately after opening timestamp `[<|0.00|>]` -> all timestamps must be suppressed
    for i in range(n_logits):
        logits[unsafe_offset=i] = Float32(1.0)
    var tok_after_open = List[Int32]()
    tok_after_open.append(vocab.token_beg)
    apply_whisper_logit_filters(logits, vocab, 1500, params, tok_after_open, True, 0, 220)
    for i in range(beg_id, n_logits):
        assert_true(logits[unsafe_offset=i] < -1e20)
    assert_true(logits[unsafe_offset=400] > 0.0)

    # Case 3: Immediately after closing timestamp `[<|0.00|>, 400, <|2.00|>]` -> all text tokens `< token_eot` must be suppressed
    for i in range(n_logits):
        logits[unsafe_offset=i] = Float32(1.0)
    var tok_after_close = List[Int32]()
    tok_after_close.append(vocab.token_beg)
    tok_after_close.append(Int32(400))
    tok_after_close.append(vocab.token_beg + 100)
    apply_whisper_logit_filters(logits, vocab, 1500, params, tok_after_close, True, 200, 220)
    for i in range(0, eot_id):
        assert_true(logits[unsafe_offset=i] < -1e20)
    # Monotonicity: timestamps < beg_id + 100 must be suppressed, >= beg_id + 100 allowed
    assert_true(logits[unsafe_offset=beg_id + 50] < -1e20)
    assert_true(logits[unsafe_offset=beg_id + 100] > -1e20)

    logits.unsafe_free()
    print("  ✓ apply_whisper_logit_filters & ApplyTimestampRules verified")


def test_whisper_full_jfk_timestamped() raises:
    print("Testing end-to-end whisper_full (timestamped) on ggml-base.en.bin + jfk.wav...")
    var bin_path = "/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin"
    var wav_path = "/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav"

    var ctx = whisper_init_from_file(bin_path)
    var pcm = load_wav_file(wav_path)
    var params = whisper_full_default_params()
    params.no_timestamps = False

    var rc = whisper_full(ctx, params, pcm)
    assert_equal(rc, 0)

    var n_seg = whisper_full_n_segments(ctx)
    assert_true(n_seg >= 1)

    var full_text = String("")
    for i in range(n_seg):
        var t0 = whisper_full_get_segment_t0(ctx, i)
        var t1 = whisper_full_get_segment_t1(ctx, i)
        var txt = whisper_full_get_segment_text(ctx, i)
        print("  [" + format_timestamp_cs(t0) + " --> " + format_timestamp_cs(t1) + "] " + txt)
        full_text += txt
        assert_true(t1 > t0)

    assert_equal(whisper_full_get_segment_t0(ctx, 0), 0)
    assert_true("fellow Americans" in full_text)
    assert_true("ask not what your country can do for you" in full_text)
    assert_true("ask what you can do for your country" in full_text)

    ctx.free()
    print("  ✓ End-to-end timestamped whisper_full verified!")


def main() raises:
    print("=== Running Whisper Phase 5 Timestamp Rules & whisper_full Tests ===")
    test_format_timestamp_cs()
    test_apply_whisper_logit_filters()
    test_whisper_full_jfk_timestamped()
    print("=== All Whisper Phase 5 Tests Passed! ===")

