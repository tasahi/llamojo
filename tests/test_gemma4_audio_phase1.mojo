""" Unit & parity tests for Gemma 4 Audio Phase 1 DSP Frontend (`src/llamojo/gemma4_audio/frontend.mojo`).

Tests:
1. `Gemma4AudioFrontendCache`: 320-sample periodic Hann window zero-padded to 512,
   128-bin HTK Mel filterbank (`0..8000 Hz`, no Slaney area norm).
2. `gemma4_fft_512` vs `gemma4_dft_512`: exact agreement (`atol=1e-3`) across all 257 bins.
3. Frame & token count formulas (`1 s` -> 99 frames -> 25 tokens, `30 s` -> 2999 -> 750,
   `31 s` -> 2 chunks).
4. Single-thread vs multi-thread parity and log-floor bounds on a synthetic 440 Hz + 1 kHz signal.
5. Real WAV file integration (`jfk.wav` if present).
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.math import sin, cos, log, isnan, isinf
from std.os.path import exists

from llamojo import (
    GEMMA4_AUDIO_SAMPLE_RATE,
    GEMMA4_AUDIO_N_MEL,
    GEMMA4_AUDIO_N_FFT,
    GEMMA4_AUDIO_N_FFT_BINS,
    GEMMA4_AUDIO_WINDOW_LEN,
    GEMMA4_AUDIO_HOP_LENGTH,
    GEMMA4_AUDIO_PAD_LEFT,
    GEMMA4_AUDIO_CHUNK_SAMPLES,
    GEMMA4_AUDIO_MEL_FLOOR,
    Gemma4AudioMel,
    Gemma4AudioFrontendCache,
    gemma4_dft_512,
    gemma4_fft_512,
    gemma4_audio_pt_frames,
    gemma4_audio_n_tokens_for_frames,
    gemma4_audio_chunk_mel,
    gemma4_audio_preprocess,
    gemma4_audio_preprocess_wav,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime M_PI_F64: Float64 = 3.14159265358979323846


def test_cache_and_htk_filters() raises:
    print("Testing Gemma4AudioFrontendCache (Hann window & 128-bin HTK Mel filterbank)...")
    var cache = Gemma4AudioFrontendCache()

    assert_equal(cache.n_mel, 128)
    assert_equal(cache.n_fft, 512)
    assert_equal(cache.n_fft_bins, 257)

    # Periodic Hann window of length 320: w[0]=0.0, w[160]=1.0, w[320..511]=0.0
    assert_almost_equal(cache.hann_window[unsafe_offset=0], Float32(0.0), atol=1e-6)
    assert_almost_equal(cache.hann_window[unsafe_offset=160], Float32(1.0), atol=1e-6)
    for i in range(320, 512):
        assert_almost_equal(cache.hann_window[unsafe_offset=i], Float32(0.0), atol=1e-7)

    # HTK Mel filterbank invariants: non-negative, <= 1.0 (no Slaney scaling > 1).
    # Note: bin 0 spans [0 Hz, 27.89 Hz] < bin_hz_step (31.25 Hz), so row 0 has sum 0.0
    # (matching mtmd-audio.cpp & HF Gemma4AudioFeatureExtractor); rows 1..127 have positive sums.
    for m in range(cache.n_mel):
        var row_sum: Float64 = 0.0
        for k in range(cache.n_fft_bins):
            var v = cache.mel_filters[unsafe_offset=m * cache.n_fft_bins + k]
            assert_true(v >= 0.0 and v <= 1.00001)
            row_sum += Float64(v)
        if m == 0:
            assert_almost_equal(Float32(row_sum), Float32(0.0), atol=1e-7)
        else:
            assert_true(row_sum > 0.0)

    cache.free()
    print("  ✓ Hann window (320 zero-padded to 512) and 128-bin HTK Mel filterbank verified")


def test_fft_512_vs_dft() raises:
    print("Testing 512-point iterative radix-2 FFT vs naive DFT...")
    var cache = Gemma4AudioFrontendCache()

    var inp = unsafe_alloc[Float32](GEMMA4_AUDIO_N_FFT)
    var fft_out = unsafe_alloc[Float32](GEMMA4_AUDIO_N_FFT * 2)
    var dft_out = unsafe_alloc[Float32](GEMMA4_AUDIO_N_FFT * 2)

    for i in range(GEMMA4_AUDIO_N_FFT):
        var s = Float32(
            sin(Float64(i) * 0.17) + 0.6 * cos(Float64(i) * 0.41) - 0.3 * sin(Float64(i) * 0.83)
        ) * cache.hann_window[unsafe_offset=i]
        inp[unsafe_offset=i] = s

    gemma4_fft_512(inp, fft_out, cache.sin_vals, cache.cos_vals)
    gemma4_dft_512(inp, dft_out, cache.sin_vals, cache.cos_vals)

    for k in range(GEMMA4_AUDIO_N_FFT_BINS):
        assert_almost_equal(fft_out[unsafe_offset=2 * k + 0], dft_out[unsafe_offset=2 * k + 0], atol=1e-3)
        assert_almost_equal(fft_out[unsafe_offset=2 * k + 1], dft_out[unsafe_offset=2 * k + 1], atol=1e-3)

    inp.unsafe_free()
    fft_out.unsafe_free()
    dft_out.unsafe_free()
    cache.free()
    print("  ✓ 512-point radix-2 FFT matches naive DFT (< 1e-3 across all 257 bins)")


def test_frame_and_chunk_counts() raises:
    print("Testing semicausal frame counts, Conformer token counts, and 30s chunking...")
    assert_equal(gemma4_audio_pt_frames(160), 0)
    assert_equal(gemma4_audio_pt_frames(161), 1)
    assert_equal(gemma4_audio_pt_frames(16000), 99)
    assert_equal(gemma4_audio_pt_frames(480000), 2999)

    assert_equal(gemma4_audio_n_tokens_for_frames(99), 25)
    assert_equal(gemma4_audio_n_tokens_for_frames(2999), 750)

    var cache = Gemma4AudioFrontendCache()
    # 31 seconds -> 496,000 samples -> 2 chunks (2999 frames, 99 frames)
    var n_samples = 496000
    var samples = unsafe_alloc[Float32](n_samples)
    for i in range(n_samples):
        samples[unsafe_offset=i] = Float32(0.1 * sin(2.0 * M_PI_F64 * 440.0 * Float64(i) / 16000.0))

    var chunks = gemma4_audio_preprocess(samples, n_samples, cache, 2)
    assert_equal(len(chunks), 2)
    assert_equal(chunks[0].n_mel, 128)
    assert_equal(chunks[0].n_len, 2999)
    assert_equal(chunks[1].n_mel, 128)
    assert_equal(chunks[1].n_len, 99)

    for idx in range(len(chunks)):
        chunks[idx].free()
    samples.unsafe_free()
    cache.free()
    print("  ✓ 1s (99 frames -> 25 tok), 30s (2999 frames -> 750 tok), and 31s (2 chunks) verified")


def test_multithreaded_mel_parity() raises:
    print("Testing 1-thread vs 4-thread Gemma 4 Log-Mel spectrogram parity...")
    var cache = Gemma4AudioFrontendCache()
    var n_samples = 16000
    var samples = unsafe_alloc[Float32](n_samples)
    for i in range(n_samples):
        var t = Float64(i) / 16000.0
        samples[unsafe_offset=i] = Float32(
            0.5 * sin(2.0 * M_PI_F64 * 440.0 * t) + 0.3 * sin(2.0 * M_PI_F64 * 1000.0 * t)
        )

    var mel_1t = Gemma4AudioMel()
    var mel_4t = Gemma4AudioMel()

    var ok1 = gemma4_audio_chunk_mel(samples, n_samples, cache, mel_1t, 1)
    var ok4 = gemma4_audio_chunk_mel(samples, n_samples, cache, mel_4t, 4)
    assert_true(ok1 and ok4)
    assert_equal(mel_1t.n_mel, 128)
    assert_equal(mel_1t.n_len, 99)
    assert_equal(mel_4t.n_len, 99)

    var floor_log = Float32(log(GEMMA4_AUDIO_MEL_FLOOR))
    var total = mel_1t.n_mel * mel_1t.n_len
    for idx in range(total):
        var v1 = mel_1t.data[unsafe_offset=idx]
        var v4 = mel_4t.data[unsafe_offset=idx]
        assert_true(not isnan(v1) and not isinf(v1))
        assert_true(v1 >= floor_log - 1e-4)
        assert_almost_equal(v1, v4, atol=1e-5)

    # Cross-check against reference mtmd_audio_preprocessor_gemma4a / HF Gemma4AudioFeatureExtractor values
    var n_len = mel_1t.n_len
    assert_almost_equal(mel_1t.data[unsafe_offset=0 * n_len + 0], Float32(-6.9077553), atol=1e-4)
    assert_almost_equal(mel_1t.data[unsafe_offset=15 * n_len + 10], Float32(-2.4016533), atol=1e-4)
    assert_almost_equal(mel_1t.data[unsafe_offset=30 * n_len + 50], Float32(-1.3582773), atol=1e-4)
    assert_almost_equal(mel_1t.data[unsafe_offset=31 * n_len + 50], Float32(-2.0519731), atol=1e-4)
    assert_almost_equal(mel_1t.data[unsafe_offset=64 * n_len + 98], Float32(-6.1377015), atol=1e-4)

    mel_1t.free()
    mel_4t.free()
    samples.unsafe_free()
    cache.free()
    print("  ✓ 1-thread and 4-thread Log-Mel outputs match within 1e-5 and satisfy log floor")


def test_real_wav_integration() raises:
    var wav_path = String("/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav")
    if not exists(wav_path):
        print("  (Skipping real WAV integration test; jfk.wav not found)")
        return

    print("Testing real WAV preprocessing on jfk.wav...")
    var cache = Gemma4AudioFrontendCache()
    var chunks = gemma4_audio_preprocess_wav(wav_path, cache, 4)
    assert_equal(len(chunks), 1)
    assert_equal(chunks[0].n_mel, 128)
    assert_true(chunks[0].n_len > 1000 and chunks[0].n_len <= 2999)

    var floor_log = Float32(log(GEMMA4_AUDIO_MEL_FLOOR))
    var min_v: Float32 = 1e10
    var max_v: Float32 = -1e10
    for idx in range(chunks[0].n_mel * chunks[0].n_len):
        var v = chunks[0].data[unsafe_offset=idx]
        assert_true(not isnan(v) and not isinf(v))
        assert_true(v >= floor_log - 1e-4)
        if v < min_v:
            min_v = v
        if v > max_v:
            max_v = v

    print(
        "  ✓ jfk.wav ->",
        chunks[0].n_len,
        "mel frames (",
        gemma4_audio_n_tokens_for_frames(chunks[0].n_len),
        "audio tokens ), range [",
        min_v,
        ",",
        max_v,
        "]",
    )

    chunks[0].free()
    cache.free()


def main() raises:
    print("============================================================")
    print("Gemma 4 Audio Phase 1 DSP Frontend Tests")
    print("============================================================")
    test_cache_and_htk_filters()
    test_fft_512_vs_dft()
    test_frame_and_chunk_counts()
    test_multithreaded_mel_parity()
    test_real_wav_integration()
    print("All Gemma 4 Audio Phase 1 tests passed! ✓")
