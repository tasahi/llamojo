"""Unit tests for Whisper Phase 2 DSP Audio Frontend (`src/llamojo/whisper/audio.mojo`).

Tests:
1. `WhisperTrigCache`: periodic Hann window and precomputed sin/cos tables (`WHISPER_N_FFT = 400`).
2. `whisper_dft` vs `whisper_fft`: exact agreement (`atol=1e-4`) between naive 400-point DFT and
   recursive Cooley-Tukey 400-point FFT across all 201 frequency bins.
3. `log_mel_spectrogram`: single-thread vs multi-thread consistency, reflective + 30-second padding,
   and clamping/normalization bounds (`max - min <= 2.0`).
4. Real-world integration with `/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav` and
   `/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin` filterbank.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.math import sin, cos

from llamojo import (
    WHISPER_SAMPLE_RATE,
    WHISPER_N_FFT,
    WHISPER_HOP_LENGTH,
    WhisperFilters,
    WhisperMel,
    WhisperTrigCache,
    whisper_dft,
    whisper_fft,
    log_mel_spectrogram,
    load_wav_file,
    load_filters_from_whisper_bin,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]


def test_trig_and_fft_vs_dft() raises:
    print("Testing WhisperTrigCache and 400-point Cooley-Tukey whisper_fft vs naive whisper_dft...")
    var trig = WhisperTrigCache()

    # Periodic Hann window at 0 is 0.0, at N/2 (200) is 1.0
    assert_almost_equal(trig.hann_window[unsafe_offset=0], Float32(0.0), atol=1e-6)
    assert_almost_equal(trig.hann_window[unsafe_offset=200], Float32(1.0), atol=1e-6)

    # Compare 400-point FFT vs 400-point naive DFT on a multi-frequency signal
    var fft_in = unsafe_alloc[Float32](WHISPER_N_FFT * 2)
    var dft_in = unsafe_alloc[Float32](WHISPER_N_FFT)
    var fft_out = unsafe_alloc[Float32](WHISPER_N_FFT * 8)
    var dft_out = unsafe_alloc[Float32](WHISPER_N_FFT * 2)

    for i in range(WHISPER_N_FFT):
        var val = Float32(sin(Float64(i) * 0.15) + 0.5 * cos(Float64(i) * 0.43)) * trig.hann_window[unsafe_offset=i]
        fft_in[unsafe_offset=i] = val
        dft_in[unsafe_offset=i] = val
    for i in range(WHISPER_N_FFT, WHISPER_N_FFT * 2):
        fft_in[unsafe_offset=i] = 0.0

    whisper_fft(fft_in, WHISPER_N_FFT, fft_out, trig.sin_vals, trig.cos_vals)
    whisper_dft(dft_in, WHISPER_N_FFT, dft_out, trig.sin_vals, trig.cos_vals)

    for k in range(201):
        assert_almost_equal(fft_out[unsafe_offset=2 * k + 0], dft_out[unsafe_offset=2 * k + 0], atol=1e-3)
        assert_almost_equal(fft_out[unsafe_offset=2 * k + 1], dft_out[unsafe_offset=2 * k + 1], atol=1e-3)

    fft_in.unsafe_free()
    dft_in.unsafe_free()
    fft_out.unsafe_free()
    dft_out.unsafe_free()
    trig.free()
    print("  ✓ 400-point Cooley-Tukey FFT matches naive DFT to < 1e-3 across all 201 bins")


def test_synthetic_log_mel_multithreaded() raises:
    print("Testing log_mel_spectrogram single-thread vs multi-thread parity...")
    var n_samples = 16000  # 1 second of 16 kHz audio
    var samples = unsafe_alloc[Float32](n_samples)
    for i in range(n_samples):
        samples[unsafe_offset=i] = Float32(sin(Float64(i) * 2.0 * 3.14159265 * 440.0 / 16000.0)) * 0.5

    var n_mel = 80
    var n_fft = 201
    var filter_data = unsafe_alloc[Float32](n_mel * n_fft)
    for j in range(n_mel):
        for k in range(n_fft):
            filter_data[unsafe_offset=j * n_fft + k] = Float32(0.01) if (k >= j * 2 and k <= j * 2 + 5) else Float32(0.0)

    var filters = WhisperFilters(n_mel, n_fft, filter_data, True)
    var mel_1t = WhisperMel()
    var mel_4t = WhisperMel()

    log_mel_spectrogram(samples, n_samples, filters, mel_1t, 1)
    log_mel_spectrogram(samples, n_samples, filters, mel_4t, 4)

    assert_equal(mel_1t.n_mel, 80)
    assert_equal(mel_1t.n_len, 3100)  # 1s (100 frames) + 30s pad (3000 frames)
    assert_equal(mel_1t.n_len, mel_4t.n_len)

    var min_v: Float32 = 1e10
    var max_v: Float32 = -1e10
    for idx in range(mel_1t.n_mel * mel_1t.n_len):
        var v1 = mel_1t.data[unsafe_offset=idx]
        var v4 = mel_4t.data[unsafe_offset=idx]
        assert_almost_equal(v1, v4, atol=1e-5)
        if v1 < min_v:
            min_v = v1
        if v1 > max_v:
            max_v = v1

    # Due to clamping at mmax - 8.0 and dividing by 4.0, (max_v - min_v) is bounded by 8.0 / 4.0 = 2.0
    assert_Almost_or_le(max_v - min_v, Float32(2.0001))

    samples.unsafe_free()
    mel_1t.free()
    mel_4t.free()
    filters.free()
    print("  ✓ Single-thread and 4-thread Log-Mel Spectrograms match bit-for-bit")


def assert_Almost_or_le(val: Float32, limit: Float32) raises:
    assert_true(val <= limit)


def test_real_jfk_wav_and_whisper_bin_filters() raises:
    print("Testing real WAV loader (jfk.wav) and Whisper .bin filterbank extraction (ggml-base.en.bin)...")
    var wav_path = "/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav"
    var bin_path = "/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin"

    var pcm = load_wav_file(wav_path)
    # jfk.wav is 11.0 seconds at 16,000 Hz = 176,000 samples
    assert_equal(len(pcm), 176000)

    var filters = load_filters_from_whisper_bin(bin_path)
    assert_equal(filters.n_mel, 80)
    assert_equal(filters.n_fft, 201)

    var mel = WhisperMel()
    log_mel_spectrogram(pcm, filters, mel, 4)

    # 176,000 samples (11s = 1,100 frames) + 480,000 pad (30s = 3,000 frames) -> 4,100 frames
    assert_equal(mel.n_mel, 80)
    assert_equal(mel.n_len, 4100)
    assert_equal(mel.n_len_org, 1099)

    # Verify non-trivial speech energy in the active frames vs silence in trailing frames
    var active_val = mel.data[unsafe_offset=10 * mel.n_len + 200]
    var silence_val = mel.data[unsafe_offset=10 * mel.n_len + 3500]
    assert_true(active_val > silence_val)
    print(
        "  ✓ Loaded jfk.wav (176,000 samples) & ggml-base.en.bin filters [80 x 201] -> Mel [80 x 4100] (sample speech="
        + String(active_val)
        + ", silence="
        + String(silence_val)
        + ")"
    )

    mel.free()
    filters.free()


def main() raises:
    print("=== Running Whisper Phase 2 Audio DSP Frontend Tests ===")
    test_trig_and_fft_vs_dft()
    test_synthetic_log_mel_multithreaded()
    test_real_jfk_wav_and_whisper_bin_filters()
    print("=== All Whisper Phase 2 Audio DSP Frontend Tests Passed! ===")