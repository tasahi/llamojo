---
type: reference
title: "Gemma 4 Audio Support: Phase 1 Implementation Documentation"
description: "Implementation and verification of the pure-Mojo Gemma 4 E2B (gemma4a) DSP Audio Frontend: 320-sample periodic Hann window, 512-point radix-2 FFT, 128-bin HTK Mel filterbank, semicausal padding, and 30-second chunking."
status: completed
tags: [gemma4, gemma4a, audio, dsp, fft, mel-spectrogram, phase1, mojo]
timestamp: 2026-11-25T00:00:00Z
---

# Gemma 4 Audio Support: Phase 1 Implementation Documentation

## 1. Overview

Phase 1 of the [Gemma 4 E2B Audio Transcription Implementation Plan](./gemma4_audio_transcription_plan.md) delivers the pure-Mojo **Audio DSP Frontend** in [`src/llamojo/gemma4_audio/frontend.mojo`](../../src/llamojo/gemma4_audio/frontend.mojo) (exported via [`src/llamojo/gemma4_audio/__init__.mojo`](../../src/llamojo/gemma4_audio/__init__.mojo) and [`src/llamojo/__init__.mojo`](../../src/llamojo/__init__.mojo)), matching the exact numerical formulation of Hugging Face `Gemma4AudioFeatureExtractor` and `llama.cpp`'s `mtmd_audio_preprocessor_gemma4a`.

### Parameter Comparison vs. Whisper Frontend

| Feature | Gemma 4 `gemma4a` (`frontend.mojo`) | Whisper (`whisper/audio.mojo`) |
| :--- | :--- | :--- |
| **Sample Rate** | `16,000` Hz (mono `Float32` in `[-1, 1]`) | `16,000` Hz |
| **Mel Bins** | `128` (`0..8000 Hz`) | `80` / `128` |
| **Window Length** | `320` samples (20 ms periodic Hann, zero-padded to `512`) | `400` samples (25 ms) |
| **FFT Size (`n_fft`)** | `512` (`257` frequency bins, `31.25 Hz` spacing) | `400` (`201` frequency bins) |
| **Hop Length** | `160` samples (10 ms) | `160` samples (10 ms) |
| **Mel Scale & Norm** | **HTK** ($2595 \log_{10}(1 + f/700)$), no Slaney area norm (`scale = 1.0`) | Slaney |
| **Spectrum & Log** | Magnitude $\|X\| = \sqrt{\text{Re}^2 + \text{Im}^2}$, natural $\ln(\max(\text{mel}, 0.001))$ | Power $\|X\|^2$, $\log_{10}$ + Whisper clamp/norm |
| **Padding & Chunking** | Semicausal left-pad `160`, `pt_frames = (N + 160 - 321) // 160 + 1`, 30 s (`480,000` sample) chunks | 200-sample reflective left-pad + 30 s trailing zero-pad |

---

## 2. DSP Architecture & Implementation Details

### 2.1 Precomputed Cache (`Gemma4AudioFrontendCache`)
1. **Trigonometric Tables (`sin_vals`, `cos_vals`)**: Precomputes $\sin(2\pi i / 512)$ and $\cos(2\pi i / 512)$ for $i \in [0, 512)$.
2. **Zero-Padded Periodic Hann Window (`hann_window`)**: Computes $w[i] = 0.5(1 - \cos(2\pi i / 320))$ for $0 \le i < 320$ and zero-fills $320 \le i < 512$, matching `mtmd_audio_preprocessor_gemma4a::initialize()` where the window is left-aligned (not center-padded).
3. **128-Bin HTK Mel Filterbank (`mel_filters`)**: Builds a row-major `[128, 257]` triangular filterbank on the HTK scale from `0.0 Hz` to `8000.0 Hz` without Slaney area normalization:
   - $m(f) = 2595 \log_{10}(1 + f / 700)$, $f(m) = 700 (10^{m / 2595} - 1)$.
   - Note: Because bin `0` spans $[0.0, 27.89]\text{ Hz}$ while the FFT bin spacing is $16000 / 512 = 31.25\text{ Hz}$, row `0` evaluates to `0.0` across all FFT bins—matching upstream `mtmd-audio.cpp` and explaining why the `0.001` floor is required before taking `ln`.

### 2.2 Iterative 512-Point Radix-2 Cooley-Tukey FFT (`gemma4_fft_512`)
- Because $N_{\text{FFT}} = 512 = 2^9$, `gemma4_fft_512` executes an iterative 9-bit bit-reversal copy followed by 9 in-place radix-2 butterfly stages with zero per-call heap allocations.
- A naive reference `gemma4_dft_512` is also provided for unit verification.

### 2.3 Semicausal Padding, 30s Chunking & Multi-Threaded STFT
- **Frame Count (`gemma4_audio_pt_frames`)**:
  $$\text{pt\_frames} = \left\lfloor \frac{\text{chunk\_len} + 160 - 321}{160} \right\rfloor + 1$$
  Yields `99` frames for 1 s (`16,000` samples) and `2,999` frames for 30 s (`480,000` samples), which map to `25` and `750` Conformer tokens respectively (`gemma4_audio_n_tokens_for_frames`).
- **Padding**: Left-pads each chunk by `160` zeros and right-pads to at least `(pt_frames - 1) * 160 + 512` samples.
- **Worker Dispatch (`gemma4_audio_chunk_mel`)**: Distributes frames across `parallelize` worker threads (`num_performance_cores()` by default), computing $|X[k]|$, 4-way unrolled `Float64` mel filterbank dot products, and $\ln(\max(\text{sum}, 0.001))$ written to `[128, pt_frames]`.
- **WAV & Multi-Chunk Entrypoints**: `gemma4_audio_preprocess` splits arbitrary-length PCM into `480,000`-sample chunks; `gemma4_audio_preprocess_wav` reuses `load_wav_file` from [`src/llamojo/whisper/audio.mojo`](../../src/llamojo/whisper/audio.mojo).

---

## 3. Public API Surface

Exported from `llamojo.gemma4_audio` and re-exported at top-level `llamojo`:
- Constants: `GEMMA4_AUDIO_SAMPLE_RATE`, `GEMMA4_AUDIO_N_MEL`, `GEMMA4_AUDIO_N_FFT`, `GEMMA4_AUDIO_N_FFT_BINS`, `GEMMA4_AUDIO_WINDOW_LEN`, `GEMMA4_AUDIO_HOP_LENGTH`, `GEMMA4_AUDIO_PAD_LEFT`, `GEMMA4_AUDIO_CHUNK_SECONDS`, `GEMMA4_AUDIO_CHUNK_SAMPLES`, `GEMMA4_AUDIO_MEL_FLOOR`
- Structs: `Gemma4AudioMel`, `Gemma4AudioFrontendCache`
- Functions: `gemma4_dft_512`, `gemma4_fft_512`, `gemma4_audio_pt_frames`, `gemma4_audio_n_tokens_for_frames`, `gemma4_audio_chunk_mel`, `gemma4_audio_preprocess`, `gemma4_audio_preprocess_wav`

---

## 4. Verification Results

Verified via [`tests/test_gemma4_audio_phase1.mojo`](../../tests/test_gemma4_audio_phase1.mojo) (registered in [`run-tests.sh`](../../run-tests.sh)):
- **Hann Window & HTK Filterbank**: Verified `hann_window[0] == 0.0`, `hann_window[160] == 1.0`, `hann_window[320..511] == 0.0`, and HTK filterbank invariants across all `128` rows.
- **512-Point FFT vs. Naive DFT**: Verified agreement within `< 1e-3` across all `257` frequency bins.
- **Frame, Token & Chunk Counts**: Verified `1 s` (`99` frames $\rightarrow$ `25` tokens), `30 s` (`2,999` frames $\rightarrow$ `750` tokens), and `31 s` (`2` chunks).
- **Multi-Threaded & Reference Parity**: Verified `workers=1` vs `workers=4` agreement within `1e-5` and exact parity (`atol=1e-4`) against `mtmd_audio_preprocessor_gemma4a` reference values on a synthetic 440 Hz + 1 kHz signal.
- **Real WAV Integration**: Preprocessed `/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav` into `1` chunk of `[128, 1099]` (`275` audio tokens, value range `[-6.9077554, 3.5981128]`).
- **Whisper Regression Check**: Re-ran [`tests/test_whisper_phase2.mojo`](../../tests/test_whisper_phase2.mojo) with all assertions passing.
