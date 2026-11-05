---
type: reference
title: "Whisper ASR Support: Phase 2 Implementation Documentation"
description: "Implementation and verification of the pure-Mojo DSP Audio Frontend for Whisper ASR: periodic Hann window, 400-point Cooley-Tukey FFT, multi-threaded STFT + Log-Mel Spectrogram, and RIFF WAV loader."
status: completed
tags: [whisper, phase2, audio-dsp, stft, fft, log-mel-spectrogram, wav-loader]
timestamp: 2026-11-05T00:00:00Z
---

# Whisper ASR Support: Phase 2 Implementation Documentation

## 1. Overview

Phase 2 of the Whisper ASR integration into `llamojo` implements the complete **DSP Audio Frontend** in pure Mojo (`src/llamojo/whisper/audio.mojo`), matching the numerical behavior of `whisper.cpp` (`whisper_pcm_to_mel` and `log_mel_spectrogram`):

1. **Precomputed Trigonometric & Periodic Hann Window Tables (`WhisperTrigCache`)**:
   - Precomputes $\sin(2\pi i / 400)$, $\cos(2\pi i / 400)$, and the periodic Hann window $w[i] = 0.5(1 - \cos(2\pi i / 400))$ for $N_{\text{FFT}} = 400$ (25 ms window at 16 kHz).
2. **Cooley-Tukey FFT & Base-Case DFT (`whisper_fft` & `whisper_dft`)**:
   - Because $400 = 2^4 \times 25$, `whisper_fft` recursively splits even and odd indices 4 times down to $N = 25$, where it invokes `whisper_dft` using stride-stepped table lookups into `sin_vals` and `cos_vals`—matching `whisper.cpp` lines 3107–3171.
3. **Multi-Threaded Log-Mel Spectrogram (`log_mel_spectrogram`)**:
   - Applies 200-sample reflective padding on the left and 30-second ($480,000$-sample) + 200-sample zero-padding on the right.
   - Dispatches STFT frames ($160$-sample hop = 10 ms step) across `parallelize` worker threads.
   - Computes power spectrum $|X[k]|^2 = \text{Re}^2 + \text{Im}^2$ for the $n_{\text{fft}} = 201$ positive frequency bins.
   - Projects each frame through the $[n_{\text{mel}} \times 201]$ Mel filterbank (`n_mel = 80` or `128`) with 4-way loop unrolling, applies $\log_{10}(\max(\text{sum}, 10^{-10}))$, clamps to $\max - 8.0$, and normalizes via $(\text{val} + 4.0) / 4.0$.
4. **Zero-Dependency RIFF `.wav` Loader & `.bin` Filterbank Extractor**:
   - `load_wav_file(path)`: Memory-maps `.wav` files via POSIX `mmap64`, parses `RIFF`/`WAVE`/`fmt `/`data` chunks, and converts 16-bit PCM or 32-bit float audio into mono 16 kHz `Float32` samples in $[-1.0, 1.0]$.
   - `load_filters_from_whisper_bin(path)`: Reads the embedded $[n_{\text{mel}} \times n_{\text{fft}}]$ Mel filterbank directly from `whisper.cpp` `.bin` model files (`magic == 0x67676d6c`).

---

## 2. Verification Results

Verified in [`tests/test_whisper_phase2.mojo`](../../tests/test_whisper_phase2.mojo):
- **400-Point FFT vs. Naive DFT**: Verified across all 201 frequency bins with error $< 10^{-3}$.
- **Multi-Threaded Determinism**: Verified bit-for-bit parity between single-thread (`workers=1`) and multi-thread (`workers=4`) execution.
- **Real Audio & Model Integration**: Loaded `/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav` ($176,000$ samples = 11.0 seconds) and extracted the `[80 x 201]` filterbank from `/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin`, producing the exact `[80 x 4100]` Log-Mel spectrogram (`n_len_org = 1099`).
