---
type: reference
title: "Whisper ASR Support: Phase 5 Implementation Documentation"
description: "Implementation and verification of Whisper paired timestamp logit rules, the 30-second sliding-window transcription loop (whisper_full), C-ABI shared library exports, and Python Whisper / LlamojoForSpeechSeq2Seq wrappers."
status: completed
tags: [whisper, phase5, timestamps, logit-filters, c-abi, python-bindings, hf-compat, asr]
timestamp: 2026-11-05T00:00:00Z
---

# Whisper ASR Support: Phase 5 Implementation Documentation

## 1. Overview

Phase 5 completes the five-phase integration of **OpenAI Whisper Automatic Speech Recognition (ASR)** into `llamojo`, delivering:
1. **Timestamp Logit Rules & Token Suppression ([`src/llamojo/whisper/transcribe.mojo`](../../src/llamojo/whisper/transcribe.mojo))**:
   - `suppress_blank`: Suppresses `<|endoftext|>` (`token_eot`) and the blank space token (`" "`, ID `220`) on the first generated token of a segment.
   - Control & Language Suppression: Masks `token_sot`, `token_not`, `token_nosp`, `token_solm`, `token_translate`, `token_transcribe`, `token_prev`, and all language ID tokens during text generation.
   - Paired Timestamp Constraints (`ApplyTimestampRules`):
     - Enforces that timestamps appear in pairs (`<|t0|> text ... <|t1|> <|t1|> text ...`).
     - Bounds the opening timestamp via `max_initial_ts` (default $1.0\text{ s} \implies \text{tid}_0 \le 50$).
     - Enforces monotonically non-decreasing timestamps within each 30-second window ($\text{tid} \ge \text{seek\_delta} / 2$).
     - Implements the **Timestamp Log-Sum-Exp Dominance Rule**: when the marginal log-probability over all timestamp tokens $\log \sum_{t \ge \text{beg}} \exp(\text{logprob}_t)$ exceeds the maximum single text-token log-probability $\max_{w < \text{beg}} \text{logprob}_w$, all text tokens are masked to force timestamp emission.
2. **Full 30-Second Seek Loop (`whisper_full`) & `WhisperContext`**:
   - Automatically computes the 80/128-bin Log-Mel spectrogram, steps through 30-second windows via `seek += seek_delta`, runs `whisper_encode` + `whisper_decode`, and splits paired timestamp spans into `WhisperSegment(t0, t1, text)` entries.
3. **C-ABI Shared Library Exports ([`src/llamojo/c_api.mojo`](../../src/llamojo/c_api.mojo#L632-L735))**:
   - Exposes `whisper_init_from_file`, `whisper_free`, `whisper_is_multilingual`, `whisper_full_c`, `whisper_full_wav_c`, `whisper_full_n_segments`, `whisper_full_get_segment_t0`, `whisper_full_get_segment_t1`, and `whisper_full_get_segment_text` in `libllamojo.so`.
4. **Python `Whisper` & Hugging Face `LlamojoForSpeechSeq2Seq` Wrappers**:
   - [`python/llamojo/_lib.py`](../../python/llamojo/_lib.py): Registers ctypes signatures for all `whisper_*` C-ABI symbols.
   - [`python/llamojo/whisper.py`](../../python/llamojo/whisper.py): Provides `Whisper` and `WhisperSegmentResult`.
   - [`python/llamojo/hf_compat.py`](../../python/llamojo/hf_compat.py#L236-L281): Provides `LlamojoForSpeechSeq2Seq` (`from_pretrained`, `transcribe`, `__call__`).

---

## 2. Verification Results

Verified in [`tests/test_whisper_phase5.mojo`](../../tests/test_whisper_phase5.mojo):
- **`format_timestamp_cs`**: Verified centisecond formatting (`0` $\to$ `00:00:00.000`, `1100` $\to$ `00:00:11.000`, `366150` $\to$ `01:01:01.500`).
- **`apply_whisper_logit_filters`**: Verified initial blank suppression, `max_initial_ts` cutoff, paired timestamp state transitions, monotonic `seek_delta` masking, and log-sum-exp dominance.
- **End-to-End Timestamped Transcription (`ggml-base.en.bin` + `jfk.wav`)**:
  - Verified automatic timestamped segment extraction `[00:00:00.000 --> 00:00:11.000]` with full JFK speech transcription.

