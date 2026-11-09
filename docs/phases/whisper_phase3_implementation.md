---
type: reference
title: "Whisper ASR Support: Phase 3 Implementation Documentation"
description: "Implementation and verification of the Dual Model Loader for Whisper ASR in llamojo: zero-copy mmap support for legacy whisper.cpp .bin files (0x67676d6c) and GGUF v2/v3 files, vocabulary synthesis, and structured encoder/decoder weight binding."
status: completed
tags: [whisper, phase3, model-loader, gguf, ggml-bin, mmap, vocabulary]
timestamp: 2026-11-05T00:00:00Z
---

# Whisper ASR Support: Phase 3 Implementation Documentation

## 1. Overview

Phase 3 of the Whisper ASR integration into `llamojo` implements the **Dual Model Loader & Structured Weight Binder** (`src/llamojo/whisper/loader.mojo`), allowing `llamojo` to transparently load both:
1. **Legacy `whisper.cpp` `.bin` models (`magic == 0x67676d6c`)**: Zero-copy memory-mapped via POSIX `mmap64` (`_load_whisper_from_bin`).
2. **Modern GGUF v2/v3 Whisper models (`magic == 0x46554747`)**: Loaded via `llamojo.gguf.gguf_load_from_file` (`_load_whisper_from_gguf`).

---

## 2. Architecture & Data Structures

### 2.1 `WhisperHParams` & `WhisperVocab`
- **`WhisperHParams`**: Captures all 11 model hyperparameters (`n_vocab`, `n_audio_ctx`, `n_audio_state`, `n_audio_head`, `n_audio_layer`, `n_text_ctx`, `n_text_state`, `n_text_head`, `n_text_layer`, `n_mels`, `ftype`, `eps`).
- **`WhisperVocab`**:
  - Parses the base GPT-2 BPE vocabulary ($50,257$ tokens in `.bin` files) and automatically computes dynamic special token offsets (`token_eot`, `token_sot`, `token_translate`, `token_transcribe`, `token_solm`, `token_prev`, `token_nosp`, `token_not`, `token_beg`) for both English-only ($n_{\text{vocab}} = 51,864$) and multilingual ($n_{\text{vocab}} \ge 51,865$, $99+$ languages) models.
  - Synthesizes the $1,500+$ timestamp tokens (`[_BEG_]`, `[_TT_1]` .. `[_TT_1500]`) matching `whisper.cpp` lines 1659–1690.

### 2.2 `WhisperEncoderLayer`, `WhisperDecoderLayer` & `WhisperModel`
- **`WhisperEncoderLayer`**: Binds all 15 per-layer encoder tensors (`attn_ln`, `attn.query` w/b, `attn.key` w, `attn.value` w/b, `attn.out` w/b, `mlp_ln`, `mlp.0` w/b, `mlp.2` w/b) along with individual `GGMLType` quantization descriptors.
- **`WhisperDecoderLayer`**: Binds all 24 per-layer decoder tensors spanning Causal Self-Attention (`attn.*`), Encoder-Decoder Cross-Attention (`cross_attn.*`), and Feed-Forward MLP (`mlp.*`).
- **`WhisperModel`**: Centralizes global tensors (`encoder.positional_embedding`, `encoder.conv1`, `encoder.conv2`, `encoder.ln_post`, `decoder.positional_embedding`, `decoder.token_embedding.weight`, `decoder.ln`), embedded `WhisperFilters` (`[80 x 201]` or `[128 x 201]`), and the layer arrays.

---

## 3. Verification Results

Verified in [`tests/test_whisper_phase3.mojo`](../../tests/test_whisper_phase3.mojo) against `/mnt/c/Documents/Programming/whisper.cpp/ggml-base.en.bin` ($147.96\text{ MB}$):
- **Hyperparameters**: Verified `n_vocab = 51864`, `n_audio_ctx = 1500`, `n_audio_state = 512`, `n_audio_head = 8`, `n_audio_layer = 6`, `n_text_ctx = 448`, `n_text_layer = 6`, `n_mels = 80`, `ftype = 1` (`F16`).
- **Zero-Copy Tensor Binding**: All **245 tensors** (`15 + 15*6 + 24*6`) bound directly into the POSIX `mmap64` region with zero heap duplication.
- **Vocabulary & Timestamps**: Verified all $51,864$ tokens including `<|endoftext|>` (`50256`), `[_SOT_]` (`50257`), and `[_BEG_]` (`50363`), plus multilingual offset shifting for $n_{\text{vocab}} = 51,865$.

