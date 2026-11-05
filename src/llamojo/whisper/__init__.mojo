""" Whisper ASR sub-package for llamojo.

Provides:
- Audio DSP frontend (`WhisperFilters`, `WhisperMel`, `WhisperTrigCache`, `whisper_dft`, `whisper_fft`,
  `log_mel_spectrogram`, `load_wav_file`, `load_filters_from_whisper_bin`)
- Dual Model Loader (`WhisperHParams`, `WhisperVocab`, `WhisperEncoderLayer`, `WhisperDecoderLayer`,
  `WhisperModel`, `whisper_load_model_from_file`)
- Encoder & Decoder Execution (`WhisperState`, `get_embedding_row`, `whisper_encode_layer`,
  `whisper_build_cross_kv`, `whisper_encode`, `whisper_decode_step`, `whisper_decode`)
- Timestamp Logit Rules & High-Level Transcription (`WhisperSegment`, `WhisperFullParams`,
  `whisper_full_default_params`, `format_timestamp_cs`, `apply_whisper_logit_filters`,
  `WhisperContext`, `whisper_init_from_file`, `whisper_full`, `whisper_full_n_segments`,
  `whisper_full_get_segment_text`, `whisper_full_get_segment_t0`, `whisper_full_get_segment_t1`)
"""

from .audio import (
    WHISPER_SAMPLE_RATE,
    WHISPER_N_FFT,
    WHISPER_HOP_LENGTH,
    WHISPER_CHUNK_SIZE,
    WHISPER_N_SAMPLES,
    WhisperFilters,
    WhisperMel,
    WhisperTrigCache,
    whisper_dft,
    whisper_fft,
    log_mel_spectrogram,
    load_wav_file,
    load_filters_from_whisper_bin,
)
from .loader import (
    WHISPER_BIN_MAGIC,
    WhisperHParams,
    WhisperVocab,
    WhisperEncoderLayer,
    WhisperDecoderLayer,
    WhisperTensorEntry,
    WhisperModel,
    whisper_load_model_from_file,
)
from .model import (
    WhisperState,
    get_embedding_row,
    whisper_encode_layer,
    whisper_build_cross_kv,
    whisper_encode,
    whisper_decode_step,
    whisper_decode,
)
from .transcribe import (
    WhisperSegment,
    WhisperFullParams,
    whisper_full_default_params,
    format_timestamp_cs,
    apply_whisper_logit_filters,
    WhisperContext,
    whisper_init_from_file,
    whisper_full,
    whisper_full_n_segments,
    whisper_full_get_segment_text,
    whisper_full_get_segment_t0,
    whisper_full_get_segment_t1,
)
