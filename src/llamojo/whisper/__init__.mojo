""" Whisper ASR sub-package for llamojo.

Provides:
- Audio DSP frontend (`WhisperFilters`, `WhisperMel`, `WhisperTrigCache`, `whisper_dft`, `whisper_fft`,
  `log_mel_spectrogram`, `load_wav_file`, `load_filters_from_whisper_bin`)
- Dual Model Loader (`WhisperHParams`, `WhisperVocab`, `WhisperEncoderLayer`, `WhisperDecoderLayer`,
  `WhisperModel`, `whisper_load_model_from_file`)
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
