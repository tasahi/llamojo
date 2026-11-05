""" High-level Python wrapper for the llamojo Whisper ASR engine (`libllamojo.so`)."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from ._lib import get_lib

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


@dataclass
class WhisperSegmentResult:
    """ A single transcribed speech segment with timestamps in seconds (`start`, `end`) and centiseconds (`t0`, `t1`)."""

    t0: int
    t1: int
    text: str

    @property
    def start(self) -> float:
        """ Segment start timestamp in seconds."""
        return self.t0 / 100.0

    @property
    def end(self) -> float:
        """ Segment end timestamp in seconds."""
        return self.t1 / 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


class Whisper:
    """ High-level Python interface to `llamojo`'s pure-Mojo Whisper ASR engine.\n
    Supports both legacy `whisper.cpp` `.bin` (`0x67676d6c`) and GGUF v2/v3 Whisper models.
    """

    def __init__(self, model_path: str, n_threads: int = 0):
        self.model_path = os.path.abspath(model_path)
        self.n_threads = int(n_threads)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper model file not found: {self.model_path}")

        self._lib = get_lib()
        if not hasattr(self._lib, "whisper_init_from_file"):
            raise RuntimeError(
                "Loaded libllamojo.so does not expose whisper_* C-ABI symbols. "
                "Rebuild via './tools/build_shared_lib.sh'."
            )

        self._ctx = self._lib.whisper_init_from_file(self.model_path.encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"Failed to load Whisper model from: {self.model_path}")

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.whisper_free(self._ctx)
            self._ctx = None

    @property
    def is_multilingual(self) -> bool:
        if not self._ctx:
            return False
        return bool(self._lib.whisper_is_multilingual(self._ctx))

    def _collect_segments(self) -> List[WhisperSegmentResult]:
        n_seg = int(self._lib.whisper_full_n_segments(self._ctx))
        segments: List[WhisperSegmentResult] = []
        for i in range(n_seg):
            t0 = int(self._lib.whisper_full_get_segment_t0(self._ctx, i))
            t1 = int(self._lib.whisper_full_get_segment_t1(self._ctx, i))
            raw_txt = self._lib.whisper_full_get_segment_text(self._ctx, i)
            txt = raw_txt.decode("utf-8", errors="replace") if raw_txt else ""
            segments.append(WhisperSegmentResult(t0=t0, t1=t1, text=txt))
        return segments

    def transcribe( self,
            audio: Union[str, Sequence[float], Any],
            no_timestamps: bool = False,
            translate: bool = False,
            n_threads: Optional[int] = None,
        ) -> Dict[str, Any]:
        """ Transcribe a `.wav` file path or a 16 kHz mono float32 PCM sequence/ndarray.

        Returns a dictionary with ``"text"`` (full transcript string) and
        ``"segments"`` (list of :class:`WhisperSegmentResult` dicts).
        """
        if not self._ctx:
            raise RuntimeError("Whisper context is closed")

        threads = self.n_threads if n_threads is None else int(n_threads)
        no_ts_flag = 1 if no_timestamps else 0
        tr_flag = 1 if translate else 0

        if isinstance(audio, (str, os.PathLike)):
            wav_path = os.path.abspath(os.fspath(audio))
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"WAV file not found: {wav_path}")
            rc = int(
                self._lib.whisper_full_wav_c(
                    self._ctx,
                    wav_path.encode("utf-8"),
                    no_ts_flag,
                    tr_flag,
                    threads,
                )
            )
            if rc != 0:
                raise RuntimeError(f"whisper_full_wav_c failed with code {rc}")
        else:
            if _np is not None and isinstance(audio, _np.ndarray):
                arr = _np.ascontiguousarray(audio, dtype=_np.float32).ravel()
                n_samples = int(arr.size)
                c_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                rc = int(
                    self._lib.whisper_full_c(
                        self._ctx,
                        c_ptr,
                        n_samples,
                        no_ts_flag,
                        tr_flag,
                        threads,
                    )
                )
            else:
                samples_list = [float(x) for x in audio]
                n_samples = len(samples_list)
                FloatArray = ctypes.c_float * n_samples
                c_arr = FloatArray(*samples_list)
                rc = int(
                    self._lib.whisper_full_c(
                        self._ctx,
                        ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_float)),
                        n_samples,
                        no_ts_flag,
                        tr_flag,
                        threads,
                    )
                )
            if rc != 0:
                raise RuntimeError(f"whisper_full_c failed with code {rc}")

        segs = self._collect_segments()
        full_text = "".join(s.text for s in segs)
        return {
            "text": full_text,
            "segments": [s.to_dict() for s in segs],
        }

    __call__ = transcribe
