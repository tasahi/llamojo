""" High-level Python wrapper for the llamojo Whisper ASR engine (`libllamojo.so`)."""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from ._lib import get_lib

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:
    import av as _av
except Exception:  # pragma: no cover
    _av = None


def _resolve_path(raw_path: Union[str, os.PathLike]) -> str:
    """ Resolve a file path without erroneously prepending cwd when a leading '/' is omitted
    (e.g. ``'mnt/g/...'`` -> ``'/mnt/g/...'``) or when a Windows drive letter is used under WSL.
    """
    p = os.path.expanduser(os.fspath(raw_path))
    if os.path.exists(p):
        return p

    # If the user omitted the leading slash on an absolute Linux/WSL path (e.g. 'mnt/g/...' or 'home/...')
    rooted = "/" + p.lstrip("/")
    if os.path.exists(rooted):
        return rooted

    # If a Windows path like 'G:/My Drive/...' or 'G:\My Drive\...' was passed under WSL
    if len(p) >= 3 and p[1] == ":" and p[2] in ("/", "\\"):
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        wsl_path = f"/mnt/{drive}/{rest}"
        if os.path.exists(wsl_path):
            return wsl_path

    return p


def load_audio_16k(file_path: str) -> Any:
    """ Decode any audio file (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, etc.) into a
    16,000 Hz mono float32 1D array using PyAV (`av`) or `ffmpeg`.
    """
    if _av is not None and _np is not None:
        container = _av.open(file_path)
        try:
            resampler = _av.AudioResampler(format="fltp", layout="mono", rate=16000)
            chunks = []
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray()[0])
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray()[0])
            if not chunks:
                return _np.zeros((0,), dtype=_np.float32)
            return _np.concatenate(chunks).astype(_np.float32, copy=False)
        finally:
            container.close()

    # Fallback to ffmpeg CLI if installed
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to decode audio file '{file_path}'. Install 'av' (PyAV) or 'ffmpeg'."
        )
    if _np is not None:
        return (
            _np.frombuffer(proc.stdout, dtype=_np.int16).astype(_np.float32)
            / 32768.0
        )
    import struct

    n_samples = len(proc.stdout) // 2
    ints = struct.unpack(f"<{n_samples}h", proc.stdout)
    return [x / 32768.0 for x in ints]


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
    Supports both legacy `whisper.cpp` `.bin` (`0x67676d6c`) and GGUF v2/v3 Whisper models,
    and automatically decodes `.mp3`, `.wav`, `.flac`, `.m4a`, and `.ogg` audio files to 16 kHz mono PCM.
    """

    def __init__(self, model_path: str, n_threads: int = 0):
        self.model_path = _resolve_path(model_path)
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

    def transcribe(
        self,
        audio: Union[str, Sequence[float], Any],
        language: str = "auto",
        initial_prompt: Optional[str] = None,
        no_timestamps: bool = False,
        no_context: bool = False,
        translate: bool = False,
        offset_ms: int = 0,
        duration_ms: int = 0,
        verbose: bool = True,
        n_threads: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ Transcribe an audio file path (`.mp3`, `.wav`, `.flac`, `.m4a`, etc.) or a
        16 kHz mono float32 PCM sequence/ndarray.

        Parameters
        ----------
        audio : str or 1D float32 sequence/ndarray
            Path to an audio file or 16 kHz mono float32 waveform.
        language : str, default="auto"
            ISO 639-1 language code (e.g. ``"es"``, ``"en"``, ``"pt"``) or ``"auto"`` for automatic detection.
        no_timestamps : bool, default=False
            If True, disables timestamp token generation.
        no_context : bool, default=False
            If False (default), carries text tokens across consecutive 30-second windows
            to maintain grammatical and conversational continuity. Set to True to process each window independently.
        translate : bool, default=False
            If True, translates speech into English.
        offset_ms : int, default=0
            Start offset in milliseconds.
        duration_ms : int, default=0
            Duration to transcribe in milliseconds (0 = full file).
        verbose : bool, default=True
            If True, prints each `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text` segment in real time as each 30s window finishes.
        """
        if not self._ctx:
            raise RuntimeError("Whisper context is closed")

        threads = self.n_threads if n_threads is None else int(n_threads)
        no_ts_flag = 1 if no_timestamps else 0
        no_ctx_flag = 1 if no_context else 0
        tr_flag = 1 if translate else 0
        rt_flag = 1 if verbose else 0
        lang_bytes = (language or "auto").encode("utf-8")

        if isinstance(audio, (str, os.PathLike)):
            audio_path = _resolve_path(audio)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            pcm_16k = load_audio_16k(audio_path)
            return self.transcribe(
                pcm_16k,
                language=language,
                no_timestamps=no_timestamps,
                no_context=no_context,
                translate=translate,
                offset_ms=offset_ms,
                duration_ms=duration_ms,
                verbose=verbose,
                n_threads=threads,
            )
        else:
            if _np is not None and isinstance(audio, _np.ndarray):
                arr = _np.ascontiguousarray(audio, dtype=_np.float32).ravel()
                n_samples = int(arr.size)
                c_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                samples_list = [float(x) for x in audio]
                n_samples = len(samples_list)
                FloatArray = ctypes.c_float * n_samples
                c_arr = FloatArray(*samples_list)
                c_ptr = ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_float))

            p_tokens_ptr = None
            n_p_tokens = 0
            if initial_prompt:
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("gpt2")
                    p_toks = enc.encode(" " + initial_prompt.strip())
                    if p_toks:
                        n_p_tokens = len(p_toks)
                        IntArray = ctypes.c_int32 * n_p_tokens
                        c_p_arr = IntArray(*p_toks)
                        p_tokens_ptr = ctypes.cast(c_p_arr, ctypes.POINTER(ctypes.c_int32))
                except Exception:
                    pass

            if hasattr(self._lib, "whisper_full_ex_c"):
                rc = int(
                    self._lib.whisper_full_ex_c(
                        self._ctx,
                        c_ptr,
                        n_samples,
                        lang_bytes,
                        no_ts_flag,
                        tr_flag,
                        threads,
                        int(offset_ms),
                        int(duration_ms),
                        rt_flag,
                        no_ctx_flag,
                        p_tokens_ptr,
                        n_p_tokens,
                    )
                )
            else:
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
            if rc != 0:
                raise RuntimeError(f"whisper_full_c failed with code {rc}")

        segs = self._collect_segments()
        full_text = "".join(s.text for s in segs)
        return {
            "text": full_text,
            "segments": [s.to_dict() for s in segs],
        }

    __call__ = transcribe
