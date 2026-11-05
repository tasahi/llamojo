""" Hugging Face / PyTorch-style facade over :class:`llamojo.Llama`.

Provides ``LlamojoForCausalLM`` (``generate``, ``__call__`` -> logits) and
``LlamojoTokenizer`` (``encode`` / ``decode`` / ``__call__``) so code written
against ``transformers`` causal-LM idioms can run on a GGUF model.

Design notes
------------
* The facade wraps any object exposing the small ``Llama`` surface it uses
  (``reset``, ``eval``, ``sample``, ``scores``, ``tokenize``, ``detokenize``,
  ``token_eos``, ``token_bos``, ``n_vocab``), so it is testable without a model.
* NumPy and PyTorch are optional. If an input is a ``torch.Tensor`` the output
  is a ``torch.Tensor``; otherwise plain nested lists are returned. Logits are
  NumPy arrays when NumPy is installed, else lists.
* llamojo is stateful (KV cache, one sequence). Each batch row is processed
  independently after a ``reset()``; there is no true batched forward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

try:  # optional
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # optional
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

IdsLike = Union[Sequence[int], Sequence[Sequence[int]], Any]


def _to_rows(input_ids: IdsLike) -> "tuple[List[List[int]], bool]":
    """ Normalise ids to a list of rows. Returns (rows, was_flat)."""
    if hasattr(input_ids, "tolist"):  # numpy / torch
        input_ids = input_ids.tolist()
    ids = list(input_ids)
    if len(ids) == 0:
        raise ValueError("input_ids must not be empty")
    if isinstance(ids[0], (list, tuple)):
        return [[int(t) for t in row] for row in ids], False
    return [[int(t) for t in ids]], True


def _wrap_like(rows: List[List[int]], like: Any) -> Any:
    if _torch is not None and isinstance(like, _torch.Tensor):
        width = max(len(r) for r in rows)
        pad = [r + [r[-1]] * (width - len(r)) for r in rows]
        return _torch.tensor(pad, dtype=like.dtype)
    return rows


@dataclass
class CausalLMOutput:
    """ Mimics ``transformers`` ``CausalLMOutput`` (``logits`` only).\n
    ``logits`` has shape ``[batch, vocab]``: llamojo exposes the logits of the
    last evaluated token, not the full ``[batch, seq, vocab]`` tensor.
    """

    logits: Any

    def __getitem__(self, key: Union[int, str]) -> Any:
        if key in (0, "logits"):
            return self.logits
        raise KeyError(key)


class LlamojoTokenizer:
    """ Minimal ``PreTrainedTokenizer``-like wrapper around ``Llama``."""

    def __init__(self, llm: Any):
        self._llm = llm

    @property
    def eos_token_id(self) -> int:
        return int(self._llm.token_eos())

    @property
    def bos_token_id(self) -> int:
        return int(self._llm.token_bos())

    @property
    def vocab_size(self) -> int:
        return int(self._llm.n_vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=add_special_tokens))

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._llm.detokenize(list(ids), special=not skip_special_tokens).decode(
            "utf-8", errors="ignore"
        )

    def batch_decode(self, rows: Sequence[Sequence[int]], **kw: Any) -> List[str]:
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        return [self.decode(r, **kw) for r in rows]

    def __call__(self, text: Union[str, List[str]], **kw: Any) -> Dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self.encode(t, kw.get("add_special_tokens", True)) for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(r) for r in ids]}


class LlamojoForCausalLM:
    """ ``AutoModelForCausalLM``-style wrapper over a llamojo ``Llama`` instance."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.tokenizer = LlamojoTokenizer(llm)
        self.training = False

    # -- construction -----------------------------------------------------
    @classmethod
    def from_pretrained(
            cls, model_path: str, n_ctx: int = 2048, **kwargs: Any
        ) -> "LlamojoForCausalLM":
        """ Load a local GGUF file or Hugging Face Safetensors checkpoint directory."""
        from .llama import Llama

        model_path = os.path.abspath(model_path)
        # If model_path is a directory containing model.safetensors or config.json
        if os.path.isdir(model_path):
            return cls(Llama.from_safetensors(model_path, n_ctx=n_ctx, **kwargs))

        return cls(Llama(model_path=model_path, n_ctx=n_ctx, **kwargs))

    # -- torch.nn.Module-like no-ops --------------------------------------
    def eval(self) -> "LlamojoForCausalLM":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "LlamojoForCausalLM":
        return self  # CPU only

    def close(self) -> None:
        self.llm.close()

    @property
    def config(self) -> Dict[str, Any]:
        return {"vocab_size": self.tokenizer.vocab_size,
                "eos_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": self.tokenizer.bos_token_id}

    # -- forward ----------------------------------------------------------
    def _last_logits(self, ids: List[int]) -> Any:
        self.llm.reset()
        self.llm.eval(ids)
        scores = self.llm.scores
        return scores.copy() if hasattr(scores, "copy") else list(scores)

    def _all_logits(self, ids: List[int]) -> Any:
        self.llm.reset()
        self.llm.eval(ids, all_logits=True)
        if hasattr(self.llm, "logits_all"):
            return self.llm.logits_all(len(ids))
        scores = self.llm.scores
        row = scores.copy() if hasattr(scores, "copy") else list(scores)
        return [row] * len(ids)

    def __call__( self,
            input_ids: IdsLike,
            output_all_logits: bool = False,
            **kwargs: Any,
        ) -> CausalLMOutput:
        """ Return logits for each row.\n
        When output_all_logits=False (default): returns [batch, vocab] for the last token.
        When output_all_logits=True: returns [batch, seq, vocab] across all tokens.
        """
        rows, _ = _to_rows(input_ids)
        if output_all_logits:
            out = [self._all_logits(r) for r in rows]
        else:
            out = [self._last_logits(r) for r in rows]
        logits = _np.stack(out) if _np is not None else out
        if _torch is not None and isinstance(input_ids, _torch.Tensor) and _np is not None:
            logits = _torch.from_numpy(logits)
        return CausalLMOutput(logits=logits)

    forward = __call__

    # -- generation -------------------------------------------------------
    def generate( self,
            input_ids: IdsLike,
            max_new_tokens: int = 20,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 1.0,
            eos_token_id: Optional[int] = None,
            seed: Optional[int] = None,
            **kwargs: Any,
        ) -> Any:
        """ HF-style generation. Returns prompt + new tokens (same type as input).\n
        ``do_sample=False`` is greedy decoding (temperature 0). ``num_beams``
        and other unsupported kwargs raise ``NotImplementedError`` rather than
        being silently ignored.
        """
        if kwargs.get("num_beams", 1) != 1:
            raise NotImplementedError("beam search is not supported")
        unsupported = set(kwargs) - {"num_beams", "pad_token_id", "attention_mask"}
        if unsupported:
            raise NotImplementedError(f"unsupported generate() args: {sorted(unsupported)}")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if seed is not None:
            self.llm.set_seed(seed)
        eos = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        temp = temperature if do_sample else 0.0

        rows, was_flat = _to_rows(input_ids)
        results: List[List[int]] = []
        for prompt in rows:
            self.llm.reset()
            self.llm.eval(prompt)
            seq = list(prompt)
            for _ in range(max_new_tokens):
                tok = int(self.llm.sample(top_k=top_k, top_p=top_p, temp=temp))
                seq.append(tok)
                if tok == eos:
                    break
                self.llm.eval([tok])
            results.append(seq)

        if was_flat and not (_torch is not None and isinstance(input_ids, _torch.Tensor)):
            return results[0] if isinstance(input_ids, (list, tuple)) else results
        return _wrap_like(results, input_ids)


class LlamojoForSpeechSeq2Seq:
    """ ``AutoModelForSpeechSeq2Seq`` / ASR pipeline-style wrapper over :class:`llamojo.Whisper`."""

    def __init__(self, whisper_engine: Any):
        self.whisper = whisper_engine
        self.training = False

    @classmethod
    def from_pretrained(
        cls, model_path: str, n_threads: int = 0, **kwargs: Any
    ) -> "LlamojoForSpeechSeq2Seq":
        from .whisper import Whisper

        return cls(Whisper(model_path=os.path.abspath(model_path), n_threads=n_threads))

    def eval(self) -> "LlamojoForSpeechSeq2Seq":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "LlamojoForSpeechSeq2Seq":
        return self

    def close(self) -> None:
        self.whisper.close()

    def transcribe( self,
        audio: Union[str, Sequence[float], Any],
            return_timestamps: bool = True,
            translate: bool = False,
            **kwargs: Any,
        ) -> Dict[str, Any]:
        """ Transcribe audio and return ``{"text": ..., "chunks": ...}`` matching HF ASR pipeline conventions."""
        res = self.whisper.transcribe(
            audio,
            no_timestamps=not return_timestamps,
            translate=translate,
        )
        chunks = [
            {"timestamp": (seg["start"], seg["end"]), "text": seg["text"]}
            for seg in res.get("segments", [])
        ]
        return {
            "text": res.get("text", ""),
            "chunks": chunks,
            "segments": res.get("segments", []),
        }

    __call__ = transcribe
