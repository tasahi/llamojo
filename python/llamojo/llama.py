"""llamojo: High-level Python interface compatible with llama-cpp-python."""

import ctypes
import os
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from ._lib import get_lib


class Llama:
    """High-level Python wrapper around the compiled llamojo engine.

    Mirrors the `llama_cpp.Llama` API closely for drop-in compatibility.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs: Any,
    ):
        self.model_path = os.path.abspath(model_path)
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads or 2
        self.verbose = verbose

        self._lib = get_lib()

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        c_path = self.model_path.encode("utf-8")
        self._model = self._lib.llama_model_load_from_file(c_path, n_gpu_layers)
        if not self._model:
            raise RuntimeError(f"Failed to load model from: {self.model_path}")

        self._vocab = self._lib.llama_model_get_vocab(self._model)
        if not self._vocab:
            self._lib.llama_model_free(self._model)
            raise RuntimeError("Failed to extract model vocabulary")

        self._ctx = self._lib.llama_init_from_model(
            self._model, self.n_ctx, self.n_batch, self.n_threads
        )
        if not self._ctx:
            self._lib.llama_vocab_free(self._vocab)
            self._lib.llama_model_free(self._model)
            raise RuntimeError("Failed to create inference context")

        self.n_vocab = self._lib.llama_model_n_vocab(self._model)
        self.n_embd = self._lib.llama_model_n_embd(self._model)
        self.n_layer = self._lib.llama_model_n_layer(self._model)

    def __del__(self):
        self.close()

    def close(self):
        """Release all allocated native resources."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if hasattr(self, "_vocab") and self._vocab:
            self._lib.llama_vocab_free(self._vocab)
            self._vocab = None
        if hasattr(self, "_model") and self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def tokenize(
        self, text: Union[str, bytes], add_bos: bool = True, special: bool = False
    ) -> List[int]:
        """Convert a text string to a list of token IDs."""
        if isinstance(text, str):
            b_text = text.encode("utf-8")
        else:
            b_text = text

        max_toks = max(len(b_text) + 16, 256)
        tok_arr = (ctypes.c_int32 * max_toks)()

        n = self._lib.llama_tokenize(
            self._vocab,
            b_text,
            len(b_text),
            ctypes.cast(tok_arr, ctypes.c_void_p),
            max_toks,
            add_bos,
            special,
        )
        if n < 0:
            # Need larger buffer
            max_toks = -n
            tok_arr = (ctypes.c_int32 * max_toks)()
            n = self._lib.llama_tokenize(
                self._vocab,
                b_text,
                len(b_text),
                ctypes.cast(tok_arr, ctypes.c_void_p),
                max_toks,
                add_bos,
                special,
            )

        return [tok_arr[i] for i in range(max(0, n))]

    def detokenize(self, tokens: List[int], special: bool = False) -> bytes:
        """Convert a list of token IDs back into bytes."""
        out = bytearray()
        buf = ctypes.create_string_buffer(256)
        for tok in tokens:
            plen = self._lib.llama_token_to_piece(
                self._vocab, tok, buf, 256, 0, special
            )
            if plen > 0:
                out.extend(buf.raw[:plen])
        return bytes(out)

    def token_to_piece(self, token: int, special: bool = False) -> str:
        """Convert a single token ID into its string piece."""
        buf = ctypes.create_string_buffer(256)
        plen = self._lib.llama_token_to_piece(self._vocab, token, buf, 256, 0, special)
        if plen > 0:
            return buf.raw[:plen].decode("utf-8", errors="replace")
        return ""

    def __call__(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        echo: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Callable shorthand for create_completion."""
        return self.create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            stream=stream,
            stop=stop,
            echo=echo,
            **kwargs,
        )

    def create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        echo: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Generate text completion from a prompt matching llama-cpp-python's schema."""
        if isinstance(prompt, str):
            tokens = self.tokenize(prompt, add_bos=True)
            prompt_str = prompt
        else:
            tokens = list(prompt)
            prompt_str = self.detokenize(tokens).decode("utf-8", errors="replace")

        if max_tokens is None:
            max_tokens = 128

        stop_sequences: List[str] = []
        if isinstance(stop, str):
            stop_sequences = [stop]
        elif isinstance(stop, list):
            stop_sequences = stop

        if stream:
            return self._stream_generator(
                tokens, prompt_str, max_tokens, temperature, top_k, top_p, stop_sequences
            )

        # Batch decode prompt tokens
        batch = self._lib.llama_batch_init(1, 0, 1)
        for i, tok in enumerate(tokens):
            self._lib.llama_batch_set_token(
                batch, 0, tok, i, 0, (i == len(tokens) - 1)
            )
            self._lib.llama_batch_set_n_tokens(batch, 1)
            ret = self._lib.llama_decode(self._ctx, batch)
            if ret != 0:
                self._lib.llama_batch_free(batch)
                raise RuntimeError(f"Decode failed at prompt token {i}")

        # Setup sampler
        if temperature <= 0.0:
            smpl = self._lib.llama_sampler_init_greedy()
        else:
            smpl = self._lib.llama_sampler_chain_init()
            if top_k > 0:
                self._lib.llama_sampler_chain_add(
                    smpl, self._lib.llama_sampler_init_top_k(top_k)
                )
            if top_p < 1.0:
                self._lib.llama_sampler_chain_add(
                    smpl, self._lib.llama_sampler_init_top_p(top_p, 1)
                )
            self._lib.llama_sampler_chain_add(
                smpl, self._lib.llama_sampler_init_temp(temperature)
            )
            self._lib.llama_sampler_chain_add(
                smpl, self._lib.llama_sampler_init_dist(int(time.time()))
            )

        gen_tokens: List[int] = []
        gen_text = ""
        finish_reason = "length"

        for step in range(max_tokens):
            logits = self._lib.llama_get_logits_ith(self._ctx, 0)
            next_tok = self._lib.llama_sampler_sample_token(
                smpl, logits, self.n_vocab, temperature
            )
            if next_tok < 0 or self._lib.llama_vocab_is_eog(self._vocab, next_tok):
                finish_reason = "stop"
                break

            gen_tokens.append(next_tok)
            piece = self.token_to_piece(next_tok)
            gen_text += piece

            # Stop sequence check
            stopped = False
            for s in stop_sequences:
                if s in gen_text:
                    cutoff = gen_text.index(s)
                    gen_text = gen_text[:cutoff]
                    finish_reason = "stop"
                    stopped = True
                    break
            if stopped:
                break

            # Forward pass for next token
            self._lib.llama_batch_set_token(
                batch, 0, next_tok, len(tokens) + step, 0, True
            )
            self._lib.llama_batch_set_n_tokens(batch, 1)
            ret = self._lib.llama_decode(self._ctx, batch)
            if ret != 0:
                break

        # Cleanup batch & sampler
        self._lib.llama_batch_free(batch)
        self._lib.llama_sampler_free(smpl)

        full_text = (prompt_str + gen_text) if echo else gen_text

        response_id = f"cmpl-{uuid.uuid4()}"
        return {
            "id": response_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [
                {
                    "text": full_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(gen_tokens),
                "total_tokens": len(tokens) + len(gen_tokens),
            },
        }

    def _stream_generator(
        self,
        tokens: List[int],
        prompt_str: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_sequences: List[str],
    ) -> Iterator[Dict[str, Any]]:
        """Streaming token generator."""
        batch = self._lib.llama_batch_init(1, 0, 1)
        for i, tok in enumerate(tokens):
            self._lib.llama_batch_set_token(
                batch, 0, tok, i, 0, (i == len(tokens) - 1)
            )
            self._lib.llama_batch_set_n_tokens(batch, 1)
            self._lib.llama_decode(self._ctx, batch)

        if temperature <= 0.0:
            smpl = self._lib.llama_sampler_init_greedy()
        else:
            smpl = self._lib.llama_sampler_chain_init()
            if top_k > 0:
                self._lib.llama_sampler_chain_add(
                    smpl, self._lib.llama_sampler_init_top_k(top_k)
                )
            if top_p < 1.0:
                self._lib.llama_sampler_chain_add(
                    smpl, self._lib.llama_sampler_init_top_p(top_p, 1)
                )
            self._lib.llama_sampler_chain_add(
                smpl, self._lib.llama_sampler_init_temp(temperature)
            )
            self._lib.llama_sampler_chain_add(
                smpl, self._lib.llama_sampler_init_dist(int(time.time()))
            )

        resp_id = f"cmpl-{uuid.uuid4()}"
        accumulated = ""

        try:
            for step in range(max_tokens):
                logits = self._lib.llama_get_logits_ith(self._ctx, 0)
                next_tok = self._lib.llama_sampler_sample_token(
                    smpl, logits, self.n_vocab, temperature
                )
                if next_tok < 0 or self._lib.llama_vocab_is_eog(self._vocab, next_tok):
                    yield {
                        "id": resp_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "choices": [
                            {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                        ],
                    }
                    break

                piece = self.token_to_piece(next_tok)
                accumulated += piece

                # Check stops
                stopped = False
                for s in stop_sequences:
                    if s in accumulated:
                        stopped = True
                        break
                if stopped:
                    yield {
                        "id": resp_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "choices": [
                            {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                        ],
                    }
                    break

                yield {
                    "id": resp_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "text": piece,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }

                self._lib.llama_batch_set_token(
                    batch, 0, next_tok, len(tokens) + step, 0, True
                )
                self._lib.llama_batch_set_n_tokens(batch, 1)
                self._lib.llama_decode(self._ctx, batch)
        finally:
            self._lib.llama_batch_free(batch)
            self._lib.llama_sampler_free(smpl)

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Basic Chat Completion matching OpenAI / llama-cpp-python chat format."""
        formatted = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"

        res = self.create_completion(formatted, **kwargs)
        if isinstance(res, dict):
            choice = res["choices"][0]
            choice["message"] = {"role": "assistant", "content": choice["text"]}
            res["object"] = "chat.completion"
        return res

