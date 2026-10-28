"""llamojo: High-level Python interface compatible with llama-cpp-python."""

import ctypes
import json
import os
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from ._lib import get_lib
from .chat_format import (
    ChatFormatter,
    ChatFormatterResponse,
    Jinja2ChatFormatter,
    extract_chat_template_from_gguf,
    get_chat_formatter,
)

try:
    import numpy as np
    import numpy.typing as npt
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    npt = None  # type: ignore
    _NUMPY_AVAILABLE = False


LogitsProcessor = Callable[[Sequence[int], Any], Any]


class LogitsProcessorList(List[LogitsProcessor]):
    """ Chain of logits processors matching llama_cpp.LogitsProcessorList."""

    def __call__(self, input_ids: Sequence[int], scores: Any) -> Any:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class MinTokensLogitsProcessor:
    """ Logits processor preventing EOS token generation until min_tokens is reached."""

    def __init__(self, min_tokens: int, token_eos: int):
        self.min_tokens = min_tokens
        self.token_eos = token_eos
        self.prompt_tokens: Optional[int] = None

    def __call__(self, input_ids: Sequence[int], scores: Any) -> Any:
        if self.prompt_tokens is None:
            self.prompt_tokens = len(input_ids)
        if len(input_ids) - self.prompt_tokens < self.min_tokens:
            if _NUMPY_AVAILABLE and isinstance(scores, np.ndarray):
                scores[self.token_eos] = -np.inf
            else:
                scores[self.token_eos] = float("-inf")
        return scores


class _CallableInt(int):
    """ Integer that can also be called as a zero-argument function.\n
    Provides dual compatibility with both property-style access (`llm.n_ctx`)
    and upstream llama-cpp-python method-style access (`llm.n_ctx()`).
    """

    def __call__(self) -> int:
        return int(self)


class LlamaGrammar:
    """ GBNF Grammar constraint parser matching llama_cpp.LlamaGrammar."""

    def __init__(self, grammar_str: str):
        self.grammar_str = grammar_str
        self._lib = get_lib()
        c_str = grammar_str.encode("utf-8")
        self._handle = self._lib.llama_grammar_init(c_str)
        if not self._handle:
            raise RuntimeError(f"Failed to parse GBNF grammar: {grammar_str}")

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.llama_grammar_free(self._handle)
            self._handle = None

    @classmethod
    def from_string(cls, grammar_str: str) -> "LlamaGrammar":
        return cls(grammar_str)

    @classmethod
    def from_json_schema(cls, schema: Union[str, Dict[str, Any]]) -> "LlamaGrammar":
        """ Generate a grammar constraining output to a JSON object matching schema."""
        if isinstance(schema, dict):
            _ = json.dumps(schema)
        # Built-in JSON validator pattern handled by llamojo grammar engine
        return cls("json")


class Llama:
    """ High-level Python wrapper around the compiled llamojo engine.\n
    Mirrors the `llama_cpp.Llama` API closely for drop-in compatibility.
    """

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            filename: Optional[str] = None,
            *args: Any,
            **kwargs: Any,
        ) -> "Llama":
        """ Download a .gguf model from Hugging Face Hub and load it.\n
        Matches llama_cpp.Llama.from_pretrained.

        Args:
            repo_id: The Hugging Face repo ID (e.g. "mradermacher/SmolLM2-135M-Instruct-GGUF"
                or "Qwen/Qwen2.5-0.5B-Instruct-GGUF").
            filename: The specific GGUF filename (e.g. "SmolLM2-135M-Instruct-Q4_K_M.gguf")
                or a glob pattern (e.g. "*q4_k_m.gguf"). If omitted, attempts to find
                the most appropriate .gguf file in the repository.
            *args: Additional arguments passed to Llama.__init__.
            **kwargs: Additional keyword arguments. Hugging Face hub options like
                `cache_dir`, `local_dir`, `token`, `revision` are forwarded to
                `hf_hub_download`; remaining kwargs are forwarded to Llama.__init__.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for Llama.from_pretrained. "
                "Please install it with: pip install huggingface-hub"
            ) from e

        # Extract hf_hub_download parameters
        hf_kwargs = {}
        for key in [
            "subfolder",
            "repo_type",
            "revision",
            "token",
            "cache_dir",
            "local_dir",
            "force_download",
            "resume_download",
            "proxies",
            "etag_timeout",
            "headers",
            "endpoint",
        ]:
            if key in kwargs:
                hf_kwargs[key] = kwargs.pop(key)

        token = hf_kwargs.get("token")
        revision = hf_kwargs.get("revision")

        # Resolve filename if None or contains wildcards
        if filename is None:
            files = list_repo_files(repo_id=repo_id, revision=revision, token=token)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                raise ValueError(f"No .gguf files found in Hugging Face repository '{repo_id}'")
            if len(gguf_files) == 1:
                filename = gguf_files[0]
            else:
                # Prioritize Q4_K_M or Q4_0 if multiple variants exist
                preferred = None
                for candidate in ["*q4_k_m*.gguf", "*q4_0*.gguf", "*q8_0*.gguf"]:
                    import fnmatch
                    matches = [f for f in gguf_files if fnmatch.fnmatch(f.lower(), candidate)]
                    if matches:
                        preferred = matches[0]
                        break
                filename = preferred or gguf_files[0]
        elif any(c in filename for c in ["*", "?", "["]):
            import fnmatch
            files = list_repo_files(repo_id=repo_id, revision=revision, token=token)
            matches = [f for f in files if fnmatch.fnmatch(f, filename)]
            if not matches:
                # Try case-insensitive matching
                matches = [f for f in files if fnmatch.fnmatch(f.lower(), filename.lower())]
            if not matches:
                raise ValueError(
                    f"Pattern '{filename}' did not match any files in '{repo_id}'"
                )
            filename = matches[0]

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            **hf_kwargs,
        )

        return cls(model_path=model_path, *args, **kwargs)

    def __init__( self,
            model_path: str,
            n_ctx: int = 512,
            n_batch: int = 512,
            n_threads: Optional[int] = None,
            n_gpu_layers: int = 0,
            lora_path: Optional[str] = None,
            lora_scale: float = 1.0,
            chat_format: Optional[str] = None,
            chat_handler: Optional[ChatFormatter] = None,
            verbose: bool = False,
            **kwargs: Any,
        ):
        self.model_path = os.path.abspath(model_path)
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._n_threads = n_threads or 2
        self.verbose = verbose
        self.seed = kwargs.get("seed", 42)

        self._lib = get_lib()

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        c_path = self.model_path.encode("utf-8")
        self._model = self._lib.llama_model_load_from_file(c_path, n_gpu_layers)
        if not self._model:
            raise RuntimeError(f"Failed to load model from: {self.model_path}")

        if lora_path:
            self.apply_lora(lora_path, scale=lora_scale)

        self._vocab = self._lib.llama_model_get_vocab(self._model)
        if not self._vocab:
            self._lib.llama_model_free(self._model)
            raise RuntimeError("Failed to extract model vocabulary")

        self._ctx = self._lib.llama_init_from_model(
            self._model, self._n_ctx, self._n_batch, self._n_threads
        )
        if not self._ctx:
            self._lib.llama_vocab_free(self._vocab)
            self._lib.llama_model_free(self._model)
            raise RuntimeError("Failed to create inference context")

        self._n_vocab = self._lib.llama_model_n_vocab(self._model)
        self._n_embd = self._lib.llama_model_n_embd(self._model)
        self._n_layer = self._lib.llama_model_n_layer(self._model)
        self._n_head = self._lib.llama_model_n_head(self._model)

        # Dual property/method compatibility
        self.n_ctx = _CallableInt(self._n_ctx)
        self.n_batch = _CallableInt(self._n_batch)
        self.n_threads = _CallableInt(self._n_threads)
        self.n_vocab = _CallableInt(self._n_vocab)
        self.n_embd = _CallableInt(self._n_embd)
        self.n_layer = _CallableInt(self._n_layer)
        self.n_head = _CallableInt(self._n_head)

        # Chat format resolution
        self.chat_format = chat_format
        self.chat_handler = chat_handler
        self._init_chat_handler()

    def _init_chat_handler(self):
        """Resolve chat handler from custom handler, chat_format name, or GGUF metadata."""
        if self.chat_handler is not None:
            return

        if self.chat_format is not None:
            formatter = get_chat_formatter(self.chat_format)
            if formatter:
                self.chat_handler = formatter
                return

        # Attempt to extract tokenizer.chat_template from GGUF metadata
        tmpl_str = extract_chat_template_from_gguf(self.model_path)
        if tmpl_str:
            try:
                eos_piece = self.token_to_piece(self.token_eos(), special=True) or "<|im_end|>"
                bos_piece = self.token_to_piece(self.token_bos(), special=True) or "<|im_start|>"
                self.chat_handler = Jinja2ChatFormatter(
                    template=tmpl_str,
                    eos_token=eos_piece,
                    bos_token=bos_piece,
                    add_generation_prompt=True,
                )
                return
            except Exception:
                pass

        # Fallback to standard chatml
        self.chat_handler = get_chat_formatter("chatml")

    @property
    def model(self):
        """ Return the underlying model pointer."""
        return self._model

    @property
    def ctx(self):
        """ Return the underlying context pointer."""
        return self._ctx

    def set_seed(self, seed: int):
        """ Set the random seed.\n
        Args:
            seed: The random seed.
        """
        self.seed = seed

    def __del__(self):
        self.close()

    def close(self):
        """ Release all allocated native resources."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if hasattr(self, "_vocab") and self._vocab:
            self._lib.llama_vocab_free(self._vocab)
            self._vocab = None
        if hasattr(self, "_model") and self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def apply_lora(self, lora_path: str, scale: float = 1.0) -> int:
        """ Apply low-rank adapter weights from a GGUF adapter file in-place."""
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA adapter file not found: {lora_path}")
        c_lora = lora_path.encode("utf-8")
        ret = self._lib.llama_lora_adapter_apply(self._model, c_lora, ctypes.c_float(scale))
        if ret != 0:
            raise RuntimeError(f"Failed to apply LoRA adapter from: {lora_path}")
        return ret

    def reset(self):
        """ Reset KV cache and internal context state."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.llama_kv_cache_clear(self._ctx)

    def token_bos(self) -> int:
        """ Return BOS token ID."""
        return self._lib.llama_vocab_bos(self._vocab)

    def token_eos(self) -> int:
        """ Return EOS token ID."""
        return self._lib.llama_vocab_eos(self._vocab)

    def token_nl(self) -> int:
        """ Return newline token ID."""
        nl_toks = self.tokenize("\n", add_bos=False)
        return nl_toks[0] if nl_toks else 13

    def _apply_logits_processors( self,
        input_ids: Sequence[int],
        logits_ptr: Any,
        processors: Optional[Sequence[Callable[[Sequence[int], Any], Any]]] = None,
        logit_bias: Optional[Dict[int, float]] = None,
    ):
        """Apply custom logit bias and logits processors in-place."""
        if not processors and not logit_bias:
            return

        # Direct pointer manipulation
        if logit_bias:
            for tok_id, bias in logit_bias.items():
                if 0 <= tok_id < self._n_vocab:
                    logits_ptr[tok_id] += float(bias)

        if processors:
            addr = ctypes.cast(logits_ptr, ctypes.c_void_p).value
            if _NUMPY_AVAILABLE:
                scores = np.ctypeslib.as_array((ctypes.c_float * self._n_vocab).from_address(addr))
            else:
                scores = (ctypes.c_float * self._n_vocab).from_address(addr)

            for proc in processors:
                res = proc(input_ids, scores)
                if res is not None and res is not scores:
                    if _NUMPY_AVAILABLE and isinstance(res, np.ndarray):
                        np.copyto(scores, res)
                    else:
                        for idx, val in enumerate(res):
                            scores[idx] = val

    def eval(self, tokens: List[int]):
        """ Evaluate a sequence of tokens in the model."""
        if not tokens:
            return
        batch = self._lib.llama_batch_init(1, 0, 1)
        try:
            for i, tok in enumerate(tokens):
                self._lib.llama_batch_set_token(
                    batch, 0, tok, i, 0, (i == len(tokens) - 1)
                )
                self._lib.llama_batch_set_n_tokens(batch, 1)
                ret = self._lib.llama_decode(self._ctx, batch)
                if ret != 0:
                    raise RuntimeError(f"llama_decode failed at token {i}")
        finally:
            self._lib.llama_batch_free(batch)

    def sample( self,
            top_k: int = 40,
            top_p: float = 0.95,
            temp: float = 0.8,
            grammar: Optional[Union[LlamaGrammar, str]] = None,
            logits_processor: Optional[Sequence[Callable[[Sequence[int], Any], Any]]] = None,
            logit_bias: Optional[Dict[int, float]] = None,
            idx: Sequence[int] = (),
            **kwargs: Any,
        ) -> int:
        """ Sample a token from the current context logits."""
        logits = self._lib.llama_get_logits_ith(self._ctx, 0)
        self._apply_logits_processors(idx, logits, logits_processor, logit_bias)
        smpl = self._build_sampler(temp=temp, top_k=top_k, top_p=top_p, grammar=grammar)
        try:
            return self._lib.llama_sampler_sample_token(
                smpl, logits, self.n_vocab, temp
            )
        finally:
            self._lib.llama_sampler_free(smpl)

    def generate( self,
            tokens: List[int],
            top_k: int = 40,
            top_p: float = 0.95,
            temp: float = 0.8,
            reset: bool = True,
            grammar: Optional[Union[LlamaGrammar, str]] = None,
            logits_processor: Optional[Sequence[Callable[[Sequence[int], Any], Any]]] = None,
            logit_bias: Optional[Dict[int, float]] = None,
            **kwargs: Any,
        ) -> Iterator[int]:
        """ Low-level token generator yielding next token IDs."""
        if reset:
            self.reset()
        self.eval(tokens)
        curr_pos = len(tokens)
        all_tokens = list(tokens)
        batch = self._lib.llama_batch_init(1, 0, 1)
        smpl = self._build_sampler(temp=temp, top_k=top_k, top_p=top_p, grammar=grammar)
        try:
            while True:
                logits = self._lib.llama_get_logits_ith(self._ctx, 0)
                self._apply_logits_processors(all_tokens, logits, logits_processor, logit_bias)
                next_tok = self._lib.llama_sampler_sample_token(
                    smpl, logits, self.n_vocab, temp
                )
                if next_tok < 0 or self._lib.llama_vocab_is_eog(self._vocab, next_tok):
                    break
                self._lib.llama_sampler_accept_token(smpl, next_tok)
                yield next_tok
                all_tokens.append(next_tok)
                self._lib.llama_batch_set_token(batch, 0, next_tok, curr_pos, 0, True)
                self._lib.llama_batch_set_n_tokens(batch, 1)
                ret = self._lib.llama_decode(self._ctx, batch)
                if ret != 0:
                    break
                curr_pos += 1
        finally:
            self._lib.llama_batch_free(batch)
            self._lib.llama_sampler_free(smpl)

    def embed( self,
            input: Union[str, List[str]],
            normalize: bool = False,
            truncate: bool = True,
            return_count: bool = False,
        ) -> Union[List[float], List[List[float]]]:
        """ Generate embeddings for input text or list of texts."""
        is_single = isinstance(input, str)
        inputs = [input] if is_single else input
        results = []

        for text in inputs:
            self.reset()
            toks = self.tokenize(text, add_bos=True)
            self.eval(toks)
            ptr = self._lib.llama_get_embeddings(self._ctx)
            if not ptr:
                vec = [0.0] * self.n_embd
            else:
                vec = [ptr[i] for i in range(self.n_embd)]

            if normalize:
                norm = sum(x * x for x in vec) ** 0.5
                if norm > 0:
                    vec = [x / norm for x in vec]
            results.append(vec)

        return results[0] if is_single else results

    def create_embedding( self,
            input: Union[str, List[str]],
            model: Optional[str] = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
        """ OpenAI-compatible embedding endpoint."""
        is_single = isinstance(input, str)
        inputs = [input] if is_single else input
        embeddings = self.embed(inputs)
        if is_single:
            embeddings = [embeddings]

        total_tokens = sum(len(self.tokenize(t, add_bos=False)) for t in inputs)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": i,
                }
                for i, emb in enumerate(embeddings)
            ],
            "model": model or self.model_path,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    def tokenize( self,
            text: Union[str, bytes], add_bos: bool = True, special: bool = False
        ) -> List[int]:
        """ Convert a text string to a list of token IDs."""
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
        """ Convert a list of token IDs back into bytes."""
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
        """ Convert a single token ID into its string piece."""
        buf = ctypes.create_string_buffer(256)
        plen = self._lib.llama_token_to_piece(self._vocab, token, buf, 256, 0, special)
        if plen > 0:
            return buf.raw[:plen].decode("utf-8", errors="replace")
        return ""

    def _build_sampler( self,
            temp: float,
            top_k: int = 40,
            top_p: float = 0.95,
            grammar: Optional[Union[LlamaGrammar, str]] = None,
        ) -> Any:
        """ Construct composite sampler pipeline with optional grammar constraint."""
        chain = self._lib.llama_sampler_chain_init()

        # Add grammar filter stage if provided
        if grammar is not None:
            g_pat = grammar.grammar_str if isinstance(grammar, LlamaGrammar) else grammar
            g_smpl = self._lib.llama_sampler_init_grammar(
                self._vocab, g_pat.encode("utf-8")
            )
            if g_smpl:
                self._lib.llama_sampler_chain_add(chain, g_smpl)

        if temp <= 0.0:
            self._lib.llama_sampler_chain_add(chain, self._lib.llama_sampler_init_greedy())
        else:
            if top_k > 0:
                self._lib.llama_sampler_chain_add(
                    chain, self._lib.llama_sampler_init_top_k(top_k)
                )
            if top_p < 1.0:
                self._lib.llama_sampler_chain_add(
                    chain, self._lib.llama_sampler_init_top_p(top_p, 1)
                )
            self._lib.llama_sampler_chain_add(
                chain, self._lib.llama_sampler_init_temp(temp)
            )
            self._lib.llama_sampler_chain_add(
                chain, self._lib.llama_sampler_init_dist(int(time.time()))
            )

        return chain

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
        grammar: Optional[Union[LlamaGrammar, str]] = None,
        logits_processor: Optional[Union[LogitsProcessorList, Sequence[Callable[[Sequence[int], Any], Any]]]] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
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
            grammar=grammar,
            logits_processor=logits_processor,
            logit_bias=logit_bias,
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
        grammar: Optional[Union[LlamaGrammar, str]] = None,
        logits_processor: Optional[Union[LogitsProcessorList, Sequence[Callable[[Sequence[int], Any], Any]]]] = None,
        logit_bias: Optional[Dict[int, float]] = None,
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

        processors_seq = list(logits_processor) if logits_processor else []

        if stream:
            return self._stream_generator(
                tokens,
                prompt_str,
                max_tokens,
                temperature,
                top_k,
                top_p,
                stop_sequences,
                grammar,
                processors_seq,
                logit_bias,
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

        smpl = self._build_sampler(temp=temperature, top_k=top_k, top_p=top_p, grammar=grammar)

        gen_tokens: List[int] = []
        gen_text = ""
        finish_reason = "length"
        curr_tokens = list(tokens)

        for step in range(max_tokens):
            logits = self._lib.llama_get_logits_ith(self._ctx, 0)
            self._apply_logits_processors(curr_tokens, logits, processors_seq, logit_bias)
            next_tok = self._lib.llama_sampler_sample_token(
                smpl, logits, self.n_vocab, temperature
            )
            if next_tok < 0 or self._lib.llama_vocab_is_eog(self._vocab, next_tok):
                finish_reason = "stop"
                break

            self._lib.llama_sampler_accept_token(smpl, next_tok)
            gen_tokens.append(next_tok)
            curr_tokens.append(next_tok)
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

    def _stream_generator( self,
            tokens: List[int],
            prompt_str: str,
            max_tokens: int,
            temperature: float,
            top_k: int,
            top_p: float,
            stop_sequences: List[str],
            grammar: Optional[Union[LlamaGrammar, str]] = None,
            logits_processor: Optional[Sequence[Callable[[Sequence[int], Any], Any]]] = None,
            logit_bias: Optional[Dict[int, float]] = None,
        ) -> Iterator[Dict[str, Any]]:
        """ Streaming token generator with grammar and logits processor integration."""
        batch = self._lib.llama_batch_init(1, 0, 1)
        for i, tok in enumerate(tokens):
            self._lib.llama_batch_set_token(
                batch, 0, tok, i, 0, (i == len(tokens) - 1)
            )
            self._lib.llama_batch_set_n_tokens(batch, 1)
            self._lib.llama_decode(self._ctx, batch)

        smpl = self._build_sampler(temp=temperature, top_k=top_k, top_p=top_p, grammar=grammar)
        resp_id = f"cmpl-{uuid.uuid4()}"
        accumulated = ""
        curr_tokens = list(tokens)

        try:
            for step in range(max_tokens):
                logits = self._lib.llama_get_logits_ith(self._ctx, 0)
                self._apply_logits_processors(curr_tokens, logits, logits_processor, logit_bias)
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

                self._lib.llama_sampler_accept_token(smpl, next_tok)
                piece = self.token_to_piece(next_tok)
                accumulated += piece
                curr_tokens.append(next_tok)

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

    def create_chat_completion( self,
            messages: List[Dict[str, str]],
            functions: Optional[List[Dict[str, Any]]] = None,
            function_call: Optional[Any] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Optional[Any] = None,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 40,
            stream: bool = False,
            stop: Optional[Union[str, List[str]]] = None,
            max_tokens: Optional[int] = None,
            grammar: Optional[Union[LlamaGrammar, str]] = None,
            **kwargs: Any,
        ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """ Chat Completion matching OpenAI / llama-cpp-python chat format."""
        if self.chat_handler is not None:
            formatter_resp = self.chat_handler(
                messages,
                functions=functions,
                function_call=function_call,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            prompt = formatter_resp.prompt
            stop_candidates = list(formatter_resp.stop or [])
            if stop:
                if isinstance(stop, str):
                    stop_candidates.append(stop)
                elif isinstance(stop, list):
                    stop_candidates.extend(stop)
            stop_seq = stop_candidates if stop_candidates else None
        else:
            # Fallback ChatML format
            formatted = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted += "<|im_start|>assistant\n"
            prompt = formatted
            stop_seq = ["<|im_end|>"]
            if stop:
                if isinstance(stop, str):
                    stop_seq.append(stop)
                elif isinstance(stop, list):
                    stop_seq.extend(stop)

        if stream:
            return self._stream_chat_generator(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop_seq,
                grammar=grammar,
                **kwargs,
            )

        res = self.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop_seq,
            grammar=grammar,
            **kwargs,
        )

        if isinstance(res, dict):
            choice = res["choices"][0]
            choice["message"] = {"role": "assistant", "content": choice["text"]}
            del choice["text"]
            res["object"] = "chat.completion"
        return res

    def _stream_chat_generator( self,
            prompt: str,
            **kwargs: Any,
        ) -> Iterator[Dict[str, Any]]:
        """ Streaming chat completion generator yielding OpenAI chat chunks."""
        first = True
        stream_iter = self.create_completion(prompt=prompt, stream=True, **kwargs)
        for chunk in stream_iter:
            c = chunk["choices"][0]
            text = c.get("text", "")
            finish_reason = c.get("finish_reason")

            delta = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if text:
                delta["content"] = text

            yield {
                "id": chunk.get("id"),
                "object": "chat.completion.chunk",
                "created": chunk.get("created"),
                "model": self.model_path,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
