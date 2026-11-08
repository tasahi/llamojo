""" Low-level ctypes bindings for libllamojo.so."""

import ctypes
import os
import pathlib
from typing import List, Optional

_lib: Optional[ctypes.CDLL] = None


def find_library() -> pathlib.Path:
    """Locate libllamojo.so relative to repo root, package dir, or environment."""
    env_path = os.environ.get("LLAMOJO_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return pathlib.Path(env_path).resolve()

    pkg_dir = pathlib.Path(__file__).parent.resolve()
    candidates = [
        pkg_dir / "libllamojo.so",
        pkg_dir.parent.parent / "build" / "libllamojo.so",
        pathlib.Path.cwd() / "build" / "libllamojo.so",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not find libllamojo.so. Please run './tools/build_shared_lib.sh' "
        "or set LLAMOJO_LIB_PATH environment variable."
    )


def get_lib() -> ctypes.CDLL:
    """Initialize and return the CDLL instance."""
    global _lib
    if _lib is not None:
        return _lib

    # Automatically resolve modular runtime lib dir if needed
    modular_lib_dirs = [
        pathlib.Path(p)
        for p in [
            "/home/tasahi/miniconda/envs/moj/lib/python3.14/site-packages/modular/lib",
            os.path.expanduser("~/.modular/lib"),
        ]
        if os.path.isdir(p)
    ]
    for d in modular_lib_dirs:
        curr = os.environ.get("LD_LIBRARY_PATH", "")
        if str(d) not in curr:
            os.environ["LD_LIBRARY_PATH"] = f"{d}:{curr}" if curr else str(d)
        for sub in d.glob("libAsyncRT*.so"):
            try:
                ctypes.CDLL(str(sub))
            except Exception:
                pass

    lib_path = find_library()
    lib = ctypes.CDLL(str(lib_path))

    # Backend lifecycle
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None

    lib.llama_backend_free.argtypes = []
    lib.llama_backend_free.restype = None

    # Model lifecycle & introspection
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.llama_model_load_from_file.restype = ctypes.c_void_p

    lib.llama_model_free.argtypes = [ctypes.c_void_p]
    lib.llama_model_free.restype = None

    lib.llama_model_n_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_vocab.restype = ctypes.c_int32

    lib.llama_model_n_ctx_train.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_ctx_train.restype = ctypes.c_int32

    lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_embd.restype = ctypes.c_int32

    lib.llama_model_n_layer.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_layer.restype = ctypes.c_int32

    lib.llama_model_n_head.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_head.restype = ctypes.c_int32

    # Vocab & Tokenization
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p

    lib.llama_vocab_free.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_free.restype = None

    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_n_tokens.restype = ctypes.c_int32

    lib.llama_vocab_bos.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_bos.restype = ctypes.c_int32

    lib.llama_vocab_eos.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_eos.restype = ctypes.c_int32

    lib.llama_vocab_eot.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_eot.restype = ctypes.c_int32

    lib.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_vocab_is_eog.restype = ctypes.c_bool

    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32

    lib.llama_token_to_piece.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32

    # Context lifecycle
    lib.llama_init_from_model.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    lib.llama_init_from_model.restype = ctypes.c_void_p

    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free.restype = None

    # Batch and Inference
    lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.llama_batch_init.restype = ctypes.c_void_p

    lib.llama_batch_free.argtypes = [ctypes.c_void_p]
    lib.llama_batch_free.restype = None

    lib.llama_batch_set_token.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    lib.llama_batch_set_token.restype = None

    lib.llama_batch_set_n_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_batch_set_n_tokens.restype = None

    lib.llama_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llama_decode.restype = ctypes.c_int32

    lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    lib.llama_get_embeddings.argtypes = [ctypes.c_void_p]
    lib.llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

    lib.llama_kv_cache_clear.argtypes = [ctypes.c_void_p]
    lib.llama_kv_cache_clear.restype = None

    # Context State Serialization (Gap 4)
    lib.llama_state_get_size.argtypes = [ctypes.c_void_p]
    lib.llama_state_get_size.restype = ctypes.c_uint64

    lib.llama_state_get_data.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.llama_state_get_data.restype = ctypes.c_uint64

    lib.llama_state_set_data.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.llama_state_set_data.restype = ctypes.c_uint64

    # Samplers
    lib.llama_sampler_chain_init.argtypes = []
    lib.llama_sampler_chain_init.restype = ctypes.c_void_p

    lib.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llama_sampler_chain_add.restype = None

    lib.llama_sampler_init_greedy.argtypes = []
    lib.llama_sampler_init_greedy.restype = ctypes.c_void_p

    lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    lib.llama_sampler_init_temp.restype = ctypes.c_void_p

    lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    lib.llama_sampler_init_top_k.restype = ctypes.c_void_p

    lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_int32]
    lib.llama_sampler_init_top_p.restype = ctypes.c_void_p

    lib.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    lib.llama_sampler_init_dist.restype = ctypes.c_void_p

    lib.llama_sampler_sample_token.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
    ]
    lib.llama_sampler_sample_token.restype = ctypes.c_int32

    lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
    lib.llama_sampler_free.restype = None

    # GBNF Grammar
    lib.llama_grammar_init.argtypes = [ctypes.c_char_p]
    lib.llama_grammar_init.restype = ctypes.c_void_p

    lib.llama_grammar_free.argtypes = [ctypes.c_void_p]
    lib.llama_grammar_free.restype = None

    lib.llama_sampler_init_grammar.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.llama_sampler_init_grammar.restype = ctypes.c_void_p

    lib.llama_sampler_accept_token.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_sampler_accept_token.restype = None

    # LoRA Adapters
    lib.llama_lora_adapter_init.argtypes = [ctypes.c_char_p]
    lib.llama_lora_adapter_init.restype = ctypes.c_void_p

    lib.llama_lora_adapter_free.argtypes = [ctypes.c_void_p]
    lib.llama_lora_adapter_free.restype = None

    lib.llama_lora_adapter_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
    ]
    lib.llama_lora_adapter_apply.restype = ctypes.c_int32

    # Part 1: Zero-Copy Views
    lib.llamojo_get_logits_view.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llamojo_get_logits_view.restype = ctypes.c_int32

    lib.llamojo_get_embeddings_view.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llamojo_get_embeddings_view.restype = ctypes.c_int32

    lib.llamojo_get_hidden_view.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llamojo_get_hidden_view.restype = ctypes.c_int32

    lib.llamojo_get_kv_view.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    lib.llamojo_get_kv_view.restype = ctypes.c_int32

    # Whisper ASR C-ABI exports (optional if older libllamojo.so is loaded)
    if hasattr(lib, "whisper_init_from_file"):
        lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
        lib.whisper_init_from_file.restype = ctypes.c_void_p

        lib.whisper_free.argtypes = [ctypes.c_void_p]
        lib.whisper_free.restype = None

        lib.whisper_is_multilingual.argtypes = [ctypes.c_void_p]
        lib.whisper_is_multilingual.restype = ctypes.c_int32

        lib.whisper_full_c.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.whisper_full_c.restype = ctypes.c_int32

        if hasattr(lib, "whisper_full_ex_c"):
            lib.whisper_full_ex_c.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32,
                ctypes.c_char_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
            ]
            lib.whisper_full_ex_c.restype = ctypes.c_int32

        lib.whisper_full_wav_c.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.whisper_full_wav_c.restype = ctypes.c_int32

        lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        lib.whisper_full_n_segments.restype = ctypes.c_int32

        lib.whisper_full_get_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.whisper_full_get_segment_t0.restype = ctypes.c_int64

        lib.whisper_full_get_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.whisper_full_get_segment_t1.restype = ctypes.c_int64

        lib.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.whisper_full_get_segment_text.restype = ctypes.c_char_p

    lib.llama_backend_init()
    _lib = lib
    return _lib
