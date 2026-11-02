"""Converter utility: Hugging Face Safetensors checkpoint directory to GGUF format."""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Optional

import numpy as np

from .safetensors import load_safetensors_dir
from .weight_maps import hf_to_gguf_name, permute_rope_hf_to_ggml


GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8


def _align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_str(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def convert_safetensors_to_gguf(
    model_dir: str,
    output_gguf_path: str,
    arch: str = "llama",
    custom_tokens: Optional[List[str]] = None,
) -> str:
    """Convert Hugging Face safetensors directory + config.json to a runnable GGUF file."""
    model_dir = os.path.abspath(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load all weights from directory (single or sharded)
    hf_weights = load_safetensors_dir(model_dir)

    # Extract architectural hyperparameters
    hidden_dim = cfg.get("hidden_size", 768)
    intermediate_dim = cfg.get("intermediate_size", 2048)
    n_heads = cfg.get("num_attention_heads", 12)
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    n_layers = cfg.get("num_hidden_layers", 12)
    vocab_size = cfg.get("vocab_size", 32000)
    norm_eps = cfg.get("rms_norm_eps", 1e-5)
    rope_theta = cfg.get("rope_theta", 10000.0)
    head_dim = hidden_dim // n_heads

    # Map and translate weights to internal GGUF format
    gguf_tensors: Dict[str, np.ndarray] = {}
    for hf_name, tensor in hf_weights.items():
        gguf_name = hf_to_gguf_name(hf_name, arch=arch)
        if gguf_name is None:
            continue

        # Convert to float32 if in float16/bfloat16
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        # Apply RoPE permutation to Q and K projections if needed
        if "attn_q.weight" in gguf_name or "attn_k.weight" in gguf_name:
            heads = n_heads if "attn_q" in gguf_name else n_kv_heads
            tensor = permute_rope_hf_to_ggml(tensor, n_heads=heads, head_dim=head_dim)

        gguf_tensors[gguf_name] = tensor

    # Ensure output.weight exists; if tied, copy from token_embd.weight
    if "output.weight" not in gguf_tensors and "token_embd.weight" in gguf_tensors:
        gguf_tensors["output.weight"] = gguf_tensors["token_embd.weight"]

    # Vocabulary tokens
    tokens = custom_tokens or [f"tok_{i}" for i in range(vocab_size)]

    # Assemble GGUF binary
    header_buf = bytearray()
    header_buf.extend(struct.pack("<I", GGUF_MAGIC))
    header_buf.extend(struct.pack("<I", GGUF_VERSION))
    header_buf.extend(struct.pack("<Q", len(gguf_tensors)))  # n_tensors

    # Count KV pairs
    kv_items = [
        ("general.architecture", GGUF_TYPE_STRING, _pack_str(arch)),
        ("general.name", GGUF_TYPE_STRING, _pack_str(cfg.get("model_type", arch))),
        (f"{arch}.context_length", GGUF_TYPE_UINT32, struct.pack("<I", cfg.get("max_position_embeddings", 2048))),
        (f"{arch}.embedding_length", GGUF_TYPE_UINT32, struct.pack("<I", hidden_dim)),
        (f"{arch}.feed_forward_length", GGUF_TYPE_UINT32, struct.pack("<I", intermediate_dim)),
        (f"{arch}.block_count", GGUF_TYPE_UINT32, struct.pack("<I", n_layers)),
        (f"{arch}.attention.head_count", GGUF_TYPE_UINT32, struct.pack("<I", n_heads)),
        (f"{arch}.attention.head_count_kv", GGUF_TYPE_UINT32, struct.pack("<I", n_kv_heads)),
        (f"{arch}.rope.freq_base", GGUF_TYPE_FLOAT32, struct.pack("<f", float(rope_theta))),
        (f"{arch}.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, struct.pack("<f", float(norm_eps))),
    ]

    # Add tokenizer tokens array
    tok_buf = bytearray()
    tok_buf.extend(struct.pack("<I", GGUF_TYPE_STRING))
    tok_buf.extend(struct.pack("<Q", len(tokens)))
    for tok in tokens:
        tok_buf.extend(_pack_str(tok))
    kv_items.append(("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, tok_buf))

    header_buf.extend(struct.pack("<Q", len(kv_items)))  # n_kv

    # Write KV entries
    for key, val_type, payload in kv_items:
        header_buf.extend(_pack_str(key))
        header_buf.extend(struct.pack("<I", val_type))
        header_buf.extend(payload)

    # Compute tensor offsets
    tensor_headers = bytearray()
    tensor_data = bytearray()
    current_data_offset = 0

    sorted_names = sorted(list(gguf_tensors.keys()))
    for name in sorted_names:
        arr = gguf_tensors[name]
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        raw_bytes = arr.tobytes()
        n_dims = len(arr.shape)

        # In GGUF ne[0] is fastest / inner dimension (reverse of numpy shape)
        shape_reversed = list(reversed(arr.shape))

        tensor_headers.extend(_pack_str(name))
        tensor_headers.extend(struct.pack("<I", n_dims))
        for d in shape_reversed:
            tensor_headers.extend(struct.pack("<Q", int(d)))
        tensor_headers.extend(struct.pack("<I", GGML_TYPE_F32))
        tensor_headers.extend(struct.pack("<Q", current_data_offset))

        # Append to tensor data with alignment padding
        tensor_data.extend(raw_bytes)
        padded_offset = _align_offset(current_data_offset + len(raw_bytes))
        pad_len = padded_offset - (current_data_offset + len(raw_bytes))
        if pad_len > 0:
            tensor_data.extend(b"\x00" * pad_len)
        current_data_offset = padded_offset

    header_buf.extend(tensor_headers)

    # Write full GGUF file
    os.makedirs(os.path.dirname(os.path.abspath(output_gguf_path)), exist_ok=True)
    with open(output_gguf_path, "wb") as f:
        f.write(header_buf)
        header_end = f.tell()
        data_start = _align_offset(header_end, GGUF_DEFAULT_ALIGNMENT)
        pad = data_start - header_end
        if pad > 0:
            f.write(b"\x00" * pad)
        f.write(tensor_data)

    return output_gguf_path

