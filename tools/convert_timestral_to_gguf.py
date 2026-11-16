#!/usr/bin/env python3
""" convert_timestral_to_gguf.py - Convert Mistral Timestral models to GGUF v3 format.\n
Supports:
1. Converting Hugging Face pretrained models (e.g. `mistralai/Timestral-28M`, `mistralai/Timestral-209M`)
   from either a local directory or directly from Hugging Face Hub.
2. Generating synthetic miniature models (--synthetic) for fast unit testing and CI verification.
3. Quantization to FP32 (default), FP16, or Q8_0.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("Error: 'gguf' Python package not found. Install via: pip install gguf")

DEFAULT_QUANTILE_KNOTS = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99
]

HF_TO_GGUF_NAME_MAP = {
    # Patch projection
    "representation.patch_proj.hidden_layer.weight": "timestral.patch_proj.hidden.weight",
    "representation.patch_proj.hidden_layer.bias": "timestral.patch_proj.hidden.bias",
    "representation.patch_proj.output_layer.weight": "timestral.patch_proj.output.weight",
    "representation.patch_proj.output_layer.bias": "timestral.patch_proj.output.bias",
    "representation.patch_proj.residual_layer.weight": "timestral.patch_proj.residual.weight",
    "representation.patch_proj.residual_layer.bias": "timestral.patch_proj.residual.bias",
    # Horizon query token
    "representation.forecast_query": "timestral.forecast_query",
    # Final normalization
    "transformer.out_norm.scale": "timestral.out_norm.weight",
    # Quantile projection head
    "head.projection.hidden_layer.weight": "timestral.head.hidden.weight",
    "head.projection.output_layer.weight": "timestral.head.output.weight",
    "head.projection.residual_layer.weight": "timestral.head.residual.weight",
}


def map_layer_tensor_name(hf_name: str) -> Optional[str]:
    """ Map transformer layer tensor names from HF to canonical GGUF names."""
    # Pattern: transformer.layers.{i}.*
    parts = hf_name.split(".")
    if len(parts) < 4 or parts[0] != "transformer" or parts[1] != "layers":
        return None
    
    layer_idx = parts[2]
    sub = ".".join(parts[3:])

    # Time attention (attns.0)
    if sub == "attn_norms.0.scale":
        return f"blk.{layer_idx}.time_attn_norm.weight"
    if sub == "attns.0._gate.weight":
        return f"blk.{layer_idx}.time_attn_gate.weight"
    if sub == "attns.0.wQKV.weight":
        return f"blk.{layer_idx}.time_attn_qkv.weight"
    if sub == "attns.0.wO.weight":
        return f"blk.{layer_idx}.time_attn_o.weight"

    # Variate attention (attns.1)
    if sub == "attn_norms.1.scale":
        return f"blk.{layer_idx}.variate_attn_norm.weight"
    if sub == "attns.1._gate.weight":
        return f"blk.{layer_idx}.variate_attn_gate.weight"
    if sub == "attns.1.wQKV.weight":
        return f"blk.{layer_idx}.variate_attn_qkv.weight"
    if sub == "attns.1.wO.weight":
        return f"blk.{layer_idx}.variate_attn_o.weight"

    # MLP / Feed-Forward
    if sub == "mlp_norm.scale":
        return f"blk.{layer_idx}.ffn_norm.weight"
    if sub == "mlp.w12.weight":
        return f"blk.{layer_idx}.ffn_gate_up.weight"
    if sub == "mlp.w3.weight":
        return f"blk.{layer_idx}.ffn_down.weight"

    return None


def quantize_to_q8_0(data: np.ndarray) -> np.ndarray:
    """ Quantize 1D or 2D float32 array to Q8_0 blocks matching ggml format."""
    orig_shape = data.shape
    flat = data.astype(np.float32).reshape(-1)
    n_elements = flat.shape[0]
    block_size = 32
    if n_elements % block_size != 0:
        # Cannot quantize non-multiple of 32 to Q8_0; return as float32
        return data

    n_blocks = n_elements // block_size
    reshaped = flat.reshape(n_blocks, block_size)
    d = np.max(np.abs(reshaped), axis=1) / 127.0
    d = np.where(d == 0, 1.0, d).astype(np.float16)

    # Scale and round to int8
    d_f32 = d.astype(np.float32)[:, None]
    qs = np.clip(np.round(reshaped / d_f32), -128, 127).astype(np.int8)

    # Pack into byte buffer: for each block: 2 bytes fp16 delta + 32 bytes int8
    dt = np.dtype([('d', np.float16), ('qs', np.int8, (32,))])
    out = np.empty(n_blocks, dtype=dt)
    out['d'] = d
    out['qs'] = qs
    return out


def write_timestral_gguf(
        output_path: str,
        config: Dict[str, Any],
        tensors: Dict[str, np.ndarray],
        model_name: str = "timestral",
        out_type: str = "f32",
    ):
    """ Write GGUF file with Timestral metadata and tensors."""
    writer = gguf.GGUFWriter(output_path, "timestral")

    # 1. Architecture & General Metadata
    writer.add_name(model_name)
    writer.add_description("Mistral Timestral Time-Series Foundation Model in GGUF format")

    # 2. Hyperparameters
    patch_size = int(config.get("patch_size", 32))
    d_model = int(config.get("d_model", 384))
    num_heads = int(config.get("num_heads", 6))
    mlp_hidden_dim = int(config.get("mlp_hidden_dim", 1024))
    context_length = int(config.get("context_length", 32768))
    block_schedule = list(config.get("block_schedule", ["time"]))
    num_layers = len(block_schedule)

    writer.add_uint32("timestral.patch_size", patch_size)
    writer.add_uint32("timestral.d_model", d_model)
    writer.add_uint32("timestral.num_heads", num_heads)
    writer.add_uint32("timestral.mlp_hidden_dim", mlp_hidden_dim)
    writer.add_uint32("timestral.num_layers", num_layers)
    writer.add_uint32("timestral.context_length", context_length)

    # Block schedule (temporal vs. variate attention order)
    writer.add_array("timestral.block_schedule", block_schedule)

    # RoPE & Normalization
    writer.add_float32("timestral.rope_theta", float(config.get("rope_theta", 10000.0)))
    writer.add_float32("timestral.rope_partial_factor", float(config.get("rope_partial_factor", 0.5)))
    writer.add_float32("timestral.norm_eps", float(config.get("norm_eps", 1e-6)))

    # Quantiles
    quantile_knots = [float(x) for x in config.get("quantile_knots", DEFAULT_QUANTILE_KNOTS)]
    writer.add_array("timestral.quantile_knots", quantile_knots)

    # Scaling & Decoupled Horizon Forecasting flags
    writer.add_bool("timestral.use_asinh", bool(config.get("use_asinh", True)))
    writer.add_float32("timestral.minimum_scale", float(config.get("minimum_scale", 1e-6)))
    writer.add_float32("timestral.quantile_width_factor", float(config.get("quantile_width_factor", 3.0)))
    writer.add_bool("timestral.split_context_horizon_attn", bool(config.get("split_context_horizon_attn", True)))
    writer.add_bool("timestral.attend_missing_patches", bool(config.get("attend_missing_patches", True)))
    writer.add_bool("timestral.covariate_enabled", bool(config.get("covariate_enabled", True)))

    # 3. Add Tensors
    print(f"Writing {len(tensors)} tensors with precision '{out_type}'...")
    for name, tensor in sorted(tensors.items()):
        # Determine target dtype
        is_norm_or_bias = name.endswith(".bias") or ".norm." in name or "_norm." in name
        
        if out_type == "f16" and not is_norm_or_bias and tensor.ndim >= 2:
            data = tensor.astype(np.float16)
            writer.add_tensor(name, data)
        elif out_type == "q8_0" and not is_norm_or_bias and tensor.ndim >= 2:
            # Check if columns are multiple of 32 for Q8_0 quantization
            n_cols = tensor.shape[-1]
            if n_cols % 32 == 0:
                q_data = gguf.quants.quantize(tensor, gguf.GGMLQuantizationType.Q8_0)
                writer.add_tensor(name, q_data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                data = tensor.astype(np.float32)
                writer.add_tensor(name, data)
        else:
            data = tensor.astype(np.float32)
            writer.add_tensor(name, data)

    # 4. Finalize File
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully generated GGUF model: {output_path}")


def create_synthetic_model(
        output_path: str,
        d_model: int = 64,
        num_heads: int = 2,
        mlp_hidden_dim: int = 128,
        patch_size: int = 32,
        out_type: str = "f32",
    ):
    """ Generate a miniature synthetic Timestral model for fast unit testing."""
    block_schedule = ["time", "time_variate"]
    num_layers = len(block_schedule)
    num_quantiles = len(DEFAULT_QUANTILE_KNOTS)

    config = {
        "patch_size": patch_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "mlp_hidden_dim": mlp_hidden_dim,
        "block_schedule": block_schedule,
        "context_length": 512,
        "rope_theta": 10000.0,
        "rope_partial_factor": 0.5,
        "norm_eps": 1e-6,
        "quantile_knots": DEFAULT_QUANTILE_KNOTS,
        "use_asinh": True,
        "minimum_scale": 1e-6,
        "quantile_width_factor": 3.0,
        "split_context_horizon_attn": True,
        "attend_missing_patches": True,
        "covariate_enabled": True,
    }

    tensors: Dict[str, np.ndarray] = {}
    rng = np.random.RandomState(42)

    # Patch projection: input is 2 * patch_size (value + missing indicator)
    in_dim = patch_size * 2
    proj_hidden = d_model * 4
    tensors["timestral.patch_proj.hidden.weight"] = rng.randn(proj_hidden, in_dim).astype(np.float32) * 0.02
    tensors["timestral.patch_proj.hidden.bias"] = np.zeros(proj_hidden, dtype=np.float32)
    tensors["timestral.patch_proj.output.weight"] = rng.randn(d_model, proj_hidden).astype(np.float32) * 0.02
    tensors["timestral.patch_proj.output.bias"] = np.zeros(d_model, dtype=np.float32)
    tensors["timestral.patch_proj.residual.weight"] = rng.randn(d_model, in_dim).astype(np.float32) * 0.02
    tensors["timestral.patch_proj.residual.bias"] = np.zeros(d_model, dtype=np.float32)

    # Horizon query token
    tensors["timestral.forecast_query"] = rng.randn(d_model).astype(np.float32) * 0.02

    # Layers
    for i, btype in enumerate(block_schedule):
        # Time attention
        tensors[f"blk.{i}.time_attn_norm.weight"] = np.ones(d_model, dtype=np.float32)
        tensors[f"blk.{i}.time_attn_gate.weight"] = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        tensors[f"blk.{i}.time_attn_qkv.weight"] = rng.randn(d_model * 3, d_model).astype(np.float32) * 0.02
        tensors[f"blk.{i}.time_attn_o.weight"] = rng.randn(d_model, d_model).astype(np.float32) * 0.02

        # Variate attention (if time_variate)
        if btype == "time_variate":
            tensors[f"blk.{i}.variate_attn_norm.weight"] = np.ones(d_model, dtype=np.float32)
            tensors[f"blk.{i}.variate_attn_gate.weight"] = rng.randn(d_model, d_model).astype(np.float32) * 0.02
            tensors[f"blk.{i}.variate_attn_qkv.weight"] = rng.randn(d_model * 3, d_model).astype(np.float32) * 0.02
            tensors[f"blk.{i}.variate_attn_o.weight"] = rng.randn(d_model, d_model).astype(np.float32) * 0.02

        # MLP
        tensors[f"blk.{i}.ffn_norm.weight"] = np.ones(d_model, dtype=np.float32)
        tensors[f"blk.{i}.ffn_gate_up.weight"] = rng.randn(mlp_hidden_dim * 2, d_model).astype(np.float32) * 0.02
        tensors[f"blk.{i}.ffn_down.weight"] = rng.randn(d_model, mlp_hidden_dim).astype(np.float32) * 0.02

    # Final norm & Quantile head
    tensors["timestral.out_norm.weight"] = np.ones(d_model, dtype=np.float32)
    head_hidden = d_model * 4
    out_dim = patch_size * num_quantiles  # e.g., 32 * 21 = 672
    tensors["timestral.head.hidden.weight"] = rng.randn(head_hidden, d_model).astype(np.float32) * 0.02
    tensors["timestral.head.output.weight"] = rng.randn(out_dim, head_hidden).astype(np.float32) * 0.02
    tensors["timestral.head.residual.weight"] = rng.randn(out_dim, d_model).astype(np.float32) * 0.02

    write_timestral_gguf(
        output_path=output_path,
        config=config,
        tensors=tensors,
        model_name="timestral-synthetic",
        out_type=out_type,
    )


def convert_hf_model(
        model_source: str,
        output_path: str,
        out_type: str = "f32",
    ):
    """ Convert Hugging Face Timestral repository or directory to GGUF."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    source_path = Path(model_source)
    if source_path.is_dir() and (source_path / "config.json").exists():
        config_file = source_path / "config.json"
        weights_file = source_path / "model.safetensors"
    else:
        print(f"Fetching from Hugging Face Hub: {model_source}...")
        config_file = Path(hf_hub_download(repo_id=model_source, filename="config.json"))
        weights_file = Path(hf_hub_download(repo_id=model_source, filename="model.safetensors"))

    with open(config_file, "r") as f:
        config = json.load(f)

    print(f"Reading weights from {weights_file}...")
    tensors: Dict[str, np.ndarray] = {}
    with safe_open(weights_file, framework="numpy") as f:
        for k in f.keys():
            arr = f.get_tensor(k)
            # Map canonical name
            if k in HF_TO_GGUF_NAME_MAP:
                tensors[HF_TO_GGUF_NAME_MAP[k]] = arr
            else:
                mapped = map_layer_tensor_name(k)
                if mapped:
                    tensors[mapped] = arr
                else:
                    print(f"Warning: unmapped tensor '{k}' (shape={arr.shape})")

    model_name = Path(model_source).name or "timestral"
    write_timestral_gguf(
        output_path=output_path,
        config=config,
        tensors=tensors,
        model_name=model_name,
        out_type=out_type,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert Timestral model to GGUF format.")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face repo ID (e.g. mistralai/Timestral-28M)")
    parser.add_argument("--model-dir", type=str, default=None, help="Local directory with config.json and model.safetensors")
    parser.add_argument("--synthetic", action="store_true", help="Generate miniature synthetic model for testing")
    parser.add_argument("--out", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--precision", choices=["f32", "f16", "q8_0"], default="f32", help="Weight precision")

    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_model(args.out, out_type=args.precision)
    elif args.model_dir:
        convert_hf_model(args.model_dir, args.out, out_type=args.precision)
    elif args.model_id:
        convert_hf_model(args.model_id, args.out, out_type=args.precision)
    else:
        parser.error("Must specify one of --model-id, --model-dir, or --synthetic")


if __name__ == "__main__":
    main()
