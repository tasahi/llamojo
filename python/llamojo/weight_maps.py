"""Weight mapping tables and RoPE permutation handlers for Hugging Face to GGUF (Part 2).

Provides bidirectional conversion between standard HuggingFace / PyTorch checkpoint keys
and llamojo internal GGUF tensor representations (LLaMA, Gemma, Mistral, ModernBERT).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def hf_to_gguf_name(hf_name: str, arch: str = "llama") -> Optional[str]:
    """Map a Hugging Face parameter key to its internal GGUF equivalent."""
    # 1. Embeddings & Final Norm/Heads
    if hf_name in ("model.embed_tokens.weight", "embed_tokens.weight"):
        return "token_embd.weight"
    if hf_name in ("model.norm.weight", "norm.weight"):
        return "output_norm.weight"
    if hf_name in ("lm_head.weight",):
        return "output.weight"

    # 2. ModernBERT specific naming
    if hf_name in ("model.embeddings.tok_embeddings.weight", "embeddings.tok_embeddings.weight"):
        return "token_embd.weight"
    if hf_name in ("model.final_norm.weight", "final_norm.weight"):
        return "output_norm.weight"

    # 3. Layer-wise parameter mapping
    # Format: model.layers.{idx}.xxx or layers.{idx}.xxx
    parts = hf_name.split(".")
    if len(parts) >= 4 and parts[0] in ("model", "transformer") and parts[1] == "layers":
        layer_idx = parts[2]
        rest = ".".join(parts[3:])
    elif len(parts) >= 3 and parts[0] == "layers":
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
    else:
        return None

    prefix = f"blk.{layer_idx}"

    layer_map = {
        "input_layernorm.weight": f"{prefix}.attn_norm.weight",
        "post_attention_layernorm.weight": f"{prefix}.ffn_norm.weight",
        "pre_feedforward_layernorm.weight": f"{prefix}.ffn_norm.weight",
        "post_feedforward_layernorm.weight": f"{prefix}.ffn_post_norm.weight",
        # Attention
        "self_attn.q_proj.weight": f"{prefix}.attn_q.weight",
        "self_attn.k_proj.weight": f"{prefix}.attn_k.weight",
        "self_attn.v_proj.weight": f"{prefix}.attn_v.weight",
        "self_attn.o_proj.weight": f"{prefix}.attn_output.weight",
        "self_attn.q_proj.bias": f"{prefix}.attn_q.bias",
        "self_attn.k_proj.bias": f"{prefix}.attn_k.bias",
        "self_attn.v_proj.bias": f"{prefix}.attn_v.bias",
        # MLP (SwiGLU)
        "mlp.gate_proj.weight": f"{prefix}.ffn_gate.weight",
        "mlp.up_proj.weight": f"{prefix}.ffn_up.weight",
        "mlp.down_proj.weight": f"{prefix}.ffn_down.weight",
        # MLP (GELU / ModernBERT / standard)
        "mlp.Wi.weight": f"{prefix}.ffn_up.weight",
        "mlp.Wo.weight": f"{prefix}.ffn_down.weight",
    }

    return layer_map.get(rest)


def gguf_to_hf_name(gguf_name: str, arch: str = "llama") -> Optional[str]:
    """Map an internal GGUF tensor name to its standard Hugging Face key."""
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"

    parts = gguf_name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
        prefix = f"model.layers.{layer_idx}"

        inv_map = {
            "attn_norm.weight": f"{prefix}.input_layernorm.weight",
            "ffn_norm.weight": f"{prefix}.post_attention_layernorm.weight",
            "attn_q.weight": f"{prefix}.self_attn.q_proj.weight",
            "attn_k.weight": f"{prefix}.self_attn.k_proj.weight",
            "attn_v.weight": f"{prefix}.self_attn.v_proj.weight",
            "attn_output.weight": f"{prefix}.self_attn.o_proj.weight",
            "ffn_gate.weight": f"{prefix}.mlp.gate_proj.weight",
            "ffn_up.weight": f"{prefix}.mlp.up_proj.weight",
            "ffn_down.weight": f"{prefix}.mlp.down_proj.weight",
        }
        return inv_map.get(rest)

    return None


def permute_rope_hf_to_ggml(
    weights: np.ndarray, n_heads: int, head_dim: int
) -> np.ndarray:
    """Permute rotary embedding weight rows from HuggingFace to GGML/llamojo layout.

    HF stores complex RoPE pairs interleaved: [x0, x1, x2, x3...] where (x0, x1) is a pair.
    GGML stores real/imag blocks split: [x0, x2, ..., x1, x3...].
    """
    if len(weights.shape) != 2:
        return weights

    out_dim, in_dim = weights.shape
    if out_dim != n_heads * head_dim:
        return weights

    w = weights.reshape(n_heads, 2, head_dim // 2, in_dim)
    w = w.swapaxes(1, 2)
    return w.reshape(out_dim, in_dim)


def permute_rope_ggml_to_hf(
    weights: np.ndarray, n_heads: int, head_dim: int
) -> np.ndarray:
    """Permute rotary embedding weight rows from GGML/llamojo to HuggingFace layout."""
    if len(weights.shape) != 2:
        return weights

    out_dim, in_dim = weights.shape
    if out_dim != n_heads * head_dim:
        return weights

    w = weights.reshape(n_heads, head_dim // 2, 2, in_dim)
    w = w.swapaxes(1, 2)
    return w.reshape(out_dim, in_dim)

