---
type: reference
title: "Gemma 4 Architecture Support: Phase 2 Implementation Documentation"
description: "Implementation and numerical verification of Proportional Logit Expansion (PLE) and per-head attention norm kernels."
status: completed
tags: [gemma4, phase2, ple, attention-norm, numerical-validation]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Architecture Support: Phase 2 Implementation Documentation

This document records the design, implementation, and verification of **Phase 2: Math Kernels: GELU, Double RMSNorm & Softcapping** for the Gemma 4 architecture (`gemma4`) in **llamojo**.

---

## 1. Context & Motivation

In standard LLaMA-family models:
- The FFN sub-block uses **SwiGLU** activation: $\text{SiLU}(\text{gate}) \cdot \text{up}$.
- Layer normalization is applied once before the attention and FFN blocks (**pre-norm**).
- Attention queries ($Q$) and keys ($K$) undergo rotary positional embedding (RoPE) without normalization, and values ($V$) are stored unnormalized.
- The output logits are passed directly to top-k/top-p/greedy sampling.

In contrast, **Gemma 4** (`gemma4`) introduces four distinct mathematical operators:
1. **GELU Non-Linearity (GELU-GLU)**: The FFN layers and per-layer embedding input gates use approximate GELU rather than SiLU.
2. **Per-Head Q and K Normalization**: Queries and Keys are normalized on a per-head basis prior to RoPE rotation using separate scale weights (`blk.*.attn_q_norm.weight` and `attn_k_norm.weight`).
3. **Unweighted V-Normalisation**: Values ($V$) are normalized using unweighted RMSNorm (`ggml_rms_norm`) before insertion into the KV cache.
4. **Logit Softcapping**: Logits from the LM head are scaled and clamped via $\text{cap} \cdot \tanh(\text{logits} / \text{cap})$ with $\text{cap} = 30.0$ before sampling to prevent extreme logit drift.

Phase 2 implements these mathematical primitives in pure Mojo with zero external dependencies and full SIMD vectorization.

---

## 2. Implemented Kernels

All kernels are implemented in [`src/llamojo/backend/kernels_norm_rope.mojo`](../../src/llamojo/backend/kernels_norm_rope.mojo) and exposed through the library namespaces.

### A. Approximate GELU & Fused GELU-GLU

Implements the standard approximate GELU formulation matching `llama.cpp` and PyTorch:
$$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot x \cdot (1 + 0.044715 \cdot x^2)\right)\right)$$

```mojo
@always_inline
def gelu(x: Float32) -> Float32:
    comptime SQRT_2_OVER_PI: Float32 = 0.7978845608
    comptime GELU_COEF_A: Float32 = 0.044715
    var inner = SQRT_2_OVER_PI * x * (Float32(1.0) + GELU_COEF_A * x * x)
    return Float32(0.5) * x * (Float32(1.0) + tanh(inner))

@always_inline
def fused_gelu_mul(gate: FloatPtr, up: FloatPtr, out_ptr: FloatPtr, n: Int):
    for i in range(n):
        out_ptr[unsafe_offset=i] = gelu(gate[unsafe_offset=i]) * up[unsafe_offset=i]
```

### B. Per-Head Q & K Normalization (`rms_norm_per_head`)

Normalizes each individual attention head of dimension `head_dim` independently, using SIMD vector loads, reductions (`(xv * xv).reduce_add()`), and head-specific weight scaling:

$$\text{head}_h = \frac{\text{head}_h}{\sqrt{\frac{1}{d_{\text{head}}}\sum \text{head}_h^2 + \epsilon}} \odot \text{weight}$$

```mojo
@always_inline
def rms_norm_per_head(
        x: FloatPtr,
        weight: FloatPtr,
        n_heads: Int,
        head_dim: Int,
        eps: Float32 = 1e-6,
    ):
    for h in range(n_heads):
        var h_off = h * head_dim
        var ss: Float32 = 0.0
        var i = 0
        while i + SIMD_WIDTH <= head_dim:
            var xv = x.unsafe_offset(h_off).unsafe_load[width=SIMD_WIDTH](i)
            ss += (xv * xv).reduce_add()
            i += SIMD_WIDTH
        while i < head_dim:
            var val = x[unsafe_offset=h_off + i]
            ss += val * val
            i += 1

        var inv_rms = Float32(1.0) / sqrt(ss / Float32(head_dim) + eps)
        var inv_v = F32xW(inv_rms)

        i = 0
        while i + SIMD_WIDTH <= head_dim:
            var xv = x.unsafe_offset(h_off).unsafe_load[width=SIMD_WIDTH](i)
            var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
            x.unsafe_offset(h_off).unsafe_store[width=SIMD_WIDTH](i, xv * inv_v * wv)
            i += SIMD_WIDTH
        while i < head_dim:
            x[unsafe_offset=h_off + i] = x[unsafe_offset=h_off + i] * inv_rms * weight[unsafe_offset=i]
            i += 1
```

### C. Unweighted V-Normalization (`rms_norm_no_weight`)

Computes unweighted RMSNorm ($V_{\text{normed}} = V / \text{RMS}(V)$) directly into an output buffer using SIMD vectors without weight multiplication.

### D. Logit Softcapping (`apply_logit_softcapping`)

Scales and clamps logits in-place before sampler consumption:
$$\text{logits}[i] = \text{cap} \cdot \tanh\left(\frac{\text{logits}[i]}{\text{cap}}\right) \quad (\text{cap} = 30.0)$$

```mojo
@always_inline
def apply_logit_softcapping(logits: FloatPtr, n_vocab: Int, cap: Float32 = 30.0):
    if cap <= 0.0:
        return
    var inv_cap = Float32(1.0) / cap
    for i in range(n_vocab):
        var v = logits[unsafe_offset=i]
        logits[unsafe_offset=i] = cap * tanh(v * inv_cap)
```

---

## 3. Verification & Validation

### A. Dedicated Unit Tests (`tests/test_kernels.mojo`)

Added test coverage for all new math kernels in `tests/test_kernels.mojo`:
- **GELU Accuracy**: Tested known analytical values:
  - $\text{GELU}(0.0) = 0.0$
  - $\text{GELU}(1.0) \approx 0.8413$
  - $\text{GELU}(-1.0) \approx -0.1587$
  - $\text{GELU}(10.0) \approx 10.0$
- **Fused GELU-GLU**: Verified elementwise `gelu(gate) * up` multiplication.
- **Per-Head RMSNorm**: Validated on multi-head buffers with distinct per-head energies to verify head boundary isolation.
- **Unweighted RMSNorm**: Verified output magnitude normalization on uniform inputs.
- **Logit Softcapping**: Verified $0.0 \to 0.0$, $+300.0 \to +30.0$, and $-300.0 \to -30.0$.

### B. Full Test Suite Regression Pass

Executed `./run-tests.sh` to ensure zero regressions across existing models and APIs:
```text
============================================================
Running Phase 2 Kernels Test Suite
============================================================
Testing fused_rms_norm_residual and fused_silu_mul...
  ✓ Fused RMSNorm & SwiGLU passed
Testing GELU, Per-Head RMSNorm, and Logit Softcapping...
  ✓ GELU, Per-Head RMSNorm & Softcapping passed
Testing apply_rope_neox in-place rotary embeddings...
  ✓ RoPE rotation passed
Testing quantize_row_q8_0 and dot_q8_0_row...
  ✓ Q8_0 activation quantization and dot product passed
Testing k4_scale bit unpacking and dot_q4_k_row...
  ✓ k4_scale unpacking & Q4_K dot product passed
Testing dot_mxfp4_row...
  ✓ MXFP4 dot product passed
Testing matmul_quant_fused for QKV (n=3) and Gate/Up (n=2)...
  ✓ matmul_quant_fused QKV (n=3) passed
All Phase 2 Kernel tests passed! ✓
...
============================================================
All llamojo tests passed! ✓
============================================================
```

---

## 4. Current Status and Next Steps

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | GGUF Metadata Parser & Model Config Upgrades | **Complete ✓** |
| **Phase 2** | **Math Kernels: GELU, Double RMSNorm & Softcapping** | **Complete ✓** |
| **Phase 3** | Hybrid & Shared KV Cache Engine | *Next* |
| **Phase 4** | Dual-Stream Per-Layer Input Embedding (PLE) | Pending |
| **Phase 5** | Dedicated Gemma 4 Forward Pass Graph (`gemma4.mojo`) | Pending |
| **Phase 6** | Integration, Verification & CLI Benchmarking | Pending |

