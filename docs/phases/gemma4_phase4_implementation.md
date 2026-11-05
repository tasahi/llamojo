---
type: reference
title: "Gemma 4 Architecture Support: Phase 4 Implementation Documentation"
description: "Fused GeGLU activation functions and Gemma 4 Feed-Forward Network (FFN) block execution."
status: completed
tags: [gemma4, phase4, geglu, ffn-kernels, simd]
timestamp: 2026-11-05T00:00:00Z
---

# Gemma 4 Architecture Support: Phase 4 Implementation Documentation

This document records the design, mathematical formulation, implementation, and verification of **Phase 4: Dual-Stream Per-Layer Input Embedding (PLE)** for the Gemma 4 architecture (`gemma4`) in **llamojo**.

---

## 1. Background & Architectural Context

Standard autoregressive transformer architectures (e.g. LLaMA, Mistral, Qwen) project input token IDs through a single token embedding matrix ($W_{\text{emb}} \in \mathbb{R}^{V \times D}$) to produce the initial sequence of hidden states $x_0 \in \mathbb{R}^{N \times D}$. These states are subsequently passed sequentially through transformer decoder layers $0 \dots L-1$.

**Gemma 4 introduces a dual-stream architecture with Per-Layer Embeddings (PLE)**:
In addition to the primary token embedding stream ($D = 1536$), a parallel auxiliary stream extracts and injects layer-specific residual representations ($D_{\text{ple}} = 256$) directly into every decoder layer.

The PLE mechanism consists of two components:
1. **Token-Identity Embedding Component**:
   An auxiliary embedding table `per_layer_token_embd.weight` of dimension $(L \cdot D_{\text{ple}}) \times V = (35 \times 256) \times 262144 = 8960 \times 262,144$ (stored in quantized `Q6_K`).
2. **Context-Aware Model Projection Component**:
   A linear projection `per_layer_model_proj.weight` of dimension $D \times (L \cdot D_{\text{ple}}) = 1536 \times 8960$ (stored in `BF16` or quantized format) that projects the token embedding $x_0$ into the multi-layer stream.

---

## 2. Mathematical Formulation

For sequence length $T$ and token index $t \in [0, T-1]$:

### Step 1: Token-Identity Embedding & Scaling
For token $t$ with token ID $w_t$:
$$\text{ple\_id}_t = \text{per\_layer\_token\_embd}[w_t] \in \mathbb{R}^{L \cdot D_{\text{ple}}}$$
The raw token embedding slice is scaled by $\sqrt{D_{\text{ple}}}$:
$$\text{ple\_id}_t \leftarrow \text{ple\_id}_t \times \sqrt{D_{\text{ple}}} = \text{ple\_id}_t \times \sqrt{256} = 16.0 \cdot \text{ple\_id}_t$$

### Step 2: Context Projection, Reshaping & Projection Norm
The input token embedding $x_0(t) \in \mathbb{R}^{D}$ is projected across all layers:
$$\text{proj}_t = x_0(t) \cdot W_{\text{model\_proj}} \in \mathbb{R}^{L \cdot D_{\text{ple}}}$$
Scaled by $D^{-0.5} = \frac{1}{\sqrt{1536}}$:
$$\text{proj}_t \leftarrow \text{proj}_t \times \frac{1}{\sqrt{D}}$$
The projected vector is viewed as $L$ layer slices of dimension $D_{\text{ple}}$:
$$\text{proj}_t^{(l)} \in \mathbb{R}^{D_{\text{ple}}} \quad \text{for } l \in [0, L-1]$$
Each layer slice is normalized by the layer-shared RMSNorm `per_layer_proj_norm.weight`:
$$\text{proj}_t^{(l)} \leftarrow \text{RMSNorm}\left(\text{proj}_t^{(l)}, \gamma_{\text{proj\_norm}}\right)$$

### Step 3: Stream Fusion
The token-identity embedding and normalized context projection are fused with scale factor $2^{-0.5} = \frac{1}{\sqrt{2}}$:
$$\text{ple}_t^{(l)} = \left(\text{proj}_t^{(l)} + \text{ple\_id}_t^{(l)}\right) \times \frac{1}{\sqrt{2}}$$

### Step 4: Per-Decoder-Layer Gated Residual Injection
Within each decoder layer $l$, after attention and feed-forward MLP blocks have updated hidden state $x$:
1. **Input Gating**:
   A gating signal is computed by projecting $x$ via `blk.l.inp_gate.weight` ($D \to D_{\text{ple}}$) and applying $\text{GELU}$:
   $$g^{(l)} = \text{GELU}\left(x \cdot W_{\text{inp\_gate}}^{(l)}\right) \in \mathbb{R}^{D_{\text{ple}}}$$
2. **PLE Element-wise Modulation**:
   The layer's PLE slice is gated:
   $$h^{(l)} = g^{(l)} \odot \text{ple}_t^{(l)} \in \mathbb{R}^{D_{\text{ple}}}$$
3. **Up-Projection**:
   The modulated signal is projected back to model dimension $D$ via `blk.l.proj.weight` ($D_{\text{ple}} \to D$):
   $$u^{(l)} = h^{(l)} \cdot W_{\text{proj}}^{(l)} \in \mathbb{R}^{D}$$
4. **Post-Projection Normalization & Residual Addition**:
   The projected update is normalized via `blk.l.post_norm.weight` and accumulated into the layer state:
   $$x \leftarrow x + \text{RMSNorm}\left(u^{(l)}, \gamma_{\text{post\_norm}}^{(l)}\right)$$
5. **Layer Output Scaling**:
   Finally, if `blk.l.layer_output_scale.weight` ($\alpha^{(l)}$) is present:
   $$x \leftarrow x \times \alpha^{(l)}$$

---

## 3. Implemented Changes

### A. Weight Structures (`src/llamojo/backend/models.mojo`)
Added fields to `LayerWeights` to store Gemma 4 per-layer tensors:
- `attn_q_norm: FloatPtr`: Query per-head RMSNorm weights.
- `attn_k_norm: FloatPtr`: Key per-head RMSNorm weights.
- `post_attention_norm: FloatPtr`: RMSNorm weights following attention residual.
- `post_ffw_norm: FloatPtr`: RMSNorm weights following FFN residual.
- `inp_gate: BytePtr` & `dtype_inp_gate: UInt32`: Linear projection weights from hidden dimension ($D=1536$) to PLE dimension ($D_{\text{ple}}=256$).
- `proj: BytePtr` & `dtype_proj: UInt32`: Linear projection weights from PLE dimension ($D_{\text{ple}}=256$) to hidden dimension ($D=1536$).
- `post_norm: FloatPtr`: RMSNorm weights applied to projected PLE output.
- `layer_output_scale: Float32`: Per-layer output scaling factor.

### B. Global Model Loader (`src/llamojo/llama.mojo`)
Updated `llama_model.__init__` to load model-level PLE weights:
- `per_layer_token_embd`: Byte pointer to quantized token embedding table (`Q6_K`, $8960 \times 262144$).
- `per_layer_token_embd_type`: Data type of `per_layer_token_embd`.
- `per_layer_model_proj`: Byte pointer to context projection matrix ($1536 \times 8960$).
- `per_layer_model_proj_type`: Data type of `per_layer_model_proj`.
- `per_layer_proj_norm`: Pointer to RMSNorm scale vector ($D_{\text{ple}}=256$).
- Populated each layer's `inp_gate`, `proj`, `post_norm`, and `layer_output_scale`.

---

## 4. Verification & Testing

- Verified tensor binding against `gemma-4-E2B-it.Q4_K_M.gguf`:
  - `per_layer_model_proj.weight`: Shape `[1536, 8960]`, type `BF16` / quantized.
  - `per_layer_proj_norm.weight`: Shape `[256]`, type `F32`.
  - `per_layer_token_embd.weight`: Shape `[8960, 262144]`, type `Q6_K`.
  - `blk.*.inp_gate.weight`: Shape `[1536, 256]`, type `Q4_K`.
  - `blk.*.proj.weight`: Shape `[256, 1536]`, type `Q4_K`.
  - `blk.*.post_norm.weight`: Shape `[1536]`, type `F32`.
  - `blk.*.layer_output_scale.weight`: Shape `[1]`, type `F32`.
- Ran full regression suite `./run-tests.sh` across all 11 test suites; 100% passed without regression.

---

## 5. Next Steps

With Phase 4's data structures, tensor bindings, and mathematical specifications complete:
1. Proceed to **Phase 5: Dedicated Gemma 4 Forward Pass Graph (`gemma4.mojo`)**, implementing:
   - Token & PLE projection kernel (`project_per_layer_inputs`).
   - Per-layer forward pass combining SWA/Global attention, double RMSNorm, MoE/dense FFN, and PLE gating.
   - Tied LM-head projection with logit softcapping.
2. Route `llama_decode` to the dedicated Gemma 4 pipeline when `config.arch == "gemma4"`.

