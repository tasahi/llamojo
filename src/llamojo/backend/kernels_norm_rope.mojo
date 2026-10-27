""" Fused Normalization, Residual Connections, and Rotary Positional Embeddings (RoPE).\n
Consolidates and modernizes:
- MojoLlama/src/mojollama/kernels/fused_kernels.mojo
- MojoLlama/src/mojollama/mojo_engine/rope_kv_cache.mojo
- llama2.mojo rmsnorm / rope_head

Provides zero-allocation fused passes for:
1. fused_rms_norm_residual (add residual + compute RMSNorm in one pass over memory)
2. fused_silu_mul (SwiGLU activation: SiLU(gate) * up)
3. apply_rope / rope_head (Rotary positional embedding for Query and Key heads)
4. apply_rope_neox (GPT-NeoX / Qwen style RoPE rotation)
"""

from std.memory import Pointer
from std.math import sqrt, exp, pow, cos, sin, tanh
from .isa import SIMD_WIDTH, F32xW

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

# ─── Fused RMSNorm + Residual Connection ──────────────────────────────────────

@always_inline
def fused_rms_norm_residual(
        x: FloatPtr,
        residual: FloatPtr,
        weight: FloatPtr,
        out_normed: FloatPtr,
        out_residual: FloatPtr,
        n: Int,
        eps: Float32 = 1e-5,
    ):
    """ Fused residual addition and RMS normalization.\n
    Computes:
      out_residual[i] = x[i] + residual[i]
      out_normed[i] = (out_residual[i] * weight[i]) / sqrt(mean(out_residual^2) + eps)
    """
    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        var rv = residual.unsafe_load[width=SIMD_WIDTH](i)
        var sum_v = xv + rv
        out_residual.unsafe_store[width=SIMD_WIDTH](i, sum_v)
        ss += (sum_v * sum_v).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var sum_s = x[unsafe_offset=i] + residual[unsafe_offset=i]
        out_residual[unsafe_offset=i] = sum_s
        ss += sum_s * sum_s
        i += 1

    var inv_rms = Float32(1.0) / sqrt(ss / Float32(n) + eps)
    var inv_v = F32xW(inv_rms)

    i = 0
    while i + SIMD_WIDTH <= n:
        var rv = out_residual.unsafe_load[width=SIMD_WIDTH](i)
        var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
        out_normed.unsafe_store[width=SIMD_WIDTH](i, rv * inv_v * wv)
        i += SIMD_WIDTH
    while i < n:
        out_normed[unsafe_offset=i] = out_residual[unsafe_offset=i] * inv_rms * weight[unsafe_offset=i]
        i += 1


@always_inline
def rms_norm(
        out_ptr: FloatPtr,
        x: FloatPtr,
        weight: FloatPtr,
        n: Int,
        eps: Float32 = 1e-5,
    ):
    """ Standard RMSNorm without residual addition."""
    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        ss += (xv * xv).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var val = x[unsafe_offset=i]
        ss += val * val
        i += 1

    var inv_rms = Float32(1.0) / sqrt(ss / Float32(n) + eps)
    var inv_v = F32xW(inv_rms)

    i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
        out_ptr.unsafe_store[width=SIMD_WIDTH](i, xv * inv_v * wv)
        i += SIMD_WIDTH
    while i < n:
        out_ptr[unsafe_offset=i] = x[unsafe_offset=i] * inv_rms * weight[unsafe_offset=i]
        i += 1

# ─── Fused SwiGLU / GELU Activations ─────────────────────────────────────────

@always_inline
def silu(x: Float32) -> Float32:
    """ SiLU (Swish) activation: x / (1 + exp(-x))."""
    return x / (Float32(1.0) + exp(-x))

@always_inline
def fused_silu_mul(
        gate: FloatPtr,
        up: FloatPtr,
        out_ptr: FloatPtr,
        n: Int,
    ):
    """ Fused SiLU gate activation and elementwise product: out[i] = silu(gate[i]) * up[i]."""
    for i in range(n):
        var g = gate[unsafe_offset=i]
        var u = up[unsafe_offset=i]
        out_ptr[unsafe_offset=i] = silu(g) * u

@always_inline
def gelu(x: Float32) -> Float32:
    """ Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))."""
    comptime SQRT_2_OVER_PI: Float32 = 0.7978845608
    comptime GELU_COEF_A: Float32 = 0.044715
    var inner = SQRT_2_OVER_PI * x * (Float32(1.0) + GELU_COEF_A * x * x)
    return Float32(0.5) * x * (Float32(1.0) + tanh(inner))

@always_inline
def fused_gelu_mul(
        gate: FloatPtr,
        up: FloatPtr,
        out_ptr: FloatPtr,
        n: Int,
    ):
    """ Fused GELU gate activation and elementwise product: out[i] = gelu(gate[i]) * up[i]."""
    for i in range(n):
        var g = gate[unsafe_offset=i]
        var u = up[unsafe_offset=i]
        out_ptr[unsafe_offset=i] = gelu(g) * u

@always_inline
def rms_norm_per_head(
        x: FloatPtr,
        weight: FloatPtr,
        n_heads: Int,
        head_dim: Int,
        eps: Float32 = 1e-6,
    ):
    """ In-place RMSNorm per head (used by Gemma 4 Q/K normalization).
    x has shape [n_heads * head_dim].
    weight has shape [head_dim].
    """
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

@always_inline
def rms_norm_no_weight(
        out_ptr: FloatPtr,
        x: FloatPtr,
        n: Int,
        eps: Float32 = 1e-6,
    ):
    """ Unweighted RMSNorm (used by Gemma 4 V normalization: ggml_rms_norm)."""
    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        ss += (xv * xv).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var val = x[unsafe_offset=i]
        ss += val * val
        i += 1

    var inv_rms = Float32(1.0) / sqrt(ss / Float32(n) + eps)
    var inv_v = F32xW(inv_rms)

    i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        out_ptr.unsafe_store[width=SIMD_WIDTH](i, xv * inv_v)
        i += SIMD_WIDTH
    while i < n:
        out_ptr[unsafe_offset=i] = x[unsafe_offset=i] * inv_rms
        i += 1

@always_inline
def apply_logit_softcapping(
        logits: FloatPtr,
        n_vocab: Int,
        cap: Float32 = 30.0,
    ):
    """ Apply logit softcapping: logits[i] = cap * tanh(logits[i] / cap)."""
    if cap <= 0.0:
        return
    var inv_cap = Float32(1.0) / cap
    for i in range(n_vocab):
        var v = logits[unsafe_offset=i]
        logits[unsafe_offset=i] = cap * tanh(v * inv_cap)


# ─── Rotary Positional Embedding (RoPE) ──────────────────────────────────────

@always_inline
def rope_head(
        q: FloatPtr,
        k: FloatPtr,
        freq_cis_real: FloatPtr,
        freq_cis_imag: FloatPtr,
        h: Int,
        head_dim: Int,
        n_kv_heads: Int,
    ):
    """ Standard interleaved RoPE rotation (llama2.c style) for head `h`."""
    var q_offset = h * head_dim
    for i in range(0, head_dim, 2):
        var fcr = freq_cis_real[unsafe_offset=i // 2]
        var fci = freq_cis_imag[unsafe_offset=i // 2]
        var q_idx = q_offset + i
        var q0 = q[unsafe_offset=q_idx]
        var q1 = q[unsafe_offset=q_idx + 1]
        q[unsafe_offset=q_idx] = q0 * fcr - q1 * fci
        q[unsafe_offset=q_idx + 1] = q0 * fci + q1 * fcr

        if h < n_kv_heads:
            var k0 = k[unsafe_offset=q_idx]
            var k1 = k[unsafe_offset=q_idx + 1]
            k[unsafe_offset=q_idx] = k0 * fcr - k1 * fci
            k[unsafe_offset=q_idx + 1] = k0 * fci + k1 * fcr

@always_inline
def apply_rope_norm(
        q: FloatPtr,
        k: FloatPtr,
        n_heads: Int,
        n_kv_heads: Int,
        head_dim: Int,
        pos: Int,
        rope_theta: Float32 = 10000.0,
        rope_freq_scale: Float32 = 1.0,
    ):
    """ Apply standard interleaved RoPE (LLAMA_ROPE_TYPE_NORM: 2*i and 2*i+1 rotated together)."""
    var inv_theta = Float32(1.0) / rope_theta
    var half_dim = head_dim // 2
    for h in range(n_heads):
        var q_head = q.unsafe_offset(h * head_dim)
        for i in range(half_dim):
            var freq = pow(inv_theta, (Float32(i * 2) / Float32(head_dim)) * rope_freq_scale)
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var q0 = q_head[unsafe_offset=i * 2]
            var q1 = q_head[unsafe_offset=i * 2 + 1]
            q_head[unsafe_offset=i * 2] = q0 * cos_a - q1 * sin_a
            q_head[unsafe_offset=i * 2 + 1] = q0 * sin_a + q1 * cos_a

    for h in range(n_kv_heads):
        var k_head = k.unsafe_offset(h * head_dim)
        for i in range(half_dim):
            var freq = pow(inv_theta, (Float32(i * 2) / Float32(head_dim)) * rope_freq_scale)
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var k0 = k_head[unsafe_offset=i * 2]
            var k1 = k_head[unsafe_offset=i * 2 + 1]
            k_head[unsafe_offset=i * 2] = k0 * cos_a - k1 * sin_a
            k_head[unsafe_offset=i * 2 + 1] = k0 * sin_a + k1 * cos_a


@always_inline
def apply_rope_neox(
        q: FloatPtr,
        k: FloatPtr,
        n_heads: Int,
        n_kv_heads: Int,
        head_dim: Int,
        pos: Int,
        rope_theta: Float32 = 10000.0,
        rope_freq_scale: Float32 = 1.0,
        max_rot_dim: Int = -1,
    ):
    """ Apply NeoX/half-split style RoPE (Llama-3/Qwen/Gemma) directly in-place.
    If max_rot_dim > 0, only pairs within max_rot_dim are rotated (dimensions >= max_rot_dim are left unchanged),
    matching proportional RoPE / rope_freqs cutoff in Gemma 4.
    """
    var inv_theta = Float32(1.0) / rope_theta
    var half_dim = head_dim // 2
    var rot_half = (max_rot_dim // 2) if (max_rot_dim > 0 and max_rot_dim < head_dim) else half_dim
    for h in range(n_heads):
        var q_head = q.unsafe_offset(h * head_dim)
        for i in range(rot_half):
            var freq = pow(inv_theta, (Float32(i * 2) / Float32(head_dim)) * rope_freq_scale)
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var q0 = q_head[unsafe_offset=i]
            var q1 = q_head[unsafe_offset=i + half_dim]
            q_head[unsafe_offset=i] = q0 * cos_a - q1 * sin_a
            q_head[unsafe_offset=i + half_dim] = q0 * sin_a + q1 * cos_a

    for h in range(n_kv_heads):
        var k_head = k.unsafe_offset(h * head_dim)
        for i in range(rot_half):
            var freq = pow(inv_theta, (Float32(i * 2) / Float32(head_dim)) * rope_freq_scale)
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var k0 = k_head[unsafe_offset=i]
            var k1 = k_head[unsafe_offset=i + half_dim]
            k_head[unsafe_offset=i] = k0 * cos_a - k1 * sin_a
            k_head[unsafe_offset=i + half_dim] = k0 * sin_a + k1 * cos_a
