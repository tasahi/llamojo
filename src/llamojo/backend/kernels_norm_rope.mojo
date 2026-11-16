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
from std.memory.alloc import unsafe_alloc
from std.math import sqrt, exp, pow, cos, sin, tanh, log, floor, ceil
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


# ─── Standard LayerNorm (Mean & Variance + Scale & Bias) ─────────────────────

@always_inline
def layer_norm(
        out_ptr: FloatPtr,
        x: FloatPtr,
        weight: FloatPtr,
        bias: FloatPtr,
        n: Int,
        eps: Float32 = 1e-5,
    ):
    """ Standard LayerNorm (ggml_norm + scale + bias, used by Whisper & BERT).
    Computes:
      mean = sum(x) / n
      var  = sum((x - mean)^2) / n
      out[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
    Supports nullable weight (defaults to 1.0) and nullable bias (defaults to 0.0).
    """
    var sum_x: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        sum_x += xv.reduce_add()
        i += SIMD_WIDTH
    while i < n:
        sum_x += x[unsafe_offset=i]
        i += 1

    var mean = sum_x / Float32(n)
    var mean_v = F32xW(mean)

    var sum_sq: Float32 = 0.0
    i = 0
    while i + SIMD_WIDTH <= n:
        var diff = x.unsafe_load[width=SIMD_WIDTH](i) - mean_v
        sum_sq += (diff * diff).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var diff = x[unsafe_offset=i] - mean
        sum_sq += diff * diff
        i += 1

    var inv_std = Float32(1.0) / sqrt(sum_sq / Float32(n) + eps)
    var inv_std_v = F32xW(inv_std)

    var has_w = Int(weight) != 0
    var has_b = Int(bias) != 0

    i = 0
    if has_w and has_b:
        while i + SIMD_WIDTH <= n:
            var xv = x.unsafe_load[width=SIMD_WIDTH](i)
            var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
            var bv = bias.unsafe_load[width=SIMD_WIDTH](i)
            out_ptr.unsafe_store[width=SIMD_WIDTH](i, (xv - mean_v) * inv_std_v * wv + bv)
            i += SIMD_WIDTH
        while i < n:
            out_ptr[unsafe_offset=i] = (x[unsafe_offset=i] - mean) * inv_std * weight[unsafe_offset=i] + bias[unsafe_offset=i]
            i += 1
    elif has_w:
        while i + SIMD_WIDTH <= n:
            var xv = x.unsafe_load[width=SIMD_WIDTH](i)
            var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
            out_ptr.unsafe_store[width=SIMD_WIDTH](i, (xv - mean_v) * inv_std_v * wv)
            i += SIMD_WIDTH
        while i < n:
            out_ptr[unsafe_offset=i] = (x[unsafe_offset=i] - mean) * inv_std * weight[unsafe_offset=i]
            i += 1
    elif has_b:
        while i + SIMD_WIDTH <= n:
            var xv = x.unsafe_load[width=SIMD_WIDTH](i)
            var bv = bias.unsafe_load[width=SIMD_WIDTH](i)
            out_ptr.unsafe_store[width=SIMD_WIDTH](i, (xv - mean_v) * inv_std_v + bv)
            i += SIMD_WIDTH
        while i < n:
            out_ptr[unsafe_offset=i] = (x[unsafe_offset=i] - mean) * inv_std + bias[unsafe_offset=i]
            i += 1
    else:
        while i + SIMD_WIDTH <= n:
            var xv = x.unsafe_load[width=SIMD_WIDTH](i)
            out_ptr.unsafe_store[width=SIMD_WIDTH](i, (xv - mean_v) * inv_std_v)
            i += SIMD_WIDTH
        while i < n:
            out_ptr[unsafe_offset=i] = (x[unsafe_offset=i] - mean) * inv_std
            i += 1


@always_inline
def fused_layer_norm_residual(
        x: FloatPtr,
        residual: FloatPtr,
        weight: FloatPtr,
        bias: FloatPtr,
        out_normed: FloatPtr,
        out_residual: FloatPtr,
        n: Int,
        eps: Float32 = 1e-5,
    ):
    """ Fused residual addition and LayerNorm in minimal memory passes.
    Computes:
      out_residual[i] = x[i] + residual[i]
      out_normed = LayerNorm(out_residual, weight, bias, eps)
    """
    var sum_x: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.unsafe_load[width=SIMD_WIDTH](i)
        var rv = residual.unsafe_load[width=SIMD_WIDTH](i)
        var sum_v = xv + rv
        out_residual.unsafe_store[width=SIMD_WIDTH](i, sum_v)
        sum_x += sum_v.reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var sum_s = x[unsafe_offset=i] + residual[unsafe_offset=i]
        out_residual[unsafe_offset=i] = sum_s
        sum_x += sum_s
        i += 1

    var mean = sum_x / Float32(n)
    var mean_v = F32xW(mean)

    var sum_sq: Float32 = 0.0
    i = 0
    while i + SIMD_WIDTH <= n:
        var diff = out_residual.unsafe_load[width=SIMD_WIDTH](i) - mean_v
        sum_sq += (diff * diff).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var diff = out_residual[unsafe_offset=i] - mean
        sum_sq += diff * diff
        i += 1

    var inv_std = Float32(1.0) / sqrt(sum_sq / Float32(n) + eps)
    var inv_std_v = F32xW(inv_std)

    var has_w = Int(weight) != 0
    var has_b = Int(bias) != 0

    i = 0
    if has_w and has_b:
        while i + SIMD_WIDTH <= n:
            var rv = out_residual.unsafe_load[width=SIMD_WIDTH](i)
            var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
            var bv = bias.unsafe_load[width=SIMD_WIDTH](i)
            out_normed.unsafe_store[width=SIMD_WIDTH](i, (rv - mean_v) * inv_std_v * wv + bv)
            i += SIMD_WIDTH
        while i < n:
            out_normed[unsafe_offset=i] = (out_residual[unsafe_offset=i] - mean) * inv_std * weight[unsafe_offset=i] + bias[unsafe_offset=i]
            i += 1
    elif has_w:
        while i + SIMD_WIDTH <= n:
            var rv = out_residual.unsafe_load[width=SIMD_WIDTH](i)
            var wv = weight.unsafe_load[width=SIMD_WIDTH](i)
            out_normed.unsafe_store[width=SIMD_WIDTH](i, (rv - mean_v) * inv_std_v * wv)
            i += SIMD_WIDTH
        while i < n:
            out_normed[unsafe_offset=i] = (out_residual[unsafe_offset=i] - mean) * inv_std * weight[unsafe_offset=i]
            i += 1
    else:
        while i + SIMD_WIDTH <= n:
            var rv = out_residual.unsafe_load[width=SIMD_WIDTH](i)
            out_normed.unsafe_store[width=SIMD_WIDTH](i, (rv - mean_v) * inv_std_v)
            i += SIMD_WIDTH
        while i < n:
            out_normed[unsafe_offset=i] = (out_residual[unsafe_offset=i] - mean) * inv_std
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


@always_inline
def apply_rope_partial(
        q: FloatPtr,
        k: FloatPtr,
        n_heads: Int,
        n_kv_heads: Int,
        head_dim: Int,
        rope_dim: Int,
        pos: Int,
        rope_theta: Float32 = 10000.0,
    ):
    """ Apply partial RoPE rotation (NeoX style) matching Timestral / SinCosRoPE.
    Only the first `rope_dim` dimensions of each head rotate (pairs (i, i + head_dim//2) for i < rope_dim//2).
    The remaining dimensions are left completely unrotated (identity rotation).
    """
    var inv_theta = Float32(1.0) / rope_theta
    var half_head = head_dim // 2
    var rot_half = (rope_dim // 2) if (rope_dim > 0 and rope_dim < head_dim) else half_head

    for h in range(n_heads):
        var q_head = q.unsafe_offset(h * head_dim)
        for i in range(rot_half):
            var freq = pow(inv_theta, Float32(i * 2) / Float32(rope_dim))
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var q0 = q_head[unsafe_offset=i]
            var q1 = q_head[unsafe_offset=i + half_head]
            q_head[unsafe_offset=i] = q0 * cos_a - q1 * sin_a
            q_head[unsafe_offset=i + half_head] = q0 * sin_a + q1 * cos_a

    for h in range(n_kv_heads):
        var k_head = k.unsafe_offset(h * head_dim)
        for i in range(rot_half):
            var freq = pow(inv_theta, Float32(i * 2) / Float32(rope_dim))
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var k0 = k_head[unsafe_offset=i]
            var k1 = k_head[unsafe_offset=i + half_head]
            k_head[unsafe_offset=i] = k0 * cos_a - k1 * sin_a
            k_head[unsafe_offset=i + half_head] = k0 * sin_a + k1 * cos_a



# ─── YaRN (Yet another RoPE extensioN) Implementation ────────────────────────

@always_inline
def rope_yarn_corr_factor(n_dims: Int, n_ctx_orig: Int, n_rot: Float32, base: Float32) -> Float32:
    """ Compute YaRN correction dimension factor from base wavelength."""
    var pi_2 = Float32(6.283185307179586)
    var num = log(Float32(n_ctx_orig) / (n_rot * pi_2))
    var den = Float32(2.0) * log(base)
    return Float32(n_dims) * (num / den)

@always_inline
def rope_yarn_corr_dims(
        n_dims: Int,
        n_ctx_orig: Int,
        freq_base: Float32,
        beta_fast: Float32,
        beta_slow: Float32,
        mut dims_low: Float32,
        mut dims_high: Float32,
    ):
    """ Compute lower and upper correction dimension bounds [dims_low, dims_high]."""
    var f_low = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)
    var f_high = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)
    dims_low = max(Float32(0.0), floor(f_low))
    dims_high = min(Float32(n_dims - 1), ceil(f_high))

@always_inline
def rope_yarn_ramp(low: Float32, high: Float32, i0: Int) -> Float32:
    """ Compute smooth linear ramp between interpolation and extrapolation."""
    var span = max(Float32(0.001), high - low)
    var y = (Float32(i0) - low) / span
    var clamped = min(Float32(1.0), max(Float32(0.0), y))
    return Float32(1.0) - clamped

@always_inline
def apply_rope_yarn(
    q: FloatPtr,
    k: FloatPtr,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    pos: Int,
    rope_theta: Float32 = 10000.0,
    rope_freq_scale: Float32 = 1.0,
    n_ctx_orig: Int = 16384,
    beta_fast: Float32 = 32.0,
    beta_slow: Float32 = 1.0,
    ext_factor: Float32 = 1.0,
    attn_factor: Float32 = 1.0,
    is_neox: Bool = False,
):
    """ Apply YaRN (Yet another RoPE extensioN) position embedding with frequency blending.

    Partitions the head_dim spectrum into:
    - High frequency (extrapolated / unmodified)
    - Low frequency (interpolated by rope_freq_scale)
    - Mid frequency (smooth ramp blend via beta_fast and beta_slow)
    Scales sin/cos components by mscale = attn_factor * (1.0 + 0.1 * log(scale)).
    """
    var inv_theta = Float32(1.0) / rope_theta
    var half_dim = head_dim // 2

    # Compute correction dimension bounds
    var corr_low: Float32 = 0.0
    var corr_high: Float32 = 0.0
    var orig_ctx = n_ctx_orig if n_ctx_orig > 0 else 16384
    rope_yarn_corr_dims(head_dim, orig_ctx, rope_theta, beta_fast, beta_slow, corr_low, corr_high)

    # Compute attention magnitude scaling factor
    var mscale = attn_factor
    var scale = Float32(1.0) / rope_freq_scale
    if scale > 1.0:
        mscale *= Float32(1.0) + Float32(0.1) * log(scale)

    # Precompute cos and sin tables for all half_dim pairs
    var cos_arr = unsafe_alloc[Float32](half_dim)
    var sin_arr = unsafe_alloc[Float32](half_dim)

    for i in range(half_dim):
        var dim_idx = i * 2
        var freq_extrap = pow(inv_theta, Float32(dim_idx) / Float32(head_dim))
        var freq_interp = freq_extrap * rope_freq_scale
        var freq = freq_interp

        if ext_factor != 0.0:
            var ramp_mix = rope_yarn_ramp(corr_low, corr_high, dim_idx) * ext_factor
            freq = freq_interp * (Float32(1.0) - ramp_mix) + freq_extrap * ramp_mix

        var angle = Float32(pos) * freq
        cos_arr[unsafe_offset=i] = cos(angle) * mscale
        sin_arr[unsafe_offset=i] = sin(angle) * mscale

    # Apply to Query heads
    for h in range(n_heads):
        var q_head = q.unsafe_offset(h * head_dim)
        for i in range(half_dim):
            var cos_a = cos_arr[unsafe_offset=i]
            var sin_a = sin_arr[unsafe_offset=i]
            if is_neox:
                var q0 = q_head[unsafe_offset=i]
                var q1 = q_head[unsafe_offset=i + half_dim]
                q_head[unsafe_offset=i] = q0 * cos_a - q1 * sin_a
                q_head[unsafe_offset=i + half_dim] = q0 * sin_a + q1 * cos_a
            else:
                var q0 = q_head[unsafe_offset=i * 2]
                var q1 = q_head[unsafe_offset=i * 2 + 1]
                q_head[unsafe_offset=i * 2] = q0 * cos_a - q1 * sin_a
                q_head[unsafe_offset=i * 2 + 1] = q0 * sin_a + q1 * cos_a

    # Apply to Key heads
    for h in range(n_kv_heads):
        var k_head = k.unsafe_offset(h * head_dim)
        for i in range(half_dim):
            var cos_a = cos_arr[unsafe_offset=i]
            var sin_a = sin_arr[unsafe_offset=i]
            if is_neox:
                var k0 = k_head[unsafe_offset=i]
                var k1 = k_head[unsafe_offset=i + half_dim]
                k_head[unsafe_offset=i] = k0 * cos_a - k1 * sin_a
                k_head[unsafe_offset=i + half_dim] = k0 * sin_a + k1 * cos_a
            else:
                var k0 = k_head[unsafe_offset=i * 2]
                var k1 = k_head[unsafe_offset=i * 2 + 1]
                k_head[unsafe_offset=i * 2] = k0 * cos_a - k1 * sin_a
                k_head[unsafe_offset=i * 2 + 1] = k0 * sin_a + k1 * cos_a

    cos_arr.unsafe_free()
    sin_arr.unsafe_free()
