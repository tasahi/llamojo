""" Timestral Transformer Model Architecture, Dual-Axis Attention, and Quantile Head (`src/llamojo/timestral/model.mojo`).\n
Implements Phase 3 of the Timestral Engine:
1. `TimestralState`: Pre-allocated activation workspace arena for multi-patch multi-variate execution.
2. `forward_time_attention`: Multi-head time attention with partial RoPE, DHF causal attention mask,
   sdpa_sigmoid gating, and output projection.
3. `forward_variate_attention`: Cross-variate multi-head attention across series rows, sdpa_sigmoid gating,
   and output projection.
4. `forward_timestral_block`: Transformer layer evaluation (pre-RMSNorm + time attention + optional variate attention + SwiGLU MLP).
5. `monotone_quantiles`: Non-crossing quantile cumsum from median channel out to 21 quantiles.
6. `forward_quantile_head`: Projects horizon patch tokens to 21 quantiles and inverts affine asinh scaling.
7. `forward_timestral`: End-to-end forward pass evaluating all transformer blocks and decoding forecasted quantiles.
"""

from std.math import sqrt, exp, log
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc

from ..types import null_ptr, GGMLType, ggml_type_size, ggml_block_size
from ..backend.isa import SIMD_WIDTH, F32xW
from ..backend.kernels_norm_rope import rms_norm, silu, fused_silu_mul, apply_rope_partial
from ..backend.kernels_quant import BytePtr, FloatPtr, dot_quant_row
from .config import TimestralConfig
from .loader import AttentionWeights, TimestralLayerWeights, QuantileHeadWeights, TimestralModelWeights
from .patch import PatchBuffer, EmbeddedTokens, invert_affine_asinh


# ─── Activation Helpers ───────────────────────────────────────────────────────

@always_inline
def _sigmoid(x: Float32) -> Float32:
    if x >= 20.0:
        return 1.0
    if x <= -20.0:
        return 0.0
    return 1.0 / (1.0 + exp(-x))


@always_inline
def _softplus(x: Float32) -> Float32:
    if x > 20.0:
        return x
    if x < -20.0:
        return exp(x)
    return log(1.0 + exp(x))


@always_inline
def _get_row_ptr(weight_base: BytePtr, dtype: UInt32, row_idx: Int, n_cols: Int) -> BytePtr:
    """ Computes the byte offset for a specific matrix row based on quantization block size."""
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var n_blocks = (n_cols + blk_sz - 1) // blk_sz
    var row_bytes = n_blocks * type_sz
    return weight_base.unsafe_offset(row_idx * row_bytes)


# ─── Timestral Activation Scratchpad / State Arena ────────────────────────────

struct TimestralState(Movable):
    """ Pre-allocated scratchpad buffers for evaluating Timestral transformer blocks."""
    var max_rows: Int
    var max_patches: Int
    var d_model: Int
    var num_heads: Int
    var head_dim: Int
    var mlp_hidden_dim: Int

    var norm_x: FloatPtr       # (max_rows * max_patches, d_model)
    var qkv_buf: FloatPtr      # (3 * d_model)
    var gate_buf: FloatPtr     # (d_model)
    var attn_out: FloatPtr     # (d_model)
    var attn_scores: FloatPtr  # (max_patches) or (max_rows)
    var mlp_gate_up: FloatPtr  # (2 * mlp_hidden_dim)
    var mlp_act: FloatPtr      # (mlp_hidden_dim)
    var mlp_out: FloatPtr      # (d_model)
    var head_scratch: FloatPtr # (4 * d_model)

    def __init__(out self, config: TimestralConfig, max_rows: Int = 16, max_patches: Int = 128):
        self.max_rows = max_rows
        self.max_patches = max_patches
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.mlp_hidden_dim = config.mlp_hidden_dim

        var total_tokens = max_rows * max_patches
        self.norm_x = unsafe_alloc[Float32](total_tokens * self.d_model)
        self.qkv_buf = unsafe_alloc[Float32](3 * self.d_model)
        self.gate_buf = unsafe_alloc[Float32](self.d_model)
        self.attn_out = unsafe_alloc[Float32](self.d_model)

        var max_dim = max_patches if max_patches > max_rows else max_rows
        self.attn_scores = unsafe_alloc[Float32](max_dim)

        self.mlp_gate_up = unsafe_alloc[Float32](2 * self.mlp_hidden_dim)
        self.mlp_act = unsafe_alloc[Float32](self.mlp_hidden_dim)
        self.mlp_out = unsafe_alloc[Float32](self.d_model)
        self.head_scratch = unsafe_alloc[Float32](4 * self.d_model)

    def free(mut self):
        if Int(self.norm_x) != 0:
            self.norm_x.unsafe_free()
            self.norm_x = null_ptr[Float32]()
        if Int(self.qkv_buf) != 0:
            self.qkv_buf.unsafe_free()
            self.qkv_buf = null_ptr[Float32]()
        if Int(self.gate_buf) != 0:
            self.gate_buf.unsafe_free()
            self.gate_buf = null_ptr[Float32]()
        if Int(self.attn_out) != 0:
            self.attn_out.unsafe_free()
            self.attn_out = null_ptr[Float32]()
        if Int(self.attn_scores) != 0:
            self.attn_scores.unsafe_free()
            self.attn_scores = null_ptr[Float32]()
        if Int(self.mlp_gate_up) != 0:
            self.mlp_gate_up.unsafe_free()
            self.mlp_gate_up = null_ptr[Float32]()
        if Int(self.mlp_act) != 0:
            self.mlp_act.unsafe_free()
            self.mlp_act = null_ptr[Float32]()
        if Int(self.mlp_out) != 0:
            self.mlp_out.unsafe_free()
            self.mlp_out = null_ptr[Float32]()
        if Int(self.head_scratch) != 0:
            self.head_scratch.unsafe_free()
            self.head_scratch = null_ptr[Float32]()


# ─── Softmax Helper ──────────────────────────────────────────────────────────

@always_inline
def _softmax(scores: FloatPtr, n: Int):
    """ In-place numerically stable softmax over scores[0..n-1]."""
    if n <= 0:
        return
    var max_val = scores[unsafe_offset=0]
    for i in range(1, n):
        var v = scores[unsafe_offset=i]
        if v > max_val:
            max_val = v

    var sum_exp: Float32 = 0.0
    for i in range(n):
        var ev = exp(scores[unsafe_offset=i] - max_val)
        scores[unsafe_offset=i] = ev
        sum_exp += ev

    var inv_sum: Float32 = 1.0 / sum_exp if sum_exp > 0.0 else 0.0
    for i in range(n):
        scores[unsafe_offset=i] *= inv_sum


# ─── Time-Axis Multi-Head Attention (with partial RoPE & DHF Mask) ────────────

def forward_time_attention(
        attn: AttentionWeights,
        tokens_in: FloatPtr,
        tokens_out: FloatPtr,
        n_rows: Int,
        n_patches: Int,
        n_hist_patches: Int,
        config: TimestralConfig,
        mut state: TimestralState,
    ):
    """ Evaluates Multi-Head Time Attention along the patch sequence for each row independently.
    Applies:
      1. Pre-norm using attn.norm_w
      2. wQKV projection with (3 * d_model) layout
      3. Partial RoPE on first (head_dim * rope_partial_factor) elements
      4. DHF Mask:
         - Context queries (p < n_hist_patches) attend to keys <= p
         - Horizon queries (p >= n_hist_patches) attend to all context keys (0..n_hist_patches-1)
           and causally to horizon keys (n_hist_patches..p)
      5. SDPA Sigmoid gate: out = out * sigmoid(W_gate @ x_normed)
      6. Output projection W_o
      7. Residual connection: tokens_out[r, p] += projected_out
    """
    var d_model = config.d_model
    var num_heads = config.num_heads
    var head_dim = config.head_dim
    var rope_dim = config.rope_dim()
    var inv_scale = Float32(1.0) / sqrt(Float32(head_dim))
    var has_gate = Int(attn.gate_w) != 0

    var q_all = unsafe_alloc[Float32](n_patches * d_model)
    var k_all = unsafe_alloc[Float32](n_patches * d_model)
    var v_all = unsafe_alloc[Float32](n_patches * d_model)
    var row_norm = unsafe_alloc[Float32](n_patches * d_model)

    for r in range(n_rows):
        var row_offset = r * n_patches * d_model

        # 1. Compute Pre-RMSNorm and Q, K, V for all patches in this row
        for p in range(n_patches):
            var x_ptr = tokens_in.unsafe_offset(row_offset + p * d_model)
            var norm_ptr = row_norm.unsafe_offset(p * d_model)
            rms_norm(norm_ptr, x_ptr, attn.norm_w, d_model, config.norm_eps)

            # Compute QKV: wQKV shape is (3 * d_model, d_model)
            for row_qkv in range(3 * d_model):
                var rptr = _get_row_ptr(attn.qkv_w, attn.dtype_qkv, row_qkv, d_model)
                var dot_v = dot_quant_row(attn.dtype_qkv, rptr, norm_ptr, d_model)
                state.qkv_buf[unsafe_offset=row_qkv] = dot_v

            # Map from wQKV layout: row = (chunk * head_dim + d) * num_heads + h
            for h in range(num_heads):
                for d in range(head_dim):
                    var row_q = (0 * head_dim + d) * num_heads + h
                    var row_k = (1 * head_dim + d) * num_heads + h
                    var row_v = (2 * head_dim + d) * num_heads + h

                    var head_idx = (p * num_heads + h) * head_dim + d
                    q_all[unsafe_offset=head_idx] = state.qkv_buf[unsafe_offset=row_q]
                    k_all[unsafe_offset=head_idx] = state.qkv_buf[unsafe_offset=row_k]
                    v_all[unsafe_offset=head_idx] = state.qkv_buf[unsafe_offset=row_v]

            # Apply partial RoPE to Q and K for this patch position `p`
            for h in range(num_heads):
                var q_head = q_all.unsafe_offset((p * num_heads + h) * head_dim)
                var k_head = k_all.unsafe_offset((p * num_heads + h) * head_dim)
                apply_rope_partial(
                    q=q_head,
                    k=k_head,
                    n_heads=1,
                    n_kv_heads=1,
                    head_dim=head_dim,
                    rope_dim=rope_dim,
                    pos=p,
                    rope_theta=config.rope_theta,
                )

        # 2. Scaled Dot-Product Attention for each query patch `qp`
        for qp in range(n_patches):
            var norm_ptr = row_norm.unsafe_offset(qp * d_model)

            # If sdpa_sigmoid gate is present, compute sigmoid(W_gate @ norm_ptr)
            if has_gate:
                for row_g in range(d_model):
                    var rptr = _get_row_ptr(attn.gate_w, attn.dtype_gate, row_g, d_model)
                    var dot_g = dot_quant_row(attn.dtype_gate, rptr, norm_ptr, d_model)
                    state.gate_buf[unsafe_offset=row_g] = _sigmoid(dot_g)

            # Determine valid key range under DHF rule:
            # Context queries (qp < n_hist_patches): attend to keys kp <= qp
            # Horizon queries (qp >= n_hist_patches): attend to all context keys (0..n_hist_patches-1)
            # and causally to horizon keys (n_hist_patches..qp)
            var max_key = qp + 1

            for h in range(num_heads):
                var q_head = q_all.unsafe_offset((qp * num_heads + h) * head_dim)
                var out_head = state.attn_out.unsafe_offset(h * head_dim)

                # Compute attention scores for allowed keys
                var score_idx = 0
                for kp in range(max_key):
                    var k_head = k_all.unsafe_offset((kp * num_heads + h) * head_dim)
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        score += q_head[unsafe_offset=d] * k_head[unsafe_offset=d]
                    state.attn_scores[unsafe_offset=score_idx] = score * inv_scale
                    score_idx += 1

                # Softmax over active scores
                _softmax(state.attn_scores, score_idx)

                # Weighted sum over values
                for d in range(head_dim):
                    out_head[unsafe_offset=d] = 0.0

                for kp in range(score_idx):
                    var weight = state.attn_scores[unsafe_offset=kp]
                    var v_head = v_all.unsafe_offset((kp * num_heads + h) * head_dim)
                    for d in range(head_dim):
                        out_head[unsafe_offset=d] += weight * v_head[unsafe_offset=d]

            # If gated, multiply element-wise by sigmoid(gate)
            if has_gate:
                for d in range(d_model):
                    state.attn_out[unsafe_offset=d] *= state.gate_buf[unsafe_offset=d]

            # Output projection W_o: out = W_o @ attn_out
            var out_token = tokens_out.unsafe_offset(row_offset + qp * d_model)
            for row_o in range(d_model):
                var rptr = _get_row_ptr(attn.o_w, attn.dtype_o, row_o, d_model)
                var dot_o = dot_quant_row(attn.dtype_o, rptr, state.attn_out, d_model)
                out_token[unsafe_offset=row_o] += dot_o

    q_all.unsafe_free()
    k_all.unsafe_free()
    v_all.unsafe_free()
    row_norm.unsafe_free()


# ─── Variate-Axis Multi-Head Attention (Cross-Variate) ────────────────────────

def forward_variate_attention(
        attn: AttentionWeights,
        tokens_in: FloatPtr,
        tokens_out: FloatPtr,
        n_rows: Int,
        n_patches: Int,
        config: TimestralConfig,
        mut state: TimestralState,
    ):
    """ Evaluates Cross-Variate Multi-Head Attention across rows for each patch independently.
    Permutes logical shape from (R, N, D) to (N, R, D), runs unmasked self-attention across R,
    applies sdpa_sigmoid gating, and adds residual output.
    """
    var d_model = config.d_model
    var num_heads = config.num_heads
    var head_dim = config.head_dim
    var inv_scale = Float32(1.0) / sqrt(Float32(head_dim))
    var has_gate = Int(attn.gate_w) != 0

    var q_all = unsafe_alloc[Float32](n_rows * d_model)
    var k_all = unsafe_alloc[Float32](n_rows * d_model)
    var v_all = unsafe_alloc[Float32](n_rows * d_model)
    var patch_norm = unsafe_alloc[Float32](n_rows * d_model)

    for p in range(n_patches):
        # 1. Compute Pre-RMSNorm and Q, K, V for all rows at this patch `p`
        for r in range(n_rows):
            var x_ptr = tokens_in.unsafe_offset(r * n_patches * d_model + p * d_model)
            var norm_ptr = patch_norm.unsafe_offset(r * d_model)
            rms_norm(norm_ptr, x_ptr, attn.norm_w, d_model, config.norm_eps)

            for row_qkv in range(3 * d_model):
                var rptr = _get_row_ptr(attn.qkv_w, attn.dtype_qkv, row_qkv, d_model)
                var dot_v = dot_quant_row(attn.dtype_qkv, rptr, norm_ptr, d_model)
                state.qkv_buf[unsafe_offset=row_qkv] = dot_v

            for h in range(num_heads):
                for d in range(head_dim):
                    var row_q = (0 * head_dim + d) * num_heads + h
                    var row_k = (1 * head_dim + d) * num_heads + h
                    var row_v = (2 * head_dim + d) * num_heads + h

                    var head_idx = (r * num_heads + h) * head_dim + d
                    q_all[unsafe_offset=head_idx] = state.qkv_buf[unsafe_offset=row_q]
                    k_all[unsafe_offset=head_idx] = state.qkv_buf[unsafe_offset=row_k]
                    v_all[unsafe_offset=head_idx] = state.qkv_buf[unsafe_offset=row_v]

        # 2. Attention across rows for each query row `qr`
        for qr in range(n_rows):
            var norm_ptr = patch_norm.unsafe_offset(qr * d_model)
            if has_gate:
                for row_g in range(d_model):
                    var rptr = _get_row_ptr(attn.gate_w, attn.dtype_gate, row_g, d_model)
                    var dot_g = dot_quant_row(attn.dtype_gate, rptr, norm_ptr, d_model)
                    state.gate_buf[unsafe_offset=row_g] = _sigmoid(dot_g)

            for h in range(num_heads):
                var q_head = q_all.unsafe_offset((qr * num_heads + h) * head_dim)
                var out_head = state.attn_out.unsafe_offset(h * head_dim)

                for kr in range(n_rows):
                    var k_head = k_all.unsafe_offset((kr * num_heads + h) * head_dim)
                    var score: Float32 = 0.0
                    for d in range(head_dim):
                        score += q_head[unsafe_offset=d] * k_head[unsafe_offset=d]
                    state.attn_scores[unsafe_offset=kr] = score * inv_scale

                _softmax(state.attn_scores, n_rows)

                for d in range(head_dim):
                    out_head[unsafe_offset=d] = 0.0

                for kr in range(n_rows):
                    var weight = state.attn_scores[unsafe_offset=kr]
                    var v_head = v_all.unsafe_offset((kr * num_heads + h) * head_dim)
                    for d in range(head_dim):
                        out_head[unsafe_offset=d] += weight * v_head[unsafe_offset=d]

            if has_gate:
                for d in range(d_model):
                    state.attn_out[unsafe_offset=d] *= state.gate_buf[unsafe_offset=d]

            var out_token = tokens_out.unsafe_offset(qr * n_patches * d_model + p * d_model)
            for row_o in range(d_model):
                var rptr = _get_row_ptr(attn.o_w, attn.dtype_o, row_o, d_model)
                var dot_o = dot_quant_row(attn.dtype_o, rptr, state.attn_out, d_model)
                out_token[unsafe_offset=row_o] += dot_o

    q_all.unsafe_free()
    k_all.unsafe_free()
    v_all.unsafe_free()
    patch_norm.unsafe_free()


# ─── SwiGLU MLP Block ────────────────────────────────────────────────────────

def forward_swiglu_mlp(
        lw: TimestralLayerWeights,
        tokens_in: FloatPtr,
        tokens_out: FloatPtr,
        n_tokens: Int,
        config: TimestralConfig,
        mut state: TimestralState,
    ):
    """ Evaluates SwiGLU MLP for each token:
      1. Pre-norm using lw.ffn_norm_w
      2. Fused gate & up projection via lw.ffn_gate_up_w (size 2 * mlp_hidden_dim)
      3. Activation: act = gate * silu(up)
      4. Down projection: out = w3 @ act
      5. Residual add: tokens_out += out
    """
    var d_model = config.d_model
    var hidden_dim = config.mlp_hidden_dim

    for t in range(n_tokens):
        var x_ptr = tokens_in.unsafe_offset(t * d_model)
        var norm_ptr = state.norm_x.unsafe_offset(t * d_model)
        rms_norm(norm_ptr, x_ptr, lw.ffn_norm_w, d_model, config.norm_eps)

        # 2. Gate + Up: shape (2 * hidden_dim, d_model)
        for r in range(2 * hidden_dim):
            var rptr = _get_row_ptr(lw.ffn_gate_up_w, lw.dtype_gate_up, r, d_model)
            var dot_val = dot_quant_row(lw.dtype_gate_up, rptr, norm_ptr, d_model)
            state.mlp_gate_up[unsafe_offset=r] = dot_val

        # 3. Activation: gate = [0..hidden_dim-1], up = [hidden_dim..2*hidden_dim-1]
        for i in range(hidden_dim):
            var gate_val = state.mlp_gate_up[unsafe_offset=i]
            var up_val = state.mlp_gate_up[unsafe_offset=hidden_dim + i]
            state.mlp_act[unsafe_offset=i] = gate_val * silu(up_val)

        # 4. Down: shape (d_model, hidden_dim)
        var out_ptr = tokens_out.unsafe_offset(t * d_model)
        for r in range(d_model):
            var rptr = _get_row_ptr(lw.ffn_down_w, lw.dtype_down, r, hidden_dim)
            var dot_down = dot_quant_row(lw.dtype_down, rptr, state.mlp_act, hidden_dim)
            out_ptr[unsafe_offset=r] += dot_down


# ─── Full Timestral Transformer Block ────────────────────────────────────────

def forward_timestral_block(
        layer_idx: Int,
        lw: TimestralLayerWeights,
        tokens: EmbeddedTokens,
        n_hist_patches: Int,
        config: TimestralConfig,
        mut state: TimestralState,
    ):
    """ Evaluates a single transformer block in-place on `tokens`:
      1. Time attention + residual
      2. If configured as 'time_variate', variate attention + residual
      3. SwiGLU MLP + residual
    """
    var n_rows = tokens.n_rows
    var n_patches = tokens.n_patches
    var n_tokens = n_rows * n_patches

    # 1. Time attention
    forward_time_attention(
        attn=lw.time_attn,
        tokens_in=tokens.data,
        tokens_out=tokens.data,
        n_rows=n_rows,
        n_patches=n_patches,
        n_hist_patches=n_hist_patches,
        config=config,
        state=state,
    )

    # 2. Variate attention (if scheduled for this layer)
    if lw.has_variate_attn:
        forward_variate_attention(
            attn=lw.variate_attn,
            tokens_in=tokens.data,
            tokens_out=tokens.data,
            n_rows=n_rows,
            n_patches=n_patches,
            config=config,
            state=state,
        )

    # 3. SwiGLU MLP
    forward_swiglu_mlp(
        lw=lw,
        tokens_in=tokens.data,
        tokens_out=tokens.data,
        n_tokens=n_tokens,
        config=config,
        state=state,
    )


# ─── Quantile Head Projection & Monotonic Cumsum ──────────────────────────────

def monotone_quantiles(raw: FloatPtr, out_q: FloatPtr, width_factor: Float32 = 3.0):
    """ Transforms 21 raw predictions into monotonically non-decreasing quantiles.
    Canonical 21 knots centered at median knot index 10 (q = 0.50):
      out[10] = median
      out[i] = out[i-1] + width_factor * delta_p * softplus(raw[i]) for i in 11..20
      out[i] = out[i+1] - width_factor * delta_p * softplus(raw[i]) for i in 9..0
    """
    var m = 10  # index for 0.50 median

    var inc = unsafe_alloc[Float32](21)
    for i in range(21):
        var w: Float32 = 0.05
        if i == 0 or i == 20:
            w = 0.04
        elif i == 10:
            w = 0.0
        inc[unsafe_offset=i] = width_factor * w * _softplus(raw[unsafe_offset=i])

    out_q[unsafe_offset=m] = raw[unsafe_offset=m]

    # Upper branch: fan out upwards
    for i in range(m + 1, 21):
        out_q[unsafe_offset=i] = out_q[unsafe_offset=i - 1] + inc[unsafe_offset=i]

    # Lower branch: fan out downwards
    for i in range(m - 1, -1, -1):
        out_q[unsafe_offset=i] = out_q[unsafe_offset=i + 1] - inc[unsafe_offset=i]

    inc.unsafe_free()


def forward_quantile_head(
        head: QuantileHeadWeights,
        out_norm_w: FloatPtr,
        tokens: EmbeddedTokens,
        patch_buf: PatchBuffer,
        n_hist_patches: Int,
        n_horizon_patches: Int,
        config: TimestralConfig,
        out_predictions: FloatPtr,
        mut state: TimestralState,
    ):
    """ Evaluates the Quantile Projection Head on horizon tokens and inverts affine asinh scaling.
    Args:
      head: Weights of quantile projection ResidualBlock
      out_norm_w: Weights of final timestral.out_norm
      tokens: Embedded tokens from the transformer stack
      patch_buf: Container with causal running loc and scale statistics
      n_hist_patches: Number of historical context patches
      n_horizon_patches: Number of forecasted horizon patches
      config: Timestral configuration
      out_predictions: Destination float pointer of shape (n_rows, n_horizon_patches * 32, 21)
      state: Pre-allocated scratchpad state
    """
    var d_model = config.d_model
    var ps = config.patch_size
    var n_quantiles = config.num_quantiles()
    var head_hidden = 4 * d_model
    var head_out_dim = ps * n_quantiles  # 32 * 21 = 672

    var raw_buf = unsafe_alloc[Float32](head_out_dim)
    var norm_token = unsafe_alloc[Float32](d_model)

    for r in range(tokens.n_rows):
        # Anchor loc and scale are taken from the last historical patch
        var anchor_loc = patch_buf.get_loc(r, n_hist_patches - 1)
        var anchor_scale = patch_buf.get_scale(r, n_hist_patches - 1)

        for hp in range(n_horizon_patches):
            var patch_idx = n_hist_patches + hp
            var token_ptr = tokens.token_ptr(r, patch_idx)

            # 1. Final RMSNorm
            rms_norm(norm_token, token_ptr, out_norm_w, d_model, config.norm_eps)

            # 2. ResidualBlock: out = output(silu(hidden(x))) + residual(x)
            # Hidden layer: (head_hidden, d_model)
            for i in range(head_hidden):
                var rptr = _get_row_ptr(head.hidden_w, head.dtype_hidden, i, d_model)
                var dot_h = dot_quant_row(head.dtype_hidden, rptr, norm_token, d_model)
                state.head_scratch[unsafe_offset=i] = silu(dot_h)

            # Output layer: (head_out_dim, head_hidden) + Residual layer: (head_out_dim, d_model)
            for i in range(head_out_dim):
                var rptr_o = _get_row_ptr(head.output_w, head.dtype_output, i, head_hidden)
                var rptr_r = _get_row_ptr(head.residual_w, head.dtype_residual, i, d_model)
                var dot_o = dot_quant_row(head.dtype_output, rptr_o, state.head_scratch, head_hidden)
                var dot_r = dot_quant_row(head.dtype_residual, rptr_r, norm_token, d_model)
                raw_buf[unsafe_offset=i] = dot_o + dot_r

            # 3. Apply monotone_quantiles and invert affine asinh scaling for each step in this patch
            for step in range(ps):
                var raw_step_ptr = raw_buf.unsafe_offset(step * n_quantiles)
                var out_step_ptr = out_predictions.unsafe_offset(
                    (r * n_horizon_patches * ps + hp * ps + step) * n_quantiles
                )

                # Monotone fan out in normalized space
                monotone_quantiles(raw_step_ptr, out_step_ptr, config.quantile_width_factor)

                # Invert affine asinh scaling into original physical measurement units
                for q in range(n_quantiles):
                    var scaled_val = out_step_ptr[unsafe_offset=q]
                    out_step_ptr[unsafe_offset=q] = invert_affine_asinh(
                        y=scaled_val,
                        loc=anchor_loc,
                        scale=anchor_scale,
                        use_asinh=config.use_asinh,
                    )

    raw_buf.unsafe_free()
    norm_token.unsafe_free()


# ─── End-to-End Timestral Forward Pass ───────────────────────────────────────

def forward_timestral(
        weights: TimestralModelWeights,
        mut tokens: EmbeddedTokens,
        patch_buf: PatchBuffer,
        n_hist_patches: Int,
        n_horizon_patches: Int,
        config: TimestralConfig,
        out_predictions: FloatPtr,
        mut state: TimestralState,
    ):
    """ Evaluates the complete multi-layer Timestral transformer and quantile head.
    Tokens are mutated in-place through all layers in `weights.layers`, then passed through
    `forward_quantile_head` to write denormalized physical quantiles into `out_predictions`.
    """
    for layer_idx in range(len(weights.layers)):
        forward_timestral_block(
            layer_idx=layer_idx,
            lw=weights.layers[layer_idx],
            tokens=tokens,
            n_hist_patches=n_hist_patches,
            config=config,
            state=state,
        )

    forward_quantile_head(
        head=weights.head,
        out_norm_w=weights.out_norm_w,
        tokens=tokens,
        patch_buf=patch_buf,
        n_hist_patches=n_hist_patches,
        n_horizon_patches=n_horizon_patches,
        config=config,
        out_predictions=out_predictions,
        state=state,
    )
