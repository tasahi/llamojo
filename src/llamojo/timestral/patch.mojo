""" Time-Series Patchification, Causal Normalization, and Missingness Processing (`src/llamojo/timestral/patch.mojo`).\n
Implements Phase 2 of the Timestral Engine:
1. `TimeSeriesBatch`: Multi-variate input container holding targets, past covariates,
   and known-future covariates with NaN-aware observation checking.
2. `CausalPatchStats`: Causal per-patch Welford running mean and variance computation.
3. `apply_affine_asinh` and `invert_affine_asinh`: Reversible non-linear squashing.
4. `PatchBuffer`: Left-padding to multiple of 32, packing `[values_32, missing_32]` (64 dims).
5. `forward_patch_proj`: Evaluates the patch projection ResidualBlock (`output(silu(hidden(x))) + residual(x)`).
6. `PatchifyResult` and `patchify_and_embed`: End-to-end data preparation pipeline.
"""

from std.math import asinh, sinh, isnan, isinf, sqrt
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc

from ..types import null_ptr, GGMLType, ggml_type_size, ggml_block_size
from ..backend.isa import SIMD_WIDTH, F32xW
from ..backend.kernels_norm_rope import silu
from ..backend.kernels_quant import BytePtr, FloatPtr, dot_quant_row
from .config import TimestralConfig
from .loader import PatchProjWeights

comptime ROLE_TARGET: Int = 0
comptime ROLE_PAST_COV: Int = 1
comptime ROLE_KNOWN_COV: Int = 2


# ─── Time-Series Input Batch Container ───────────────────────────────────────

struct TimeSeriesBatch(Movable):
    """Container for multi-variate time-series inputs: Targets and Covariates."""
    var batch_size: Int
    var n_targets: Int
    var t_context: Int
    var t_horizon: Int
    var n_past_cov: Int
    var n_known_cov: Int

    var context_ptr: FloatPtr
    var past_cov_ptr: FloatPtr
    var known_cov_ptr: FloatPtr
    var owns_memory: Bool

    def __init__( out self,
            batch_size: Int,
            n_targets: Int,
            t_context: Int,
            t_horizon: Int,
            n_past_cov: Int = 0,
            n_known_cov: Int = 0,
        ):
        """ Allocate an owned TimeSeriesBatch buffer."""
        self.batch_size = batch_size
        self.n_targets = n_targets
        self.t_context = t_context
        self.t_horizon = t_horizon
        self.n_past_cov = n_past_cov
        self.n_known_cov = n_known_cov
        self.owns_memory = True

        var ctx_size = batch_size * n_targets * t_context
        self.context_ptr = unsafe_alloc[Float32](ctx_size)
        unsafe_memset_zero(self.context_ptr, ctx_size)

        if n_past_cov > 0:
            var past_size = batch_size * n_past_cov * t_context
            self.past_cov_ptr = unsafe_alloc[Float32](past_size)
            unsafe_memset_zero(self.past_cov_ptr, past_size)
        else:
            self.past_cov_ptr = null_ptr[Float32]()

        if n_known_cov > 0:
            var known_size = batch_size * n_known_cov * (t_context + t_horizon)
            self.known_cov_ptr = unsafe_alloc[Float32](known_size)
            unsafe_memset_zero(self.known_cov_ptr, known_size)
        else:
            self.known_cov_ptr = null_ptr[Float32]()

    @staticmethod
    def wrap(
            batch_size: Int,
            n_targets: Int,
            t_context: Int,
            t_horizon: Int,
            context_ptr: FloatPtr,
            past_cov_ptr: FloatPtr = null_ptr[Float32](),
            n_past_cov: Int = 0,
            known_cov_ptr: FloatPtr = null_ptr[Float32](),
            n_known_cov: Int = 0,
        ) -> Self:
        """ Create a non-owning wrapper over existing float buffers."""
        var b = Self(
            batch_size=batch_size,
            n_targets=n_targets,
            t_context=t_context,
            t_horizon=t_horizon,
            n_past_cov=0,
            n_known_cov=0,
        )
        # Free default allocated context_ptr and replace with non-owning pointers
        if Int(b.context_ptr) != 0:
            b.context_ptr.unsafe_free()
        b.context_ptr = context_ptr
        b.past_cov_ptr = past_cov_ptr
        b.known_cov_ptr = known_cov_ptr
        b.n_past_cov = n_past_cov
        b.n_known_cov = n_known_cov
        b.owns_memory = False
        return b^

    def free(mut self):
        if self.owns_memory:
            if Int(self.context_ptr) != 0:
                self.context_ptr.unsafe_free()
                self.context_ptr = null_ptr[Float32]()
            if Int(self.past_cov_ptr) != 0:
                self.past_cov_ptr.unsafe_free()
                self.past_cov_ptr = null_ptr[Float32]()
            if Int(self.known_cov_ptr) != 0:
                self.known_cov_ptr.unsafe_free()
                self.known_cov_ptr = null_ptr[Float32]()

    @always_inline
    def is_observed(self, val: Float32) -> Bool:
        """ Check if a float value is valid and observed (not NaN and not Inf)."""
        return not isnan(val) and not isinf(val)

    @always_inline
    def get_target(self, b: Int, v: Int, t: Int) -> Float32:
        var idx = (b * self.n_targets + v) * self.t_context + t
        return self.context_ptr[unsafe_offset=idx]

    @always_inline
    def set_target(mut self, b: Int, v: Int, t: Int, val: Float32):
        var idx = (b * self.n_targets + v) * self.t_context + t
        self.context_ptr[unsafe_offset=idx] = val

    @always_inline
    def get_past_cov(self, b: Int, v: Int, t: Int) -> Float32:
        var idx = (b * self.n_past_cov + v) * self.t_context + t
        return self.past_cov_ptr[unsafe_offset=idx]

    @always_inline
    def set_past_cov(mut self, b: Int, v: Int, t: Int, val: Float32):
        var idx = (b * self.n_past_cov + v) * self.t_context + t
        self.past_cov_ptr[unsafe_offset=idx] = val

    @always_inline
    def get_known_cov(self, b: Int, v: Int, t: Int) -> Float32:
        var idx = (b * self.n_known_cov + v) * (self.t_context + self.t_horizon) + t
        return self.known_cov_ptr[unsafe_offset=idx]

    @always_inline
    def set_known_cov(mut self, b: Int, v: Int, t: Int, val: Float32):
        var idx = (b * self.n_known_cov + v) * (self.t_context + self.t_horizon) + t
        self.known_cov_ptr[unsafe_offset=idx] = val

    def total_rows(self) -> Int:
        """ Total number of variates per series: targets + past_cov + known_cov."""
        return self.n_targets + self.n_past_cov + self.n_known_cov


# ─── Causal Normalization & Reversible asinh Functions ───────────────────────

@always_inline
def apply_affine_asinh(
        x: Float32,
        loc: Float32,
        scale: Float32,
        is_obs: Bool,
        use_asinh: Bool,
    ) -> Float32:
    """ Apply normalized asinh squash: asinh((x - loc) / scale) on observed; 0 on missing."""
    if not is_obs:
        return 0.0
    var s = scale
    if s < 1e-6:
        s = 1e-6
    var z = (x - loc) / s
    if use_asinh:
        return asinh(z)
    return z


@always_inline
def invert_affine_asinh(
        y: Float32,
        loc: Float32,
        scale: Float32,
        use_asinh: Bool,
    ) -> Float32:
    """ Invert normalized prediction back to physical units: loc + scale * sinh(y)."""
    var z: Float32
    if use_asinh:
        z = sinh(y)
    else:
        z = y
    return loc + scale * z


struct CausalPatchStats(Movable):
    """ Stores per-patch running location (mean) and scale (std)."""
    var n_patches: Int
    var loc: FloatPtr
    var scale: FloatPtr

    def __init__(out self, n_patches: Int):
        self.n_patches = n_patches
        self.loc = unsafe_alloc[Float32](n_patches)
        self.scale = unsafe_alloc[Float32](n_patches)
        unsafe_memset_zero(self.loc, n_patches)
        for i in range(n_patches):
            self.scale[unsafe_offset=i] = 1.0

    def free(mut self):
        if Int(self.loc) != 0:
            self.loc.unsafe_free()
            self.loc = null_ptr[Float32]()
        if Int(self.scale) != 0:
            self.scale.unsafe_free()
            self.scale = null_ptr[Float32]()

    def compute(
            mut self,
            series: FloatPtr,
            t_len: Int,
            pad_len: Int,
            patch_size: Int,
            minimum_scale: Float32,
        ):
        """ Compute causal running Welford statistics along time axis matching upstream scaler.py."""
        var w: Float32 = 0.0
        var mean: Float32 = 0.0
        var m2: Float32 = 0.0

        var cur_patch = 0
        var t_total = pad_len + t_len

        for t in range(t_total):
            # Check if this position is within the original series (not padding)
            if t >= pad_len:
                var val = series[unsafe_offset=t - pad_len]
                if not isnan(val) and not isinf(val):
                    w += 1.0
                    var delta = val - mean
                    mean += delta / w
                    m2 += delta * (val - mean)

            # At the boundary of each patch (end of step 31, 63, etc.)
            if (t + 1) % patch_size == 0 and cur_patch < self.n_patches:
                self.loc[unsafe_offset=cur_patch] = mean
                var var_val: Float32 = 0.0
                if w > 1.0:
                    var_val = m2 / (w - 1.0)
                var s = sqrt(var_val)
                if s < minimum_scale:
                    s = minimum_scale
                self.scale[unsafe_offset=cur_patch] = s
                cur_patch += 1


# ─── Patch Buffer & ResidualBlock Projection ─────────────────────────────────

struct PatchBuffer(Movable):
    """ Stores packed length-64 patch vectors [32 values, 32 missingness flags] and patch statistics."""
    var n_rows: Int
    var n_patches: Int
    var patch_size: Int
    var in_dim: Int  # 2 * patch_size = 64
    var packed_data: FloatPtr
    var loc_data: FloatPtr
    var scale_data: FloatPtr

    def __init__(out self, n_rows: Int, n_patches: Int, patch_size: Int = 32):
        self.n_rows = n_rows
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.in_dim = patch_size * 2

        var total_packed = n_rows * n_patches * self.in_dim
        self.packed_data = unsafe_alloc[Float32](total_packed)
        unsafe_memset_zero(self.packed_data, total_packed)

        var total_stats = n_rows * n_patches
        self.loc_data = unsafe_alloc[Float32](total_stats)
        self.scale_data = unsafe_alloc[Float32](total_stats)
        unsafe_memset_zero(self.loc_data, total_stats)
        for i in range(total_stats):
            self.scale_data[unsafe_offset=i] = 1.0

    def free(mut self):
        if Int(self.packed_data) != 0:
            self.packed_data.unsafe_free()
            self.packed_data = null_ptr[Float32]()
        if Int(self.loc_data) != 0:
            self.loc_data.unsafe_free()
            self.loc_data = null_ptr[Float32]()
        if Int(self.scale_data) != 0:
            self.scale_data.unsafe_free()
            self.scale_data = null_ptr[Float32]()

    @always_inline
    def patch_ptr(self, r: Int, p: Int) -> FloatPtr:
        var offset = (r * self.n_patches + p) * self.in_dim
        return self.packed_data.unsafe_offset(offset)

    @always_inline
    def get_loc(self, r: Int, p: Int) -> Float32:
        return self.loc_data[unsafe_offset=r * self.n_patches + p]

    @always_inline
    def get_scale(self, r: Int, p: Int) -> Float32:
        return self.scale_data[unsafe_offset=r * self.n_patches + p]

    @always_inline
    def set_stat(mut self, r: Int, p: Int, loc: Float32, scale: Float32):
        var idx = r * self.n_patches + p
        self.loc_data[unsafe_offset=idx] = loc
        self.scale_data[unsafe_offset=idx] = scale


@always_inline
def _get_patch_row_ptr(weight_base: BytePtr, dtype: UInt32, row_idx: Int, n_cols: Int) -> BytePtr:
    """ Computes the byte offset for a specific matrix row based on quantization block size."""
    var blk_sz = Int(ggml_block_size(dtype))
    var type_sz = Int(ggml_type_size(dtype))
    var n_blocks = (n_cols + blk_sz - 1) // blk_sz
    var row_bytes = n_blocks * type_sz
    return weight_base.unsafe_offset(row_idx * row_bytes)


def forward_patch_proj(
        weights: PatchProjWeights,
        in_vec: FloatPtr,
        out_vec: FloatPtr,
        scratch_hidden: FloatPtr,
        d_model: Int,
        in_dim: Int = 64,
    ):
    """ Execute the patch projection ResidualBlock: output(silu(hidden(x))) + residual(x).\n
    Matches upstream layers.ResidualBlock(input_dims=64, hidden_dims=4*d_model, output_dims=d_model).
    """
    var hidden_dims = 4 * d_model

    # 1. hidden = silu(hidden_layer(x))
    for j in range(hidden_dims):
        var row_ptr = _get_patch_row_ptr(weights.hidden_w, weights.dtype_hidden, j, in_dim)
        var dot_val = dot_quant_row(weights.dtype_hidden, row_ptr, in_vec, in_dim)
        var bias_val: Float32 = 0.0
        if Int(weights.hidden_b) != 0:
            bias_val = weights.hidden_b[unsafe_offset=j]
        scratch_hidden[unsafe_offset=j] = silu(dot_val + bias_val)

    # 2. out = output_layer(hidden)
    for i in range(d_model):
        var row_ptr = _get_patch_row_ptr(weights.output_w, weights.dtype_output, i, hidden_dims)
        var dot_val = dot_quant_row(weights.dtype_output, row_ptr, scratch_hidden, hidden_dims)
        var bias_val: Float32 = 0.0
        if Int(weights.output_b) != 0:
            bias_val = weights.output_b[unsafe_offset=i]
        out_vec[unsafe_offset=i] = dot_val + bias_val

    # 3. res = residual_layer(x) and add to out
    for i in range(d_model):
        var row_ptr = _get_patch_row_ptr(weights.residual_w, weights.dtype_residual, i, in_dim)
        var dot_val = dot_quant_row(weights.dtype_residual, row_ptr, in_vec, in_dim)
        var bias_val: Float32 = 0.0
        if Int(weights.residual_b) != 0:
            bias_val = weights.residual_b[unsafe_offset=i]
        out_vec[unsafe_offset=i] += dot_val + bias_val


# ─── Embedded Tokens Output Container ────────────────────────────────────────

struct EmbeddedTokens(Movable):
    """ Container for embedded representation tokens across all rows, patches, and d_model dimensions."""
    var n_rows: Int
    var n_patches: Int
    var d_model: Int
    var data: FloatPtr

    def __init__(out self, n_rows: Int, n_patches: Int, d_model: Int):
        self.n_rows = n_rows
        self.n_patches = n_patches
        self.d_model = d_model
        var total = n_rows * n_patches * d_model
        self.data = unsafe_alloc[Float32](total)
        unsafe_memset_zero(self.data, total)

    def free(mut self):
        if Int(self.data) != 0:
            self.data.unsafe_free()
            self.data = null_ptr[Float32]()

    @always_inline
    def token_ptr(self, r: Int, p: Int) -> FloatPtr:
        var offset = (r * self.n_patches + p) * self.d_model
        return self.data.unsafe_offset(offset)


struct PatchifyResult(Movable):
    """ Result of patchify_and_embed: embedded tokens and patch normalization buffer."""
    var tokens: EmbeddedTokens
    var patch_buf: PatchBuffer

    def __init__(out self, var tokens: EmbeddedTokens, var patch_buf: PatchBuffer):
        self.tokens = tokens^
        self.patch_buf = patch_buf^

    def free(mut self):
        self.tokens.free()
        self.patch_buf.free()


# ─── End-to-End Patchify & Embed Pipeline ────────────────────────────────────

def patchify_and_embed(
        batch: TimeSeriesBatch,
        config: TimestralConfig,
        patch_proj: PatchProjWeights,
        forecast_query: FloatPtr,
    ) raises -> PatchifyResult:
    """ Execute complete Phase 2 preprocessing:\n
    1. Left-pads history to multiple of 32 steps.
    2. Computes causal Welford statistics (loc, scale) per patch.
    3. Normalizes context with reversible asinh squash.
    4. Packs [values_32, missing_32] 64-dim vectors.
    5. Runs ResidualBlock patch projection for context tokens.
    6. Appends horizon patches (learned forecast_query for targets, projected for known covariates).
    """
    var ps = config.patch_size
    var pad_len = 0
    if batch.t_context % ps != 0:
        pad_len = ps - (batch.t_context % ps)

    var t_padded = pad_len + batch.t_context
    var n_hist_patches = t_padded // ps
    var n_horizon_patches = (batch.t_horizon + ps - 1) // ps
    var n_total_patches = n_hist_patches + n_horizon_patches

    var total_rows = batch.batch_size * batch.total_rows()
    var patch_buf = PatchBuffer(total_rows, n_total_patches, ps)
    var tokens = EmbeddedTokens(total_rows, n_total_patches, config.d_model)

    var scratch_hidden = unsafe_alloc[Float32](4 * config.d_model)

    var cur_row = 0

    # Process each series in the batch
    for b in range(batch.batch_size):
        # 1. Process Targets
        for v in range(batch.n_targets):
            var series_ptr = batch.context_ptr.unsafe_offset((b * batch.n_targets + v) * batch.t_context)

            # A. Compute causal stats
            var stats = CausalPatchStats(n_hist_patches)
            stats.compute(series_ptr, batch.t_context, pad_len, ps, config.minimum_scale)

            # Store stats into patch_buf
            for p in range(n_hist_patches):
                patch_buf.set_stat(cur_row, p, stats.loc[unsafe_offset=p], stats.scale[unsafe_offset=p])

            # For horizon patches of targets, copy the last context patch's loc/scale
            var last_loc: Float32 = 0.0
            var last_scale: Float32 = 1.0
            if n_hist_patches > 0:
                last_loc = stats.loc[unsafe_offset=n_hist_patches - 1]
                last_scale = stats.scale[unsafe_offset=n_hist_patches - 1]
            for p in range(n_hist_patches, n_total_patches):
                patch_buf.set_stat(cur_row, p, last_loc, last_scale)

            # B. Pack historical patches
            for p in range(n_hist_patches):
                var p_vec = patch_buf.patch_ptr(cur_row, p)
                var p_loc = patch_buf.get_loc(cur_row, p)
                var p_scale = patch_buf.get_scale(cur_row, p)

                for k in range(ps):
                    var t_idx = p * ps + k
                    var val: Float32 = 0.0
                    var is_obs = False
                    if t_idx >= pad_len:
                        var raw_val = series_ptr[unsafe_offset=t_idx - pad_len]
                        if not isnan(raw_val) and not isinf(raw_val):
                            val = raw_val
                            is_obs = True

                    # Channel 1: Scaled value (or 0.0)
                    p_vec[unsafe_offset=k] = apply_affine_asinh(val, p_loc, p_scale, is_obs, config.use_asinh)
                    # Channel 2: Missingness indicator (1.0 for missing/pad, 0.0 for observed)
                    p_vec[unsafe_offset=ps + k] = 0.0 if is_obs else 1.0

                # Project context patch -> token
                var out_token = tokens.token_ptr(cur_row, p)
                forward_patch_proj(patch_proj, p_vec, out_token, scratch_hidden, config.d_model, ps * 2)

            # C. Horizon patches: Inject learned forecast_query token for targets
            for p in range(n_hist_patches, n_total_patches):
                var out_token = tokens.token_ptr(cur_row, p)
                unsafe_memcpy(dest=out_token, src=forecast_query, count=config.d_model)

            stats.free()
            cur_row += 1

        # 2. Process Past-Only Covariates
        for v in range(batch.n_past_cov):
            var series_ptr = batch.past_cov_ptr.unsafe_offset((b * batch.n_past_cov + v) * batch.t_context)

            var stats = CausalPatchStats(n_hist_patches)
            stats.compute(series_ptr, batch.t_context, pad_len, ps, config.minimum_scale)

            for p in range(n_hist_patches):
                patch_buf.set_stat(cur_row, p, stats.loc[unsafe_offset=p], stats.scale[unsafe_offset=p])

            var last_loc: Float32 = 0.0
            var last_scale: Float32 = 1.0
            if n_hist_patches > 0:
                last_loc = stats.loc[unsafe_offset=n_hist_patches - 1]
                last_scale = stats.scale[unsafe_offset=n_hist_patches - 1]
            for p in range(n_hist_patches, n_total_patches):
                patch_buf.set_stat(cur_row, p, last_loc, last_scale)

            # Pack context patches
            for p in range(n_hist_patches):
                var p_vec = patch_buf.patch_ptr(cur_row, p)
                var p_loc = patch_buf.get_loc(cur_row, p)
                var p_scale = patch_buf.get_scale(cur_row, p)

                for k in range(ps):
                    var t_idx = p * ps + k
                    var val: Float32 = 0.0
                    var is_obs = False
                    if t_idx >= pad_len:
                        var raw_val = series_ptr[unsafe_offset=t_idx - pad_len]
                        if not isnan(raw_val) and not isinf(raw_val):
                            val = raw_val
                            is_obs = True

                    p_vec[unsafe_offset=k] = apply_affine_asinh(val, p_loc, p_scale, is_obs, config.use_asinh)
                    p_vec[unsafe_offset=ps + k] = 0.0 if is_obs else 1.0

                var out_token = tokens.token_ptr(cur_row, p)
                forward_patch_proj(patch_proj, p_vec, out_token, scratch_hidden, config.d_model, ps * 2)

            # For past covariates in horizon: mask as all-missing
            for p in range(n_hist_patches, n_total_patches):
                var p_vec = patch_buf.patch_ptr(cur_row, p)
                for k in range(ps):
                    p_vec[unsafe_offset=k] = 0.0
                    p_vec[unsafe_offset=ps + k] = 1.0  # missing
                var out_token = tokens.token_ptr(cur_row, p)
                forward_patch_proj(patch_proj, p_vec, out_token, scratch_hidden, config.d_model, ps * 2)

            stats.free()
            cur_row += 1

        # 3. Process Known-Future Covariates (spans T_context + T_horizon)
        for v in range(batch.n_known_cov):
            var total_known_len = batch.t_context + batch.t_horizon
            var series_ptr = batch.known_cov_ptr.unsafe_offset((b * batch.n_known_cov + v) * total_known_len)

            # Global / instance mean & std over observed positions
            var cnt: Float32 = 0.0
            var sum_v: Float32 = 0.0
            for t in range(total_known_len):
                var val = series_ptr[unsafe_offset=t]
                if not isnan(val) and not isinf(val):
                    cnt += 1.0
                    sum_v += val

            var inst_loc: Float32 = 0.0
            if cnt > 0.0:
                inst_loc = sum_v / cnt

            var sum_sq: Float32 = 0.0
            for t in range(total_known_len):
                var val = series_ptr[unsafe_offset=t]
                if not isnan(val) and not isinf(val):
                    var diff = val - inst_loc
                    sum_sq += diff * diff

            var inst_scale = config.minimum_scale
            if cnt > 1.0:
                var v_scale = sqrt(sum_sq / cnt)
                if v_scale > inst_scale:
                    inst_scale = v_scale

            # Set stats for all patches (history + horizon)
            for p in range(n_total_patches):
                patch_buf.set_stat(cur_row, p, inst_loc, inst_scale)

            # Pack all patches (both history and future horizon)
            for p in range(n_total_patches):
                var p_vec = patch_buf.patch_ptr(cur_row, p)

                for k in range(ps):
                    var t_idx = p * ps + k
                    var val: Float32 = 0.0
                    var is_obs = False
                    if t_idx >= pad_len and (t_idx - pad_len) < total_known_len:
                        var raw_val = series_ptr[unsafe_offset=t_idx - pad_len]
                        if not isnan(raw_val) and not isinf(raw_val):
                            val = raw_val
                            is_obs = True

                    p_vec[unsafe_offset=k] = apply_affine_asinh(val, inst_loc, inst_scale, is_obs, config.use_asinh)
                    p_vec[unsafe_offset=ps + k] = 0.0 if is_obs else 1.0

                var out_token = tokens.token_ptr(cur_row, p)
                forward_patch_proj(patch_proj, p_vec, out_token, scratch_hidden, config.d_model, ps * 2)

            cur_row += 1

    scratch_hidden.unsafe_free()
    return PatchifyResult(tokens^, patch_buf^)
