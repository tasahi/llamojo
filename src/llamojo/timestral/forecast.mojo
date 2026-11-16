""" High-Level Forecasting & Imputation API for Timestral (`src/llamojo/timestral/forecast.mojo`).¦n
Implements Phase 4 of the Timestral Engine:
1. `ForecastResult`: Ergonomic output container holding the (B, V, H, 21) quantile array,
   with convenient `.median()` access and arbitrary `.quantile(q)` lookup/interpolation.
2. `TimestralForecaster`: End-to-end model handle loaded from GGUF v3 file, holding configuration,
   bound model weights, and pre-allocated zero-allocation inference states.
3. `.forecast(...)`: Accepts raw continuous targets, optional past covariates, and known-future covariates.
4. `.impute(...)`: Fills missing values (NaNs) in time-series using causal iterative forecasting.
"""

from std.math import nan, isnan, isinf
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.collections import List

from ..types import null_ptr
from ..gguf import GGUFContext, gguf_load_from_file
from ..backend.kernels_quant import FloatPtr
from .config import TimestralConfig
from .loader import TimestralModelWeights, load_timestral_weights
from .patch import (
    TimeSeriesBatch,
    PatchBuffer,
    EmbeddedTokens,
    PatchifyResult,
    patchify_and_embed,
)
from .model import TimestralState, forward_timestral


# ─── Forecast Result Container ────────────────────────────────────────────────

struct ForecastResult(Movable):
    """ Container for predicted non-crossing quantiles with dimension (batch_size, n_targets, horizon, 21)."""
    var batch_size: Int
    var n_targets: Int
    var horizon: Int
    var n_quantiles: Int
    var quantile_knots: List[Float32]
    var data: FloatPtr
    var owns_memory: Bool

    def __init__(
            out self,
            batch_size: Int,
            n_targets: Int,
            horizon: Int,
            config: TimestralConfig,
        ):
        self.batch_size = batch_size
        self.n_targets = n_targets
        self.horizon = horizon
        self.n_quantiles = config.num_quantiles()
        self.quantile_knots = List[Float32]()
        for i in range(len(config.quantile_knots)):
            self.quantile_knots.append(config.quantile_knots[i])

        var total = batch_size * n_targets * horizon * self.n_quantiles
        self.data = unsafe_alloc[Float32](total)
        unsafe_memset_zero(self.data, total)
        self.owns_memory = True

    def free(mut self):
        if self.owns_memory and Int(self.data) != 0:
            self.data.unsafe_free()
            self.data = null_ptr[Float32]()

    @always_inline
    def get(self, b: Int, v: Int, h: Int, q_idx: Int) -> Float32:
        """ Direct access to quantile value at knot index `q_idx`."""
        var idx = ((b * self.n_targets + v) * self.horizon + h) * self.n_quantiles + q_idx
        return self.data[unsafe_offset=idx]

    @always_inline
    def set(mut self, b: Int, v: Int, h: Int, q_idx: Int, val: Float32):
        var idx = ((b * self.n_targets + v) * self.horizon + h) * self.n_quantiles + q_idx
        self.data[unsafe_offset=idx] = val

    @always_inline
    def median(self, b: Int, v: Int, h: Int) -> Float32:
        """ Returns the median prediction (q = 0.50, knot index 10)."""
        return self.get(b, v, h, 10)

    def quantile(self, b: Int, v: Int, h: Int, q: Float32) -> Float32:
        """ Returns the predicted value for a target probability level `q` (e.g. 0.10, 0.50, 0.90).
        If `q` matches a knot exactly, returns it; otherwise linearly interpolates between adjacent knots.
        """
        if q <= self.quantile_knots[0]:
            return self.get(b, v, h, 0)
        var last_idx = len(self.quantile_knots) - 1
        if q >= self.quantile_knots[last_idx]:
            return self.get(b, v, h, last_idx)

        # Find interval [knot_i, knot_{i+1}]
        for i in range(last_idx):
            var k0 = self.quantile_knots[i]
            var k1 = self.quantile_knots[i + 1]
            if q >= k0 and q <= k1:
                var v0 = self.get(b, v, h, i)
                var v1 = self.get(b, v, h, i + 1)
                var t = (q - k0) / (k1 - k0)
                return v0 + t * (v1 - v0)

        return self.median(b, v, h)

    def copy_median_to(self, b: Int, v: Int, dest: FloatPtr):
        """ Copies the median forecast slice across horizon steps into destination pointer."""
        for h in range(self.horizon):
            dest[unsafe_offset=h] = self.median(b, v, h)


# ─── Timestral Forecaster Model Handle ────────────────────────────────────────

struct TimestralForecaster(Movable):
    """ High-level forecaster handle for Mistral Timestral models."""
    var ctx: GGUFContext
    var config: TimestralConfig
    var weights: TimestralModelWeights
    var state: TimestralState

    def __init__(out self, path: String, max_rows: Int = 16, max_patches: Int = 128) raises:
        """ Load Timestral model from GGUF file path and pre-allocate inference state."""
        self.ctx = gguf_load_from_file(path, use_mmap=True)
        self.config = TimestralConfig.from_gguf(self.ctx)
        self.weights = load_timestral_weights(self.ctx, self.config)
        self.state = TimestralState(self.config, max_rows=max_rows, max_patches=max_patches)

    @staticmethod
    def from_gguf(path: String, max_rows: Int = 16, max_patches: Int = 128) raises -> Self:
        return Self(path, max_rows=max_rows, max_patches=max_patches)

    def free(mut self):
        self.state.free()

    def forecast_batch(mut self, mut batch: TimeSeriesBatch) raises -> ForecastResult:
        """ Run forecast on a prepared `TimeSeriesBatch` container."""
        var ps = self.config.patch_size
        var t_context = batch.t_context
        var t_horizon = batch.t_horizon

        # Compute number of patches
        var n_hist_patches = (t_context + ps - 1) // ps
        var n_horizon_patches = (t_horizon + ps - 1) // ps
        var n_total_patches = n_hist_patches + n_horizon_patches

        # Re-allocate state if dimensions exceed current scratchpad
        var total_rows = batch.n_targets + batch.n_past_cov + batch.n_known_cov
        if total_rows > self.state.max_rows or n_total_patches > self.state.max_patches:
            self.state.free()
            var new_rows = total_rows if total_rows > self.state.max_rows else self.state.max_rows
            var new_patches = n_total_patches if n_total_patches > self.state.max_patches else self.state.max_patches
            self.state = TimestralState(self.config, max_rows=new_rows, max_patches=new_patches)

        # 1. Patchify, causally normalize, and embed tokens
        var res = patchify_and_embed(batch, self.config, self.weights.patch_proj, self.weights.forecast_query)
        ref tokens = res.tokens
        ref patch_buf = res.patch_buf

        # 2. Allocate output predictions for full patch blocks
        var total_pred_floats = total_rows * n_horizon_patches * ps * self.config.num_quantiles()
        var full_patch_preds = unsafe_alloc[Float32](total_pred_floats)
        unsafe_memset_zero(full_patch_preds, total_pred_floats)

        # 3. Transformer forward pass & quantile head evaluation
        forward_timestral(
            weights=self.weights,
            tokens=tokens,
            patch_buf=patch_buf,
            n_hist_patches=n_hist_patches,
            n_horizon_patches=n_horizon_patches,
            config=self.config,
            out_predictions=full_patch_preds,
            state=self.state,
        )

        # 4. Copy target predictions into ForecastResult sliced to exact `t_horizon`
        var result = ForecastResult(
            batch_size=batch.batch_size,
            n_targets=batch.n_targets,
            horizon=t_horizon,
            config=self.config,
        )

        var n_q = self.config.num_quantiles()
        for b in range(batch.batch_size):
            for v in range(batch.n_targets):
                for h in range(t_horizon):
                    for q in range(n_q):
                        # Row in full_patch_preds is `v` (target rows come first)
                        var src_idx = (v * n_horizon_patches * ps + h) * n_q + q
                        var val = full_patch_preds[unsafe_offset=src_idx]
                        result.set(b, v, h, q, val)

        full_patch_preds.unsafe_free()
        res.free()
        return result^

    def forecast(
            mut self,
            context: FloatPtr,
            t_context: Int,
            prediction_length: Int,
            batch_size: Int = 1,
            n_targets: Int = 1,
            past_covariates: FloatPtr = null_ptr[Float32](),
            n_past_cov: Int = 0,
            known_covariates: FloatPtr = null_ptr[Float32](),
            n_known_cov: Int = 0,
        ) raises -> ForecastResult:
        """ Zero-shot probabilistic forecast from raw float pointer arrays.
        Matches the ergonomics described in `timestral_about.md`:
          forecast = forecaster.forecast(context, prediction_length=32)
        """
        var batch = TimeSeriesBatch.wrap(
            batch_size=batch_size,
            n_targets=n_targets,
            t_context=t_context,
            t_horizon=prediction_length,
            context_ptr=context,
            past_cov_ptr=past_covariates,
            n_past_cov=n_past_cov,
            known_cov_ptr=known_covariates,
            n_known_cov=n_known_cov,
        )
        return self.forecast_batch(batch)

    def impute(
            mut self,
            data: FloatPtr,
            t_total: Int,
            batch_size: Int = 1,
            n_targets: Int = 1,
        ) raises:
        """ In-place causal imputation of missing values (NaN) across a continuous time-series.
        Implements Strategy A from `timestral_about.md`:
        Detects contiguous gaps of missing values, forecasts the missing segment from
        the preceding clean historical context, and fills the gap with the predicted median.
        """
        for b in range(batch_size):
            for v in range(n_targets):
                var series_offset = (b * n_targets + v) * t_total
                var series_ptr = data.unsafe_offset(series_offset)

                var t = 0
                while t < t_total:
                    var val = series_ptr[unsafe_offset=t]
                    if isnan(val) or isinf(val):
                        # Found start of missing gap
                        var gap_start = t
                        while t < t_total and (isnan(series_ptr[unsafe_offset=t]) or isinf(series_ptr[unsafe_offset=t])):
                            t += 1
                        var gap_end = t
                        var gap_len = gap_end - gap_start

                        if gap_start > 0:
                            # We have prior historical context [0..gap_start)
                            # Run forecast for prediction_length = gap_len
                            var forecast_res = self.forecast(
                                context=series_ptr,
                                t_context=gap_start,
                                prediction_length=gap_len,
                                batch_size=1,
                                n_targets=1,
                            )
                            # Fill gap with predicted median values
                            for h in range(gap_len):
                                series_ptr[unsafe_offset=gap_start + h] = forecast_res.median(0, 0, h)
                            forecast_res.free()
                        else:
                            # Leading gap without prior context: forward fill with 0.0 or first observed
                            # Find next observed value
                            var next_obs: Float32 = 0.0
                            if gap_end < t_total and not isnan(series_ptr[unsafe_offset=gap_end]):
                                next_obs = series_ptr[unsafe_offset=gap_end]
                            for h in range(gap_len):
                                series_ptr[unsafe_offset=gap_start + h] = next_obs
                    else:
                        t += 1
