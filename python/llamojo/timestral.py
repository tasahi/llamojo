""" High-level Python wrapper for the llamojo Timestral engine (`libllamojo.so`).

Supports:
- Loading Timestral GGUF models directly in Python
- Forecasting multi-variate continuous telemetry with 21 non-crossing quantiles
- Passing past and known-future covariates
- Causal missing-data imputation (`.impute(...)`)
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Union

from ._lib import get_lib

try:
    import numpy as np
except ImportError:
    np = None


def target_encode(
        target_context: np.ndarray,
        past_codes: np.ndarray,
        future_codes: Optional[np.ndarray] = None,
        smooth: float = 1.0,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """ Per-series smoothed mean-target encoding.\n
    Matches upstream `timestral.covariate_encoding.target_encode`:
        (smooth * mean_i + sum_{i,c}) / (smooth + count_{i,c})
    Missing values (NaN) form their own category slot.
    Unseen categories in `future_codes` resolve to the series mean.

    Args:
        target_context: (T,) target values. Non-finite entries are left out of the means.
        past_codes: (T,) category codes aligned with `target_context`; NaN marks missing.
        future_codes: Optional (H,) codes over the horizon (known covariates).
        smooth: Weight of the series mean in each category mean (default 1.0).

    Returns:
        (encoded_past, encoded_future) float32 arrays; encoded_future is None when future_codes is None.
    """
    if np is None:
        raise ImportError("numpy is required for target encoding.")

    target = np.asarray(target_context, dtype=np.float64)
    past_codes = np.asarray(past_codes, dtype=np.float64)
    assert target.shape == past_codes.shape, f"target {target.shape} and codes {past_codes.shape} must align"

    observed = np.isfinite(target)
    target = np.where(observed, target, 0.0)
    item_count = float(observed.sum())
    item_mean = float(target.sum() / item_count) if item_count > 0 else 0.0

    categories = np.unique(past_codes[np.isfinite(past_codes)])
    nan_slot = len(categories)

    def _slots(codes: np.ndarray, cats: np.ndarray, n_slot: int, unseen: Optional[int] = None) -> np.ndarray:
        idx = np.searchsorted(cats, np.nan_to_num(codes, nan=0.0))
        idx = np.minimum(idx, max(len(cats) - 1, 0))
        known = np.isfinite(codes) & (len(cats) > 0)
        if len(cats) > 0:
            known &= cats[idx] == codes
        return np.where(np.isnan(codes), n_slot, np.where(known, idx, unseen if unseen is not None else 0)).astype(np.intp)

    past_slots = _slots(past_codes, categories, nan_slot)
    sums = np.bincount(past_slots, weights=target * observed, minlength=nan_slot + 1)
    counts = np.bincount(past_slots, weights=observed.astype(np.float64), minlength=nan_slot + 1)
    lookup = np.append((smooth * item_mean + sums) / (smooth + counts), item_mean)

    encoded_past = lookup[past_slots].astype(np.float32)
    if future_codes is None:
        return encoded_past, None

    future_slots = _slots(np.asarray(future_codes, dtype=np.float64), categories, nan_slot, unseen=nan_slot + 1)
    return encoded_past, lookup[future_slots].astype(np.float32)


class Forecast:
    """ Probabilistic forecast output containing 21 quantiles."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p, shape: tuple[int, int, int, int]):
        self._lib = lib
        self._handle = handle
        self.shape = shape  # (batch_size, n_targets, horizon, 21)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.llamojo_timestral_forecast_free(self._handle)
            self._handle = None

    @property
    def quantiles(self):
        """Returns the full (Batch, Targets, Horizon, 21) numpy array view."""
        if np is None:
            raise ImportError("numpy is required to extract quantile arrays.")
        ptr = self._lib.llamojo_timestral_forecast_get_quantiles(self._handle)
        total_size = 1
        for dim in self.shape:
            total_size *= dim
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(total_size,))
        return arr.reshape(self.shape)

    @property
    def median(self):
        """Returns the median prediction (q = 0.50, knot 10) of shape (Batch, Targets, Horizon)."""
        return self.quantiles[..., 10]

    def quantile(self, q: float):
        """Returns or linearly interpolates the prediction at probability quantile level `q`."""
        b, v, h, _ = self.shape
        out = np.empty((b, v, h), dtype=np.float32)
        for bi in range(b):
            for vi in range(v):
                for hi in range(h):
                    out[bi, vi, hi] = self._lib.llamojo_timestral_forecast_quantile(
                        self._handle, bi, vi, hi, ctypes.c_float(q)
                    )
        return out


class Timestral:
    """High-level Python wrapper for the llamojo Timestral time-series foundation model."""

    def __init__(self, model_path: Union[str, os.PathLike], max_rows: int = 16, max_patches: int = 128):
        self._lib = get_lib()
        self._setup_c_signatures()

        path_str = str(model_path).encode("utf-8")
        self._handle = self._lib.llamojo_timestral_load(path_str, ctypes.c_int32(max_rows), ctypes.c_int32(max_patches))
        if not self._handle:
            raise RuntimeError(f"Failed to load Timestral model from {model_path}")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.llamojo_timestral_free(self._handle)
            self._handle = None

    def _setup_c_signatures(self):
        self._lib.llamojo_timestral_load.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
        self._lib.llamojo_timestral_load.restype = ctypes.c_void_p

        self._lib.llamojo_timestral_free.argtypes = [ctypes.c_void_p]
        self._lib.llamojo_timestral_free.restype = None

        self._lib.llamojo_timestral_forecast.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ]
        self._lib.llamojo_timestral_forecast.restype = ctypes.c_void_p

        self._lib.llamojo_timestral_forecast_free.argtypes = [ctypes.c_void_p]
        self._lib.llamojo_timestral_forecast_free.restype = None

        self._lib.llamojo_timestral_forecast_get_quantiles.argtypes = [ctypes.c_void_p]
        self._lib.llamojo_timestral_forecast_get_quantiles.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.llamojo_timestral_forecast_median.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
        ]
        self._lib.llamojo_timestral_forecast_median.restype = ctypes.c_float

        self._lib.llamojo_timestral_forecast_quantile.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float
        ]
        self._lib.llamojo_timestral_forecast_quantile.restype = ctypes.c_float

        self._lib.llamojo_timestral_impute.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
        ]
        self._lib.llamojo_timestral_impute.restype = ctypes.c_int32

    def forecast( self,
            context: np.ndarray,
            prediction_length: int,
            past_covariates: Optional[np.ndarray] = None,
            known_covariates: Optional[np.ndarray] = None,
            categorical_past_covariates: Optional[np.ndarray] = None,
            categorical_known_covariates: Optional[np.ndarray] = None,
            smooth: float = 1.0,
        ) -> Forecast:
        """ Run probabilistic zero-shot forecast on context array.\n
        Supports continuous targets, numeric past/known covariates, and automatic
        smoothed target encoding for categorical covariates.
        """
        if np is None:
            raise ImportError("numpy is required for timestral forecasting.")

        context = np.ascontiguousarray(context, dtype=np.float32)
        if context.ndim == 1:
            context = context.reshape(1, 1, -1)
        elif context.ndim == 2:
            context = context.reshape(1, context.shape[0], context.shape[1])

        batch_size, n_targets, t_context = context.shape

        # Handle categorical past covariates via target_encode
        if categorical_past_covariates is not None:
            cat_past = np.asarray(categorical_past_covariates, dtype=np.float32)
            if cat_past.ndim == 1:
                cat_past = cat_past.reshape(1, 1, -1)
            elif cat_past.ndim == 2:
                cat_past = cat_past.reshape(1, cat_past.shape[0], cat_past.shape[1])

            n_cat_past = cat_past.shape[1]
            encoded_list = []
            for b in range(batch_size):
                b_enc = []
                # Use target mean from target channel 0 for encoding covariates
                tgt_ctx = context[b, 0]
                for c in range(n_cat_past):
                    enc, _ = target_encode(tgt_ctx, cat_past[b, c], smooth=smooth)
                    b_enc.append(enc)
                encoded_list.append(b_enc)
            encoded_past_arr = np.array(encoded_list, dtype=np.float32)
            if past_covariates is not None:
                past_covariates = np.concatenate([past_covariates, encoded_past_arr], axis=1)
            else:
                past_covariates = encoded_past_arr

        # Handle categorical known covariates via target_encode
        if categorical_known_covariates is not None:
            cat_known = np.asarray(categorical_known_covariates, dtype=np.float32)
            if cat_known.ndim == 1:
                cat_known = cat_known.reshape(1, 1, -1)
            elif cat_known.ndim == 2:
                cat_known = cat_known.reshape(1, cat_known.shape[0], cat_known.shape[1])

            n_cat_known = cat_known.shape[1]
            encoded_known_list = []
            for b in range(batch_size):
                b_enc = []
                tgt_ctx = context[b, 0]
                for c in range(n_cat_known):
                    p_codes = cat_known[b, c, :t_context]
                    f_codes = cat_known[b, c, t_context : t_context + prediction_length]
                    enc_p, enc_f = target_encode(tgt_ctx, p_codes, f_codes, smooth=smooth)
                    b_enc.append(np.concatenate([enc_p, enc_f]))
                encoded_known_list.append(b_enc)
            encoded_known_arr = np.array(encoded_known_list, dtype=np.float32)
            if known_covariates is not None:
                known_covariates = np.concatenate([known_covariates, encoded_known_arr], axis=1)
            else:
                known_covariates = encoded_known_arr

        c_ptr = context.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        p_ptr = None
        n_past = 0
        if past_covariates is not None:
            past_covariates = np.ascontiguousarray(past_covariates, dtype=np.float32)
            if past_covariates.ndim == 2:
                past_covariates = past_covariates.reshape(1, past_covariates.shape[0], past_covariates.shape[1])
            n_past = past_covariates.shape[1]
            p_ptr = past_covariates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        k_ptr = None
        n_known = 0
        if known_covariates is not None:
            known_covariates = np.ascontiguousarray(known_covariates, dtype=np.float32)
            if known_covariates.ndim == 2:
                known_covariates = known_covariates.reshape(1, known_covariates.shape[0], known_covariates.shape[1])
            n_known = known_covariates.shape[1]
            k_ptr = known_covariates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        f_handle = self._lib.llamojo_timestral_forecast(
            self._handle,
            c_ptr,
            ctypes.c_int32(t_context),
            ctypes.c_int32(prediction_length),
            ctypes.c_int32(batch_size),
            ctypes.c_int32(n_targets),
            p_ptr,
            ctypes.c_int32(n_past),
            k_ptr,
            ctypes.c_int32(n_known),
        )
        if not f_handle:
            raise RuntimeError("Forecast failed in llamojo C API")

        return Forecast(self._lib, f_handle, (batch_size, n_targets, prediction_length, 21))

    def impute(self, data: np.ndarray) -> np.ndarray:
        """In-place causal imputation of NaN gaps. Returns mutated array."""
        if np is None:
            raise ImportError("numpy is required for imputation.")

        data = np.ascontiguousarray(data, dtype=np.float32)
        orig_shape = data.shape
        if data.ndim == 1:
            data_3d = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            data_3d = data.reshape(1, data.shape[0], data.shape[1])
        else:
            data_3d = data

        b, v, t = data_3d.shape
        d_ptr = data_3d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.llamojo_timestral_impute(
            self._handle, d_ptr, ctypes.c_int32(t), ctypes.c_int32(b), ctypes.c_int32(v)
        )
        if ret != 0:
            raise RuntimeError("Imputation failed in llamojo C API")
        return data.reshape(orig_shape)
