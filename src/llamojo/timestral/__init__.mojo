""" Pure Mojo Timestral Engine for llamojo (`src/llamojo/timestral/__init__.mojo`).

Public exports for Timestral time-series foundation model support:
- `TimestralConfig`: Configuration hyperparameters and metadata parser.
- `TimestralModelWeights`: Zero-copy GGUF tensor weight accessors.
- `load_timestral_weights`: Tensor loader and binding validator.
- `ROLE_TARGET`, `ROLE_PAST_COV`, `ROLE_KNOWN_COV`: Covariate structural roles.
- `TimeSeriesBatch`: Multi-variate input batch container.
- `CausalPatchStats`: Causal per-patch Welford statistics.
- `PatchBuffer`: Packed length-64 patch representations.
- `EmbeddedTokens`: Projected patch tokens.
- `apply_affine_asinh`, `invert_affine_asinh`: Reversible non-linear scaling.
- `forward_patch_proj`: Patch projection ResidualBlock evaluation.
- `patchify_and_embed`: End-to-end preprocessing pipeline.
"""

from .config import TimestralConfig
from .loader import (
    PatchProjWeights,
    AttentionWeights,
    TimestralLayerWeights,
    QuantileHeadWeights,
    TimestralModelWeights,
    load_timestral_weights,
)
from .patch import (
    ROLE_TARGET,
    ROLE_PAST_COV,
    ROLE_KNOWN_COV,
    TimeSeriesBatch,
    CausalPatchStats,
    PatchBuffer,
    EmbeddedTokens,
    PatchifyResult,
    apply_affine_asinh,
    invert_affine_asinh,
    forward_patch_proj,
    patchify_and_embed,
)
from .model import (
    TimestralState,
    forward_time_attention,
    forward_variate_attention,
    forward_swiglu_mlp,
    forward_timestral_block,
    monotone_quantiles,
    forward_quantile_head,
    forward_timestral,
)
from .covariates import (
    TargetEncodingResult,
    target_encode,
)
from .forecast import (
    ForecastResult,
    TimestralForecaster,
)
