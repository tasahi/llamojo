""" Timestral Architecture Configuration and Metadata Parser (`src/llamojo/timestral/config.mojo`).

Represents Timestral-specific hyperparameters, block schedules, and quantile knots
extracted from GGUF v3 metadata.
"""

from std.collections import List
from ..gguf import GGUFContext, GGUFValueType


struct TimestralConfig(Copyable, Movable):
    """Configuration hyperparameters for Mistral Timestral models."""
    var arch: String
    var name: String
    var patch_size: Int
    var d_model: Int
    var num_heads: Int
    var head_dim: Int
    var mlp_hidden_dim: Int
    var num_layers: Int
    var context_length: Int
    var rope_theta: Float32
    var rope_partial_factor: Float32
    var norm_eps: Float32
    var use_asinh: Bool
    var minimum_scale: Float32
    var quantile_width_factor: Float32
    var split_context_horizon_attn: Bool
    var attend_missing_patches: Bool
    var covariate_enabled: Bool
    var block_schedule: List[String]
    var quantile_knots: List[Float32]

    def __init__(out self):
        self.arch = "timestral"
        self.name = "timestral-28m"
        self.patch_size = 32
        self.d_model = 384
        self.num_heads = 6
        self.head_dim = 64
        self.mlp_hidden_dim = 1024
        self.num_layers = 12
        self.context_length = 32768
        self.rope_theta = 10000.0
        self.rope_partial_factor = 0.5
        self.norm_eps = 1e-6
        self.use_asinh = True
        self.minimum_scale = 1e-6
        self.quantile_width_factor = 3.0
        self.split_context_horizon_attn = True
        self.attend_missing_patches = True
        self.covariate_enabled = True
        self.block_schedule = List[String]()
        self.quantile_knots = List[Float32]()
        
        # Populate standard 21 quantile knots
        self.quantile_knots.append(0.01)
        self.quantile_knots.append(0.05)
        self.quantile_knots.append(0.10)
        self.quantile_knots.append(0.15)
        self.quantile_knots.append(0.20)
        self.quantile_knots.append(0.25)
        self.quantile_knots.append(0.30)
        self.quantile_knots.append(0.35)
        self.quantile_knots.append(0.40)
        self.quantile_knots.append(0.45)
        self.quantile_knots.append(0.50)
        self.quantile_knots.append(0.55)
        self.quantile_knots.append(0.60)
        self.quantile_knots.append(0.65)
        self.quantile_knots.append(0.70)
        self.quantile_knots.append(0.75)
        self.quantile_knots.append(0.80)
        self.quantile_knots.append(0.85)
        self.quantile_knots.append(0.90)
        self.quantile_knots.append(0.95)
        self.quantile_knots.append(0.99)

    @staticmethod
    def from_gguf(ctx: GGUFContext) raises -> Self:
        """ Parse Timestral configuration from loaded GGUF context metadata."""
        var cfg = Self()
        cfg.arch = ctx.get_string("general.architecture", "timestral")
        cfg.name = ctx.get_string("general.name", "timestral")
        cfg.patch_size = ctx.get_int("timestral.patch_size", 32)
        cfg.d_model = ctx.get_int("timestral.d_model", 384)
        cfg.num_heads = ctx.get_int("timestral.num_heads", 6)
        if cfg.num_heads > 0:
            cfg.head_dim = cfg.d_model // cfg.num_heads
        else:
            cfg.head_dim = 64
        cfg.mlp_hidden_dim = ctx.get_int("timestral.mlp_hidden_dim", 1024)
        cfg.num_layers = ctx.get_int("timestral.num_layers", 12)
        cfg.context_length = ctx.get_int("timestral.context_length", 32768)
        cfg.rope_theta = ctx.get_float("timestral.rope_theta", 10000.0)
        cfg.rope_partial_factor = ctx.get_float("timestral.rope_partial_factor", 0.5)
        cfg.norm_eps = ctx.get_float("timestral.norm_eps", 1e-6)
        cfg.use_asinh = ctx.get_bool("timestral.use_asinh", True)
        cfg.minimum_scale = ctx.get_float("timestral.minimum_scale", 1e-6)
        cfg.quantile_width_factor = ctx.get_float("timestral.quantile_width_factor", 3.0)
        cfg.split_context_horizon_attn = ctx.get_bool("timestral.split_context_horizon_attn", True)
        cfg.attend_missing_patches = ctx.get_bool("timestral.attend_missing_patches", True)
        cfg.covariate_enabled = ctx.get_bool("timestral.covariate_enabled", True)

        # Parse block_schedule array
        var b_opt = ctx.metadata.find("timestral.block_schedule")
        if b_opt:
            ref val = b_opt.value()
            if val.vtype == GGUFValueType.ARRAY and len(val.str_arr) > 0:
                cfg.block_schedule = List[String]()
                for i in range(len(val.str_arr)):
                    cfg.block_schedule.append(val.str_arr[i])
        if len(cfg.block_schedule) == 0:
            cfg.block_schedule = List[String]()
            for _ in range(cfg.num_layers):
                cfg.block_schedule.append("time")

        # Parse quantile_knots array
        var q_opt = ctx.metadata.find("timestral.quantile_knots")
        if q_opt:
            ref val = q_opt.value()
            if val.vtype == GGUFValueType.ARRAY and len(val.float_arr) > 0:
                cfg.quantile_knots = List[Float32]()
                for i in range(len(val.float_arr)):
                    cfg.quantile_knots.append(val.float_arr[i])

        return cfg^

    def has_variate_attn(self, layer_idx: Int) -> Bool:
        """ Return True if the specified layer block includes cross-variate attention."""
        if layer_idx < 0 or layer_idx >= len(self.block_schedule):
            return False
        return self.block_schedule[layer_idx] == "time_variate"

    def num_quantiles(self) -> Int:
        """ Return number of quantile knots (typically 21)."""
        return len(self.quantile_knots)
