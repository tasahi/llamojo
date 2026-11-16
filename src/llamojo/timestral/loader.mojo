""" GGUF Tensor Loader and Weight Structs for Timestral (`src/llamojo/timestral/loader.mojo`).

Maps zero-copy mmap pointers from `GGUFContext` into structured weight accessors for:
1. `PatchProjWeights`: Linear projection of (time_step + missingness_indicator) into `d_model`.
2. `AttentionWeights`: Time-axis and cross-variate Multi-Head Attention weights with SDPA sigmoid gating.
3. `TimestralLayerWeights`: Layer normalization, dual-axis attention, and SwiGLU MLP.
4. `QuantileHeadWeights`: Final MLP projecting hidden states into 21 non-crossing quantiles.
"""

from std.collections import List
from std.memory import Pointer
from ..types import null_ptr, GGMLType
from ..gguf import BytePtr, GGUFContext, GGUFTensorInfo
from .config import TimestralConfig

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]


struct PatchProjWeights(ImplicitlyCopyable, Copyable, Movable):
    """Pointers to the patch projection MLP (hidden, output, and residual layers)."""
    var hidden_w: BytePtr
    var hidden_b: FloatPtr
    var output_w: BytePtr
    var output_b: FloatPtr
    var residual_w: BytePtr
    var residual_b: FloatPtr
    var dtype_hidden: UInt32
    var dtype_output: UInt32
    var dtype_residual: UInt32

    def __init__(out self):
        var z: Int = 0
        self.hidden_w = BytePtr(unsafe_from_address=z)
        self.hidden_b = FloatPtr(unsafe_from_address=z)
        self.output_w = BytePtr(unsafe_from_address=z)
        self.output_b = FloatPtr(unsafe_from_address=z)
        self.residual_w = BytePtr(unsafe_from_address=z)
        self.residual_b = FloatPtr(unsafe_from_address=z)
        self.dtype_hidden = GGMLType.F32
        self.dtype_output = GGMLType.F32
        self.dtype_residual = GGMLType.F32


struct AttentionWeights(ImplicitlyCopyable, Copyable, Movable):
    """Pointers to Attention weights (RMSNorm, fused QKV, SDPA sigmoid Gate, and Output)."""
    var norm_w: FloatPtr
    var qkv_w: BytePtr
    var gate_w: BytePtr
    var o_w: BytePtr
    var dtype_qkv: UInt32
    var dtype_gate: UInt32
    var dtype_o: UInt32

    def __init__(out self):
        var z: Int = 0
        self.norm_w = FloatPtr(unsafe_from_address=z)
        self.qkv_w = BytePtr(unsafe_from_address=z)
        self.gate_w = BytePtr(unsafe_from_address=z)
        self.o_w = BytePtr(unsafe_from_address=z)
        self.dtype_qkv = GGMLType.F32
        self.dtype_gate = GGMLType.F32
        self.dtype_o = GGMLType.F32


struct TimestralLayerWeights(ImplicitlyCopyable, Copyable, Movable):
    """ Pointers to all weights in a single Timestral transformer block."""
    var time_attn: AttentionWeights
    var variate_attn: AttentionWeights
    var has_variate_attn: Bool

    var ffn_norm_w: FloatPtr
    var ffn_gate_up_w: BytePtr  # w12 (fused gate + up)
    var ffn_down_w: BytePtr     # w3 (down projection)
    var dtype_gate_up: UInt32
    var dtype_down: UInt32

    def __init__(out self):
        var z: Int = 0
        self.time_attn = AttentionWeights()
        self.variate_attn = AttentionWeights()
        self.has_variate_attn = False
        self.ffn_norm_w = FloatPtr(unsafe_from_address=z)
        self.ffn_gate_up_w = BytePtr(unsafe_from_address=z)
        self.ffn_down_w = BytePtr(unsafe_from_address=z)
        self.dtype_gate_up = GGMLType.F32
        self.dtype_down = GGMLType.F32


struct QuantileHeadWeights(ImplicitlyCopyable, Copyable, Movable):
    """ Pointers to the quantile projection head MLP (hidden, output, residual)."""
    var hidden_w: BytePtr
    var output_w: BytePtr
    var residual_w: BytePtr
    var dtype_hidden: UInt32
    var dtype_output: UInt32
    var dtype_residual: UInt32

    def __init__(out self):
        var z: Int = 0
        self.hidden_w = BytePtr(unsafe_from_address=z)
        self.output_w = BytePtr(unsafe_from_address=z)
        self.residual_w = BytePtr(unsafe_from_address=z)
        self.dtype_hidden = GGMLType.F32
        self.dtype_output = GGMLType.F32
        self.dtype_residual = GGMLType.F32


struct TimestralModelWeights(Movable):
    """ Encapsulates all bound tensor weights for a loaded Timestral model."""
    var patch_proj: PatchProjWeights
    var forecast_query: FloatPtr
    var layers: List[TimestralLayerWeights]
    var out_norm_w: FloatPtr
    var head: QuantileHeadWeights
    var total_bytes: UInt64
    var n_tensors: Int

    def __init__(out self):
        var z: Int = 0
        self.patch_proj = PatchProjWeights()
        self.forecast_query = FloatPtr(unsafe_from_address=z)
        self.layers = List[TimestralLayerWeights]()
        self.out_norm_w = FloatPtr(unsafe_from_address=z)
        self.head = QuantileHeadWeights()
        self.total_bytes = 0
        self.n_tensors = 0


def _get_tensor_or_fail(ctx: GGUFContext, name: String) raises -> GGUFTensorInfo:
    """ Retrieve tensor info by name or raise descriptive error."""
    if not ctx.has_tensor(name):
        raise Error("Missing required Timestral tensor in GGUF: " + name)
    return ctx.get_tensor(name)


def _as_float_ptr(t: GGUFTensorInfo) -> FloatPtr:
    return t.data_ptr.unsafe_bitcast[Float32]()


def load_timestral_weights(ctx: GGUFContext, config: TimestralConfig) raises -> TimestralModelWeights:
    """ Extract and validate all required Timestral tensor pointers from GGUFContext."""
    var weights = TimestralModelWeights()
    weights.total_bytes = ctx.file_size
    weights.n_tensors = len(ctx.tensors)

    # 1. Patch Projection Weights
    var pp_hw = _get_tensor_or_fail(ctx, "timestral.patch_proj.hidden.weight")
    var pp_hb = _get_tensor_or_fail(ctx, "timestral.patch_proj.hidden.bias")
    var pp_ow = _get_tensor_or_fail(ctx, "timestral.patch_proj.output.weight")
    var pp_ob = _get_tensor_or_fail(ctx, "timestral.patch_proj.output.bias")
    var pp_rw = _get_tensor_or_fail(ctx, "timestral.patch_proj.residual.weight")
    var pp_rb = _get_tensor_or_fail(ctx, "timestral.patch_proj.residual.bias")

    weights.patch_proj.hidden_w = pp_hw.data_ptr
    weights.patch_proj.hidden_b = _as_float_ptr(pp_hb)
    weights.patch_proj.output_w = pp_ow.data_ptr
    weights.patch_proj.output_b = _as_float_ptr(pp_ob)
    weights.patch_proj.residual_w = pp_rw.data_ptr
    weights.patch_proj.residual_b = _as_float_ptr(pp_rb)
    weights.patch_proj.dtype_hidden = pp_hw.dtype
    weights.patch_proj.dtype_output = pp_ow.dtype
    weights.patch_proj.dtype_residual = pp_rw.dtype

    # 2. Horizon Query Token
    var fq = _get_tensor_or_fail(ctx, "timestral.forecast_query")
    weights.forecast_query = _as_float_ptr(fq)

    # 3. Transformer Layers
    for i in range(config.num_layers):
        var lw = TimestralLayerWeights()
        var prefix = "blk." + String(i) + "."

        # Time Attention
        var ta_norm = _get_tensor_or_fail(ctx, prefix + "time_attn_norm.weight")
        var ta_gate = _get_tensor_or_fail(ctx, prefix + "time_attn_gate.weight")
        var ta_qkv = _get_tensor_or_fail(ctx, prefix + "time_attn_qkv.weight")
        var ta_o = _get_tensor_or_fail(ctx, prefix + "time_attn_o.weight")

        lw.time_attn.norm_w = _as_float_ptr(ta_norm)
        lw.time_attn.gate_w = ta_gate.data_ptr
        lw.time_attn.qkv_w = ta_qkv.data_ptr
        lw.time_attn.o_w = ta_o.data_ptr
        lw.time_attn.dtype_gate = ta_gate.dtype
        lw.time_attn.dtype_qkv = ta_qkv.dtype
        lw.time_attn.dtype_o = ta_o.dtype

        # Variate Attention (if configured in block_schedule)
        lw.has_variate_attn = config.has_variate_attn(i)
        if lw.has_variate_attn:
            var va_norm = _get_tensor_or_fail(ctx, prefix + "variate_attn_norm.weight")
            var va_gate = _get_tensor_or_fail(ctx, prefix + "variate_attn_gate.weight")
            var va_qkv = _get_tensor_or_fail(ctx, prefix + "variate_attn_qkv.weight")
            var va_o = _get_tensor_or_fail(ctx, prefix + "variate_attn_o.weight")

            lw.variate_attn.norm_w = _as_float_ptr(va_norm)
            lw.variate_attn.gate_w = va_gate.data_ptr
            lw.variate_attn.qkv_w = va_qkv.data_ptr
            lw.variate_attn.o_w = va_o.data_ptr
            lw.variate_attn.dtype_gate = va_gate.dtype
            lw.variate_attn.dtype_qkv = va_qkv.dtype
            lw.variate_attn.dtype_o = va_o.dtype

        # MLP / Feed-Forward
        var ffn_norm = _get_tensor_or_fail(ctx, prefix + "ffn_norm.weight")
        var ffn_gate_up = _get_tensor_or_fail(ctx, prefix + "ffn_gate_up.weight")
        var ffn_down = _get_tensor_or_fail(ctx, prefix + "ffn_down.weight")

        lw.ffn_norm_w = _as_float_ptr(ffn_norm)
        lw.ffn_gate_up_w = ffn_gate_up.data_ptr
        lw.ffn_down_w = ffn_down.data_ptr
        lw.dtype_gate_up = ffn_gate_up.dtype
        lw.dtype_down = ffn_down.dtype

        weights.layers.append(lw^)

    # 4. Final Output Norm
    var out_norm = _get_tensor_or_fail(ctx, "timestral.out_norm.weight")
    weights.out_norm_w = _as_float_ptr(out_norm)

    # 5. Quantile Projection Head
    var head_h = _get_tensor_or_fail(ctx, "timestral.head.hidden.weight")
    var head_o = _get_tensor_or_fail(ctx, "timestral.head.output.weight")
    var head_r = _get_tensor_or_fail(ctx, "timestral.head.residual.weight")

    weights.head.hidden_w = head_h.data_ptr
    weights.head.output_w = head_o.data_ptr
    weights.head.residual_w = head_r.data_ptr
    weights.head.dtype_hidden = head_h.dtype
    weights.head.dtype_output = head_o.dtype
    weights.head.dtype_residual = head_r.dtype

    return weights^
