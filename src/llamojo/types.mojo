""" Core llama.h and GGUF type definitions for llamojo.

Ported from MojoLlama's `gguf/types.mojo` and `kernels/engine_v3.mojo` to Mojo 1.1.0,
with corrected upstream `ggml.h` quantization enum IDs and byte-per-block sizes,
plus `llama.h`-compatible parameter structs and opaque pointer handles.
"""

from std.memory import Pointer

@always_inline
def null_ptr[T: AnyType]() -> Pointer[T, MutUntrackedOrigin]:
    """Return a zero-initialized untracked pointer (for C ABI / uninitialized handles)."""
    var z: Int = 0
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=z)

# ─── GGUF Constants ──────────────────────────────────────────────────────────

comptime GGUF_MAGIC: UInt32 = 0x46554747  # "GGUF" in little-endian
comptime GGUF_VERSION_V2: UInt32 = 2
comptime GGUF_VERSION: UInt32 = 3
comptime GGUF_DEFAULT_ALIGNMENT: UInt64 = 32

# ─── GGUF Metadata Value Types ───────────────────────────────────────────────

struct GGUFValueType:
    comptime UINT8: UInt32 = 0
    comptime INT8: UInt32 = 1
    comptime UINT16: UInt32 = 2
    comptime INT16: UInt32 = 3
    comptime UINT32: UInt32 = 4
    comptime INT32: UInt32 = 5
    comptime FLOAT32: UInt32 = 6
    comptime BOOL: UInt32 = 7
    comptime STRING: UInt32 = 8
    comptime ARRAY: UInt32 = 9
    comptime UINT64: UInt32 = 10
    comptime INT64: UInt32 = 11
    comptime FLOAT64: UInt32 = 12

# ─── GGML Tensor Data Types (matching upstream ggml.h enum ggml_type) ────────

struct GGMLType:
    comptime F32: UInt32 = 0
    comptime F16: UInt32 = 1
    comptime Q4_0: UInt32 = 2
    comptime Q4_1: UInt32 = 3
    comptime Q5_0: UInt32 = 6
    comptime Q5_1: UInt32 = 7
    comptime Q8_0: UInt32 = 8
    comptime Q8_1: UInt32 = 9
    comptime Q2_K: UInt32 = 10
    comptime Q3_K: UInt32 = 11
    comptime Q4_K: UInt32 = 12
    comptime Q5_K: UInt32 = 13
    comptime Q6_K: UInt32 = 14
    comptime Q8_K: UInt32 = 15
    comptime IQ2_XXS: UInt32 = 16
    comptime IQ2_XS: UInt32 = 17
    comptime IQ3_XXS: UInt32 = 18
    comptime IQ1_S: UInt32 = 19
    comptime IQ4_NL: UInt32 = 20
    comptime IQ3_S: UInt32 = 21
    comptime IQ2_S: UInt32 = 22
    comptime IQ4_XS: UInt32 = 23
    comptime I8: UInt32 = 24
    comptime I16: UInt32 = 25
    comptime I32: UInt32 = 26
    comptime I64: UInt32 = 27
    comptime F64: UInt32 = 28
    comptime IQ1_M: UInt32 = 29
    comptime BF16: UInt32 = 30
    comptime TQ1_0: UInt32 = 34
    comptime TQ2_0: UInt32 = 35
    comptime MXFP4: UInt32 = 39


@always_inline
def ggml_type_size(t: UInt32) -> UInt64:
    """ Return the byte size of a single block for the given GGMLType."""
    if t == GGMLType.F32:
        return 4
    elif t == GGMLType.F16:
        return 2
    elif t == GGMLType.Q4_0:
        return 18      # 2B f16 scale + 16B nibbles (32 elements)
    elif t == GGMLType.Q4_1:
        return 20      # 2B d + 2B m + 16B nibbles (32 elements)
    elif t == GGMLType.Q5_0:
        return 22      # 2B d + 4B qh + 16B ql (32 elements)
    elif t == GGMLType.Q5_1:
        return 24      # 2B d + 2B m + 4B qh + 16B ql (32 elements)
    elif t == GGMLType.Q8_0:
        return 34      # 2B f16 scale + 32B int8 (32 elements)
    elif t == GGMLType.Q8_1:
        return 36      # 2B d + 2B s + 32B int8 (32 elements)
    elif t == GGMLType.Q2_K:
        return 84      # 256 elements
    elif t == GGMLType.Q3_K:
        return 110     # 256 elements
    elif t == GGMLType.Q4_K:
        return 144     # 2B d + 2B dmin + 12B scales + 128B qs (256 elements)
    elif t == GGMLType.Q5_K:
        return 176     # 2B d + 2B dmin + 12B scales + 32B qh + 128B qs (256 elements)
    elif t == GGMLType.Q6_K:
        return 210     # 128B ql + 64B qh + 16B scales + 2B d (256 elements)
    elif t == GGMLType.Q8_K:
        return 292     # 4B d + 256B qs + 32B bsums (256 elements)
    elif t == GGMLType.IQ2_XXS:
        return 66
    elif t == GGMLType.IQ2_XS:
        return 74
    elif t == GGMLType.IQ3_XXS:
        return 98
    elif t == GGMLType.IQ1_S:
        return 50
    elif t == GGMLType.IQ4_NL:
        return 18      # 32 elements
    elif t == GGMLType.IQ3_S:
        return 110
    elif t == GGMLType.IQ2_S:
        return 82
    elif t == GGMLType.IQ4_XS:
        return 136     # 256 elements
    elif t == GGMLType.I8:
        return 1
    elif t == GGMLType.I16:
        return 2
    elif t == GGMLType.I32:
        return 4
    elif t == GGMLType.I64:
        return 8
    elif t == GGMLType.F64:
        return 8
    elif t == GGMLType.IQ1_M:
        return 56
    elif t == GGMLType.BF16:
        return 2
    elif t == GGMLType.TQ1_0:
        return 54
    elif t == GGMLType.TQ2_0:
        return 66
    elif t == GGMLType.MXFP4:
        return 17      # 16B nibbles + 1B E8M0 shared exponent (32 elements)
    else:
        return 0

@always_inline
def ggml_block_size(t: UInt32) -> UInt64:
    """ Return the number of elements per quantization block for the given GGMLType."""
    if t == GGMLType.F32 or t == GGMLType.F16 or t == GGMLType.BF16:
        return 1
    elif t == GGMLType.I8 or t == GGMLType.I16 or t == GGMLType.I32 or t == GGMLType.I64 or t == GGMLType.F64:
        return 1
    elif (
        t == GGMLType.Q4_0
        or t == GGMLType.Q4_1
        or t == GGMLType.Q5_0
        or t == GGMLType.Q5_1
        or t == GGMLType.Q8_0
        or t == GGMLType.Q8_1
        or t == GGMLType.IQ4_NL
        or t == GGMLType.MXFP4
    ):
        return 32
    elif (
        t == GGMLType.Q2_K
        or t == GGMLType.Q3_K
        or t == GGMLType.Q4_K
        or t == GGMLType.Q5_K
        or t == GGMLType.Q6_K
        or t == GGMLType.Q8_K
        or t == GGMLType.IQ2_XXS
        or t == GGMLType.IQ2_XS
        or t == GGMLType.IQ3_XXS
        or t == GGMLType.IQ1_S
        or t == GGMLType.IQ3_S
        or t == GGMLType.IQ2_S
        or t == GGMLType.IQ4_XS
        or t == GGMLType.IQ1_M
        or t == GGMLType.TQ1_0
        or t == GGMLType.TQ2_0
    ):
        return 256
    else:
        return 0

@always_inline
def ggml_type_size_in_bytes(dtype: UInt32, n_elements: UInt64) -> UInt64:
    """ Calculate total byte size for a tensor of `n_elements` with type `dtype`."""
    var blk_sz = ggml_block_size(dtype)
    if blk_sz == 0:
        return 0
    var n_blocks = (n_elements + blk_sz - 1) // blk_sz
    return n_blocks * ggml_type_size(dtype)

def ggml_type_name(t: UInt32) -> String:
    """ Return human-readable name for a GGMLType."""
    if t == GGMLType.F32:
        return "F32"
    elif t == GGMLType.F16:
        return "F16"
    elif t == GGMLType.Q4_0:
        return "Q4_0"
    elif t == GGMLType.Q4_1:
        return "Q4_1"
    elif t == GGMLType.Q5_0:
        return "Q5_0"
    elif t == GGMLType.Q5_1:
        return "Q5_1"
    elif t == GGMLType.Q8_0:
        return "Q8_0"
    elif t == GGMLType.Q8_1:
        return "Q8_1"
    elif t == GGMLType.Q2_K:
        return "Q2_K"
    elif t == GGMLType.Q3_K:
        return "Q3_K"
    elif t == GGMLType.Q4_K:
        return "Q4_K"
    elif t == GGMLType.Q5_K:
        return "Q5_K"
    elif t == GGMLType.Q6_K:
        return "Q6_K"
    elif t == GGMLType.Q8_K:
        return "Q8_K"
    elif t == GGMLType.IQ4_NL:
        return "IQ4_NL"
    elif t == GGMLType.IQ4_XS:
        return "IQ4_XS"
    elif t == GGMLType.BF16:
        return "BF16"
    elif t == GGMLType.MXFP4:
        return "MXFP4"
    else:
        return "UNKNOWN(" + String(t) + ")"

# ─── Core llama.h Primitive Aliases & Enums ──────────────────────────────────

comptime llama_token = Int32
comptime llama_pos = Int32
comptime llama_seq_id = Int32

struct llama_split_mode:
    comptime LLAMA_SPLIT_MODE_NONE: Int32 = 0
    comptime LLAMA_SPLIT_MODE_LAYER: Int32 = 1
    comptime LLAMA_SPLIT_MODE_ROW: Int32 = 2

struct llama_rope_scaling_type:
    comptime LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED: Int32 = -1
    comptime LLAMA_ROPE_SCALING_TYPE_NONE: Int32 = 0
    comptime LLAMA_ROPE_SCALING_TYPE_LINEAR: Int32 = 1
    comptime LLAMA_ROPE_SCALING_TYPE_YARN: Int32 = 2
    comptime LLAMA_ROPE_SCALING_TYPE_LONGROPE: Int32 = 3

# ─── llama.h Parameter & Batch Structs ───────────────────────────────────────

@fieldwise_init
struct llama_model_params(ImplicitlyCopyable, Copyable, Movable):
    var n_gpu_layers: Int32
    var split_mode: Int32
    var main_gpu: Int32
    var vocab_only: Bool
    var use_mmap: Bool
    var use_mlock: Bool
    var check_tensors: Bool

@fieldwise_init
struct llama_context_params(ImplicitlyCopyable, Copyable, Movable):
    var n_ctx: UInt32
    var n_batch: UInt32
    var n_ubatch: UInt32
    var n_seq_max: UInt32
    var n_threads: Int32
    var n_threads_batch: Int32
    var rope_scaling_type: Int32
    var rope_freq_base: Float32
    var rope_freq_scale: Float32
    var type_k: UInt32          # GGMLType for K cache (F16, F32, Q8_0, Q4_0)
    var type_v: UInt32          # GGMLType for V cache
    var embeddings: Bool
    var offload_kqv: Bool
    var flash_attn: Bool

@fieldwise_init
struct llama_model_quantize_params(ImplicitlyCopyable, Copyable, Movable):
    var nthread: Int32
    var ftype: UInt32           # Target GGMLType
    var output_tensor_type: UInt32
    var token_embedding_type: UInt32
    var allow_requantize: Bool
    var quantize_output_tensor: Bool
    var only_copy: Bool
    var pure: Bool
    var keep_split: Bool

@fieldwise_init
struct llama_batch(ImplicitlyCopyable, Copyable, Movable):
    var n_tokens: Int32
    var token: Pointer[llama_token, MutUntrackedOrigin]
    var embd: Pointer[Float32, MutUntrackedOrigin]
    var pos: Pointer[llama_pos, MutUntrackedOrigin]
    var n_seq_id: Pointer[Int32, MutUntrackedOrigin]
    var seq_id: Pointer[Pointer[llama_seq_id, MutUntrackedOrigin], MutUntrackedOrigin]
    var logits: Pointer[Int8, MutUntrackedOrigin]

@fieldwise_init
struct llama_token_data(ImplicitlyCopyable, Copyable, Movable):
    var id: llama_token
    var logit: Float32
    var p: Float32

@fieldwise_init
struct llama_token_data_array(ImplicitlyCopyable, Copyable, Movable):
    var data: Pointer[llama_token_data, MutUntrackedOrigin]
    var size: Int
    var selected: Int64
    var sorted: Bool

    def __init__( out self,
        data: Pointer[llama_token_data, MutUntrackedOrigin],
        size: Int,
        sorted: Bool = False,
        selected: Int64 = -1,
    ):
        self.data = data
        self.size = size
        self.selected = selected
        self.sorted = sorted

# ─── Dynamic Runtime Model Configuration ─────────────────────────────────────

@fieldwise_init
struct ModelConfig(ImplicitlyCopyable, Copyable, Movable):
    """ Runtime model configuration populated from GGUF metadata."""
    var arch: String             # e.g. "llama", "qwen2", "gemma2", "gemma4", "mamba"
    var name: String             # general.name
    var hidden_dim: Int          # {arch}.embedding_length
    var intermediate_dim: Int    # {arch}.feed_forward_length
    var n_heads: Int             # {arch}.attention.head_count
    var n_kv_heads: Int          # {arch}.attention.head_count_kv
    var head_dim: Int            # {arch}.attention.key_length or hidden_dim // n_heads
    var n_layers: Int            # {arch}.block_count
    var vocab_size: Int          # tokenizer.ggml.tokens count or {arch}.vocab_size
    var max_seq_len: Int         # {arch}.context_length
    var rope_theta: Float32      # {arch}.rope.freq_base (default 10000.0)
    var rope_freq_scale: Float32 # {arch}.rope.scaling.factor (default 1.0)
    var norm_eps: Float32        # {arch}.attention.layer_norm_rms_epsilon (default 1e-5)
    var rope_type: Int32         # 0: NORM (interleaved), 2: NEOX (half-split)
    var n_experts: Int           # {arch}.expert_count (0 for dense)
    var n_experts_used: Int      # {arch}.expert_used_count (0 for dense)
    var quant_type: UInt32       # Primary weight GGMLType

    # Gemma 4 specific architectural metadata
    var shared_kv_layers: Int    # Number of upper layers that reuse KV cache (e.g. 20)
    var sliding_window: Int      # Sliding window attention size (e.g. 512)
    var ple_dim: Int             # Per-layer input embedding dim (e.g. 256)
    var logit_softcap: Float32   # Final logit softcapping factor (e.g. 30.0)
    var head_dim_swa: Int        # SWA head dim (e.g. 256)
    var rope_theta_swa: Float32  # SWA rope frequency base (e.g. 10000.0)
    var is_tied_embeddings: Bool # Whether output.weight is tied to token_embd.weight

    def __init__(out self):
        self.arch = "llama"
        self.name = "unknown"
        self.hidden_dim = 0
        self.intermediate_dim = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.n_layers = 0
        self.vocab_size = 0
        self.max_seq_len = 2048
        self.rope_theta = 10000.0
        self.rope_freq_scale = 1.0
        self.norm_eps = 1e-5
        self.rope_type = 0
        self.n_experts = 0
        self.n_experts_used = 0
        self.quant_type = GGMLType.F32

        self.shared_kv_layers = 0
        self.sliding_window = 0
        self.ple_dim = 0
        self.logit_softcap = 0.0
        self.head_dim_swa = 0
        self.rope_theta_swa = 10000.0
        self.is_tied_embeddings = False
