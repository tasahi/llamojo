""" Public llama.h-compatible API for llamojo (Phase 1).

Implements:
  - `llama_backend_init` / `llama_backend_free` / `llama_print_system_info`
  - `llama_model_default_params` / `llama_context_default_params` / `llama_model_quantize_default_params`
  - `llama_model` struct and `llama_model_p` opaque handle
  - `llama_model_load_from_file` / `llama_model_free`
  - Model metadata and architecture inspection API (`llama_model_*`)
"""

from std.collections import List
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores
from std import time

from .types import (
    GGMLType,
    ggml_type_name,
    llama_token,
    llama_pos,
    llama_seq_id,
    llama_split_mode,
    llama_rope_scaling_type,
    llama_model_params,
    llama_context_params,
    llama_model_quantize_params,
    llama_batch,
    llama_token_data,
    llama_token_data_array,
    ModelConfig,
    null_ptr,
)
from .gguf import (
    BytePtr,
    GGUFValue,
    GGUFTensorInfo,
    GGUFContext,
    gguf_load_from_file,
)
from .backend.isa import simd_info, isa_info, SIMD_WIDTH

# ─── Opaque Handle Types (matching llama.h) ──────────────────────────────────

struct llama_model(Movable):
    """Internal representation of a loaded GGUF model (`struct llama_model`)."""
    var gguf: GGUFContext
    var config: ModelConfig
    var params: llama_model_params
    var vocab_tokens: List[String]
    var vocab_scores: List[Float32]
    var vocab_types: List[Int32]
    var bos_token_id: llama_token
    var eos_token_id: llama_token
    var eot_token_id: llama_token
    var pad_token_id: llama_token
    var n_params: UInt64
    var total_bytes: UInt64

    def __init__(
        out self,
        var gguf: GGUFContext,
        var config: ModelConfig,
        params: llama_model_params,
    ):
        self. vocab_tokens = List[String]()
        self.vocab_scores = List[Float32]()
        self.vocab_types = List[Int32]()

        var tok_opt = gguf.metadata.find("tokenizer.ggml.tokens")
        if tok_opt:
            self.vocab_tokens = tok_opt.value().str_arr.copy()

        var score_opt = gguf.metadata.find("tokenizer.ggml.scores")
        if score_opt:
            self.vocab_scores = score_opt.value().float_arr.copy()

        var type_opt = gguf.metadata.find("tokenizer.ggml.token_type")
        if type_opt:
            var t_arr = type_opt.value().int_arr.copy()
            for i in range(len(t_arr)):
                self.vocab_types.append(Int32(t_arr[i]))

        self.bos_token_id = Int32(gguf.get_int("tokenizer.ggml.bos_token_id", 1))
        self.eos_token_id = Int32(gguf.get_int("tokenizer.ggml.eos_token_id", 2))
        self.eot_token_id = Int32(gguf.get_int("tokenizer.ggml.eot_token_id", -1))
        self.pad_token_id = Int32(gguf.get_int("tokenizer.ggml.padding_token_id", -1))
        self.n_params = gguf.total_parameters()
        self.total_bytes = gguf.total_tensor_bytes()
        self.gguf = gguf^
        self.config = config^
        self.params = params

comptime llama_model_p = Pointer[llama_model, MutUntrackedOrigin]

# ─── Backend Lifecycle & System Capabilities ─────────────────────────────────

def llama_backend_init():
    """Initialize the llamojo backend (ISA flags, timing, thread defaults)."""
    _ = SIMD_WIDTH
    _ = num_performance_cores()

def llama_backend_free():
    """Free global llamojo backend resources (no-op in pure Mojo)."""
    pass

def llama_numa_init(numa: Int32):
    """Initialize NUMA policy (no-op placeholder for llama.h compatibility)."""
    pass

@always_inline
def llama_time_us() -> Int64:
    """Return monotonic wall-clock timestamp in microseconds."""
    return Int64(time.perf_counter_ns() // 1000)

@always_inline
def llama_max_devices() -> Int:
    return 1

@always_inline
def llama_supports_mmap() -> Bool:
    return True

@always_inline
def llama_supports_mlock() -> Bool:
    return True

@always_inline
def llama_supports_gpu_offload() -> Bool:
    return False

def llama_print_system_info() -> String:
    """Return system capability string matching `llama_print_system_info()`."""
    return "llamojo (Pure Mojo 1.1.0) | " + simd_info() + " | mmap=1 | mlock=1"

# ─── Default Parameter Initializers (matching llama.cpp defaults) ────────────

def llama_model_default_params() -> llama_model_params:
    """Return default `llama_model_params` matching `llama.cpp`."""
    return llama_model_params(
        n_gpu_layers=0,
        split_mode=llama_split_mode.LLAMA_SPLIT_MODE_NONE,
        main_gpu=0,
        vocab_only=False,
        use_mmap=True,
        use_mlock=False,
        check_tensors=True,
    )

def llama_context_default_params() -> llama_context_params:
    """Return default `llama_context_params` matching `llama.cpp`."""
    var hw_threads = Int32(max(1, num_performance_cores()))
    return llama_context_params(
        n_ctx=512,
        n_batch=2048,
        n_ubatch=512,
        n_seq_max=1,
        n_threads=hw_threads,
        n_threads_batch=hw_threads,
        rope_scaling_type=llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        rope_freq_base=0.0,
        rope_freq_scale=0.0,
        type_k=GGMLType.F16,
        type_v=GGMLType.F16,
        embeddings=False,
        offload_kqv=True,
        flash_attn=False,
    )

def llama_model_quantize_default_params() -> llama_model_quantize_params:
    """Return default `llama_model_quantize_params` matching `llama.cpp`."""
    return llama_model_quantize_params(
        nthread=Int32(max(1, num_performance_cores())),
        ftype=GGMLType.Q4_K,
        output_tensor_type=GGMLType.F16,
        token_embedding_type=GGMLType.F16,
        allow_requantize=False,
        quantize_output_tensor=True,
        only_copy=False,
        pure=False,
        keep_split=False,
    )

# ─── Model Load & Free API ───────────────────────────────────────────────────

def llama_model_load_from_file(
    path_model: String,
    params: llama_model_params,
) -> llama_model_p:
    """Load a GGUF model from `path_model` and return an opaque `llama_model_p`.

    Returns a null pointer (`Int(ptr) == 0`) if loading fails, matching `llama.h`.
    """
    try:
        var gguf = gguf_load_from_file(
            path_model,
            use_mmap=params.use_mmap,
            use_mlock=params.use_mlock,
            vocab_only=params.vocab_only,
            check_tensors=params.check_tensors,
        )
        var cfg = gguf.build_model_config()
        var model_ptr = unsafe_alloc[llama_model](1)
        model_ptr.unsafe_write(llama_model(gguf^, cfg^, params.copy()))
        return model_ptr
    except e:
        print("llama_model_load_from_file error:", e)
        return null_ptr[llama_model]()

def llama_load_model_from_file(
    path_model: String,
    params: llama_model_params,
) -> llama_model_p:
    """Alias for `llama_model_load_from_file` (legacy `llama.h` symbol)."""
    return llama_model_load_from_file(path_model, params)

def llama_model_free(model: llama_model_p):
    """Free all resources associated with `model` (`munmap` + heap dealloc)."""
    if Int(model) == 0:
        return
    model[].gguf.free()
    model.unsafe_deinit_pointee()
    model.unsafe_free()

def llama_free_model(model: llama_model_p):
    """Alias for `llama_model_free` (legacy `llama.h` symbol)."""
    llama_model_free(model)

# ─── Model Metadata & Architecture Inspection Functions ──────────────────────

@always_inline
def llama_model_n_params(model: llama_model_p) -> UInt64:
    if Int(model) == 0:
        return 0
    return model[].n_params

@always_inline
def llama_model_size(model: llama_model_p) -> UInt64:
    if Int(model) == 0:
        return 0
    return model[].total_bytes

@always_inline
def llama_model_n_embd(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(model[].config.hidden_dim)

@always_inline
def llama_model_n_layer(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(model[].config.n_layers)

@always_inline
def llama_model_n_head(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(model[].config.n_heads)

@always_inline
def llama_model_n_head_kv(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(model[].config.n_kv_heads)

@always_inline
def llama_model_n_ctx_train(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(model[].config.max_seq_len)

@always_inline
def llama_model_n_vocab(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(model[].config.vocab_size)

@always_inline
def llama_model_rope_freq_scale_train(model: llama_model_p) -> Float32:
    if Int(model) == 0:
        return 1.0
    return model[].config.rope_freq_scale

def llama_model_desc(model: llama_model_p) -> String:
    """Return a concise description of the model (arch, layers, embd, quant)."""
    if Int(model) == 0:
        return "null"
    var cfg = model[].config.copy()
    return (
        cfg.arch
        + " ("
        + cfg.name
        + ") "
        + String(cfg.n_layers)
        + "L x "
        + String(cfg.hidden_dim)
        + "D ["
        + ggml_type_name(cfg.quant_type)
        + "]"
    )

def llama_model_meta_val_str(model: llama_model_p, key: String) -> String:
    """Retrieve a string representation of metadata entry `key` from `model`."""
    if Int(model) == 0:
        return ""
    var opt = model[].gguf.metadata.find(key)
    if not opt:
        return ""
    var val = opt.value().copy()
    if val.vtype == 8:
        return val.str_val
    elif val.vtype == 6 or val.vtype == 12:
        return String(val.float_val)
    elif val.vtype == 7:
        return "true" if val.bool_val else "false"
    else:
        return String(val.int_val)

@always_inline
def llama_model_meta_count(model: llama_model_p) -> Int32:
    if Int(model) == 0:
        return 0
    return Int32(len(model[].gguf.metadata))

@always_inline
def llama_model_has_encoder(model: llama_model_p) -> Bool:
    if Int(model) == 0:
        return False
    return model[].config.arch == "t5"

@always_inline
def llama_model_has_decoder(model: llama_model_p) -> Bool:
    if Int(model) == 0:
        return False
    return True

@always_inline
def llama_model_is_recurrent(model: llama_model_p) -> Bool:
    if Int(model) == 0:
        return False
    var arch = model[].config.arch
    return arch == "mamba" or arch == "rwkv" or "ssm" in arch