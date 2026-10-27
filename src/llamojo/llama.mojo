""" Public llama.h-compatible API for llamojo.\n
Phases 1-3 complete implementation:
  - llama_backend_init / llama_backend_free / llama_print_system_info
  - llama_model_default_params / llama_context_default_params / llama_model_quantize_default_params
  - llama_model_load_from_file / llama_model_free
  - llama_init_from_model / llama_free (Context lifecycle & activation workspace)
  - llama_batch_init / llama_batch_get_one / llama_batch_free
  - llama_decode / llama_encode (Multi-layer transformer decode engine)
  - llama_get_logits / llama_get_logits_ith / llama_get_embeddings
  - llama_kv_cache_clear / llama_kv_cache_seq_*
  - Model metadata and architecture reflection API
"""

from std.collections import List
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores
from std import time
from max.algorithm import parallelize

from .types import (
    GGMLType,
    ggml_type_name,
    ggml_block_size,
    ggml_type_size,
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
from .kv_cache import LlamaKVCache
from .lora import llama_lora_adapter, compute_lora_delta
from .backend import (
    simd_info,
    isa_info,
    SIMD_WIDTH,
    LlamaWorkerPool,
    LlamaPoolSharedPtr,
    llama_barrier,
    chunk_start,
    chunk_end,
    LayerWeights,
    forward_layer_dense,
    forward_gemma4_single_token,
    apply_logit_softcapping,
    rms_norm,
    dot_quant_row,
    dequantize_row,
    f16_to_f32,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

# ─── Model & Context Handles (matching llama.h) ──────────────────────────────

struct llama_model(Movable):
    """ Internal representation of a loaded GGUF model (`struct llama_model`)."""
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
    var token_embd: BytePtr
    var token_embd_type: UInt32
    var output_norm: FloatPtr
    var output_weight: BytePtr
    var output_weight_type: UInt32
    var layers: List[LayerWeights]

    # Gemma 4 global per-layer embedding tensors
    var per_layer_token_embd: BytePtr
    var per_layer_token_embd_type: UInt32
    var per_layer_model_proj: BytePtr
    var per_layer_model_proj_type: UInt32
    var per_layer_proj_norm: FloatPtr

    def __init__( out self,
            var gguf: GGUFContext,
            var config: ModelConfig,
            params: llama_model_params,
        ):
        self.vocab_tokens = List[String]()
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

        # Bind embeddings and output head
        self.token_embd = gguf.get_tensor_ptr("token_embd.weight")
        self.token_embd_type = GGMLType.F32
        if gguf.has_tensor("token_embd.weight"):
            try:
                self.token_embd_type = gguf.get_tensor("token_embd.weight").dtype
            except:
                pass

        self.output_norm = gguf.get_tensor_ptr("output_norm.weight").unsafe_bitcast[Float32]()
        if Int(self.output_norm) == 0:
            self.output_norm = gguf.get_tensor_ptr("blk.0.attn_norm.weight").unsafe_bitcast[Float32]()

        self.output_weight = gguf.get_tensor_ptr("output.weight")
        self.output_weight_type = GGMLType.F32
        if gguf.has_tensor("output.weight"):
            try:
                self.output_weight_type = gguf.get_tensor("output.weight").dtype
            except:
                pass
        elif Int(self.output_weight) == 0:
            self.output_weight = self.token_embd
            self.output_weight_type = self.token_embd_type

        # Bind Gemma 4 per-layer embedding tensors
        self.per_layer_token_embd = gguf.get_tensor_ptr("per_layer_token_embd.weight")
        self.per_layer_token_embd_type = GGMLType.F32
        if gguf.has_tensor("per_layer_token_embd.weight"):
            try:
                self.per_layer_token_embd_type = gguf.get_tensor("per_layer_token_embd.weight").dtype
            except:
                pass

        self.per_layer_model_proj = gguf.get_tensor_ptr("per_layer_model_proj.weight")
        self.per_layer_model_proj_type = GGMLType.F32
        if gguf.has_tensor("per_layer_model_proj.weight"):
            try:
                self.per_layer_model_proj_type = gguf.get_tensor("per_layer_model_proj.weight").dtype
            except:
                pass

        self.per_layer_proj_norm = gguf.get_tensor_ptr("per_layer_proj_norm.weight").unsafe_bitcast[Float32]()

        # Populate per-layer weight descriptors
        self.layers = List[LayerWeights]()
        for l in range(config.n_layers):
            var lw = LayerWeights()
            var pfx = "blk." + String(l) + "."
            lw.attn_norm = gguf.get_tensor_ptr(pfx + "attn_norm.weight").unsafe_bitcast[Float32]()
            lw.ffn_norm = gguf.get_tensor_ptr(pfx + "ffn_norm.weight").unsafe_bitcast[Float32]()
            lw.wq = gguf.get_tensor_ptr(pfx + "attn_q.weight")
            lw.wk = gguf.get_tensor_ptr(pfx + "attn_k.weight")
            lw.wv = gguf.get_tensor_ptr(pfx + "attn_v.weight")
            lw.wo = gguf.get_tensor_ptr(pfx + "attn_output.weight")
            lw.w_gate = gguf.get_tensor_ptr(pfx + "ffn_gate.weight")
            lw.w_up = gguf.get_tensor_ptr(pfx + "ffn_up.weight")
            lw.w_down = gguf.get_tensor_ptr(pfx + "ffn_down.weight")

            # Gemma 4 per-layer norms, scales, and PLE projections
            lw.attn_q_norm = gguf.get_tensor_ptr(pfx + "attn_q_norm.weight").unsafe_bitcast[Float32]()
            lw.attn_k_norm = gguf.get_tensor_ptr(pfx + "attn_k_norm.weight").unsafe_bitcast[Float32]()
            lw.post_attention_norm = gguf.get_tensor_ptr(pfx + "post_attention_norm.weight").unsafe_bitcast[Float32]()
            lw.post_ffw_norm = gguf.get_tensor_ptr(pfx + "post_ffw_norm.weight").unsafe_bitcast[Float32]()
            lw.inp_gate = gguf.get_tensor_ptr(pfx + "inp_gate.weight")
            lw.proj = gguf.get_tensor_ptr(pfx + "proj.weight")
            lw.post_norm = gguf.get_tensor_ptr(pfx + "post_norm.weight").unsafe_bitcast[Float32]()

            var scale_ptr = gguf.get_tensor_ptr(pfx + "layer_output_scale.weight")
            if Int(scale_ptr) != 0:
                lw.layer_output_scale = scale_ptr.unsafe_bitcast[Float32]()[unsafe_offset=0]
            else:
                lw.layer_output_scale = 1.0

            if gguf.has_tensor(pfx + "inp_gate.weight"):
                try:
                    lw.dtype_inp_gate = gguf.get_tensor(pfx + "inp_gate.weight").dtype
                except:
                    pass
            if gguf.has_tensor(pfx + "proj.weight"):
                try:
                    lw.dtype_proj = gguf.get_tensor(pfx + "proj.weight").dtype
                except:
                    pass

            # MoE Router and Expert Weight Ingestion
            var router_ptr = gguf.get_tensor_ptr(pfx + "ffn_gate_inp.weight")
            if Int(router_ptr) != 0:
                lw.is_moe = True
                lw.w_router = router_ptr.unsafe_bitcast[Float32]()
                lw.n_experts = config.n_experts if config.n_experts > 0 else 8
                lw.n_experts_used = config.n_experts_used if config.n_experts_used > 0 else 2
                var g_arr = unsafe_alloc[BytePtr](lw.n_experts)
                var u_arr = unsafe_alloc[BytePtr](lw.n_experts)
                var d_arr = unsafe_alloc[BytePtr](lw.n_experts)
                for e in range(lw.n_experts):
                    var exp_pfx = pfx + "ffn_gate." + String(e) + ".weight"
                    g_arr[unsafe_offset=e] = gguf.get_tensor_ptr(exp_pfx)
                    var up_pfx = pfx + "ffn_up." + String(e) + ".weight"
                    u_arr[unsafe_offset=e] = gguf.get_tensor_ptr(up_pfx)
                    var down_pfx = pfx + "ffn_down." + String(e) + ".weight"
                    d_arr[unsafe_offset=e] = gguf.get_tensor_ptr(down_pfx)
                lw.exp_gates_ptr = g_arr
                lw.exp_ups_ptr = u_arr
                lw.exp_downs_ptr = d_arr

            lw.dtype = config.quant_type
            if gguf.has_tensor(pfx + "attn_q.weight"):
                try:
                    lw.dtype = gguf.get_tensor(pfx + "attn_q.weight").dtype
                except:
                    pass
            lw.dtype_q = lw.dtype
            lw.dtype_k = lw.dtype
            lw.dtype_v = lw.dtype
            lw.dtype_o = lw.dtype
            lw.dtype_gate = lw.dtype
            lw.dtype_up = lw.dtype
            lw.dtype_down = lw.dtype

            if gguf.has_tensor(pfx + "attn_k.weight"):
                try:
                    lw.dtype_k = gguf.get_tensor(pfx + "attn_k.weight").dtype
                except:
                    pass
            if gguf.has_tensor(pfx + "attn_v.weight"):
                try:
                    lw.dtype_v = gguf.get_tensor(pfx + "attn_v.weight").dtype
                except:
                    pass
            if gguf.has_tensor(pfx + "attn_output.weight"):
                try:
                    lw.dtype_o = gguf.get_tensor(pfx + "attn_output.weight").dtype
                except:
                    pass
            if gguf.has_tensor(pfx + "ffn_gate.weight"):
                try:
                    lw.dtype_gate = gguf.get_tensor(pfx + "ffn_gate.weight").dtype
                except:
                    pass
            if gguf.has_tensor(pfx + "ffn_up.weight"):
                try:
                    lw.dtype_up = gguf.get_tensor(pfx + "ffn_up.weight").dtype
                except:
                    pass
            if gguf.has_tensor(pfx + "ffn_down.weight"):
                try:
                    lw.dtype_down = gguf.get_tensor(pfx + "ffn_down.weight").dtype
                except:
                    pass
            self.layers.append(lw^)

        self.gguf = gguf^
        self.config = config^
        self.params = params

comptime llama_model_p = Pointer[llama_model, MutUntrackedOrigin]

struct llama_context(Movable):
    """ Runtime execution context holding activations, KV cache, worker pool, and logits."""
    var model: llama_model_p
    var params: llama_context_params
    var kv_cache: LlamaKVCache
    var pool: LlamaWorkerPool
    var n_tokens: Int
    var max_ctx: Int
    var dim: Int
    var hidden_dim: Int
    var vocab_size: Int
    var x: FloatPtr
    var xb: FloatPtr
    var xb2: FloatPtr
    var hb: FloatPtr
    var hb2: FloatPtr
    var q: FloatPtr
    var k_buf: FloatPtr
    var v_buf: FloatPtr
    var att_buf: FloatPtr
    var logits: FloatPtr
    var embeddings: FloatPtr

    # Gemma 4 auxiliary PLE and gate/proj workspace buffers
    var ple_buf: FloatPtr
    var ple_tok_buf: FloatPtr
    var ple_proj_buf: FloatPtr
    var gate_buf: FloatPtr
    var proj_buf: FloatPtr

    var allocated: Bool

    def __init__(out self, model: llama_model_p, params: llama_context_params):
        self.model = model
        self.params = params
        var cfg = model[].config
        self.max_ctx = Int(params.n_ctx) if params.n_ctx > 0 else cfg.max_seq_len
        self.dim = cfg.hidden_dim
        self.hidden_dim = cfg.intermediate_dim
        self.vocab_size = cfg.vocab_size
        self.n_tokens = 0

        # Initialize KV Cache (with shared KV layers support for architectures like Gemma 4)
        self.kv_cache = LlamaKVCache(
            max_ctx=self.max_ctx,
            n_layers=cfg.n_layers,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            type_k=params.type_k,
            type_v=params.type_v,
            enable_paged=False,
            shared_kv_layers=cfg.shared_kv_layers,
        )

        # Initialize Persistent Spin-Barrier Worker Pool
        var n_threads = Int(params.n_threads) if params.n_threads > 0 else num_performance_cores()
        self.pool = LlamaWorkerPool(workers=n_threads, dim=self.dim)

        # Allocate contiguous activation workspace buffers
        var max_q_dim = max(self.dim, cfg.n_heads * cfg.head_dim)
        var max_ffn_dim = max(self.hidden_dim, 12288 if cfg.arch == "gemma4" else self.hidden_dim)
        self.x = unsafe_alloc[Float32](self.dim)
        self.xb = unsafe_alloc[Float32](self.dim)
        self.xb2 = unsafe_alloc[Float32](max_q_dim)
        self.hb = unsafe_alloc[Float32](max_ffn_dim)
        self.hb2 = unsafe_alloc[Float32](max_ffn_dim)
        self.q = unsafe_alloc[Float32](max_q_dim)
        self.k_buf = unsafe_alloc[Float32](max(cfg.n_kv_heads * cfg.head_dim, 512))
        self.v_buf = unsafe_alloc[Float32](max(cfg.n_kv_heads * cfg.head_dim, 512))
        self.att_buf = unsafe_alloc[Float32](cfg.n_heads * self.max_ctx)

        # Gemma 4 workspace buffer allocations
        var ple_dim = cfg.ple_dim if cfg.ple_dim > 0 else 256
        var ple_total = cfg.n_layers * ple_dim
        self.ple_buf = unsafe_alloc[Float32](ple_total)
        self.ple_tok_buf = unsafe_alloc[Float32](ple_total)
        self.ple_proj_buf = unsafe_alloc[Float32](ple_total)
        self.gate_buf = unsafe_alloc[Float32](ple_dim)
        self.proj_buf = unsafe_alloc[Float32](self.dim)

        var n_batch = Int(params.n_batch) if params.n_batch > 0 else 512
        var logits_size = max(1, self.vocab_size) * n_batch
        self.logits = unsafe_alloc[Float32](logits_size)
        unsafe_memset_zero(self.logits, logits_size)

        if params.embeddings:
            self.embeddings = unsafe_alloc[Float32](self.dim * n_batch)
        else:
            self.embeddings = null_ptr[Float32]()

        self.allocated = True

    def deinit_storage(mut self):
        if self.allocated:
            self.x.unsafe_free()
            self.xb.unsafe_free()
            self.xb2.unsafe_free()
            self.hb.unsafe_free()
            self.hb2.unsafe_free()
            self.q.unsafe_free()
            self.k_buf.unsafe_free()
            self.v_buf.unsafe_free()
            self.att_buf.unsafe_free()
            self.ple_buf.unsafe_free()
            self.ple_tok_buf.unsafe_free()
            self.ple_proj_buf.unsafe_free()
            self.gate_buf.unsafe_free()
            self.proj_buf.unsafe_free()
            self.logits.unsafe_free()
            if Int(self.embeddings) != 0:
                self.embeddings.unsafe_free()
            self.kv_cache.deinit_storage()
            self.pool.free()
            self.allocated = False

comptime llama_context_p = Pointer[llama_context, MutUntrackedOrigin]

# ─── Backend Lifecycle & System Capabilities ─────────────────────────────────

def llama_backend_init():
    _ = SIMD_WIDTH
    _ = num_performance_cores()

def llama_backend_free():
    pass

def llama_numa_init(numa: Int32):
    pass

@always_inline
def llama_time_us() -> Int64:
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
    return "llamojo (Pure Mojo 1.1.0) | " + simd_info() + " | mmap=1 | mlock=1"

# ─── Default Parameters ──────────────────────────────────────────────────────

def llama_model_default_params() -> llama_model_params:
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
    return llama_model_load_from_file(path_model, params)

def llama_model_free(model: llama_model_p):
    if Int(model) == 0:
        return
    model[].gguf.free()
    model.unsafe_deinit_pointee()
    model.unsafe_free()

def llama_free_model(model: llama_model_p):
    llama_model_free(model)

# ─── Context Lifecycle API ───────────────────────────────────────────────────

def llama_init_from_model(
        model: llama_model_p,
        params: llama_context_params,
    ) -> llama_context_p:
    """ Initialize an inference execution context for `model`."""
    if Int(model) == 0:
        return null_ptr[llama_context]()
    var ctx_ptr = unsafe_alloc[llama_context](1)
    ctx_ptr.unsafe_write(llama_context(model, params))
    return ctx_ptr

def llama_new_context_with_model(
        model: llama_model_p,
        params: llama_context_params,
    ) -> llama_context_p:
    return llama_init_from_model(model, params)

def llama_free(ctx: llama_context_p):
    """Free context activations, KV cache, and thread pool."""
    if Int(ctx) == 0:
        return
    ctx[].deinit_storage()
    ctx.unsafe_deinit_pointee()
    ctx.unsafe_free()

# ─── Batch Management API ────────────────────────────────────────────────────

def llama_batch_init(n_tokens: Int32, embd: Int32, n_seq_max: Int32) -> llama_batch:
    """ Allocate a `llama_batch` holding arrays of size `n_tokens`."""
    var n = max(1, Int(n_tokens))
    var token_ptr = unsafe_alloc[llama_token](n)
    var pos_ptr = unsafe_alloc[llama_pos](n)
    var n_seq_id_ptr = unsafe_alloc[Int32](n)
    var seq_id_ptr = unsafe_alloc[Pointer[llama_seq_id, MutUntrackedOrigin]](n)
    var logits_ptr = unsafe_alloc[Int8](n)

    var embd_ptr = null_ptr[Float32]()
    if embd > 0:
        embd_ptr = unsafe_alloc[Float32](n * Int(embd))

    for i in range(n):
        token_ptr[unsafe_offset=i] = 0
        pos_ptr[unsafe_offset=i] = Int32(i)
        n_seq_id_ptr[unsafe_offset=i] = 1
        var seq_arr = unsafe_alloc[llama_seq_id](1)
        seq_arr[unsafe_offset=0] = 0
        seq_id_ptr[unsafe_offset=i] = seq_arr
        logits_ptr[unsafe_offset=i] = 1 if i == n - 1 else 0

    return llama_batch(
        n_tokens=n_tokens,
        token=token_ptr,
        embd=embd_ptr,
        pos=pos_ptr,
        n_seq_id=n_seq_id_ptr,
        seq_id=seq_id_ptr,
        logits=logits_ptr,
    )

def llama_batch_get_one(tokens: Pointer[llama_token, MutUntrackedOrigin], n_tokens: Int32) -> llama_batch:
    """ Create a view `llama_batch` over contiguous tokens with auto-incrementing positions."""
    var batch = llama_batch_init(n_tokens, embd=0, n_seq_max=1)
    for i in range(Int(n_tokens)):
        batch.token[unsafe_offset=i] = tokens[unsafe_offset=i]
        batch.pos[unsafe_offset=i] = Int32(i)
        batch.logits[unsafe_offset=i] = 1 if i == Int(n_tokens) - 1 else 0
    return batch^

def llama_batch_free(mut batch: llama_batch):
    """ Release heap arrays allocated by `llama_batch_init`."""
    if Int(batch.token) != 0:
        batch.token.unsafe_free()
    if Int(batch.pos) != 0:
        batch.pos.unsafe_free()
    if Int(batch.n_seq_id) != 0:
        batch.n_seq_id.unsafe_free()
    if Int(batch.seq_id) != 0:
        for i in range(Int(batch.n_tokens)):
            batch.seq_id[unsafe_offset=i].unsafe_free()
        batch.seq_id.unsafe_free()
    if Int(batch.logits) != 0:
        batch.logits.unsafe_free()
    if Int(batch.embd) != 0:
        batch.embd.unsafe_free()
    var z: Int = 0
    batch.token = Pointer[llama_token, MutUntrackedOrigin](unsafe_from_address=z)
    batch.n_tokens = 0

# ─── Inference: llama_decode Engine ──────────────────────────────────────────

def llama_decode(ctx: llama_context_p, batch: llama_batch) -> Int32:
    """ Run full autoregressive transformer forward pass on tokens in `batch`."""
    if Int(ctx) == 0 or batch.n_tokens <= 0:
        return -1

    var model = ctx[].model
    var cfg = model[].config
    var dim = cfg.hidden_dim
    var vocab_size = max(1, cfg.vocab_size)
    var n_tokens = Int(batch.n_tokens)
    var sh = ctx[].pool.sh
    var workers = ctx[].pool.workers

    for tok_idx in range(n_tokens):
        var token = batch.token[unsafe_offset=tok_idx]
        var pos = Int(batch.pos[unsafe_offset=tok_idx])

        if cfg.arch == "gemma4":
            # Dedicated Gemma 4 forward execution pipeline
            forward_gemma4_single_token(
                token,
                pos,
                cfg,
                model[].token_embd,
                model[].token_embd_type,
                model[].per_layer_token_embd,
                model[].per_layer_token_embd_type,
                model[].per_layer_model_proj,
                model[].per_layer_model_proj_type,
                model[].per_layer_proj_norm,
                model[].layers.unsafe_ptr(),
                ctx[].kv_cache.k_f32,
                ctx[].kv_cache.v_f32,
                ctx[].max_ctx,
                ctx[].x,
                ctx[].xb,
                ctx[].xb2,
                ctx[].hb,
                ctx[].hb2,
                ctx[].q,
                ctx[].k_buf,
                ctx[].v_buf,
                ctx[].att_buf,
                ctx[].ple_buf,
                ctx[].ple_tok_buf,
                ctx[].ple_proj_buf,
                ctx[].gate_buf,
                ctx[].proj_buf,
            )

            # Gemma 4 Final Output Norm + Tied LM-Head + Softcapping
            if batch.logits[unsafe_offset=tok_idx] != 0:
                var xbp = sh[].xbp
                rms_norm(xbp, ctx[].x, model[].output_norm, dim, cfg.norm_eps)

                var out_logits = ctx[].logits.unsafe_offset(tok_idx * vocab_size)
                var blk_sz_out = Int(ggml_block_size(model[].output_weight_type))
                var type_sz_out = Int(ggml_type_size(model[].output_weight_type))
                var bpr_out = (dim // blk_sz_out) * type_sz_out

                def gemma4_logits_worker(wid: Int) {imm}:
                    var v0 = chunk_start(vocab_size, wid, workers)
                    var v1 = chunk_end(vocab_size, wid, workers)
                    for r in range(v0, v1):
                        var w_row = model[].output_weight.unsafe_offset(r * bpr_out)
                        out_logits[unsafe_offset=r] = dot_quant_row(model[].output_weight_type, w_row, xbp, dim)

                if workers == 1:
                    gemma4_logits_worker(0)
                else:
                    parallelize(gemma4_logits_worker, workers, workers)

                # Gemma 4 Logit Softcapping
                if cfg.logit_softcap > 0.0:
                    apply_logit_softcapping(out_logits, vocab_size, cfg.logit_softcap)
        else:
            # Step 1: Token embedding lookup into ctx.x
            if Int(batch.embd) != 0:
                var embd_token = batch.embd.unsafe_offset(tok_idx * dim)
                unsafe_memcpy(dest=ctx[].x, src=embd_token, count=dim)
            else:
                var emb_ptr = model[].token_embd
                var emb_type = model[].token_embd_type
                var blk_sz = Int(ggml_block_size(emb_type))
                var type_sz = Int(ggml_type_size(emb_type))
                var bpr = (dim // blk_sz) * type_sz
                var row_ptr = emb_ptr.unsafe_offset(Int(token) * bpr)
                dequantize_row(emb_type, row_ptr, ctx[].x, dim)

            # Step 2: Multi-layer transformer forward passes
            for l in range(cfg.n_layers):
                var lw = model[].layers[l].copy()
                def worker_fn(wid: Int) {imm}:
                    forward_layer_dense(
                        l,
                        pos,
                        lw,
                        cfg,
                        ctx[].x,
                        ctx[].xb,
                        ctx[].xb2,
                        ctx[].hb,
                        ctx[].hb2,
                        ctx[].q,
                        ctx[].k_buf,
                        ctx[].v_buf,
                        ctx[].att_buf,
                        ctx[].kv_cache.k_f32,
                        ctx[].kv_cache.v_f32,
                        ctx[].max_ctx,
                        wid,
                        sh,
                    )

                if workers == 1:
                    worker_fn(0)
                else:
                    parallelize(worker_fn, workers, workers)

            # Step 3: Compute final logits if requested for this token index
            if batch.logits[unsafe_offset=tok_idx] != 0:
                var xbp = sh[].xbp
                rms_norm(xbp, ctx[].x, model[].output_norm, dim, cfg.norm_eps)

                var out_logits = ctx[].logits.unsafe_offset(tok_idx * vocab_size)
                var blk_sz_out = Int(ggml_block_size(model[].output_weight_type))
                var type_sz_out = Int(ggml_type_size(model[].output_weight_type))
                var bpr_out = (dim // blk_sz_out) * type_sz_out

                def logits_worker(wid: Int) {imm}:
                    var v0 = chunk_start(vocab_size, wid, workers)
                    var v1 = chunk_end(vocab_size, wid, workers)
                    for r in range(v0, v1):
                        var w_row = model[].output_weight.unsafe_offset(r * bpr_out)
                        out_logits[unsafe_offset=r] = dot_quant_row(model[].output_weight_type, w_row, xbp, dim)

                if workers == 1:
                    logits_worker(0)
                else:
                    parallelize(logits_worker, workers, workers)

                if cfg.logit_softcap > 0.0:
                    apply_logit_softcapping(out_logits, vocab_size, cfg.logit_softcap)

        # Record embeddings if requested
        if Int(ctx[].embeddings) != 0:
            var embd_dest = ctx[].embeddings.unsafe_offset(tok_idx * dim)
            unsafe_memcpy(dest=embd_dest, src=ctx[].x, count=dim)

    ctx[].n_tokens += n_tokens
    return 0

def llama_encode(ctx: llama_context_p, batch: llama_batch) -> Int32:
    """ Alias for encoder-only or prompt prefill forward pass."""
    return llama_decode(ctx, batch)

# ─── Logits & Embeddings Extraction ──────────────────────────────────────────

@always_inline
def llama_get_logits(ctx: llama_context_p) -> FloatPtr:
    """ Return pointer to output logits for the last token decoded."""
    if Int(ctx) == 0:
        return null_ptr[Float32]()
    return ctx[].logits

@always_inline
def llama_get_logits_ith(ctx: llama_context_p, i: Int32) -> FloatPtr:
    """ Return pointer to output logits for the $i$-th token in the last batch."""
    if Int(ctx) == 0:
        return null_ptr[Float32]()
    var vocab_size = max(1, ctx[].vocab_size)
    return ctx[].logits.unsafe_offset(Int(i) * vocab_size)

@always_inline
def llama_get_embeddings(ctx: llama_context_p) -> FloatPtr:
    """ Return pointer to normalized embeddings."""
    if Int(ctx) == 0:
        return null_ptr[Float32]()
    return ctx[].embeddings

# ─── KV Cache API ────────────────────────────────────────────────────────────

def llama_kv_cache_clear(ctx: llama_context_p):
    if Int(ctx) != 0:
        ctx[].kv_cache.clear()

def llama_kv_cache_seq_rm(ctx: llama_context_p, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) -> Bool:
    if Int(ctx) == 0:
        return False
    return ctx[].kv_cache.seq_rm(seq_id, p0, p1)

def llama_kv_cache_seq_cp(ctx: llama_context_p, seq_id_src: llama_seq_id, seq_id_dst: llama_seq_id, p0: llama_pos, p1: llama_pos):
    if Int(ctx) != 0:
        ctx[].kv_cache.seq_cp(seq_id_src, seq_id_dst, p0, p1)

def llama_kv_cache_seq_keep(ctx: llama_context_p, seq_id: llama_seq_id):
    if Int(ctx) != 0:
        ctx[].kv_cache.seq_keep(seq_id)

def llama_kv_cache_seq_add(ctx: llama_context_p, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos, delta: llama_pos):
    if Int(ctx) != 0:
        ctx[].kv_cache.seq_add(seq_id, p0, p1, delta)

# ─── Model Metadata & Architecture Reflection ────────────────────────────────

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

# ─── LoRA Adapter Ingestion and Online Merging (Phase 8) ─────────────────────

def llama_adapter_lora_apply(
        model: llama_model_p,
        adapter_path: String,
        scale: Float32 = 1.0,
    ) -> Int32:
    """ Load a LoRA adapter GGUF and apply dynamic additive weights to model."""
    if Int(model) == 0:
        return -1
    try:
        var adapter = llama_lora_adapter(adapter_path)
        var alpha = adapter.alpha
        var rank = adapter.rank
        var eff_scale = scale * adapter.scale
        var multiplier = (alpha / Float32(rank)) * eff_scale

        for i in range(len(adapter.pairs)):
            var pair = adapter.pairs[i]
            var tensor_name = pair.name

            # Check if this tensor matches output or token_embd
            if tensor_name == "token_embd.weight":
                if model[].token_embd_type == GGMLType.F32:
                    var f32_ptr = model[].token_embd.unsafe_bitcast[Float32]()
                    var elems = pair.b_rows * pair.a_cols
                    var delta = unsafe_alloc[Float32](elems)
                    compute_lora_delta(pair, delta, multiplier)
                    for j in range(elems):
                        f32_ptr[unsafe_offset=j] += delta[unsafe_offset=j]
                    delta.unsafe_free()
            elif tensor_name == "output.weight":
                if model[].output_weight_type == GGMLType.F32:
                    var f32_ptr = model[].output_weight.unsafe_bitcast[Float32]()
                    var elems = pair.b_rows * pair.a_cols
                    var delta = unsafe_alloc[Float32](elems)
                    compute_lora_delta(pair, delta, multiplier)
                    for j in range(elems):
                        f32_ptr[unsafe_offset=j] += delta[unsafe_offset=j]
                    delta.unsafe_free()

            # Check layer weights: blk.{l}.{sub_tensor}
            for l in range(len(model[].layers)):
                var pfx = "blk." + String(l) + "."
                if tensor_name.startswith(pfx):
                    var pfx_len = pfx.byte_length()
                    var t_len = tensor_name.byte_length()
                    var sub_span = tensor_name[byte=pfx_len : t_len]
                    var sub = String(sub_span)
                    var target_ptr: BytePtr = null_ptr[UInt8]()
                    var dtype = model[].layers[l].dtype

                    if sub == "attn_q.weight":
                        target_ptr = model[].layers[l].wq
                    elif sub == "attn_k.weight":
                        target_ptr = model[].layers[l].wk
                    elif sub == "attn_v.weight":
                        target_ptr = model[].layers[l].wv
                    elif sub == "attn_output.weight":
                        target_ptr = model[].layers[l].wo
                    elif sub == "ffn_gate.weight":
                        target_ptr = model[].layers[l].w_gate
                    elif sub == "ffn_up.weight":
                        target_ptr = model[].layers[l].w_up
                    elif sub == "ffn_down.weight":
                        target_ptr = model[].layers[l].w_down

                    if Int(target_ptr) != 0 and dtype == GGMLType.F32:
                        var f32_ptr = target_ptr.unsafe_bitcast[Float32]()
                        var elems = pair.b_rows * pair.a_cols
                        var delta = unsafe_alloc[Float32](elems)
                        compute_lora_delta(pair, delta, multiplier)
                        for j in range(elems):
                            f32_ptr[unsafe_offset=j] += delta[unsafe_offset=j]
                        delta.unsafe_free()

        adapter.free()
        return 0
    except:
        return -1
