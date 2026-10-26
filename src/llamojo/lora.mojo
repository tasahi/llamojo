""" Dynamic LoRA Adapter Ingestion and Online Merging for llamojo.\n
Implements Phase 8 of pure-CPU extension plan:
- llama_lora_adapter struct and metadata parsing (GGUF adapter files)
- Extracts low-rank adapter pairs (lora_a, lora_b) and hyperparameters (alpha, rank)
- Scaled additive merging into model weights:
    W_effective = W_base + (alpha / rank) * scale * (A x B)
- Native llama_lora_adapter_init, llama_lora_adapter_set, llama_lora_adapter_free
- Online dynamic adaptation applied to LayerWeights
"""

from std.collections import Dict, List
from std.memory import Pointer, unsafe_memcpy, unsafe_memset_zero, bitcast
from std.memory.alloc import unsafe_alloc

from .types import (
    GGMLType,
    ggml_type_size,
    ggml_block_size,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUFValueType,
    GGUF_DEFAULT_ALIGNMENT,
    null_ptr,
)
from .gguf import (
    BytePtr,
    GGUFContext,
    GGUFTensorInfo,
    gguf_load_from_file,
    align_offset,
    _append_u32,
    _append_u64,
    _append_f32,
    _append_string,
)
from .backend.isa import f16_to_f32

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

# ─── LoRA Tensor Pair ────────────────────────────────────────────────────────

struct LoraTensorPair(ImplicitlyCopyable, Copyable, Movable):
    """ Pair of low-rank matrices A and B for a specific base tensor name."""
    var name: String        # e.g. "blk.0.attn_q.weight" or "token_embd.weight"
    var a_ptr: BytePtr      # Points into adapter GGUF mmap or memory
    var a_type: UInt32      # GGMLType (F32, F16)
    var a_rows: Int         # rank r
    var a_cols: Int         # in_features
    var b_ptr: BytePtr      # Points into adapter GGUF mmap or memory
    var b_type: UInt32      # GGMLType (F32, F16)
    var b_rows: Int         # out_features
    var b_cols: Int         # rank r

    def __init__(
        out self,
        name: String,
        a_ptr: BytePtr,
        a_type: UInt32,
        a_rows: Int,
        a_cols: Int,
        b_ptr: BytePtr,
        b_type: UInt32,
        b_rows: Int,
        b_cols: Int,
    ):
        self.name = name
        self.a_ptr = a_ptr
        self.a_type = a_type
        self.a_rows = a_rows
        self.a_cols = a_cols
        self.b_ptr = b_ptr
        self.b_type = b_type
        self.b_rows = b_rows
        self.b_cols = b_cols

# ─── LoRA Adapter Object ─────────────────────────────────────────────────────

struct llama_lora_adapter(Movable):
    """ Loaded Low-Rank Adaptation (LoRA) checkpoint."""
    var gguf: GGUFContext
    var path: String
    var base_model_path: String
    var alpha: Float32
    var rank: Int
    var scale: Float32
    var pairs: List[LoraTensorPair]

    def __init__(out self, path: String) raises:
        self.path = path
        self.gguf = gguf_load_from_file(path, use_mmap=True, check_tensors=False)
        self.alpha = Float32(1.0)
        self.rank = 8
        self.scale = Float32(1.0)
        self.base_model_path = ""
        self.pairs = List[LoraTensorPair]()

        # Read adapter metadata if present
        var alpha_val = self.gguf.metadata.find("adapter.lora.alpha")
        if alpha_val:
            self.alpha = Float32(alpha_val.value().float_val)

        var base_path_val = self.gguf.metadata.find("general.base_model.path")
        if base_path_val:
            self.base_model_path = base_path_val.value().str_val

        # Discover matching .lora_a and .lora_b tensor pairs
        var a_map = Dict[String, Int]()
        var b_map = Dict[String, Int]()

        for i in range(len(self.gguf.tensors)):
            var tname = self.gguf.tensors[i].name
            var blen = tname.byte_length()
            if tname.endswith(".lora_a") or tname.endswith(".weight.lora_a"):
                var base_str = tname
                if base_str.endswith(".lora_a"):
                    var sub_span = tname[byte=0 : blen - 7]
                    base_str = String(sub_span)
                if not base_str.endswith(".weight"):
                    base_str = base_str + ".weight"
                a_map[base_str] = i
            elif tname.endswith(".lora_b") or tname.endswith(".weight.lora_b"):
                var base_str = tname
                if base_str.endswith(".lora_b"):
                    var sub_span = tname[byte=0 : blen - 7]
                    base_str = String(sub_span)
                if not base_str.endswith(".weight"):
                    base_str = base_str + ".weight"
                b_map[base_str] = i

        # Pair up matching A and B tensors
        for item in a_map.items():
            var base_name = item.key
            var a_idx = item.value
            var b_idx_opt = b_map.find(base_name)
            if b_idx_opt:
                var b_idx = b_idx_opt.value()
                var ta = self.gguf.tensors[a_idx].copy()
                var tb = self.gguf.tensors[b_idx].copy()

                var a_cols = Int(ta.dims[0]) if ta.n_dims >= 1 else 1
                var a_rows = Int(ta.dims[1]) if ta.n_dims >= 2 else 1
                var b_cols = Int(tb.dims[0]) if tb.n_dims >= 1 else 1
                var b_rows = Int(tb.dims[1]) if tb.n_dims >= 2 else 1

                if self.rank == 8 and a_rows > 0:
                    self.rank = a_rows

                var pair = LoraTensorPair(
                    name=base_name,
                    a_ptr=ta.data_ptr,
                    a_type=ta.dtype,
                    a_rows=a_rows,
                    a_cols=a_cols,
                    b_ptr=tb.data_ptr,
                    b_type=tb.dtype,
                    b_rows=b_rows,
                    b_cols=b_cols,
                )
                self.pairs.append(pair)

    def free(mut self):
        self.gguf.free()

comptime llama_lora_adapter_p = Pointer[llama_lora_adapter, MutUntrackedOrigin]

# ─── Public C-ABI and Mojo APIs ──────────────────────────────────────────────

def llama_lora_adapter_init(model_dummy: Pointer[NoneType, MutUntrackedOrigin], path: String) -> llama_lora_adapter_p:
    """ Load and initialize a LoRA adapter from a GGUF file."""
    try:
        var adapter = llama_lora_adapter(path)
        var p = unsafe_alloc[llama_lora_adapter](1)
        p.unsafe_write(adapter^)
        return p
    except:
        return null_ptr[llama_lora_adapter]()

def llama_lora_adapter_set(
        adapter: llama_lora_adapter_p,
        scale: Float32,
    ) -> Int32:
    """ Set scaling factor for the LoRA adapter."""
    if Int(adapter) == 0:
        return -1
    adapter[].scale = scale
    return 0

def llama_lora_adapter_free(adapter: llama_lora_adapter_p):
    """ Deallocate a loaded LoRA adapter."""
    if Int(adapter) != 0:
        adapter[].free()
        adapter.unsafe_deinit_pointee()
        adapter.unsafe_free()

# ─── Online Additive Weight Merging ──────────────────────────────────────────

@always_inline
def _get_elem_f32(ptr: BytePtr, dtype: UInt32, idx: Int) -> Float32:
    """Read element at index `idx` as Float32 from F32 or F16 buffer."""
    if dtype == GGMLType.F32:
        return ptr.unsafe_bitcast[Float32]()[unsafe_offset=idx]
    elif dtype == GGMLType.F16:
        var lo = UInt16(ptr[unsafe_offset=idx * 2])
        var hi = UInt16(ptr[unsafe_offset=idx * 2 + 1])
        return f16_to_f32((hi << 8) | lo)
    return Float32(0.0)

def compute_lora_delta(
        pair: LoraTensorPair,
        out_delta: FloatPtr,
        multiplier: Float32,
    ):
    """ Compute Delta W = multiplier * (B @ A).\n    
    A has shape [r, in_features] (a_rows = r, a_cols = in_features).
    B has shape [out_features, r] (b_rows = out_features, b_cols = r).
    Delta W has shape [out_features, in_features] (flattened row-major).
    """
    var r = pair.a_rows
    var in_dim = pair.a_cols
    var out_dim = pair.b_rows

    if in_dim <= 0 or out_dim <= 0 or r <= 0:
        return

    for i in range(out_dim):
        var out_row_off = i * in_dim
        for j in range(in_dim):
            var sum_val: Float32 = 0.0
            for k in range(r):
                # B[i, k]: row i, col k in B matrix
                var b_idx = i * r + k
                var b_val = _get_elem_f32(pair.b_ptr, pair.b_type, b_idx)

                # A[k, j]: row k, col j in A matrix
                var a_idx = k * in_dim + j
                var a_val = _get_elem_f32(pair.a_ptr, pair.a_type, a_idx)

                sum_val += b_val * a_val

            out_delta[unsafe_offset=out_row_off + j] = sum_val * multiplier

def apply_lora_to_f32_tensor(
        target_f32: FloatPtr,
        n_elements: Int,
        pair: LoraTensorPair,
        alpha: Float32,
        rank: Int,
        scale: Float32,
    ):
    """ Apply in-place additive LoRA adaptation:
        W = W + (alpha / rank) * scale * (B @ A)
    """
    if n_elements <= 0 or rank <= 0 or scale == 0.0:
        return

    var total_dim = pair.b_rows * pair.a_cols
    if total_dim > n_elements:
        total_dim = n_elements

    var multiplier = (alpha / Float32(rank)) * scale

    var delta = unsafe_alloc[Float32](total_dim)
    compute_lora_delta(pair, delta, multiplier)

    for i in range(total_dim):
        target_f32[unsafe_offset=i] += delta[unsafe_offset=i]

    delta.unsafe_free()

# ─── Synthetic LoRA GGUF Writer (for Unit Testing) ───────────────────────────

def write_synthetic_lora_gguf(
        path: String,
        base_name: String,
        in_features: Int,
        out_features: Int,
        rank: Int,
        alpha: Float32 = 16.0,
    ) raises:
    """ Create a minimal valid GGUF LoRA adapter file with A and B tensors."""
    var buf = List[UInt8]()

    # Magic "GGUF" (0x46554747)
    _append_u32(buf, GGUF_MAGIC)
    _append_u32(buf, GGUF_VERSION)

    # Tensor count = 2 (lora_a and lora_b)
    _append_u64(buf, UInt64(2))

    # KV count = 2 (general.type = "adapter", adapter.lora.alpha)
    _append_u64(buf, UInt64(2))

    # KV 0: general.type = "adapter"
    _append_string(buf, "general.type")
    _append_u32(buf, GGUFValueType.STRING)
    _append_string(buf, "adapter")

    # KV 1: adapter.lora.alpha = alpha (FLOAT32)
    _append_string(buf, "adapter.lora.alpha")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, alpha)

    # Tensor 0: {base_name}.lora_a [in_features, rank] F32
    var a_elems = UInt64(in_features * rank)
    var a_bytes = a_elems * 4
    var a_off: UInt64 = 0

    _append_string(buf, base_name + ".lora_a")
    _append_u32(buf, 2)
    _append_u64(buf, UInt64(in_features))
    _append_u64(buf, UInt64(rank))
    _append_u32(buf, GGMLType.F32)
    _append_u64(buf, a_off)

    # Tensor 1: {base_name}.lora_b [rank, out_features] F32
    var b_elems = UInt64(rank * out_features)
    var b_bytes = b_elems * 4
    _ = b_bytes
    var b_off = align_offset(a_off + a_bytes, GGUF_DEFAULT_ALIGNMENT)

    _append_string(buf, base_name + ".lora_b")
    _append_u32(buf, 2)
    _append_u64(buf, UInt64(rank))
    _append_u64(buf, UInt64(out_features))
    _append_u32(buf, GGMLType.F32)
    _append_u64(buf, b_off)

    # Align to 32-byte boundary
    var data_start = align_offset(UInt64(len(buf)), GGUF_DEFAULT_ALIGNMENT)
    while UInt64(len(buf)) < data_start:
        buf.append(UInt8(0))

    # Fill tensor A: all 1.0f
    for _ in range(Int(a_elems)):
        _append_f32(buf, 1.0)

    # Align to b_off
    while UInt64(len(buf)) < data_start + b_off:
        buf.append(UInt8(0))

    # Fill tensor B: all 0.5f
    for _ in range(Int(b_elems)):
        _append_f32(buf, 0.5)

    with open(path, "w") as f:
        f.write_bytes(buf)
