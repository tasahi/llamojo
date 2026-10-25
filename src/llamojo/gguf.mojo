""" Pure Mojo GGUF v2/v3 Parser, POSIX `mmap` Tensor Binder, and GGUF Writer.

Synthesizes and upgrades `MojoLlama/src/mojollama/mojo_engine/gguf_scanner.mojo`,
`gguf_extract.mojo`, and `gguf/reader.mojo` into pure Mojo 1.1.0 with:
  - Zero Python dependencies (`PythonObject` completely eliminated)
  - Full KV metadata parsing (including string/numeric arrays for vocabulary)
  - Direct zero-copy POSIX `mmap` tensor binding (with heap fallback)
  - Non-owning `llama2.Matrix` view construction for FP32 tensors
"""

from std.collections import List, Dict
from std.ffi import external_call
from std.memory import Pointer, bitcast, unsafe_memcpy, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc

from .types import (
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_VERSION_V2,
    GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType,
    GGMLType,
    ggml_type_size,
    ggml_block_size,
    ggml_type_size_in_bytes,
    ggml_type_name,
    ModelConfig,
    null_ptr,
)
from .backend.threadpool import Matrix, BufferPtrFloat32

# ─── POSIX Constants for File I/O and mmap ───────────────────────────────────

comptime O_RDONLY: Int32 = 0
comptime O_WRONLY: Int32 = 1
comptime O_CREAT: Int32 = 64
comptime O_TRUNC: Int32 = 512
comptime MODE_0644: Int32 = 420

comptime SEEK_SET: Int32 = 0
comptime SEEK_CUR: Int32 = 1
comptime SEEK_END: Int32 = 2

comptime PROT_READ: Int32 = 1
comptime MAP_SHARED: Int32 = 1
comptime MAP_PRIVATE: Int32 = 2
comptime MAP_FAILED: Int = -1

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

# ─── Low-Level Binary Readers & Alignment Helpers ────────────────────────────

@always_inline
def align_offset(offset: UInt64, alignment: UInt64) -> UInt64:
    """Align `offset` upward to the next multiple of `alignment`."""
    if alignment == 0:
        return offset
    return ((offset + alignment - 1) // alignment) * alignment

@always_inline
def _cstr_alloc(s: String) -> BytePtr:
    """Allocate a null-terminated C string from a Mojo `String`."""
    var blen = s.byte_length()
    var buf = unsafe_alloc[UInt8](blen + 1)
    var src = s.unsafe_ptr()
    for i in range(blen):
        buf[unsafe_offset=i] = src[unsafe_offset=i]
    buf[unsafe_offset=blen] = UInt8(0)
    return buf

@always_inline
def rd_u8(d: BytePtr, off: Int) -> UInt8:
    return d[unsafe_offset=off]

@always_inline
def rd_i8(d: BytePtr, off: Int) -> Int8:
    return bitcast[DType.int8](d[unsafe_offset=off])

@always_inline
def rd_u16(d: BytePtr, off: Int) -> UInt16:
    return UInt16(d[unsafe_offset=off]) | (UInt16(d[unsafe_offset=off + 1]) << 8)

@always_inline
def rd_i16(d: BytePtr, off: Int) -> Int16:
    return bitcast[DType.int16](rd_u16(d, off))

@always_inline
def rd_u32(d: BytePtr, off: Int) -> UInt32:
    return (
        UInt32(d[unsafe_offset=off])
        | (UInt32(d[unsafe_offset=off + 1]) << 8)
        | (UInt32(d[unsafe_offset=off + 2]) << 16)
        | (UInt32(d[unsafe_offset=off + 3]) << 24)
    )

@always_inline
def rd_i32(d: BytePtr, off: Int) -> Int32:
    return bitcast[DType.int32](rd_u32(d, off))

@always_inline
def rd_f32(d: BytePtr, off: Int) -> Float32:
    return Float32(bitcast[DType.float32](rd_u32(d, off)))

@always_inline
def rd_u64(d: BytePtr, off: Int) -> UInt64:
    var lo = UInt64(rd_u32(d, off))
    var hi = UInt64(rd_u32(d, off + 4))
    return lo | (hi << 32)

@always_inline
def rd_i64(d: BytePtr, off: Int) -> Int64:
    return bitcast[DType.int64](rd_u64(d, off))

@always_inline
def rd_f64(d: BytePtr, off: Int) -> Float64:
    return Float64(bitcast[DType.float64](rd_u64(d, off)))

@always_inline
def rd_string(d: BytePtr, mut pos: Int, max_len: Int) -> String:
    """Read a GGUF length-prefixed UTF-8 string (uint64 byte length + bytes)."""
    if pos + 8 > max_len:
        return String("")
    var slen = Int(rd_u64(d, pos))
    pos += 8
    if slen <= 0 or pos + slen > max_len:
        if slen > 0:
            pos = min(max_len, pos + slen)
        return String("")
    var s = String(unsafe_from_utf8=Span(unsafe_ptr=d.unsafe_offset(pos), length=slen))
    pos += slen
    return s

@always_inline
def scalar_type_byte_size(vtype: UInt32) -> Int:
    """Return byte size of a fixed-width GGUF scalar type (0 if variable)."""
    if vtype == GGUFValueType.UINT8 or vtype == GGUFValueType.INT8 or vtype == GGUFValueType.BOOL:
        return 1
    elif vtype == GGUFValueType.UINT16 or vtype == GGUFValueType.INT16:
        return 2
    elif vtype == GGUFValueType.UINT32 or vtype == GGUFValueType.INT32 or vtype == GGUFValueType.FLOAT32:
        return 4
    elif vtype == GGUFValueType.UINT64 or vtype == GGUFValueType.INT64 or vtype == GGUFValueType.FLOAT64:
        return 8
    return 0

# ─── GGUF Metadata Value Variant & Tensor Descriptor ─────────────────────────

struct GGUFValue(Copyable, Movable):
    """Typed GGUF metadata value supporting scalars and arrays."""
    var vtype: UInt32
    var int_val: Int64
    var float_val: Float64
    var bool_val: Bool
    var str_val: String
    var arr_type: UInt32
    var str_arr: List[String]
    var int_arr: List[Int64]
    var float_arr: List[Float32]

    def __init__(out self):
        self.vtype = GGUFValueType.UINT32
        self.int_val = 0
        self.float_val = 0.0
        self.bool_val = False
        self.str_val = ""
        self.arr_type = 0
        self.str_arr = List[String]()
        self.int_arr = List[Int64]()
        self.float_arr = List[Float32]()

    @staticmethod
    def from_int(vtype: UInt32, val: Int64) -> Self:
        var v = Self()
        v.vtype = vtype
        v.int_val = val
        v.float_val = Float64(val)
        v.bool_val = val != 0
        return v^

    @staticmethod
    def from_float(vtype: UInt32, val: Float64) -> Self:
        var v = Self()
        v.vtype = vtype
        v.float_val = val
        v.int_val = Int64(val)
        return v^

    @staticmethod
    def from_bool(val: Bool) -> Self:
        var v = Self()
        v.vtype = GGUFValueType.BOOL
        v.bool_val = val
        v.int_val = 1 if val else 0
        return v^

    @staticmethod
    def from_string(val: String) -> Self:
        var v = Self()
        v.vtype = GGUFValueType.STRING
        v.str_val = val
        return v^

struct GGUFTensorInfo(Copyable, Movable):
    """Descriptor and bound memory pointer for a single GGUF tensor."""
    var name: String
    var n_dims: Int
    var dims: List[UInt64]
    var dtype: UInt32
    var offset: UInt64          # Offset relative to GGUF tensor data section start
    var size_bytes: UInt64      # Computed byte footprint
    var n_elements: UInt64      # Product of dimensions
    var data_ptr: BytePtr       # Direct pointer into mmap region or heap buffer

    def __init__(
        out self,
        name: String,
        var dims: List[UInt64],
        dtype: UInt32,
        offset: UInt64,
    ):
        self.name = name
        self.n_dims = len(dims)
        var ne: UInt64 = 1
        for i in range(len(dims)):
            ne *= dims[i]
        self.dims = dims^
        self.dtype = dtype
        self.offset = offset
        self.n_elements = ne
        self.size_bytes = ggml_type_size_in_bytes(dtype, ne)
        self.data_ptr = null_ptr[UInt8]()

    def as_matrix(self) -> Matrix:
        """Return a non-owning `llama2.Matrix` view over an FP32 tensor."""
        var m_dims = List[Int]()
        # GGUF stores dimensions in column-major order (ne[0] is innermost contiguous row width).
        # Reverse dimensions when mehr than 1D so Matrix[row, col] indexes row * ne[0] + col.
        if self.n_dims == 1:
            m_dims.append(Int(self.dims[0]))
        else:
            for i in range(self.n_dims - 1, -1, -1):
                m_dims.append(Int(self.dims[i]))
        var f32_ptr = self.data_ptr.unsafe_bitcast[Float32]()
        return Matrix(f32_ptr, m_dims^)

# ─── GGUFContext: Zero-Copy In-Place Parser & mmap Loader ────────────────────

struct GGUFContext(Movable):
    """Parsed GGUF file context with metadata KV map and mmap-bound tensors."""
    var path: String
    var version: UInt32
    var alignment: UInt64
    var file_size: UInt64
    var data_offset: UInt64
    var metadata: Dict[String, GGUFValue]
    var tensors: List[GGUFTensorInfo]
    var tensor_map: Dict[String, Int]
    var raw_base: BytePtr
    var is_mmapped: Bool
    var is_mlocked: Bool
    var owns_memory: Bool
    var fd: Int32

    def __init__(out self):
        self.path = ""
        self.version = GGUF_VERSION
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.file_size = 0
        self.data_offset = 0
        self.metadata = Dict[String, GGUFValue]()
        self.tensors = List[GGUFTensorInfo]()
        self.tensor_map = Dict[String, Int]()
        self.raw_base = null_ptr[UInt8]()
        self.is_mmapped = False
        self.is_mlocked = False
        self.owns_memory = False
        self.fd = -1

    def get_string(self, key: String, default: String = "") -> String:
        var opt = self.metadata.find(key)
        if opt:
            return opt.value().str_val
        return default

    def get_int(self, key: String, default: Int = 0) -> Int:
        var opt = self.metadata.find(key)
        if opt:
            return Int(opt.value().int_val)
        return default

    def get_float(self, key: String, default: Float32 = 0.0) -> Float32:
        var opt = self.metadata.find(key)
        if opt:
            return Float32(opt.value().float_val)
        return default

    def get_bool(self, key: String, default: Bool = False) -> Bool:
        var opt = self.metadata.find(key)
        if opt:
            return opt.value().bool_val
        return default

    def has_key(self, key: String) -> Bool:
        return key in self.metadata

    def has_tensor(self, name: String) -> Bool:
        return name in self.tensor_map

    def get_tensor(self, name: String) raises -> GGUFTensorInfo:
        var idx_opt = self.tensor_map.find(name)
        if not idx_opt:
            raise Error("Tensor not found in GGUF: " + name)
        return self.tensors[idx_opt.value()].copy()

    def get_tensor_ptr(self, name: String) -> BytePtr:
        var idx_opt = self.tensor_map.find(name)
        if not idx_opt:
            return null_ptr[UInt8]()
        return self.tensors[idx_opt.value()].data_ptr

    def total_tensor_bytes(self) -> UInt64:
        var total: UInt64 = 0
        for i in range(len(self.tensors)):
            total += self.tensors[i].size_bytes
        return total

    def total_parameters(self) -> UInt64:
        var total: UInt64 = 0
        for i in range(len(self.tensors)):
            total += self.tensors[i].n_elements
        return total

    def build_model_config(self) -> ModelConfig:
        """Populate a runtime `ModelConfig` from GGUF metadata keys."""
        var cfg = ModelConfig()
        var arch = self.get_string("general.architecture", "llama")
        cfg.arch = arch
        cfg.name = self.get_string("general.name", "unknown")
        cfg.hidden_dim = self.get_int(arch + ".embedding_length", 0)
        cfg.intermediate_dim = self.get_int(arch + ".feed_forward_length", 0)
        cfg.n_heads = self.get_int(arch + ".attention.head_count", 0)
        cfg.n_kv_heads = self.get_int(arch + ".attention.head_count_kv", cfg.n_heads)
        var key_len = self.get_int(arch + ".attention.key_length", 0)
        if key_len > 0:
            cfg.head_dim = key_len
        elif cfg.n_heads > 0:
            cfg.head_dim = cfg.hidden_dim // cfg.n_heads
        cfg.n_layers = self.get_int(arch + ".block_count", 0)
        cfg.max_seq_len = self.get_int(arch + ".context_length", 2048)
        cfg.rope_theta = self.get_float(arch + ".rope.freq_base", 10000.0)
        cfg.rope_freq_scale = self.get_float(arch + ".rope.scaling.factor", 1.0)
        cfg.norm_eps = self.get_float(arch + ".attention.layer_norm_rms_epsilon", 1e-5)
        cfg.n_experts = self.get_int(arch + ".expert_count", 0)
        cfg.n_experts_used = self.get_int(arch + ".expert_used_count", 0)

        # Determine vocab_size from tokenizer.ggml.tokens array or {arch}.vocab_size
        var tok_opt = self.metadata.find("tokenizer.ggml.tokens")
        if tok_opt:
            cfg.vocab_size = len(tok_opt.value().str_arr)
        else:
            cfg.vocab_size = self.get_int(arch + ".vocab_size", 0)

        # Infer primary weight quantization type from first 2D weight tensor
        cfg.quant_type = GGMLType.F32
        for i in range(len(self.tensors)):
            if self.tensors[i].n_dims >= 2:
                cfg.quant_type = self.tensors[i].dtype
                break
        return cfg^

    def free(mut self):
        """Unmap or free backing memory and close file descriptor."""
        if self.owns_memory and Int(self.raw_base) != 0:
            if self.is_mmapped:
                _ = external_call["munmap", Int32](self.raw_base, Int(self.file_size))
            else:
                self.raw_base.unsafe_free()
            self.raw_base = null_ptr[UInt8]()
            self.owns_memory = False
            self.is_mmapped = False
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1

def _parse_kv_value(d: BytePtr, mut p: Int, max_len: Int, vtype: UInt32) raises -> GGUFValue:
    """Parse a single GGUF KV value at byte offset `p`."""
    if vtype == GGUFValueType.UINT8:
        var v = Int64(rd_u8(d, p))
        p += 1
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.INT8:
        var v = Int64(rd_i8(d, p))
        p += 1
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.UINT16:
        var v = Int64(rd_u16(d, p))
        p += 2
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.INT16:
        var v = Int64(rd_i16(d, p))
        p += 2
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.UINT32:
        var v = Int64(rd_u32(d, p))
        p += 4
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.INT32:
        var v = Int64(rd_i32(d, p))
        p += 4
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.FLOAT32:
        var v = Float64(rd_f32(d, p))
        p += 4
        return GGUFValue.from_float(vtype, v)
    elif vtype == GGUFValueType.BOOL:
        var b = rd_u8(d, p) != 0
        p += 1
        return GGUFValue.from_bool(b)
    elif vtype == GGUFValueType.STRING:
        var s = rd_string(d, p, max_len)
        return GGUFValue.from_string(s)
    elif vtype == GGUFValueType.UINT64:
        var v = bitcast[DType.int64](rd_u64(d, p))
        p += 8
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.INT64:
        var v = rd_i64(d, p)
        p += 8
        return GGUFValue.from_int(vtype, v)
    elif vtype == GGUFValueType.FLOAT64:
        var v = rd_f64(d, p)
        p += 8
        return GGUFValue.from_float(vtype, v)
    elif vtype == GGUFValueType.ARRAY:
        var arr_type = rd_u32(d, p)
        p += 4
        var arr_len = Int(rd_u64(d, p))
        p += 8
        var val = GGUFValue()
        val.vtype = GGUFValueType.ARRAY
        val.arr_type = arr_type
        if arr_type == GGUFValueType.STRING:
            for _ in range(arr_len):
                val.str_arr.append(rd_string(d, p, max_len))
        elif arr_type == GGUFValueType.FLOAT32:
            for _ in range(arr_len):
                val.float_arr.append(rd_f32(d, p))
                p += 4
        elif arr_type == GGUFValueType.FLOAT64:
            for _ in range(arr_len):
                val.float_arr.append(Float32(rd_f64(d, p)))
                p += 8
        else:
            var esz = scalar_type_byte_size(arr_type)
            if esz == 0:
                raise Error("Unsupported nested GGUF array element type: " + String(arr_type))
            for _ in range(arr_len):
                var item: Int64 = 0
                if esz == 1:
                    item = Int64(rd_u8(d, p))
                elif esz == 2:
                    item = Int64(rd_u16(d, p))
                elif esz == 4:
                    item = Int64(rd_i32(d, p))
                elif esz == 8:
                    item = rd_i64(d, p)
                val.int_arr.append(item)
                p += esz
        return val^
    else:
        raise Error("Unknown GGUF value type: " + String(vtype))

def gguf_load_from_file(
    path: String,
    use_mmap: Bool = True,
    use_mlock: Bool = False,
    vocab_only: Bool = False,
    check_tensors: Bool = True,
) raises -> GGUFContext:
    """Parse a GGUF v2/v3 file and bind all tensor pointers in pure Mojo."""
    var cpath = _cstr_alloc(path)
    var fd = external_call["open64", Int32](cpath, O_RDONLY)
    cpath.unsafe_free()
    if fd < 0:
        raise Error("Failed to open GGUF file: " + path)

    var fsz = external_call["lseek64", Int64](fd, Int64(0), SEEK_END)
    _ = external_call["lseek64", Int64](fd, Int64(0), SEEK_SET)
    if fsz < 24:
        _ = external_call["close", Int32](fd)
        raise Error("File too small to be a valid GGUF file: " + path)

    var ctx = GGUFContext()
    ctx.path = path
    ctx.file_size = UInt64(fsz)
    ctx.fd = fd

    # Map the entire file via POSIX mmap if requested, with heap read fallback
    var mapped_ok = False
    if use_mmap:
        var addr = external_call["mmap64", Int](
            Int(0),
            Int(fsz),
            PROT_READ,
            MAP_PRIVATE,
            fd,
            Int64(0),
        )
        if addr != MAP_FAILED and addr != 0:
            ctx.raw_base = BytePtr(unsafe_from_address=addr)
            ctx.is_mmapped = True
            ctx.owns_memory = True
            mapped_ok = True
            if use_mlock:
                var ml = external_call["mlock", Int32](ctx.raw_base, Int(fsz))
                ctx.is_mlocked = ml == 0

    if not mapped_ok:
        var buf = unsafe_alloc[UInt8](Int(fsz))
        var total_read: Int64 = 0
        while total_read < fsz:
            var chunk = min(Int64(67108864), fsz - total_read)
            var r = external_call["pread64", Int64](
                fd,
                buf.unsafe_offset(Int(total_read)),
                chunk,
                total_read,
            )
            if r <= 0:
                buf.unsafe_free()
                _ = external_call["close", Int32](fd)
                raise Error("Unexpected EOF while reading GGUF file: " + path)
            total_read += r
        ctx.raw_base = buf
        ctx.is_mmapped = False
        ctx.owns_memory = True
        if use_mlock:
            var ml = external_call["mlock", Int32](ctx.raw_base, Int(fsz))
            ctx.is_mlocked = ml == 0

    var d = ctx.raw_base
    var max_len = Int(fsz)

    # 1. Verify 24-byte GGUF Header
    var magic = rd_u32(d, 0)
    if magic != GGUF_MAGIC:
        ctx.free()
        raise Error("Invalid GGUF magic: " + String(magic))

    var version = rd_u32(d, 4)
    if version != GGUF_VERSION and version != GGUF_VERSION_V2:
        ctx.free()
        raise Error("Unsupported GGUF version: " + String(version))
    ctx.version = version

    var n_tensors = Int(rd_u64(d, 8))
    var n_kv = Int(rd_u64(d, 16))

    # 2. Parse KV Metadata Section
    var p: Int = 24
    for _ in range(n_kv):
        if p + 12 > max_len:
            ctx.free()
            raise Error("Truncated GGUF KV section in: " + path)
        var key = rd_string(d, p, max_len)
        var vtype = rd_u32(d, p)
        p += 4
        var val = _parse_kv_value(d, p, max_len, vtype)
        ctx.metadata[key] = val^

    # Check custom alignment
    var align_val = ctx.get_int("general.alignment", Int(GGUF_DEFAULT_ALIGNMENT))
    if align_val > 0:
        ctx.alignment = UInt64(align_val)

    # 3. Parse Tensor Info Directory
    for ti in range(n_tensors):
        if p + 16 > max_len:
            ctx.free()
            raise Error("Truncated GGUF tensor directory in: " + path)
        var tname = rd_string(d, p, max_len)
        var n_dims = Int(rd_u32(d, p))
        p += 4
        var dims = List[UInt64]()
        for _ in range(n_dims):
            dims.append(rd_u64(d, p))
            p += 8
        var dtype = rd_u32(d, p)
        p += 4
        var toffset = rd_u64(d, p)
        p += 8

        var tinfo = GGUFTensorInfo(tname, dims^, dtype, toffset)
        ctx.tensor_map[tname] = ti
        ctx.tensors.append(tinfo^)

    # 4. Compute aligned Tensor Data Offset and bind zero-copy pointers
    ctx.data_offset = align_offset(UInt64(p), ctx.alignment)

    if not vocab_only:
        for ti in range(len(ctx.tensors)):
            var abs_off = ctx.data_offset + ctx.tensors[ti].offset
            var end_off = abs_off + ctx.tensors[ti].size_bytes
            if check_tensors and end_off > ctx.file_size:
                var bad_name = ctx.tensors[ti].name
                ctx.free()
                raise Error(
                    "Tensor '"
                    + bad_name
                    + "' exceeds GGUF file bounds ("
                    + String(end_off)
                    + " > "
                    + String(ctx.file_size)
                    + ")"
                )
            ctx.tensors[ti].data_ptr = d.unsafe_offset(Int(abs_off))

    return ctx^

# ─── Synthetic GGUF v3 File Writer (for Testing & Quantization Output) ───────

@always_inline
def _append_u32(mut buf: List[UInt8], v: UInt32):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))

@always_inline
def _append_u64(mut buf: List[UInt8], v: UInt64):
    _append_u32(buf, UInt32(v & 0xFFFFFFFF))
    _append_u32(buf, UInt32((v >> 32) & 0xFFFFFFFF))

@always_inline
def _append_f32(mut buf: List[UInt8], v: Float32):
    _append_u32(buf, bitcast[DType.uint32](v))

@always_inline
def _append_string(mut buf: List[UInt8], s: String):
    var blen = s.byte_length()
    _append_u64(buf, UInt64(blen))
    var ptr = s.unsafe_ptr()
    for i in range(blen):
        buf.append(ptr[unsafe_offset=i])

def write_synthetic_gguf(
    path: String,
    arch: String = "llama",
    name: String = "tiny-test-llama",
    hidden_dim: Int = 64,
    intermediate_dim: Int = 128,
    n_heads: Int = 4,
    n_kv_heads: Int = 2,
    n_layers: Int = 2,
    vocab_size: Int = 32,
) raises:
    """Write a complete, spec-compliant GGUF v3 model file for testing."""
    var buf = List[UInt8]()

    # Header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
    _append_u32(buf, GGUF_MAGIC)
    _append_u32(buf, GGUF_VERSION)
    _append_u64(buf, 3)  # 3 tensors: token_embd.weight, blk.0.attn_norm.weight, output.weight
    _append_u64(buf, 10) # 10 metadata KV entries

    # KV 1: general.architecture (STRING)
    _append_string(buf, "general.architecture")
    _append_u32(buf, GGUFValueType.STRING)
    _append_string(buf, arch)

    # KV 2: general.name (STRING)
    _append_string(buf, "general.name")
    _append_u32(buf, GGUFValueType.STRING)
    _append_string(buf, name)

    # KV 3: {arch}.embedding_length (UINT32)
    _append_string(buf, arch + ".embedding_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(hidden_dim))

    # KV 4: {arch}.feed_forward_length (UINT32)
    _append_string(buf, arch + ".feed_forward_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(intermediate_dim))

    # KV 5: {arch}.attention.head_count (UINT32)
    _append_string(buf, arch + ".attention.head_count")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(n_heads))

    # KV 6: {arch}.attention.head_count_kv (UINT32)
    _append_string(buf, arch + ".attention.head_count_kv")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(n_kv_heads))

    # KV 7: {arch}.block_count (UINT32)
    _append_string(buf, arch + ".block_count")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(n_layers))

    # KV 8: {arch}.context_length (UINT32)
    _append_string(buf, arch + ".context_length")
    _append_u32(buf, GGUFValueType.UINT32)
    _append_u32(buf, UInt32(512))

    # KV 9: {arch}.attention.layer_norm_rms_epsilon (FLOAT32)
    _append_string(buf, arch + ".attention.layer_norm_rms_epsilon")
    _append_u32(buf, GGUFValueType.FLOAT32)
    _append_f32(buf, 1e-5)

    # KV 10: tokenizer.ggml.tokens (ARRAY of STRING)
    _append_string(buf, "tokenizer.ggml.tokens")
    _append_u32(buf, GGUFValueType.ARRAY)
    _append_u32(buf, GGUFValueType.STRING)
    _append_u64(buf, UInt64(vocab_size))
    for tok_i in range(vocab_size):
        _append_string(buf, "tok_" + String(tok_i))

    # Tensor 0: token_embd.weight [hidden_dim, vocab_size] F32
    var t0_elems = UInt64(hidden_dim * vocab_size)
    var t0_bytes = ggml_type_size_in_bytes(GGMLType.F32, t0_elems)
    var t0_off: UInt64 = 0
    _append_string(buf, "token_embd.weight")
    _append_u32(buf, 2)
    _append_u64(buf, UInt64(hidden_dim))
    _append_u64(buf, UInt64(vocab_size))
    _append_u32(buf, GGMLType.F32)
    _append_u64(buf, t0_off)

    # Tensor 1: blk.0.attn_norm.weight [hidden_dim] F32
    var t1_elems = UInt64(hidden_dim)
    var t1_bytes = ggml_type_size_in_bytes(GGMLType.F32, t1_elems)
    var t1_off = align_offset(t0_off + t0_bytes, GGUF_DEFAULT_ALIGNMENT)
    _append_string(buf, "blk.0.attn_norm.weight")
    _append_u32(buf, 1)
    _append_u64(buf, UInt64(hidden_dim))
    _append_u32(buf, GGMLType.F32)
    _append_u64(buf, t1_off)

    # Tensor 2: output.weight [hidden_dim, vocab_size] Q8_0
    var t2_elems = UInt64(hidden_dim * vocab_size)
    var t2_bytes = ggml_type_size_in_bytes(GGMLType.Q8_0, t2_elems)
    var t2_off = align_offset(t1_off + t1_bytes, GGUF_DEFAULT_ALIGNMENT)
    _append_string(buf, "output.weight")
    _append_u32(buf, 2)
    _append_u64(buf, UInt64(hidden_dim))
    _append_u64(buf, UInt64(vocab_size))
    _append_u32(buf, GGMLType.Q8_0)
    _append_u64(buf, t2_off)

    # Align to 32-byte boundary for tensor data section
    var data_start = align_offset(UInt64(len(buf)), GGUF_DEFAULT_ALIGNMENT)
    while UInt64(len(buf)) < data_start:
        buf.append(UInt8(0))

    # Populate Tensor 0 (F32): deterministic values `0.25 * (i + 1)`
    for i in range(Int(t0_elems)):
        _append_f32(buf, Float32(i + 1) * 0.25)
    while UInt64(len(buf)) < data_start + t1_off:
        buf.append(UInt8(0))

    # Populate Tensor 1 (F32): all 1.0f
    for _ in range(Int(t1_elems)):
        _append_f32(buf, 1.0)
    while UInt64(len(buf)) < data_start + t2_off:
        buf.append(UInt8(0))

    # Populate Tensor 2 (Q8_0): 34 bytes per 32 elements
    for b in range(Int(t2_bytes)):
        buf.append(UInt8(b & 0x7F))

    # Write file via Mojo stdlib file I/O
    with open(path, "w") as f:
        f.write_bytes(buf)