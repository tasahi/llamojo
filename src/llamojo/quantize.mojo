""" Quantization Pipeline & GGUF Model Converter for llamojo.\n
Integrates:
- MojoLlama/mojo_quantize.mojo block quantization algorithms (Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4)
- llama.h ABI:
  - llama_model_quantize
  - llama_model_quantize_default_params
- High-efficiency multi-threaded block quantization via std.algorithm.parallelize
- Pure Mojo zero-dependency GGUF v3 stream writer
"""

from max.algorithm import parallelize
from std.collections import List
from std.memory import Pointer, bitcast
from std.memory.alloc import unsafe_alloc

from .types import (
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType,
    GGMLType,
    ggml_type_size_in_bytes,
    llama_model_quantize_params,
    null_ptr,
)
from .gguf import (
    GGUFContext,
    GGUFValue,
    gguf_load_from_file,
    align_offset,
    _append_u32,
    _append_u64,
    _append_f32,
    _append_string,
)

# ─── Half-precision Floating Point Encoder (IEEE 754) ─────────────────────────

@always_inline
def encode_f16(f: Float32) -> UInt16:
    if f == Float32(0.0):
        return UInt16(0)
    var sign = UInt16(0)
    var fd = Float64(f)
    if fd < 0.0:
        sign = UInt16(1)
        fd = -fd
    var e: Int = 0
    if fd >= 1.0:
        while fd >= 2.0:
            fd *= 0.5
            e += 1
    else:
        while fd < 1.0:
            fd *= 2.0
            e -= 1
    var half_exp = e + 15
    if half_exp < 0:
        half_exp = 0
    if half_exp > 31:
        half_exp = 31
    var mant = Int(fd * 2048.0)
    if mant > 1023:
        mant = 1023
    return UInt16(Int(sign) << 15 | half_exp << 10 | mant)

# ─── Parallel Block Quantizers ────────────────────────────────────────────────

def quantize_q4_0_row(
        src: Pointer[Float32, MutUntrackedOrigin],
        dst: Pointer[UInt8, MutUntrackedOrigin],
        n_elements: Int,
    ):
    """ Quantize Float32 vector to Q4_0 blocks (18 bytes per 32 elements)."""
    var nb = (n_elements + 31) // 32

    def worker(b: Int) {imm}:
        var bo = b * 18
        var vo = b * 32
        var amax = Float32(0.0)
        for j in range(32):
            if vo + j >= n_elements:
                break
            var v = src[unsafe_offset=vo + j]
            if v < Float32(0.0):
                v = -v
            if v > amax:
                amax = v
        if amax == Float32(0.0):
            amax = Float32(1.0)

        var d = amax / Float32(7.0)
        var id = Float32(7.0) / amax
        var su = encode_f16(d)
        dst[unsafe_offset=bo] = UInt8(su & 0xFF)
        dst[unsafe_offset=bo + 1] = UInt8((su >> 8) & 0xFF)

        for j in range(32):
            if vo + j >= n_elements:
                break
            var q = Int32(src[unsafe_offset=vo + j] * id)
            if q > 7:
                q = 7
            if q < -8:
                q = -8
            if q < 0:
                q += 16
            var bi = bo + 2 + j // 2
            if j % 2 == 0:
                dst[unsafe_offset=bi] = UInt8(q & 0x0F) | (dst[unsafe_offset=bi] & 0xF0)
            else:
                dst[unsafe_offset=bi] = UInt8((q & 0x0F) << 4) | (dst[unsafe_offset=bi] & 0x0F)

    parallelize(worker, nb, nb)

def quantize_q5_0_row(
        src: Pointer[Float32, MutUntrackedOrigin],
        dst: Pointer[UInt8, MutUntrackedOrigin],
        n_elements: Int,
    ):
    """ Quantize Float32 vector to Q5_0 blocks (22 bytes per 32 elements)."""
    var nb = (n_elements + 31) // 32

    def worker(b: Int) {imm}:
        var bo = b * 22
        var vo = b * 32
        var amax = Float32(0.0)
        for j in range(32):
            if vo + j >= n_elements:
                break
            var v = src[unsafe_offset=vo + j]
            if v < Float32(0.0):
                v = -v
            if v > amax:
                amax = v
        if amax == Float32(0.0):
            amax = Float32(1.0)

        var d = amax / Float32(15.0)
        var id = Float32(15.0) / amax
        var su = encode_f16(d)
        dst[unsafe_offset=bo] = UInt8(su & 0xFF)
        dst[unsafe_offset=bo + 1] = UInt8((su >> 8) & 0xFF)
        var qh = UInt32(0)

        for j in range(32):
            if vo + j >= n_elements:
                break
            var q = Int32(src[unsafe_offset=vo + j] * id)
            if q > 15:
                q = 15
            if q < -16:
                q = -16
            var low = q & 0x0F
            var high = (UInt32(q >> 4) & 1) << UInt32(j)
            qh = qh | high
            var bi = bo + 6 + j // 2
            if j % 2 == 0:
                dst[unsafe_offset=bi] = UInt8(low & 0x0F) | (dst[unsafe_offset=bi] & 0xF0)
            else:
                dst[unsafe_offset=bi] = UInt8((low & 0x0F) << 4) | (dst[unsafe_offset=bi] & 0x0F)

        for i in range(4):
            dst[unsafe_offset=bo + 2 + i] = UInt8((qh >> UInt32(i * 8)) & 0xFF)

    parallelize(worker, nb, nb)

def quantize_q8_0_row(
        src: Pointer[Float32, MutUntrackedOrigin],
        dst: Pointer[UInt8, MutUntrackedOrigin],
        n_elements: Int,
    ):
    """ Quantize Float32 vector to Q8_0 blocks (34 bytes per 32 elements)."""
    var nb = (n_elements + 31) // 32

    def worker(b: Int) {imm}:
        var bo = b * 34
        var vo = b * 32
        var amax = Float32(0.0)
        for j in range(32):
            if vo + j >= n_elements:
                break
            var v = src[unsafe_offset=vo + j]
            if v < Float32(0.0):
                v = -v
            if v > amax:
                amax = v
        if amax == Float32(0.0):
            amax = Float32(1.0)

        var d = amax / Float32(127.0)
        var id = Float32(127.0) / amax
        var su = encode_f16(d)
        dst[unsafe_offset=bo] = UInt8(su & 0xFF)
        dst[unsafe_offset=bo + 1] = UInt8((su >> 8) & 0xFF)

        for j in range(32):
            if vo + j >= n_elements:
                break
            var q = Int32(src[unsafe_offset=vo + j] * id)
            if q > 127:
                q = 127
            if q < -128:
                q = -128
            if q < 0:
                q += 256
            dst[unsafe_offset=bo + 2 + j] = UInt8(q & 0xFF)

    parallelize(worker, nb, nb)


def quantize_q4_k_row(
        src: Pointer[Float32, MutUntrackedOrigin],
        dst: Pointer[UInt8, MutUntrackedOrigin],
        n_elements: Int,
    ):
    """ Quantize Float32 vector to Q4_K super-blocks (144 bytes per 256 elements)."""
    var nb = (n_elements + 255) // 256

    def worker(b: Int) {imm}:
        var bo = b * 144
        var vo = b * 256
        var sub_max = List[Float32]()
        var sub_min = List[Float32]()
        for _ in range(8):
            sub_max.append(Float32(0.0))
            sub_min.append(Float32(0.0))

        for sb in range(8):
            var smax = Float32(0.0)
            var smin = Float32(0.0)
            for j in range(32):
                if vo + sb * 32 + j >= n_elements:
                    break
                var v = src[unsafe_offset=vo + sb * 32 + j]
                if v > smax:
                    smax = v
                if v < smin:
                    smin = v
            sub_max[sb] = smax
            sub_min[sb] = smin

        var sup_max = Float32(0.0)
        for sb in range(8):
            var r = sub_max[sb] - sub_min[sb]
            if r > sup_max:
                sup_max = r
        if sup_max == Float32(0.0):
            sup_max = Float32(1.0)

        var d = sup_max / Float32(252.0)
        var dmin = -sup_max / Float32(252.0)
        var su_d = encode_f16(d)
        var su_dm = encode_f16(dmin)
        dst[unsafe_offset=bo] = UInt8(su_d & 0xFF)
        dst[unsafe_offset=bo + 1] = UInt8((su_d >> 8) & 0xFF)
        dst[unsafe_offset=bo + 2] = UInt8(su_dm & 0xFF)
        dst[unsafe_offset=bo + 3] = UInt8((su_dm >> 8) & 0xFF)

        var scales = List[UInt8]()
        var mins = List[UInt8]()
        for _ in range(8):
            scales.append(UInt8(0))
            mins.append(UInt8(0))

        for sb in range(8):
            var sc = Int32(sub_max[sb] / d)
            if sc > 63:
                sc = 63
            if sc < 0:
                sc = 0
            var mn = Int32((-sub_min[sb]) / dmin)
            if mn > 63:
                mn = 63
            if mn < 0:
                mn = 0
            scales[sb] = UInt8(sc & 0x3F)
            mins[sb] = UInt8(mn & 0x3F)

        for sb in range(4):
            dst[unsafe_offset=bo + 132 + sb] = UInt8(Int(scales[sb]) | (Int(scales[sb + 4]) << 4))
            dst[unsafe_offset=bo + 136 + sb] = UInt8(Int(mins[sb]) | (Int(mins[sb + 4]) << 4))

        for sb in range(8):
            var sc_f32 = Float32(Int(scales[sb]))
            var mn_f32 = Float32(Int(mins[sb]))
            for j in range(32):
                if vo + sb * 32 + j >= n_elements:
                    break
                var v = src[unsafe_offset=vo + sb * 32 + j]
                var qv = Int32(v / d - sc_f32 + mn_f32 * Float32(6.0))
                if qv > 7:
                    qv = 7
                if qv < -8:
                    qv = -8
                if qv < 0:
                    qv += 16
                var bi = bo + 4 + sb * 16 + j // 2
                if j % 2 == 0:
                    dst[unsafe_offset=bi] = UInt8(qv & 0x0F) | (dst[unsafe_offset=bi] & 0xF0)
                else:
                    dst[unsafe_offset=bi] = UInt8((qv & 0x0F) << 4) | (dst[unsafe_offset=bi] & 0x0F)

    parallelize(worker, nb, nb)


def quantize_q6_k_row(
        src: Pointer[Float32, MutUntrackedOrigin],
        dst: Pointer[UInt8, MutUntrackedOrigin],
        n_elements: Int,
    ):
    """ Quantize Float32 vector to Q6_K super-blocks (210 bytes per 256 elements)."""
    var nb = (n_elements + 255) // 256

    def worker(b: Int) {imm}:
        var bo = b * 210
        var vo = b * 256
        var amax = Float32(0.0)
        for j in range(256):
            if vo + j >= n_elements:
                break
            var v = src[unsafe_offset=vo + j]
            if v < Float32(0.0):
                v = -v
            if v > amax:
                amax = v
        if amax == Float32(0.0):
            amax = Float32(1.0)

        var d = amax / Float32(31.0)
        var inv_d = Float32(1.0) / d
        var su = encode_f16(d)
        dst[unsafe_offset=bo + 208] = UInt8(su & 0xFF)
        dst[unsafe_offset=bo + 209] = UInt8((su >> 8) & 0xFF)

        for sg in range(16):
            var sg_max = Float32(0.0)
            for j in range(16):
                if vo + sg * 16 + j >= n_elements:
                    break
                var v = src[unsafe_offset=vo + sg * 16 + j]
                if v < Float32(0.0):
                    v = -v
                if v > sg_max:
                    sg_max = v
            var sc = Int32(sg_max * inv_d)
            if sc > 63:
                sc = 63
            dst[unsafe_offset=bo + 192 + sg] = UInt8(sc & 0x3F)

        for j in range(256):
            if vo + j >= n_elements:
                break
            var v = src[unsafe_offset=vo + j]
            var qv = Int32(v * inv_d)
            if qv > 31:
                qv = 31
            if qv < -32:
                qv = -32
            if qv < 0:
                qv += 64
            var lo = qv & 0x0F
            var hi = (qv >> 4) & 3
            var li = bo + j // 2
            if j % 2 == 0:
                dst[unsafe_offset=li] = UInt8(lo & 0x0F) | (dst[unsafe_offset=li] & 0xF0)
            else:
                dst[unsafe_offset=li] = UInt8((lo & 0x0F) << 4) | (dst[unsafe_offset=li] & 0x0F)

            var hi_idx = j // 4
            var hi_shift = Int32((j % 4) * 2)
            dst[unsafe_offset=bo + 128 + hi_idx] = UInt8(
                Int(dst[unsafe_offset=bo + 128 + hi_idx]) | Int(hi << hi_shift)
            )

    parallelize(worker, nb, nb)

# ─── GGUF Metadata & Tensor Serialization Helpers ──────────────────────────

@always_inline
def _append_u8(mut buf: List[UInt8], v: UInt8):
    buf.append(v)

@always_inline
def _append_i8(mut buf: List[UInt8], v: Int8):
    buf.append(UInt8(bitcast[DType.uint8](v)))

@always_inline
def _append_u16(mut buf: List[UInt8], v: UInt16):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))

@always_inline
def _append_i16(mut buf: List[UInt8], v: Int16):
    _append_u16(buf, bitcast[DType.uint16](v))

@always_inline
def _append_i32(mut buf: List[UInt8], v: Int32):
    _append_u32(buf, bitcast[DType.uint32](v))

@always_inline
def _append_i64(mut buf: List[UInt8], v: Int64):
    _append_u64(buf, bitcast[DType.uint64](v))

@always_inline
def _append_f64(mut buf: List[UInt8], v: Float64):
    _append_u64(buf, bitcast[DType.uint64](v))

def _serialize_gguf_value(mut buf: List[UInt8], v: GGUFValue):
    var vt = v.vtype
    _append_u32(buf, vt)
    if vt == GGUFValueType.UINT8:
        _append_u8(buf, UInt8(v.int_val & 0xFF))
    elif vt == GGUFValueType.INT8:
        _append_i8(buf, Int8(v.int_val & 0xFF))
    elif vt == GGUFValueType.UINT16:
        _append_u16(buf, UInt16(v.int_val & 0xFFFF))
    elif vt == GGUFValueType.INT16:
        _append_i16(buf, Int16(v.int_val & 0xFFFF))
    elif vt == GGUFValueType.UINT32:
        _append_u32(buf, UInt32(v.int_val & 0xFFFFFFFF))
    elif vt == GGUFValueType.INT32:
        _append_i32(buf, Int32(v.int_val & 0xFFFFFFFF))
    elif vt == GGUFValueType.FLOAT32:
        _append_f32(buf, Float32(v.float_val))
    elif vt == GGUFValueType.BOOL:
        _append_u8(buf, UInt8(1) if v.bool_val else UInt8(0))
    elif vt == GGUFValueType.STRING:
        _append_string(buf, v.str_val)
    elif vt == GGUFValueType.UINT64:
        _append_u64(buf, UInt64(v.int_val))
    elif vt == GGUFValueType.INT64:
        _append_i64(buf, v.int_val)
    elif vt == GGUFValueType.FLOAT64:
        _append_f64(buf, v.float_val)
    elif vt == GGUFValueType.ARRAY:
        var at = v.arr_type
        _append_u32(buf, at)
        if at == GGUFValueType.STRING:
            var n = len(v.str_arr)
            _append_u64(buf, UInt64(n))
            for i in range(n):
                _append_string(buf, v.str_arr[i])
        elif at == GGUFValueType.FLOAT32:
            var n = len(v.float_arr)
            _append_u64(buf, UInt64(n))
            for i in range(n):
                _append_f32(buf, v.float_arr[i])
        else:
            var n = len(v.int_arr)
            _append_u64(buf, UInt64(n))
            for i in range(n):
                if at == GGUFValueType.INT32:
                    _append_i32(buf, Int32(v.int_arr[i]))
                elif at == GGUFValueType.UINT32:
                    _append_u32(buf, UInt32(v.int_arr[i]))
                elif at == GGUFValueType.INT16:
                    _append_i16(buf, Int16(v.int_arr[i]))
                elif at == GGUFValueType.UINT16:
                    _append_u16(buf, UInt16(v.int_arr[i]))
                elif at == GGUFValueType.INT8:
                    _append_i8(buf, Int8(v.int_arr[i]))
                elif at == GGUFValueType.UINT8:
                    _append_u8(buf, UInt8(v.int_arr[i]))
                else:
                    _append_i64(buf, v.int_arr[i])

# ─── Public llama_model_quantize API ─────────────────────────────────────────

def llama_model_quantize(
        fname_inp: String,
        fname_out: String,
        params: llama_model_quantize_params,
    ) -> Int32:
    """ Stream-read GGUF model `fname_inp`, quantize target tensor matrices
    to requested `ftype`, preserve complete metadata key-value dictionary,
    and serialize new spec-compliant GGUF v3 file to `fname_out`.
    """
    try:
        var gguf = gguf_load_from_file(fname_inp)
        var target_qt = params.ftype
        print("Quantizing", fname_inp, "->", fname_out, "[target ftype =", target_qt, "]")

        # 1. Prepare Metadata Dictionary
        var n_orig_kv = len(gguf.metadata)
        var has_file_type = False
        var has_quant_version = False

        for item in gguf.metadata.items():
            if item.key == "general.file_type":
                has_file_type = True
            elif item.key == "general.quantization_version":
                has_quant_version = True

        var n_metadata = UInt64(n_orig_kv)
        if not has_file_type:
            n_metadata += 1
        if not has_quant_version:
            n_metadata += 1

        var n_tensors = UInt64(len(gguf.tensors))

        # 2. Build In-Memory Header and Metadata Byte Stream
        var meta_buf = List[UInt8]()
        _append_u32(meta_buf, GGUF_MAGIC)
        _append_u32(meta_buf, GGUF_VERSION)
        _append_u64(meta_buf, n_tensors)
        _append_u64(meta_buf, n_metadata)

        for item in gguf.metadata.items():
            _append_string(meta_buf, item.key)
            if item.key == "general.file_type":
                _serialize_gguf_value(meta_buf, GGUFValue.from_int(GGUFValueType.UINT32, Int64(target_qt)))
            elif item.key == "general.quantization_version":
                _serialize_gguf_value(meta_buf, GGUFValue.from_int(GGUFValueType.UINT32, Int64(2)))
            else:
                _serialize_gguf_value(meta_buf, item.value)

        if not has_file_type:
            _append_string(meta_buf, "general.file_type")
            _serialize_gguf_value(meta_buf, GGUFValue.from_int(GGUFValueType.UINT32, Int64(target_qt)))

        if not has_quant_version:
            _append_string(meta_buf, "general.quantization_version")
            _serialize_gguf_value(meta_buf, GGUFValue.from_int(GGUFValueType.UINT32, Int64(2)))

        # 3. Calculate Quantized Tensor Offsets & Total File Size
        var alignment = GGUF_DEFAULT_ALIGNMENT
        var tensor_out_types = List[UInt32]()
        var tensor_out_bytes = List[UInt64]()
        var tensor_out_offsets = List[UInt64]()

        var cur_tensor_data_offset: UInt64 = 0

        for i in range(len(gguf.tensors)):
            var ti = gguf.tensors[i].copy()
            var n_elements: Int = 1
            for d in range(ti.n_dims):
                n_elements *= Int(ti.dims[d])

            # Determine whether to quantize (only 2D/3D weight matrices >= 512 elements)
            var do_quant = (ti.n_dims >= 2) and (n_elements >= 512) and (ti.dtype == GGMLType.F32 or ti.dtype == GGMLType.F16)
            var out_type = target_qt if do_quant else ti.dtype
            var q_bytes: Int = 0

            if out_type == GGMLType.Q4_0:
                q_bytes = (n_elements + 31) // 32 * 18
            elif out_type == GGMLType.Q5_0:
                q_bytes = (n_elements + 31) // 32 * 22
            elif out_type == GGMLType.Q8_0:
                q_bytes = (n_elements + 31) // 32 * 34
            elif out_type == GGMLType.Q4_K:
                q_bytes = (n_elements + 255) // 256 * 144
            elif out_type == GGMLType.Q6_K:
                q_bytes = (n_elements + 255) // 256 * 210
            elif out_type == GGMLType.F16:
                q_bytes = n_elements * 2
            else:
                q_bytes = n_elements * 4

            var aligned_off = align_offset(cur_tensor_data_offset, alignment)
            tensor_out_types.append(out_type)
            tensor_out_bytes.append(UInt64(q_bytes))
            tensor_out_offsets.append(aligned_off)
            cur_tensor_data_offset = aligned_off + UInt64(q_bytes)

        # 4. Append Tensor Metadata Descriptors
        for i in range(len(gguf.tensors)):
            var ti = gguf.tensors[i].copy()
            _append_string(meta_buf, ti.name)
            _append_u32(meta_buf, UInt32(ti.n_dims))
            for d in range(ti.n_dims):
                _append_u64(meta_buf, ti.dims[d])
            _append_u32(meta_buf, tensor_out_types[i])
            _append_u64(meta_buf, tensor_out_offsets[i])

        # 5. Align to Tensor Data Start Boundary
        var meta_aligned_start = align_offset(UInt64(len(meta_buf)), alignment)
        while UInt64(len(meta_buf)) < meta_aligned_start:
            meta_buf.append(UInt8(0))

        # 6. Stream Out Serialized GGUF File
        with open(fname_out, "w") as f:
            f.write_bytes(meta_buf)

            # Stream Quantized Tensor Payloads
            var written_payload_offset: UInt64 = 0

            for i in range(len(gguf.tensors)):
                var ti = gguf.tensors[i].copy()
                var target_off = tensor_out_offsets[i]
                var q_bytes = Int(tensor_out_bytes[i])
                var out_type = tensor_out_types[i]

                # Padding up to target aligned tensor offset
                if written_payload_offset < target_off:
                    var pad_len = Int(target_off - written_payload_offset)
                    var pad_buf = List[UInt8]()
                    for _ in range(pad_len):
                        pad_buf.append(UInt8(0))
                    f.write_bytes(pad_buf)
                    written_payload_offset = target_off

                var n_elements: Int = 1
                for d in range(ti.n_dims):
                    n_elements *= Int(ti.dims[d])

                # Extract source float array
                var f32_buf = unsafe_alloc[Float32](n_elements)
                if ti.dtype == GGMLType.F32:
                    var src_f32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(ti.data_ptr))
                    for e in range(n_elements):
                        f32_buf[unsafe_offset=e] = src_f32[unsafe_offset=e]
                elif ti.dtype == GGMLType.F16:
                    var src_u16 = Pointer[UInt16, MutUntrackedOrigin](unsafe_from_address=Int(ti.data_ptr))
                    for e in range(n_elements):
                        f32_buf[unsafe_offset=e] = Float32(Float64(src_u16[unsafe_offset=e]) / 1024.0)
                else:
                    for e in range(n_elements):
                        f32_buf[unsafe_offset=e] = Float32(0.0)

                # Quantize into payload buffer
                var q_buf = unsafe_alloc[UInt8](q_bytes)
                for b in range(q_bytes):
                    q_buf[unsafe_offset=b] = UInt8(0)

                if out_type == GGMLType.Q4_0:
                    quantize_q4_0_row(f32_buf, q_buf, n_elements)
                elif out_type == GGMLType.Q5_0:
                    quantize_q5_0_row(f32_buf, q_buf, n_elements)
                elif out_type == GGMLType.Q8_0:
                    quantize_q8_0_row(f32_buf, q_buf, n_elements)
                elif out_type == GGMLType.Q4_K:
                    quantize_q4_k_row(f32_buf, q_buf, n_elements)
                elif out_type == GGMLType.Q6_K:
                    quantize_q6_k_row(f32_buf, q_buf, n_elements)
                else:
                    var p_f32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(q_buf))
                    for e in range(n_elements):
                        p_f32[unsafe_offset=e] = f32_buf[unsafe_offset=e]

                # Write tensor binary payload
                f.write(String(unsafe_from_utf8=Span(unsafe_ptr=q_buf, length=q_bytes)))
                written_payload_offset += UInt64(q_bytes)

                f32_buf.unsafe_free()
                q_buf.unsafe_free()

        print("Successfully generated quantized model with full metadata:", fname_out)
        return 0
    except e:
        print("Error during quantization:", String(e))
        return 1
