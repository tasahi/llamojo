""" 1D Convolutional Stem & Positional Embedding Kernels for Audio / Multimodal Encoders.

Implements the exact convolutional frontend and positional embedding fusion required by
OpenAI Whisper (`whisper.cpp` `whisper_build_graph_conv` and `whisper_build_graph_encoder`):
1. `conv1d_f32`: 1D Convolution with arbitrary kernel size K, stride S, padding P,
   supporting both F32 and F16 weight tensors (`GGMLType.F32` / `GGMLType.F16`),
   optional bias addition, and optional inline GELU activation.
2. `conv1d_k3_gelu`: Specialized high-throughput 3-tap (`K=3, P=1, S=1 or S=2`) SIMD kernel
   for Whisper `encoder.conv1` and `encoder.conv2`.
3. `transpose_add_pos_embd`: Transposes the channel-first convolutional output `[C, T]`
   to sequence-first `[T, C]` (where each frame `t` is contiguous of length `C = n_audio_state`)
   while fusing `encoder.positional_embedding` addition in a single pass.
"""

from std.memory import Pointer
from std.sys.info import num_performance_cores
from max.algorithm import parallelize

from ..types import GGMLType
from .isa import SIMD_WIDTH, F32xW, f16_to_f32
from .kernels_norm_rope import gelu

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime UInt16Ptr = Pointer[UInt16, MutUntrackedOrigin]


@always_inline
def _read_conv_weight(
        w_bytes: BytePtr,
        w_dtype: UInt32,
        idx: Int,
    ) -> Float32:
    """ Read a single scalar weight from an F32 or F16 convolution weight tensor."""
    if w_dtype == GGMLType.F16:
        var u16_ptr = w_bytes.unsafe_bitcast[UInt16]()
        return f16_to_f32(u16_ptr[unsafe_offset=idx])
    else:
        var f32_ptr = w_bytes.unsafe_bitcast[Float32]()
        return f32_ptr[unsafe_offset=idx]


def conv1d_f32(
        out_ptr: FloatPtr,
        inp_ptr: FloatPtr,
        w_bytes: BytePtr,
        w_dtype: UInt32,
        bias_ptr: FloatPtr,
        c_in: Int,
        c_out: Int,
        t_in: Int,
        kernel_size: Int = 3,
        stride: Int = 1,
        padding: Int = 1,
        apply_gelu: Bool = True,
        workers: Int = 0,
    ):
    """ General 1D Convolution matching `ggml_conv_1d_ph` + bias + optional GELU.

    Memory layouts (matching `whisper.cpp` GGML 2D/3D row-major contiguous layout):
      - `inp_ptr`: `[c_in, t_in]` where `inp[ic * t_in + t]` is channel `ic` at frame `t`.
      - `w_bytes`: `[c_out, c_in, kernel_size]` (GGML `ne[0]=kernel_size, ne[1]=c_in, ne[2]=c_out`),
        so element `(oc, ic, k)` is at offset `(oc * c_in + ic) * kernel_size + k`.
      - `bias_ptr`: `[c_out]` (nullable if `Int(bias_ptr) == 0`).
      - `out_ptr`: `[c_out, t_out]` where `t_out = (t_in + 2 * padding - kernel_size) // stride + 1`.
    """
    var t_out = (t_in + 2 * padding - kernel_size) // stride + 1
    var has_bias = Int(bias_ptr) != 0
    var n_workers = workers if workers > 0 else num_performance_cores()

    if n_workers <= 1 or c_out < 16:
        for oc in range(c_out):
            var b_val: Float32 = bias_ptr[unsafe_offset=oc] if has_bias else 0.0
            var out_row = out_ptr.unsafe_offset(oc * t_out)
            var w_oc_off = oc * c_in * kernel_size

            for to in range(t_out):
                var t_base = to * stride - padding
                var acc: Float32 = b_val
                for ic in range(c_in):
                    var inp_row = inp_ptr.unsafe_offset(ic * t_in)
                    var w_ic_off = w_oc_off + ic * kernel_size
                    for k in range(kernel_size):
                        var ti = t_base + k
                        if ti >= 0 and ti < t_in:
                            var w_val = _read_conv_weight(w_bytes, w_dtype, w_ic_off + k)
                            acc += w_val * inp_row[unsafe_offset=ti]
                out_row[unsafe_offset=to] = gelu(acc) if apply_gelu else acc
        return

    var chunk = (c_out + n_workers - 1) // n_workers

    def worker_fn(wid: Int) {imm}:
        var oc_start = wid * chunk
        var oc_end = min((wid + 1) * chunk, c_out)
        for oc in range(oc_start, oc_end):
            var b_val: Float32 = bias_ptr[unsafe_offset=oc] if has_bias else 0.0
            var out_row = out_ptr.unsafe_offset(oc * t_out)
            var w_oc_off = oc * c_in * kernel_size

            for to in range(t_out):
                var t_base = to * stride - padding
                var acc: Float32 = b_val
                for ic in range(c_in):
                    var inp_row = inp_ptr.unsafe_offset(ic * t_in)
                    var w_ic_off = w_oc_off + ic * kernel_size
                    for k in range(kernel_size):
                        var ti = t_base + k
                        if ti >= 0 and ti < t_in:
                            var w_val = _read_conv_weight(w_bytes, w_dtype, w_ic_off + k)
                            acc += w_val * inp_row[unsafe_offset=ti]
                out_row[unsafe_offset=to] = gelu(acc) if apply_gelu else acc

    parallelize(worker_fn, n_workers, n_workers)


def conv1d_k3_gelu(
        out_ptr: FloatPtr,
        inp_ptr: FloatPtr,
        w_bytes: BytePtr,
        w_dtype: UInt32,
        bias_ptr: FloatPtr,
        c_in: Int,
        c_out: Int,
        t_in: Int,
        stride: Int = 1,
        workers: Int = 0,
    ):
    """ High-throughput SIMD 1D Convolution specialized for Whisper (`kernel_size=3, padding=1`) + GELU.

    Vectorizes across the time dimension `t_out` using `SIMD_WIDTH` registers for interior frames,
    streaming input channel rows `[t_in]` and accumulating 3-tap weights `(w0, w1, w2)`.
    """
    var t_out = (t_in + 2 - 3) // stride + 1
    var has_bias = Int(bias_ptr) != 0
    var n_workers = workers if workers > 0 else num_performance_cores()

    def process_oc_range(oc_start: Int, oc_end: Int) {imm}:
        for oc in range(oc_start, oc_end):
            var b_val: Float32 = bias_ptr[unsafe_offset=oc] if has_bias else 0.0
            var out_row = out_ptr.unsafe_offset(oc * t_out)
            var w_oc_off = oc * c_in * 3

            # Initialize output row with bias
            var b_vec = F32xW(b_val)
            var t = 0
            while t + SIMD_WIDTH <= t_out:
                out_row.unsafe_store[width=SIMD_WIDTH](t, b_vec)
                t += SIMD_WIDTH
            while t < t_out:
                out_row[unsafe_offset=t] = b_val
                t += 1

            # Accumulate over input channels
            for ic in range(c_in):
                var inp_row = inp_ptr.unsafe_offset(ic * t_in)
                var w_ic_off = w_oc_off + ic * 3
                var w0 = _read_conv_weight(w_bytes, w_dtype, w_ic_off + 0)
                var w1 = _read_conv_weight(w_bytes, w_dtype, w_ic_off + 1)
                var w2 = _read_conv_weight(w_bytes, w_dtype, w_ic_off + 2)

                if stride == 1:
                    # Boundary t_out = 0 (ti = -1, 0, 1)
                    if t_in >= 2:
                        out_row[unsafe_offset=0] += w1 * inp_row[unsafe_offset=0] + w2 * inp_row[unsafe_offset=1]
                    elif t_in == 1:
                        out_row[unsafe_offset=0] += w1 * inp_row[unsafe_offset=0]

                    # Interior [1, t_in - 1) vectorized with SIMD_WIDTH
                    var w0_v = F32xW(w0)
                    var w1_v = F32xW(w1)
                    var w2_v = F32xW(w2)
                    var to = 1
                    while to + SIMD_WIDTH <= t_in - 1:
                        var cur_v = out_row.unsafe_load[width=SIMD_WIDTH](to)
                        var x0_v = inp_row.unsafe_load[width=SIMD_WIDTH](to - 1)
                        var x1_v = inp_row.unsafe_load[width=SIMD_WIDTH](to)
                        var x2_v = inp_row.unsafe_load[width=SIMD_WIDTH](to + 1)
                        out_row.unsafe_store[width=SIMD_WIDTH](to, cur_v + w0_v * x0_v + w1_v * x1_v + w2_v * x2_v)
                        to += SIMD_WIDTH
                    while to < t_in - 1:
                        out_row[unsafe_offset=to] += (
                            w0 * inp_row[unsafe_offset=to - 1]
                            + w1 * inp_row[unsafe_offset=to]
                            + w2 * inp_row[unsafe_offset=to + 1]
                        )
                        to += 1

                    # Boundary t_out = t_in - 1 (ti = t_in - 2, t_in - 1, t_in)
                    if t_in >= 2:
                        out_row[unsafe_offset=t_in - 1] += (
                            w0 * inp_row[unsafe_offset=t_in - 2] + w1 * inp_row[unsafe_offset=t_in - 1]
                        )
                else:
                    # General stride (e.g. stride == 2 in Whisper conv2)
                    for to in range(t_out):
                        var t_base = to * stride - 1
                        var s: Float32 = 0.0
                        if t_base >= 0 and t_base < t_in:
                            s += w0 * inp_row[unsafe_offset=t_base]
                        if t_base + 1 >= 0 and t_base + 1 < t_in:
                            s += w1 * inp_row[unsafe_offset=t_base + 1]
                        if t_base + 2 >= 0 and t_base + 2 < t_in:
                            s += w2 * inp_row[unsafe_offset=t_base + 2]
                        out_row[unsafe_offset=to] += s

            # Apply inline GELU activation across the output row
            for to in range(t_out):
                out_row[unsafe_offset=to] = gelu(out_row[unsafe_offset=to])

    if n_workers <= 1 or c_out < 16:
        process_oc_range(0, c_out)
        return

    var chunk = (c_out + n_workers - 1) // n_workers

    def worker_fn(wid: Int) {imm}:
        var oc_start = wid * chunk
        var oc_end = min((wid + 1) * chunk, c_out)
        if oc_start < oc_end:
            process_oc_range(oc_start, oc_end)

    parallelize(worker_fn, n_workers, n_workers)


@always_inline
def transpose_add_pos_embd(
        out_seq_major: FloatPtr,
        conv_chan_major: FloatPtr,
        pos_embd: FloatPtr,
        n_state: Int,
        n_ctx: Int,
    ):
    """ Transpose channel-first conv output `[n_state, n_ctx]` into sequence-first `[n_ctx, n_state]`
    and add learned/sinusoidal positional embeddings `pos_embd` (`[n_ctx, n_state]`).

    Matches `whisper.cpp` line 2137:
      `cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));`
    where in GGML:
      - `embd_conv` has `ne[0] = n_ctx`, `ne[1] = n_state` (i.e. `[n_state, n_ctx]` in row-major C order).
      - `e_pe` has `ne[0] = n_state`, `ne[1] = n_ctx` (i.e. `[n_ctx, n_state]` in row-major C order,
        so frame `t` is contiguous of length `n_state` at `e_pe + t * n_state`).
      - `out_seq_major` has frame `t` contiguous of length `n_state` at `out_seq_major + t * n_state`.
    """
    var has_pe = Int(pos_embd) != 0
    for t in range(n_ctx):
        var dst_frame = out_seq_major.unsafe_offset(t * n_state)
        if has_pe:
            var pe_frame = pos_embd.unsafe_offset(t * n_state)
            for c in range(n_state):
                dst_frame[unsafe_offset=c] = conv_chan_major[unsafe_offset=c * n_ctx + t] + pe_frame[unsafe_offset=c]
        else:
            for c in range(n_state):
                dst_frame[unsafe_offset=c] = conv_chan_major[unsafe_offset=c * n_ctx + t]

