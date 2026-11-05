""" Pure-Mojo Audio DSP Frontend for Whisper ASR.

Implements the exact Short-Time Fourier Transform (STFT), periodic Hann window,
Cooley-Tukey FFT, Log-Mel filterbank projection, clamping/normalization, and WAV
PCM file loader matching upstream `whisper.cpp`:
- `WHISPER_SAMPLE_RATE = 16000`
- `WHISPER_N_FFT = 400` (25 ms window)
- `WHISPER_HOP_LENGTH = 160` (10 ms step)
- `WHISPER_CHUNK_SIZE = 30` (30-second audio chunks = 480,000 samples = 3,000 Mel frames)
"""

from std.collections import List
from std.ffi import external_call
from std.math import sin, cos, log10
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores
from max.algorithm import parallelize

from ..types import null_ptr
from ..gguf import (
    O_RDONLY,
    SEEK_END,
    PROT_READ,
    MAP_PRIVATE,
    MAP_FAILED,
    _cstr_alloc,
    rd_u16,
    rd_i16,
    rd_u32,
    rd_f32,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime Float64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

comptime WHISPER_SAMPLE_RATE: Int = 16000
comptime WHISPER_N_FFT: Int = 400
comptime WHISPER_HOP_LENGTH: Int = 160
comptime WHISPER_CHUNK_SIZE: Int = 30
comptime WHISPER_N_SAMPLES: Int = 480000  # 30 * 16000
comptime M_PI: Float64 = 3.14159265358979323846


# ─── Mel Filterbank & Mel Spectrogram Structs ────────────────────────────────

struct WhisperFilters(Movable):
    """ Mel filterbank weights (`[n_mel * n_fft]`, where `n_fft = 1 + WHISPER_N_FFT // 2 = 201`)."""
    var n_mel: Int
    var n_fft: Int
    var data: FloatPtr
    var owns_data: Bool

    def __init__(out self):
        self.n_mel = 80
        self.n_fft = 201
        self.data = null_ptr[Float32]()
        self.owns_data = False

    def __init__(out self, n_mel: Int, n_fft: Int, data: FloatPtr, owns_data: Bool = False):
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.data = data
        self.owns_data = owns_data

    def free(mut self):
        if self.owns_data and Int(self.data) != 0:
            self.data.unsafe_free()
            self.data = null_ptr[Float32]()
            self.owns_data = False


struct WhisperMel(Movable):
    """ Log-Mel Spectrogram output buffer (`[n_mel, n_len]` in row-major channel-first layout)."""
    var n_len: Int
    var n_len_org: Int
    var n_mel: Int
    var data: FloatPtr
    var capacity: Int

    def __init__(out self):
        self.n_len = 0
        self.n_len_org = 0
        self.n_mel = 0
        self.data = null_ptr[Float32]()
        self.capacity = 0

    def resize(mut self, n_mel: Int, n_len: Int):
        self.n_mel = n_mel
        self.n_len = n_len
        var needed = n_mel * n_len
        if needed > self.capacity:
            if Int(self.data) != 0:
                self.data.unsafe_free()
            self.data = unsafe_alloc[Float32](needed)
            self.capacity = needed

    def free(mut self):
        if Int(self.data) != 0:
            self.data.unsafe_free()
            self.data = null_ptr[Float32]()
            self.capacity = 0
            self.n_len = 0
            self.n_len_org = 0


# ─── Precomputed Trigonometric & Periodic Hann Window Tables ─────────────────

struct WhisperTrigCache(Movable):
    """ Precomputed sin/cos tables (`SIN_COS_N_COUNT = 400`) and periodic Hann window."""
    var sin_vals: FloatPtr
    var cos_vals: FloatPtr
    var hann_window: FloatPtr

    def __init__(out self):
        self.sin_vals = unsafe_alloc[Float32](WHISPER_N_FFT)
        self.cos_vals = unsafe_alloc[Float32](WHISPER_N_FFT)
        self.hann_window = unsafe_alloc[Float32](WHISPER_N_FFT)

        for i in range(WHISPER_N_FFT):
            var theta = (2.0 * M_PI * Float64(i)) / Float64(WHISPER_N_FFT)
            self.sin_vals[unsafe_offset=i] = Float32(sin(theta))
            self.cos_vals[unsafe_offset=i] = Float32(cos(theta))
            # Periodic Hann window: 0.5 * (1.0 - cos(2*pi*i / length))
            self.hann_window[unsafe_offset=i] = Float32(
                0.5 * (1.0 - cos((2.0 * M_PI * Float64(i)) / Float64(WHISPER_N_FFT)))
            )

    def free(mut self):
        if Int(self.sin_vals) != 0:
            self.sin_vals.unsafe_free()
            self.cos_vals.unsafe_free()
            self.hann_window.unsafe_free()
            self.sin_vals = null_ptr[Float32]()
            self.cos_vals = null_ptr[Float32]()
            self.hann_window = null_ptr[Float32]()


# ─── Naive DFT (Base Case for Odd Lengths N=25) & Cooley-Tukey FFT ───────────

@always_inline
def whisper_dft(
        inp: FloatPtr,
        N: Int,
        out_ptr: FloatPtr,
        sin_vals: FloatPtr,
        cos_vals: FloatPtr,
    ):
    """ Real-to-complex naive DFT for non-even base cases (e.g. N = 25 in 400 = 16 * 25).
    Matches `dft()` in `whisper.cpp:3107`.
    """
    var sin_cos_step = WHISPER_N_FFT // N
    for k in range(N):
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        for n in range(N):
            var idx = (k * n * sin_cos_step) % WHISPER_N_FFT
            var x_n = inp[unsafe_offset=n]
            re += x_n * cos_vals[unsafe_offset=idx]
            im -= x_n * sin_vals[unsafe_offset=idx]
        out_ptr[unsafe_offset=k * 2 + 0] = re
        out_ptr[unsafe_offset=k * 2 + 1] = im


def whisper_fft(
        inp: FloatPtr,
        N: Int,
        out_ptr: FloatPtr,
        sin_vals: FloatPtr,
        cos_vals: FloatPtr,
    ):
    """ Recursive Cooley-Tukey FFT matching `fft()` in `whisper.cpp:3129`.
    Requires `inp` scratch buffer of size `2 * N` floats and `out_ptr` scratch buffer of size `8 * N` floats.
    """
    if N == 1:
        out_ptr[unsafe_offset=0] = inp[unsafe_offset=0]
        out_ptr[unsafe_offset=1] = 0.0
        return

    var half_N = N // 2
    if N - half_N * 2 == 1:
        whisper_dft(inp, N, out_ptr, sin_vals, cos_vals)
        return

    var even = inp.unsafe_offset(N)
    for i in range(half_N):
        even[unsafe_offset=i] = inp[unsafe_offset=2 * i]
    var even_fft = out_ptr.unsafe_offset(2 * N)
    whisper_fft(even, half_N, even_fft, sin_vals, cos_vals)

    var odd = even
    for i in range(half_N):
        odd[unsafe_offset=i] = inp[unsafe_offset=2 * i + 1]
    var odd_fft = even_fft.unsafe_offset(N)
    whisper_fft(odd, half_N, odd_fft, sin_vals, cos_vals)

    var sin_cos_step = WHISPER_N_FFT // N
    for k in range(half_N):
        var idx = k * sin_cos_step
        var re = cos_vals[unsafe_offset=idx]
        var im = -sin_vals[unsafe_offset=idx]

        var re_odd = odd_fft[unsafe_offset=2 * k + 0]
        var im_odd = odd_fft[unsafe_offset=2 * k + 1]

        var re_even = even_fft[unsafe_offset=2 * k + 0]
        var im_even = even_fft[unsafe_offset=2 * k + 1]

        out_ptr[unsafe_offset=2 * k + 0] = re_even + re * re_odd - im * im_odd
        out_ptr[unsafe_offset=2 * k + 1] = im_even + re * im_odd + im * re_odd

        out_ptr[unsafe_offset=2 * (k + half_N) + 0] = re_even - re * re_odd + im * im_odd
        out_ptr[unsafe_offset=2 * (k + half_N) + 1] = im_even - re * im_odd - im * re_odd


# ─── Multi-Threaded Log-Mel Spectrogram Computation ──────────────────────────

def log_mel_spectrogram(
        samples: FloatPtr,
        n_samples: Int,
        filters: WhisperFilters,
        mut mel: WhisperMel,
        workers: Int = 0,
    ):
    """ Compute the Whisper Log-Mel Spectrogram from 16 kHz mono PCM samples.
    Matches `log_mel_spectrogram()` in `whisper.cpp:3239` with exact reflective padding,
    30-second trailing zero padding, power spectrum, Mel filterbank dot product, and clamping.
    """
    var frame_size = WHISPER_N_FFT
    var frame_step = WHISPER_HOP_LENGTH
    var n_mel = filters.n_mel
    var n_fft = filters.n_fft

    var trig = WhisperTrigCache()
    var hann = trig.hann_window
    var sin_vals = trig.sin_vals
    var cos_vals = trig.cos_vals

    # 1. Padding: reflective pad 200 samples on left, 30s + 200 zeros on right
    var stage_1_pad = WHISPER_SAMPLE_RATE * 30
    var stage_2_pad = frame_size // 2
    var total_padded = n_samples + stage_1_pad + stage_2_pad * 2

    var samples_padded = unsafe_alloc[Float32](total_padded)
    unsafe_memset_zero(samples_padded, total_padded)

    if n_samples > 0 and Int(samples) != 0:
        unsafe_memcpy(
            dest=samples_padded.unsafe_offset(stage_2_pad),
            src=samples,
            count=n_samples,
        )
        # Reflective pad at the beginning: reverse copy samples[1 .. 1 + n_reflect]
        var n_reflect = min(stage_2_pad, max(0, n_samples - 1))
        for i in range(n_reflect):
            samples_padded[unsafe_offset=stage_2_pad - 1 - i] = samples[unsafe_offset=1 + i]

    var n_len = (total_padded - frame_size) // frame_step
    var n_len_org = 1 + (n_samples + stage_2_pad - frame_size) // frame_step
    mel.resize(n_mel, n_len)
    mel.n_len_org = n_len_org

    var active_samples = n_samples + stage_2_pad
    var active_frames = min(active_samples // frame_step + 1, n_len)
    var zero_log = Float32(log10(Float64(1e-10)))

    var n_workers = workers if workers > 0 else num_performance_cores()
    if n_workers < 1:
        n_workers = 1

    def worker_fn(ith: Int) {imm}:
        var fft_in = unsafe_alloc[Float32](frame_size * 2)
        var fft_out = unsafe_alloc[Float32](frame_size * 8)

        var i = ith
        while i < active_frames:
            var offset = i * frame_step
            var valid_len = min(frame_size, active_samples - offset)

            for j in range(valid_len):
                fft_in[unsafe_offset=j] = hann[unsafe_offset=j] * samples_padded[unsafe_offset=offset + j]
            for j in range(valid_len, frame_size * 2):
                fft_in[unsafe_offset=j] = 0.0

            whisper_fft(fft_in, frame_size, fft_out, sin_vals, cos_vals)

            # Calculate modulus^2 of complex numbers for bins [0 .. n_fft)
            for j in range(n_fft):
                var re = fft_out[unsafe_offset=2 * j + 0]
                var im = fft_out[unsafe_offset=2 * j + 1]
                fft_out[unsafe_offset=j] = re * re + im * im

            # Mel filterbank dot product + log10
            for j in range(n_mel):
                var f_row = filters.data.unsafe_offset(j * n_fft)
                var sum_val: Float64 = 0.0
                var k = 0
                while k < n_fft - 3:
                    sum_val += (
                        Float64(fft_out[unsafe_offset=k + 0] * f_row[unsafe_offset=k + 0])
                        + Float64(fft_out[unsafe_offset=k + 1] * f_row[unsafe_offset=k + 1])
                        + Float64(fft_out[unsafe_offset=k + 2] * f_row[unsafe_offset=k + 2])
                        + Float64(fft_out[unsafe_offset=k + 3] * f_row[unsafe_offset=k + 3])
                    )
                    k += 4
                while k < n_fft:
                    sum_val += Float64(fft_out[unsafe_offset=k] * f_row[unsafe_offset=k])
                    k += 1

                if sum_val < 1e-10:
                    sum_val = 1e-10
                mel.data[unsafe_offset=j * n_len + i] = Float32(log10(sum_val))

            i += n_workers

        # Fill trailing silence frames with log10(1e-10)
        while i < n_len:
            for j in range(n_mel):
                mel.data[unsafe_offset=j * n_len + i] = zero_log
            i += n_workers

        fft_in.unsafe_free()
        fft_out.unsafe_free()

    if n_workers == 1:
        worker_fn(0)
    else:
        parallelize(worker_fn, n_workers, n_workers)

    # 2. Clamping and normalization: max(val, mmax - 8.0) -> (val + 4.0) / 4.0
    var total_mel_elems = n_mel * n_len
    var mmax: Float64 = -1e20
    for idx in range(total_mel_elems):
        var v = Float64(mel.data[unsafe_offset=idx])
        if v > mmax:
            mmax = v

    mmax -= 8.0
    for idx in range(total_mel_elems):
        var v = Float64(mel.data[unsafe_offset=idx])
        if v < mmax:
            v = mmax
        mel.data[unsafe_offset=idx] = Float32((v + 4.0) / 4.0)

    samples_padded.unsafe_free()
    trig.free()


def log_mel_spectrogram(
        samples_list: List[Float32],
        filters: WhisperFilters,
        mut mel: WhisperMel,
        workers: Int = 0,
    ):
    """ Convenience overload accepting a Mojo `List[Float32]` of 16 kHz PCM samples."""
    var n = len(samples_list)
    var ptr = FloatPtr(unsafe_from_address=Int(samples_list.unsafe_ptr()))
    log_mel_spectrogram(ptr, n, filters, mel, workers)


# ─── Zero-Dependency RIFF WAV Audio Loader ───────────────────────────────────

def load_wav_file(path: String) raises -> List[Float32]:
    """ Read a 16-bit PCM or 32-bit IEEE Float RIFF `.wav` file into a mono Float32 array in `[-1.0, 1.0]`.
    Automatically downmixes multi-channel audio to mono.
    """
    var c_path = _cstr_alloc(path)
    var fd = external_call["open", Int32](c_path, O_RDONLY)
    c_path.unsafe_free()
    if fd < 0:
        raise Error("Failed to open WAV file: " + path)

    var fsz = external_call["lseek64", Int64](fd, Int64(0), SEEK_END)
    if fsz < 44:
        _ = external_call["close", Int32](fd)
        raise Error("File too small to be a valid RIFF WAV: " + path)

    var addr = external_call["mmap64", Int](
        Int(0),
        Int(fsz),
        PROT_READ,
        MAP_PRIVATE,
        fd,
        Int64(0),
    )
    if addr == MAP_FAILED or addr == 0:
        _ = external_call["close", Int32](fd)
        raise Error("Failed to mmap WAV file: " + path)

    var d = BytePtr(unsafe_from_address=addr)
    var max_len = Int(fsz)

    # Check "RIFF" (0x46464952) and "WAVE" (0x45564157)
    var riff_magic = rd_u32(d, 0)
    var wave_magic = rd_u32(d, 8)
    if riff_magic != 0x46464952 or wave_magic != 0x45564157:
        _ = external_call["munmap", Int32](d, max_len)
        _ = external_call["close", Int32](fd)
        raise Error("Invalid RIFF/WAVE header in: " + path)

    var audio_format: Int = 1
    var num_channels: Int = 1
    var bits_per_sample: Int = 16
    var data_offset: Int = 0
    var data_size: Int = 0

    var p: Int = 12
    while p + 8 <= max_len:
        var chunk_id = rd_u32(d, p)
        var chunk_sz = Int(rd_u32(d, p + 4))
        p += 8

        # "fmt " == 0x20746D66
        if chunk_id == 0x20746D66 and p + 16 <= max_len:
            audio_format = Int(rd_u16(d, p))
            num_channels = Int(rd_u16(d, p + 2))
            _ = rd_u32(d, p + 4)  # sample_rate
            bits_per_sample = Int(rd_u16(d, p + 14))
        # "data" == 0x61746164
        elif chunk_id == 0x61746164:
            data_offset = p
            data_size = min(chunk_sz, max_len - p)
            break

        p += chunk_sz
        # RIFF chunks are word-aligned (2 bytes)
        if (chunk_sz & 1) != 0:
            p += 1

    if data_offset == 0 or data_size <= 0:
        _ = external_call["munmap", Int32](d, max_len)
        _ = external_call["close", Int32](fd)
        raise Error("No 'data' chunk found in WAV file: " + path)

    var samples = List[Float32]()
    var bytes_per_sample = bits_per_sample // 8
    var frame_bytes = bytes_per_sample * num_channels
    var n_frames = data_size // frame_bytes
    samples.reserve(n_frames)

    if audio_format == 1 and bits_per_sample == 16:
        var scale = Float32(1.0 / 32768.0)
        for f in range(n_frames):
            var f_off = data_offset + f * frame_bytes
            var acc: Float32 = 0.0
            for c in range(num_channels):
                var pcm16 = rd_i16(d, f_off + c * 2)
                acc += Float32(pcm16) * scale
            samples.append(acc / Float32(num_channels))
    elif audio_format == 3 and bits_per_sample == 32:
        for f in range(n_frames):
            var f_off = data_offset + f * frame_bytes
            var acc: Float32 = 0.0
            for c in range(num_channels):
                acc += rd_f32(d, f_off + c * 4)
            samples.append(acc / Float32(num_channels))
    else:
        _ = external_call["munmap", Int32](d, max_len)
        _ = external_call["close", Int32](fd)
        raise Error(
            "Unsupported WAV format (audio_format="
            + String(audio_format)
            + ", bits="
            + String(bits_per_sample)
            + ")"
        )

    _ = external_call["munmap", Int32](d, max_len)
    _ = external_call["close", Int32](fd)
    return samples^


# ─── Extract Mel Filterbank from Legacy Whisper `.bin` Header ────────────────

def load_filters_from_whisper_bin(path: String) raises -> WhisperFilters:
    """ Read the embedded `whisper_filters` (`n_mel`, `n_fft`, and `n_mel * n_fft` floats)
    directly from a `whisper.cpp` legacy `.bin` model file (`magic == 0x67676d6c`).
    Offset layout in `.bin`:
      - `0..4`: `uint32 magic` (`0x67676d6c`)
      - `4..48`: 11 `int32` `whisper_hparams` (44 bytes)
      - `48..52`: `int32 n_mel`
      - `52..56`: `int32 n_fft`
      - `56..56 + 4*n_mel*n_fft`: `float[n_mel * n_fft]` filterbank weights
    """
    var c_path = _cstr_alloc(path)
    var fd = external_call["open", Int32](c_path, O_RDONLY)
    c_path.unsafe_free()
    if fd < 0:
        raise Error("Failed to open Whisper .bin model file: " + path)

    var fsz = external_call["lseek64", Int64](fd, Int64(0), SEEK_END)
    if fsz < 64:
        _ = external_call["close", Int32](fd)
        raise Error("Whisper .bin file too small: " + path)

    var header_buf = unsafe_alloc[UInt8](56)
    var r = external_call["pread64", Int64](fd, header_buf, Int64(56), Int64(0))
    if r < 56:
        header_buf.unsafe_free()
        _ = external_call["close", Int32](fd)
        raise Error("Failed to read 56-byte header from: " + path)

    var magic = rd_u32(header_buf, 0)
    if magic != 0x67676D6C:
        header_buf.unsafe_free()
        _ = external_call["close", Int32](fd)
        raise Error("Invalid ggml .bin magic (expected 0x67676d6c): " + path)

    var n_mel = Int(rd_u32(header_buf, 48))
    var n_fft = Int(rd_u32(header_buf, 52))
    header_buf.unsafe_free()

    var n_floats = n_mel * n_fft
    var n_bytes = Int64(n_floats * 4)
    var data = unsafe_alloc[Float32](n_floats)
    var r_data = external_call["pread64", Int64](
        fd,
        data.unsafe_bitcast[UInt8](),
        n_bytes,
        Int64(56),
    )
    _ = external_call["close", Int32](fd)

    if r_data < n_bytes:
        data.unsafe_free()
        raise Error("Truncated mel filters in Whisper .bin file: " + path)

    return WhisperFilters(n_mel, n_fft, data, True)
