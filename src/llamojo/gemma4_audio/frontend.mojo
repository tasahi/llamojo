""" Pure-Mojo Audio DSP Frontend for Gemma 4 E2B (`gemma4a`).

Implements the exact `Gemma4AudioFeatureExtractor` / `mtmd_audio_preprocessor_gemma4a`
feature extraction pipeline:
- `GEMMA4_AUDIO_SAMPLE_RATE = 16000` (16 kHz mono Float32 in `[-1.0, 1.0]`)
- `GEMMA4_AUDIO_N_MEL = 128` (HTK Mel filterbank, `0..8000 Hz`, no Slaney area norm)
- `GEMMA4_AUDIO_WINDOW_LEN = 320` (20 ms periodic Hann window, zero-padded to `512`)
- `GEMMA4_AUDIO_N_FFT = 512` (`257` frequency bins, radix-2 Cooley-Tukey FFT)
- `GEMMA4_AUDIO_HOP_LENGTH = 160` (10 ms step)
- `GEMMA4_AUDIO_PAD_LEFT = 160` (semicausal left zero-padding)
- `GEMMA4_AUDIO_CHUNK_SAMPLES = 480000` (30-second chunks -> 2,999 frames -> 750 tokens)
- Magnitude spectrum (`|X|`) + natural logarithm (`ln(max(mel, 0.001))`)
"""

from std.collections import List
from std.math import sin, cos, log, log10, exp, sqrt
from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores
from max.algorithm import parallelize

from ..types import null_ptr
from ..whisper.audio import load_wav_file

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

comptime GEMMA4_AUDIO_SAMPLE_RATE: Int = 16000
comptime GEMMA4_AUDIO_N_MEL: Int = 128
comptime GEMMA4_AUDIO_N_FFT: Int = 512
comptime GEMMA4_AUDIO_N_FFT_BINS: Int = 257  # 1 + 512 // 2
comptime GEMMA4_AUDIO_WINDOW_LEN: Int = 320  # 20 ms
comptime GEMMA4_AUDIO_HOP_LENGTH: Int = 160  # 10 ms
comptime GEMMA4_AUDIO_PAD_LEFT: Int = 160    # WINDOW_LEN // 2
comptime GEMMA4_AUDIO_CHUNK_SECONDS: Int = 30
comptime GEMMA4_AUDIO_CHUNK_SAMPLES: Int = 480000  # 30 * 16000
comptime GEMMA4_AUDIO_MEL_FLOOR: Float64 = 0.001
comptime M_PI: Float64 = 3.14159265358979323846


# ─── Output Mel Buffer ───────────────────────────────────────────────────────

struct Gemma4AudioMel(Movable):
    """ Log-Mel Spectrogram chunk buffer (`[n_mel, n_len]` in row-major order)."""
    var n_mel: Int
    var n_len: Int
    var n_len_org: Int
    var data: FloatPtr
    var capacity: Int

    def __init__(out self):
        self.n_mel = 0
        self.n_len = 0
        self.n_len_org = 0
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
            self.n_mel = 0
            self.n_len = 0
            self.n_len_org = 0


# ─── Precomputed Trigonometric, Window & HTK Mel Filterbank Cache ────────────

@always_inline
def _htk_hz_to_mel(f_hz: Float64) -> Float64:
    return 2595.0 * log10(1.0 + f_hz / 700.0)


@always_inline
def _htk_mel_to_hz(mel: Float64) -> Float64:
    # 10^(mel / 2595) = exp(ln(10) * (mel / 2595))
    comptime LN_10: Float64 = 2.30258509299404568402
    return 700.0 * (exp(LN_10 * (mel / 2595.0)) - 1.0)


struct Gemma4AudioFrontendCache(Movable):
    """ Precomputed sin/cos tables, zero-padded 320-sample periodic Hann window,
    and 128-bin HTK Mel filterbank matrix (`[128, 257]`) for Gemma 4 `gemma4a`.
    """
    var sin_vals: FloatPtr
    var cos_vals: FloatPtr
    var hann_window: FloatPtr
    var mel_filters: FloatPtr
    var n_mel: Int
    var n_fft: Int
    var n_fft_bins: Int
    var sample_rate: Int

    def __init__(out self):
        self.n_mel = GEMMA4_AUDIO_N_MEL
        self.n_fft = GEMMA4_AUDIO_N_FFT
        self.n_fft_bins = GEMMA4_AUDIO_N_FFT_BINS
        self.sample_rate = GEMMA4_AUDIO_SAMPLE_RATE

        self.sin_vals = unsafe_alloc[Float32](self.n_fft)
        self.cos_vals = unsafe_alloc[Float32](self.n_fft)
        self.hann_window = unsafe_alloc[Float32](self.n_fft)
        self.mel_filters = unsafe_alloc[Float32](self.n_mel * self.n_fft_bins)

        # 1. Sin/Cos table for N_FFT = 512
        for i in range(self.n_fft):
            var theta = (2.0 * M_PI * Float64(i)) / Float64(self.n_fft)
            self.sin_vals[unsafe_offset=i] = Float32(sin(theta))
            self.cos_vals[unsafe_offset=i] = Float32(cos(theta))

        # 2. Periodic Hann window of length 320, zero-padded to 512
        unsafe_memset_zero(self.hann_window, self.n_fft)
        for i in range(GEMMA4_AUDIO_WINDOW_LEN):
            var w = 0.5 - 0.5 * cos((2.0 * M_PI * Float64(i)) / Float64(GEMMA4_AUDIO_WINDOW_LEN))
            self.hann_window[unsafe_offset=i] = Float32(w)

        # 3. HTK Mel filterbank (0..8000 Hz, no Slaney area normalization)
        unsafe_memset_zero(self.mel_filters, self.n_mel * self.n_fft_bins)
        var fmin: Float64 = 0.0
        var fmax: Float64 = Float64(self.sample_rate) * 0.5
        var m_lo = _htk_hz_to_mel(fmin)
        var m_hi = _htk_hz_to_mel(fmax)

        var n_pts = self.n_mel + 2
        var hz_pts = unsafe_alloc[Float64](n_pts)
        for i in range(n_pts):
            var m_pt = m_lo + (m_hi - m_lo) * (Float64(i) / Float64(self.n_mel + 1))
            hz_pts[unsafe_offset=i] = _htk_mel_to_hz(m_pt)

        var bin_hz_step = Float64(self.sample_rate) / Float64(self.n_fft)
        for m in range(self.n_mel):
            var f_left = hz_pts[unsafe_offset=m]
            var f_center = hz_pts[unsafe_offset=m + 1]
            var f_right = hz_pts[unsafe_offset=m + 2]

            var denom_l = f_center - f_left
            if denom_l < 1e-30:
                denom_l = 1e-30
            var denom_r = f_right - f_center
            if denom_r < 1e-30:
                denom_r = 1e-30

            var row_off = m * self.n_fft_bins
            for k in range(self.n_fft_bins):
                var f = Float64(k) * bin_hz_step
                var w: Float64 = 0.0
                if f >= f_left and f <= f_center:
                    w = (f - f_left) / denom_l
                elif f > f_center and f <= f_right:
                    w = (f_right - f) / denom_r
                self.mel_filters[unsafe_offset=row_off + k] = Float32(w)

        hz_pts.unsafe_free()

    def free(mut self):
        if Int(self.sin_vals) != 0:
            self.sin_vals.unsafe_free()
            self.cos_vals.unsafe_free()
            self.hann_window.unsafe_free()
            self.mel_filters.unsafe_free()
            self.sin_vals = null_ptr[Float32]()
            self.cos_vals = null_ptr[Float32]()
            self.hann_window = null_ptr[Float32]()
            self.mel_filters = null_ptr[Float32]()


# ─── 512-Point Radix-2 Cooley-Tukey FFT & Reference DFT ──────────────────────

def gemma4_dft_512(
        inp: FloatPtr,
        out_ptr: FloatPtr,
        sin_vals: FloatPtr,
        cos_vals: FloatPtr,
    ):
    """ Naive real-to-complex 512-point DFT for test verification.
    Writes `512` interleaved `(re, im)` Float32 pairs (`1024` floats) to `out_ptr`.
    """
    comptime N = GEMMA4_AUDIO_N_FFT
    for k in range(N):
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        for n in range(N):
            var idx = (k * n) & (N - 1)
            var x_n = inp[unsafe_offset=n]
            re += x_n * cos_vals[unsafe_offset=idx]
            im -= x_n * sin_vals[unsafe_offset=idx]
        out_ptr[unsafe_offset=2 * k + 0] = re
        out_ptr[unsafe_offset=2 * k + 1] = im


def gemma4_fft_512(
        inp: FloatPtr,
        out_ptr: FloatPtr,
        sin_vals: FloatPtr,
        cos_vals: FloatPtr,
    ):
    """ Iterative in-place radix-2 Cooley-Tukey FFT of length 512 (zero heap allocations).
    Reads `512` real Float32 samples from `inp` and writes `512` interleaved `(re, im)` pairs
    (`1024` Float32 values) to `out_ptr`.
    """
    comptime N = GEMMA4_AUDIO_N_FFT

    # Bit-reversal permutation copy (9 bits for N = 512)
    for i in range(N):
        var rev = 0
        var x = i
        for _ in range(9):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        out_ptr[unsafe_offset=2 * rev + 0] = inp[unsafe_offset=i]
        out_ptr[unsafe_offset=2 * rev + 1] = 0.0

    # 9 radix-2 butterfly stages (m = 2, 4, 8, ..., 512)
    var m = 2
    while m <= N:
        var half_m = m >> 1
        var step = N // m
        var k = 0
        while k < N:
            for j in range(half_m):
                var idx = j * step
                var wr = cos_vals[unsafe_offset=idx]
                var wi = -sin_vals[unsafe_offset=idx]

                var even_idx = 2 * (k + j)
                var odd_idx = 2 * (k + j + half_m)

                var u_re = out_ptr[unsafe_offset=even_idx + 0]
                var u_im = out_ptr[unsafe_offset=even_idx + 1]
                var v_re = out_ptr[unsafe_offset=odd_idx + 0]
                var v_im = out_ptr[unsafe_offset=odd_idx + 1]

                var t_re = wr * v_re - wi * v_im
                var t_im = wr * v_im + wi * v_re

                out_ptr[unsafe_offset=even_idx + 0] = u_re + t_re
                out_ptr[unsafe_offset=even_idx + 1] = u_im + t_im
                out_ptr[unsafe_offset=odd_idx + 0] = u_re - t_re
                out_ptr[unsafe_offset=odd_idx + 1] = u_im - t_im
            k += m
        m <<= 1


# ─── Frame Count, Chunking & Multi-Threaded Log-Mel Computation ──────────────

@always_inline
def gemma4_audio_pt_frames(chunk_len: Int) -> Int:
    """ Compute exact PyTorch `unfold(size=frame_length+1, step=hop)` frame count
    after semicausal left-padding (`pad_left = 160`).
    """
    var n_with_left = chunk_len + GEMMA4_AUDIO_PAD_LEFT
    if n_with_left < GEMMA4_AUDIO_WINDOW_LEN + 1:
        return 0
    return (n_with_left - (GEMMA4_AUDIO_WINDOW_LEN + 1)) // GEMMA4_AUDIO_HOP_LENGTH + 1


@always_inline
def gemma4_audio_n_tokens_for_frames(n_frames: Int) -> Int:
    """ Compute Conformer output token count after two stride-2 `Conv2D(k=3, p=1)` layers.
    E.g. 2,999 frames (30 s) -> 1,500 -> 750 tokens.
    """
    if n_frames <= 0:
        return 0
    var n = (n_frames - 1) // 2 + 1
    return (n - 1) // 2 + 1


def gemma4_audio_chunk_mel(
        samples: FloatPtr,
        chunk_len: Int,
        cache: Gemma4AudioFrontendCache,
        mut out_mel: Gemma4AudioMel,
        workers: Int = 0,
    ) -> Bool:
    """ Compute the `[128, pt_frames]` Gemma 4 HTK Log-Mel spectrogram for a single <=30 s chunk."""
    if chunk_len <= 0 or Int(samples) == 0:
        return False

    var pt_frames = gemma4_audio_pt_frames(chunk_len)
    if pt_frames <= 0:
        return False

    var n_mel = cache.n_mel
    var fft_size = cache.n_fft
    var n_fft_bins = cache.n_fft_bins
    var hop = GEMMA4_AUDIO_HOP_LENGTH
    var pad_left = GEMMA4_AUDIO_PAD_LEFT

    var n_padded_needed = (pt_frames - 1) * hop + fft_size
    var total_pad = n_padded_needed - chunk_len
    if total_pad < pad_left:
        total_pad = pad_left

    var total_samples = total_pad + chunk_len
    var padded_samples = unsafe_alloc[Float32](total_samples)
    unsafe_memset_zero(padded_samples, total_samples)
    unsafe_memcpy(
        dest=padded_samples.unsafe_offset(pad_left),
        src=samples,
        count=chunk_len,
    )

    out_mel.resize(n_mel, pt_frames)
    out_mel.n_len_org = chunk_len

    var hann = cache.hann_window
    var sin_vals = cache.sin_vals
    var cos_vals = cache.cos_vals
    var filters = cache.mel_filters
    var mel_out_ptr = out_mel.data

    var n_workers = workers if workers > 0 else num_performance_cores()
    if n_workers < 1:
        n_workers = 1
    if n_workers > pt_frames:
        n_workers = pt_frames

    def worker_fn(ith: Int) {imm}:
        var fft_in = unsafe_alloc[Float32](fft_size)
        var fft_out = unsafe_alloc[Float32](fft_size * 2)

        var i = ith
        while i < pt_frames:
            var offset = i * hop
            var valid_len = fft_size
            var rem = total_samples - offset
            if rem < valid_len:
                valid_len = rem
            var win_len = GEMMA4_AUDIO_WINDOW_LEN
            if valid_len < win_len:
                win_len = valid_len

            # Apply 320-sample periodic Hann window (hann[320..511] == 0.0)
            for j in range(win_len):
                fft_in[unsafe_offset=j] = hann[unsafe_offset=j] * padded_samples[unsafe_offset=offset + j]
            for j in range(win_len, fft_size):
                fft_in[unsafe_offset=j] = 0.0

            gemma4_fft_512(fft_in, fft_out, sin_vals, cos_vals)

            # Magnitude spectrum |X| for bins [0..256]
            for j in range(n_fft_bins):
                var re = fft_out[unsafe_offset=2 * j + 0]
                var im = fft_out[unsafe_offset=2 * j + 1]
                fft_out[unsafe_offset=j] = sqrt(re * re + im * im)

            # HTK Mel filterbank projection + ln(max(sum, 0.001))
            for m in range(n_mel):
                var f_row = filters.unsafe_offset(m * n_fft_bins)
                var sum_val: Float64 = 0.0
                var k = 0
                while k < n_fft_bins - 3:
                    sum_val += (
                        Float64(fft_out[unsafe_offset=k + 0] * f_row[unsafe_offset=k + 0])
                        + Float64(fft_out[unsafe_offset=k + 1] * f_row[unsafe_offset=k + 1])
                        + Float64(fft_out[unsafe_offset=k + 2] * f_row[unsafe_offset=k + 2])
                        + Float64(fft_out[unsafe_offset=k + 3] * f_row[unsafe_offset=k + 3])
                    )
                    k += 4
                while k < n_fft_bins:
                    sum_val += Float64(fft_out[unsafe_offset=k] * f_row[unsafe_offset=k])
                    k += 1

                if sum_val < GEMMA4_AUDIO_MEL_FLOOR:
                    sum_val = GEMMA4_AUDIO_MEL_FLOOR
                mel_out_ptr[unsafe_offset=m * pt_frames + i] = Float32(log(sum_val))

            i += n_workers

        fft_in.unsafe_free()
        fft_out.unsafe_free()

    if n_workers == 1:
        worker_fn(0)
    else:
        parallelize(worker_fn, n_workers, n_workers)

    padded_samples.unsafe_free()
    return True


def gemma4_audio_preprocess(
        samples: FloatPtr,
        n_samples: Int,
        cache: Gemma4AudioFrontendCache,
        workers: Int = 0,
    ) -> List[Gemma4AudioMel]:
    """ Split 16 kHz mono PCM samples into 30-second chunks and compute Gemma 4 Log-Mel spectrograms."""
    var chunks = List[Gemma4AudioMel]()
    if n_samples <= 0 or Int(samples) == 0:
        return chunks^

    var off = 0
    while off < n_samples:
        var chunk_len = n_samples - off
        if chunk_len > GEMMA4_AUDIO_CHUNK_SAMPLES:
            chunk_len = GEMMA4_AUDIO_CHUNK_SAMPLES
        var mel = Gemma4AudioMel()
        if gemma4_audio_chunk_mel(samples.unsafe_offset(off), chunk_len, cache, mel, workers):
            chunks.append(mel^)
        else:
            mel.free()
        off += GEMMA4_AUDIO_CHUNK_SAMPLES

    return chunks^


def gemma4_audio_preprocess(
        samples_list: List[Float32],
        cache: Gemma4AudioFrontendCache,
        workers: Int = 0,
    ) -> List[Gemma4AudioMel]:
    """ Convenience overload accepting a Mojo `List[Float32]` of 16 kHz mono PCM samples."""
    var n = len(samples_list)
    if n == 0:
        return List[Gemma4AudioMel]()
    var ptr = FloatPtr(unsafe_from_address=Int(samples_list.unsafe_ptr()))
    return gemma4_audio_preprocess(ptr, n, cache, workers)


def gemma4_audio_preprocess_wav(
        path: String,
        cache: Gemma4AudioFrontendCache,
        workers: Int = 0,
    ) raises -> List[Gemma4AudioMel]:
    """ Load a 16 kHz RIFF `.wav` file (16-bit PCM or 32-bit IEEE Float) and compute Gemma 4 Log-Mel chunks."""
    var samples = load_wav_file(path)
    return gemma4_audio_preprocess(samples, cache, workers)
