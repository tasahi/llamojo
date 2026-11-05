""" Whisper Timestamp Logit Rules, Full 30-Second Seek Loop, and High-Level Transcription API
(`src/llamojo/whisper/transcribe.mojo`).

Implements Phase 5 of the Whisper ASR Engine in `llamojo`:
1. `WhisperSegment` and `WhisperFullParams` matching `whisper.h`.
2. `apply_whisper_logit_filters`:
   - `suppress_blank` (suppresses EOT and space `" "` at step 0)
   - Special/control/language token suppression (`SOT`, `NOT`, `SOLM`, `NOSP`, `TRANSLATE`, `TRANSCRIBE`, `PREV`)
   - Paired timestamp rules (`ApplyTimestampRules`):
     * Timestamps must appear in pairs (`<|t0|> text ... <|t1|> <|t1|> text ...`)
     * Initial timestamp bounded by `max_initial_ts` (default `1.0s`)
     * Monotonically non-decreasing timestamps (`tid >= seek_delta // 2`)
     * Timestamp log-sum-exp dominance (`timestamp_logprob > max_text_token_logprob`)
3. `whisper_full`: Complete multi-window 30s seek loop (`whisper_pcm_to_mel` -> `whisper_encode` ->
   prompt conditioning -> autoregressive decoding with logit filtering -> segment splitting).
4. `WhisperContext` lifecycle and segment inspection helpers (`whisper_init_from_file`,
   `whisper_full_n_segments`, `whisper_full_get_segment_text`, `whisper_full_get_segment_t0`,
   `whisper_full_get_segment_t1`).
"""

from std.collections import List
from std.math import exp, log
from std.memory import Pointer, unsafe_memcpy

from .audio import WHISPER_CHUNK_SIZE, log_mel_spectrogram, load_wav_file
from .loader import WhisperModel, WhisperVocab, whisper_load_model_from_file
from .model import WhisperState, whisper_encode, whisper_decode, whisper_decode_step

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime NEG_INF: Float32 = -1e30


# ─── Transcription Segment & Parameters ──────────────────────────────────────

struct WhisperSegment(Copyable, Movable):
    """ A decoded speech segment with start/end timestamps in centiseconds (10 ms units)."""
    var t0: Int64
    var t1: Int64
    var text: String
    var tokens: List[Int32]
    var no_speech_prob: Float32

    def __init__(
            out self,
            t0: Int64 = 0,
            t1: Int64 = 0,
            text: String = "",
            no_speech_prob: Float32 = 0.0,
        ):
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.tokens = List[Int32]()
        self.no_speech_prob = no_speech_prob

    def __copyinit__(out self, existing: Self):
        self.t0 = existing.t0
        self.t1 = existing.t1
        self.text = existing.text
        self.tokens = existing.tokens.copy()
        self.no_speech_prob = existing.no_speech_prob


struct WhisperFullParams(Copyable, Movable):
    """ Configuration parameters for `whisper_full` (`struct whisper_full_params`)."""
    var n_threads: Int
    var offset_ms: Int
    var duration_ms: Int
    var translate: Bool
    var no_context: Bool
    var no_timestamps: Bool
    var single_segment: Bool
    var print_special: Bool
    var print_timestamps: Bool
    var max_tokens: Int
    var audio_ctx: Int
    var language: String
    var suppress_blank: Bool
    var suppress_nst: Bool
    var temperature: Float32
    var max_initial_ts: Float32
    var no_speech_thold: Float32

    def __init__(out self):
        self.n_threads = 0
        self.offset_ms = 0
        self.duration_ms = 0
        self.translate = False
        self.no_context = True
        self.no_timestamps = False
        self.single_segment = False
        self.print_special = False
        self.print_timestamps = True
        self.max_tokens = 0
        self.audio_ctx = 0
        self.language = String("en")
        self.suppress_blank = True
        self.suppress_nst = False
        self.temperature = 0.0
        self.max_initial_ts = 1.0
        self.no_speech_thold = 0.6

    def __copyinit__(out self, existing: Self):
        self.n_threads = existing.n_threads
        self.offset_ms = existing.offset_ms
        self.duration_ms = existing.duration_ms
        self.translate = existing.translate
        self.no_context = existing.no_context
        self.no_timestamps = existing.no_timestamps
        self.single_segment = existing.single_segment
        self.print_special = existing.print_special
        self.print_timestamps = existing.print_timestamps
        self.max_tokens = existing.max_tokens
        self.audio_ctx = existing.audio_ctx
        self.language = existing.language
        self.suppress_blank = existing.suppress_blank
        self.suppress_nst = existing.suppress_nst
        self.temperature = existing.temperature
        self.max_initial_ts = existing.max_initial_ts
        self.no_speech_thold = existing.no_speech_thold


def whisper_full_default_params() -> WhisperFullParams:
    """ Return default `WhisperFullParams` matching `whisper_full_default_params(WHISPER_SAMPLING_GREEDY)`."""
    return WhisperFullParams()


# ─── Timestamp Formatting Helper ─────────────────────────────────────────────

def format_timestamp_cs(t_cs: Int64) -> String:
    """ Convert centiseconds (10ms units) to `HH:MM:SS.mmm` format."""
    var msec = Int(t_cs) * 10
    var hr = msec // (1000 * 60 * 60)
    msec -= hr * (1000 * 60 * 60)
    var min_val = msec // (1000 * 60)
    msec -= min_val * (1000 * 60)
    var sec = msec // 1000
    msec -= sec * 1000

    def pad2(v: Int) -> String:
        return ("0" + String(v)) if v < 10 else String(v)

    def pad3(v: Int) -> String:
        if v < 10:
            return "00" + String(v)
        elif v < 100:
            return "0" + String(v)
        return String(v)

    return pad2(hr) + ":" + pad2(min_val) + ":" + pad2(sec) + "." + pad3(msec)


# ─── Whisper Logit Filtering & Paired Timestamp Rules (`whisper_process_logits`) ──

def apply_whisper_logit_filters(
        logits: FloatPtr,
        ref vocab: WhisperVocab,
        n_audio_ctx: Int,
        params: WhisperFullParams,
        tokens_cur: List[Int32],
        has_ts: Bool,
        seek_delta: Int,
        space_token_id: Int = 220,
    ):
    """ Apply Whisper token suppression and `ApplyTimestampRules` in-place on `logits` (`[n_vocab]`).
    Matches `whisper_process_logits` in `whisper.cpp:6380-6562`.
    """
    var n_logits = vocab.n_vocab
    var is_initial = (len(tokens_cur) == 0)
    var eot_id = Int(vocab.token_eot)
    var beg_id = Int(vocab.token_beg)

    # 1. Suppress blank at first step (`token_eot` and `" "` token 220)
    if params.suppress_blank and is_initial:
        logits[unsafe_offset=eot_id] = NEG_INF
        if space_token_id >= 0 and space_token_id < eot_id:
            logits[unsafe_offset=space_token_id] = NEG_INF

    # 2. Suppress control / task / language tokens
    logits[unsafe_offset=Int(vocab.token_not)] = NEG_INF
    logits[unsafe_offset=Int(vocab.token_sot)] = NEG_INF
    logits[unsafe_offset=Int(vocab.token_nosp)] = NEG_INF
    logits[unsafe_offset=Int(vocab.token_solm)] = NEG_INF
    logits[unsafe_offset=Int(vocab.token_translate)] = NEG_INF
    logits[unsafe_offset=Int(vocab.token_transcribe)] = NEG_INF
    logits[unsafe_offset=Int(vocab.token_prev)] = NEG_INF

    # Suppress language tokens [token_sot + 1 .. token_translate - 1]
    for lang_tok in range(Int(vocab.token_sot) + 1, Int(vocab.token_translate)):
        if lang_tok >= 0 and lang_tok < n_logits:
            logits[unsafe_offset=lang_tok] = NEG_INF

    # 3. If `no_timestamps` is True, suppress all timestamp tokens >= token_beg
    if params.no_timestamps:
        for i in range(beg_id, n_logits):
            logits[unsafe_offset=i] = NEG_INF
        return

    # 4. If `max_tokens` reached in timestamped mode, force a timestamp or EOT
    if not params.single_segment and params.max_tokens > 0 and len(tokens_cur) >= params.max_tokens:
        for i in range(0, eot_id):
            logits[unsafe_offset=i] = NEG_INF

    # 5. Paired Timestamp Rules (`ApplyTimestampRules`)
    var n_cur = len(tokens_cur)
    var last_was_timestamp = (n_cur > 0 and Int(tokens_cur[n_cur - 1]) >= beg_id)
    var penultimate_was_timestamp = (n_cur < 2 or Int(tokens_cur[n_cur - 2]) >= beg_id)

    if last_was_timestamp:
        if penultimate_was_timestamp:
            # Two timestamps in a row (or initial timestamp at pos 0) -> next MUST be text or EOT
            for i in range(beg_id, n_logits):
                logits[unsafe_offset=i] = NEG_INF
        else:
            # Closing timestamp after text -> next MUST be a new opening timestamp or EOT
            for i in range(0, eot_id):
                logits[unsafe_offset=i] = NEG_INF

    # 6. Initial timestamp cannot exceed `max_initial_ts`
    if is_initial and params.max_initial_ts > 0.0:
        var precision = Float32(WHISPER_CHUNK_SIZE) / Float32(n_audio_ctx)
        var tid0 = Int(params.max_initial_ts / precision + Float32(0.5))
        var cutoff = beg_id + tid0 + 1
        if cutoff < n_logits:
            for i in range(cutoff, n_logits):
                logits[unsafe_offset=i] = NEG_INF

    # 7. Monotonically non-decreasing timestamps within a window
    if has_ts:
        var tid0 = seek_delta // 2
        var limit = min(beg_id + tid0, n_logits)
        for i in range(beg_id, limit):
            logits[unsafe_offset=i] = NEG_INF

    # 8. Timestamp Log-Sum-Exp Dominance Rule:
    # If the marginal probability over all timestamp tokens exceeds the highest individual
    # text token probability, suppress all text tokens so a timestamp is sampled.
    var max_logit: Float32 = NEG_INF
    for i in range(n_logits):
        var v = logits[unsafe_offset=i]
        if v > max_logit:
            max_logit = v

    if max_logit <= NEG_INF / Float32(2.0):
        return

    var sum_all: Float32 = 0.0
    for i in range(n_logits):
        var v = logits[unsafe_offset=i]
        if v > NEG_INF / Float32(2.0):
            sum_all += exp(v - max_logit)
    var log_Z = log(sum_all) + max_logit

    # Max logprob among text/EOT tokens [0 .. beg_id - 1]
    var max_text_logprob: Float32 = NEG_INF
    for i in range(0, beg_id):
        var v = logits[unsafe_offset=i]
        if v > NEG_INF / Float32(2.0):
            var lp = v - log_Z
            if lp > max_text_logprob:
                max_text_logprob = lp

    # Log-sum-exp among timestamp tokens [beg_id .. n_logits - 1]
    var max_ts_logprob: Float32 = NEG_INF
    for i in range(beg_id, n_logits):
        var v = logits[unsafe_offset=i]
        if v > NEG_INF / Float32(2.0):
            var lp = v - log_Z
            if lp > max_ts_logprob:
                max_ts_logprob = lp

    var timestamp_logprob: Float32 = NEG_INF
    if max_ts_logprob > NEG_INF / Float32(2.0):
        var sum_ts_exp: Float32 = 0.0
        for i in range(beg_id, n_logits):
            var v = logits[unsafe_offset=i]
            if v > NEG_INF / Float32(2.0):
                var lp = v - log_Z
                sum_ts_exp += exp(lp - max_ts_logprob)
        if sum_ts_exp > 0.0:
            timestamp_logprob = log(sum_ts_exp) + max_ts_logprob

    if timestamp_logprob > max_text_logprob:
        for i in range(0, beg_id):
            logits[unsafe_offset=i] = NEG_INF


# ─── Whisper Context & High-Level Transcription Loop (`whisper_full`) ────────

struct WhisperContext(Movable):
    """ High-level Whisper context combining `WhisperModel`, `WhisperState`, and decoded `WhisperSegment`s."""
    var model: WhisperModel
    var state: WhisperState
    var segments: List[WhisperSegment]
    var space_token_id: Int
    var last_cstr: List[UInt8]

    def __init__(out self, var model: WhisperModel):
        var st = WhisperState(model)
        # Find `" "` token ID in vocabulary (typically 220 in GPT-2 BPE)
        var sp_id = 220
        for i in range(Int(model.vocab.token_eot)):
            if model.vocab.id_to_token[i] == " ":
                sp_id = i
                break
        self.model = model^
        self.state = st^
        self.segments = List[WhisperSegment]()
        self.space_token_id = sp_id
        self.last_cstr = List[UInt8]()
        self.last_cstr.append(0)

    def free(mut self):
        self.state.free()
        self.model.free()
        self.segments.clear()


def whisper_init_from_file(path: String) raises -> WhisperContext:
    """ Load a Whisper model (`.bin` or `.gguf`) and initialize its runtime `WhisperContext`."""
    var model = whisper_load_model_from_file(path)
    return WhisperContext(model^)


def whisper_full(
        mut ctx: WhisperContext,
        params: WhisperFullParams,
        samples: FloatPtr,
        n_samples: Int,
    ) -> Int:
    """ Run full end-to-end Whisper ASR transcription on 16 kHz mono PCM `samples[0 .. n_samples - 1]`.
    Populates `ctx.segments` and returns `0` on success.
    Matches `whisper_full_with_state` in `whisper.cpp:6984-7940`.
    """
    ctx.segments.clear()

    if n_samples > 0 and Int(samples) != 0:
        log_mel_spectrogram(
            samples,
            n_samples,
            ctx.model.filters,
            ctx.state.mel,
            workers=params.n_threads,
        )

    var seek_start = params.offset_ms // 10
    var seek_end = (
        ctx.state.mel.n_len_org
        if params.duration_ms == 0
        else seek_start + (params.duration_ms // 10)
    )
    var delta_min = 10  # 100 ms minimum audio length

    if seek_end < seek_start + delta_min:
        return 0

    # Build initial task prompt: [SOT, (lang, task), (NOT)]
    var prompt_init = List[Int32]()
    prompt_init.append(ctx.model.vocab.token_sot)
    if ctx.model.vocab.is_multilingual():
        # Default to English (lang_id = 0 -> token_sot + 1)
        prompt_init.append(ctx.model.vocab.token_sot + 1)
        if params.translate:
            prompt_init.append(ctx.model.vocab.token_translate)
        else:
            prompt_init.append(ctx.model.vocab.token_transcribe)

    if params.no_timestamps:
        prompt_init.append(ctx.model.vocab.token_not)

    var seek = seek_start
    var eot_id = Int(ctx.model.vocab.token_eot)
    var beg_id = Int(ctx.model.vocab.token_beg)
    var n_logits = ctx.model.vocab.n_vocab
    var n_max = ctx.model.hparams.n_text_ctx // 2 - 4

    while seek + delta_min < seek_end:
        # 1. Encode 30-second audio window starting at `seek`
        whisper_encode(ctx.model, ctx.state, seek, params.n_threads)

        # 2. Reset decoder self-attention KV cache and evaluate prompt_init
        ctx.state.reset_kv_self()
        whisper_decode(ctx.model, ctx.state, prompt_init, 0, params.n_threads)

        # Compute no_speech_prob from initial unfiltered logits
        var max_l0: Float32 = NEG_INF
        for i in range(n_logits):
            var v = ctx.state.logits[unsafe_offset=i]
            if v > max_l0:
                max_l0 = v
        var sum_exp0: Float32 = 0.0
        for i in range(n_logits):
            sum_exp0 += exp(ctx.state.logits[unsafe_offset=i] - max_l0)
        var nosp_id = Int(ctx.model.vocab.token_nosp)
        var no_speech_prob = exp(ctx.state.logits[unsafe_offset=nosp_id] - max_l0) / (sum_exp0 + Float32(1e-9))

        var tokens_cur = List[Int32]()
        var seek_delta = 100 * WHISPER_CHUNK_SIZE  # 3000 centiseconds (30.0s)
        var has_ts = False
        var result_len = 0
        var n_past = len(prompt_init)

        # 3. Autoregressive token generation loop
        for step in range(n_max):
            apply_whisper_logit_filters(
                ctx.state.logits,
                ctx.model.vocab,
                ctx.state.n_audio_ctx,
                params,
                tokens_cur,
                has_ts,
                seek_delta,
                ctx.space_token_id,
            )

            # Greedy argmax selection
            var best_id: Int = eot_id
            var best_val: Float32 = NEG_INF
            for i in range(n_logits):
                var v = ctx.state.logits[unsafe_offset=i]
                if v > best_val:
                    best_val = v
                    best_id = i

            tokens_cur.append(Int32(best_id))

            if best_id > beg_id:
                var seek_delta_new = 2 * (best_id - beg_id)
                if has_ts and seek_delta > seek_delta_new and result_len < step:
                    break
                seek_delta = seek_delta_new
                result_len = step + 1
                has_ts = True

            # Check termination conditions
            if (
                best_id == eot_id
                or (params.max_tokens > 0 and step >= params.max_tokens)
                or (has_ts and seek + seek_delta + delta_min >= seek_end)
            ):
                if result_len == 0 and not params.no_timestamps:
                    if seek + seek_delta + delta_min >= seek_end:
                        result_len = step + 1
                if params.single_segment or params.no_timestamps:
                    result_len = step + 1
                    seek_delta = 100 * WHISPER_CHUNK_SIZE
                break

            # Decode next step
            whisper_decode_step(ctx.model, ctx.state, Int32(best_id), n_past, True, params.n_threads)
            n_past += 1

        # 4. Extract segments from `tokens_cur[0 .. result_len - 1]`
        if result_len > 0:
            var t0 = Int64(seek)
            if Int(tokens_cur[0]) >= beg_id:
                t0 = Int64(seek + 2 * (Int(tokens_cur[0]) - beg_id))

            var text = String("")
            var seg_tokens = List[Int32]()
            var i = 0
            while i < result_len:
                var tid = Int(tokens_cur[i])
                if params.print_special or tid < eot_id:
                    text += ctx.model.vocab.id_to_token[tid]
                    seg_tokens.append(Int32(tid))

                if tid > beg_id and not params.single_segment:
                    var t1 = Int64(seek + 2 * (tid - beg_id))
                    if text.byte_length() > 0:
                        var seg = WhisperSegment(t0, t1, text, no_speech_prob)
                        seg.tokens = seg_tokens^
                        ctx.segments.append(seg^)
                        seg_tokens = List[Int32]()
                    text = String("")
                    t0 = t1
                    while i + 1 < result_len and Int(tokens_cur[i + 1]) >= beg_id:
                        i += 1
                        var next_tid = Int(tokens_cur[i])
                        t0 = Int64(seek + 2 * (next_tid - beg_id))
                i += 1

            if text.byte_length() > 0:
                var t1 = Int64(min(seek + seek_delta, seek_end))
                var seg = WhisperSegment(t0, t1, text, no_speech_prob)
                seg.tokens = seg_tokens^
                ctx.segments.append(seg^)

        seek += seek_delta

    return 0


def whisper_full(
        mut ctx: WhisperContext,
        params: WhisperFullParams,
        samples_list: List[Float32],
    ) -> Int:
    """ Overload of `whisper_full` accepting a Mojo `List[Float32]` PCM buffer."""
    var ptr = FloatPtr(unsafe_from_address=Int(samples_list.unsafe_ptr()))
    return whisper_full(ctx, params, ptr, len(samples_list))


def whisper_full_n_segments(ref ctx: WhisperContext) -> Int:
    """ Return the number of decoded speech segments in `ctx`."""
    return len(ctx.segments)


def whisper_full_get_segment_text(ref ctx: WhisperContext, i_segment: Int) -> String:
    """ Return the transcribed text of segment `i_segment`."""
    if i_segment < 0 or i_segment >= len(ctx.segments):
        return String("")
    return ctx.segments[i_segment].text


def whisper_full_get_segment_t0(ref ctx: WhisperContext, i_segment: Int) -> Int64:
    """ Return the start time (in 10ms centiseconds) of segment `i_segment`."""
    if i_segment < 0 or i_segment >= len(ctx.segments):
        return 0
    return ctx.segments[i_segment].t0


def whisper_full_get_segment_t1(ref ctx: WhisperContext, i_segment: Int) -> Int64:
    """ Return the end time (in 10ms centiseconds) of segment `i_segment`."""
    if i_segment < 0 or i_segment >= len(ctx.segments):
        return 0
    return ctx.segments[i_segment].t1
