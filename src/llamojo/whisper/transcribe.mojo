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
    var silence_thold: Float32
    var print_realtime: Bool
    var initial_prompt: String

    def __init__(out self):
        self.n_threads = 0
        self.offset_ms = 0
        self.duration_ms = 0
        self.translate = False
        self.no_context = False
        self.no_timestamps = False
        self.single_segment = False
        self.print_special = False
        self.print_realtime = False
        self.print_timestamps = True
        self.max_tokens = 0
        self.audio_ctx = 0
        self.language = String("en")
        self.initial_prompt = String("")
        self.suppress_blank = True
        self.suppress_nst = False
        self.temperature = 0.0
        self.max_initial_ts = 1.0
        self.no_speech_thold = 0.6
        self.silence_thold = 0.001

    def __copyinit__(out self, existing: Self):
        self.n_threads = existing.n_threads
        self.offset_ms = existing.offset_ms
        self.duration_ms = existing.duration_ms
        self.translate = existing.translate
        self.no_context = existing.no_context
        self.no_timestamps = existing.no_timestamps
        self.single_segment = existing.single_segment
        self.print_special = existing.print_special
        self.print_realtime = existing.print_realtime
        self.print_timestamps = existing.print_timestamps
        self.max_tokens = existing.max_tokens
        self.audio_ctx = existing.audio_ctx
        self.language = existing.language
        self.initial_prompt = existing.initial_prompt
        self.suppress_blank = existing.suppress_blank
        self.suppress_nst = existing.suppress_nst
        self.temperature = existing.temperature
        self.max_initial_ts = existing.max_initial_ts
        self.no_speech_thold = existing.no_speech_thold
        self.silence_thold = existing.silence_thold


def whisper_full_default_params() -> WhisperFullParams:
    """ Return default `WhisperFullParams` matching `whisper_full_default_params(WHISPER_SAMPLING_GREEDY)`."""
    return WhisperFullParams()


def whisper_lang_id(lang: String) -> Int:
    """ Map ISO 639-1 / language name to Whisper's 0-based language ID (`0..99`), or `-1` for `'auto'`."""
    if lang == "" or lang == "auto":
        return -1
    var codes: List[String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    ]
    for i in range(len(codes)):
        if codes[i] == lang:
            return i
    if lang == "spanish":
        return 3
    elif lang == "english":
        return 0
    elif lang == "portuguese":
        return 8
    elif lang == "french":
        return 6
    elif lang == "german":
        return 2
    return -1


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
        initial_prompt_tokens: List[Int32] = List[Int32](),
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

    var lang_id = whisper_lang_id(params.language)
    var seek = seek_start
    var eot_id = Int(ctx.model.vocab.token_eot)
    var beg_id = Int(ctx.model.vocab.token_beg)
    var n_logits = ctx.model.vocab.n_vocab
    var n_max = ctx.model.hparams.n_text_ctx // 2 - 4
    var prompt_past = List[Int32]()

    # Seed initial context prompt if provided
    for p_tok in range(len(initial_prompt_tokens)):
        prompt_past.append(initial_prompt_tokens[p_tok])

    while seek + delta_min < seek_end:
        # 0. Energy-Based Silence Skipper (VAD)
        # Evaluates peak audio energy in the 30-second window [seek .. min(seek_end, seek + 3000)].
        # If the entire window is quiet/silent, skip heavy neural network encoding & decoding.
        if params.silence_thold > 0.0 and n_samples > 0 and Int(samples) != 0:
            var samp_start = seek * 160
            var samp_end = min(n_samples, (seek + 100 * WHISPER_CHUNK_SIZE) * 160)
            if samp_end > samp_start:
                var sub_len = 8000  # 0.5s subwindow
                var max_sub_energy: Float32 = 0.0
                var pos = samp_start
                while pos < samp_end:
                    var cur_sub_end = min(samp_end, pos + sub_len)
                    var sub_cnt = cur_sub_end - pos
                    if sub_cnt > 0:
                        var sub_sum: Float32 = 0.0
                        for s_idx in range(pos, cur_sub_end):
                            var val = samples[unsafe_offset=s_idx]
                            sub_sum += val * val
                        var sub_rms = (sub_sum / Float32(sub_cnt))
                        if sub_rms > max_sub_energy:
                            max_sub_energy = sub_rms
                    pos += sub_len

                # If maximum 0.5s RMS energy is below the silence threshold (squared comparison)
                if max_sub_energy < (params.silence_thold * params.silence_thold):
                    var skip_delta = min(100 * WHISPER_CHUNK_SIZE, seek_end - seek)
                    if params.print_realtime:
                        var t0_sil = Int64(seek)
                        var t1_sil = Int64(seek + skip_delta)
                        print("[" + format_timestamp_cs(t0_sil) + " --> " + format_timestamp_cs(t1_sil) + "] [Silence]")
                    seek += skip_delta
                    continue

        # 1. Encode 30-second audio window starting at `seek`
        whisper_encode(ctx.model, ctx.state, seek, params.n_threads)

        # 2. Reset decoder self-attention KV cache and evaluate prompt_init
        ctx.state.reset_kv_self()

        # Auto-detect language on the first encoded window if multilingual and lang_id < 0
        if ctx.model.vocab.is_multilingual() and lang_id < 0:
            whisper_decode_step(ctx.model, ctx.state, ctx.model.vocab.token_sot, 0, True, params.n_threads)
            var best_lang_tok = Int(ctx.model.vocab.token_sot) + 1
            var best_lang_logit: Float32 = NEG_INF
            for lt in range(Int(ctx.model.vocab.token_sot) + 1, Int(ctx.model.vocab.token_translate)):
                var lv = ctx.state.logits[unsafe_offset=lt]
                if lv > best_lang_logit:
                    best_lang_logit = lv
                    best_lang_tok = lt
            lang_id = best_lang_tok - (Int(ctx.model.vocab.token_sot) + 1)
            ctx.state.lang_id = lang_id
            ctx.state.reset_kv_self()

        var prompt_init = List[Int32]()

        # Prepend prior text / vocabulary prompt via <|startofprev|> if available
        if not params.no_context and len(prompt_past) > 0:
            prompt_init.append(ctx.model.vocab.token_prev)
            var n_take = min(len(prompt_past), 224)
            for p_idx in range(len(prompt_past) - n_take, len(prompt_past)):
                prompt_init.append(prompt_past[p_idx])

        prompt_init.append(ctx.model.vocab.token_sot)
        if ctx.model.vocab.is_multilingual():
            var resolved_lang = lang_id if lang_id >= 0 else 0
            prompt_init.append(ctx.model.vocab.token_sot + 1 + Int32(resolved_lang))
            if params.translate:
                prompt_init.append(ctx.model.vocab.token_translate)
            else:
                prompt_init.append(ctx.model.vocab.token_transcribe)

        if params.no_timestamps:
            prompt_init.append(ctx.model.vocab.token_not)

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
        var loop_detected = False

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

            # Periodic sequence repetition detector (detects repeating phrases like "¿Eso es un tema?" or "Muy bien.")
            var cur_len = len(tokens_cur)
            if cur_len >= 12:
                # Check for repeating period P from 2 to 12 tokens
                for p in range(2, 13):
                    if cur_len >= 3 * p:
                        var matches = True
                        for k in range(2 * p):
                            if tokens_cur[cur_len - 1 - k] != tokens_cur[cur_len - 1 - k - p]:
                                matches = False
                                break
                        if matches:
                            loop_detected = True
                            break
                if loop_detected:
                    break

            if best_id > beg_id:
                var seek_delta_new = 2 * (best_id - beg_id)
                if has_ts and seek_delta > seek_delta_new and result_len < step:
                    break
                # Guard against degenerate 0-duration timestamp loops
                if has_ts and seek_delta_new <= seek_delta and result_len >= 4:
                    # Timestamp is not progressing forward in time
                    loop_detected = True
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

            # Guard against repetition loop at end of context budget (`whisper.cpp:7607`)
            if step == n_max - 1 and (result_len == 0 or seek_delta < (100 * WHISPER_CHUNK_SIZE) // 2):
                seek_delta = 100 * WHISPER_CHUNK_SIZE
                break

            # Decode next step
            whisper_decode_step(ctx.model, ctx.state, Int32(best_id), n_past, True, params.n_threads)
            n_past += 1

        if loop_detected:
            # Drop trailing repetitive tokens and advance window by 30s
            if result_len > 0 and seek_delta < (100 * WHISPER_CHUNK_SIZE) // 2:
                seek_delta = 100 * WHISPER_CHUNK_SIZE

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
                        if params.print_realtime:
                            print("[" + format_timestamp_cs(t0) + " --> " + format_timestamp_cs(t1) + "] " + text)
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
                if params.print_realtime:
                    print("[" + format_timestamp_cs(t0) + " --> " + format_timestamp_cs(t1) + "] " + text)
                var seg = WhisperSegment(t0, t1, text, no_speech_prob)
                seg.tokens = seg_tokens^
                ctx.segments.append(seg^)

            # Update rolling context for subsequent 30s windows
            if not params.no_context:
                prompt_past.clear()
                for c_tok in range(result_len):
                    var tid_c = Int(tokens_cur[c_tok])
                    if tid_c < eot_id:
                        prompt_past.append(Int32(tid_c))

        # Guarantee forward progress across windows
        if seek_delta < 100:
            seek_delta = 100 * WHISPER_CHUNK_SIZE
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
