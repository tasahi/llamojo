""" Dual Model Loader & Weight Structs for Whisper ASR (GGUF v2/v3 & Legacy `ggml` `.bin`).

Supports both:
1. Legacy `whisper.cpp` `.bin` format (`magic == 0x67676d6c`), zero-copy memory-mapped via POSIX `mmap64`,
   reading `whisper_hparams`, `whisper_filters`, vocabulary + dynamic special/timestamp tokens,
   and binding all 245+ encoder/decoder/cross-attention tensors.
2. Modern GGUF v2/v3 format (`magic == 0x46554747`), leveraging `llamojo.gguf.gguf_load_from_file`.
"""

from std.collections import List, Dict
from std.ffi import external_call
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from ..types import (
    GGUF_MAGIC,
    GGMLType,
    ggml_type_size,
    ggml_block_size,
    null_ptr,
)
from ..gguf import (
    O_RDONLY,
    SEEK_END,
    PROT_READ,
    MAP_PRIVATE,
    MAP_FAILED,
    GGUFContext,
    gguf_load_from_file,
    _cstr_alloc,
    rd_u32,
    rd_i32,
)
from .audio import WhisperFilters

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime WHISPER_BIN_MAGIC: UInt32 = 0x67676D6C  # "ggml" in little-endian


# ─── Whisper Hyperparameters & Special Token Vocabulary ──────────────────────

struct WhisperHParams(ImplicitlyCopyable, Copyable, Movable):
    """ Model hyperparameters matching `struct whisper_hparams` in `whisper.cpp:596`."""
    var n_vocab: Int
    var n_audio_ctx: Int
    var n_audio_state: Int
    var n_audio_head: Int
    var n_audio_layer: Int
    var n_text_ctx: Int
    var n_text_state: Int
    var n_text_head: Int
    var n_text_layer: Int
    var n_mels: Int
    var ftype: Int
    var eps: Float32

    def __init__(out self):
        self.n_vocab = 51864
        self.n_audio_ctx = 1500
        self.n_audio_state = 384
        self.n_audio_head = 6
        self.n_audio_layer = 4
        self.n_text_ctx = 448
        self.n_text_state = 384
        self.n_text_head = 6
        self.n_text_layer = 4
        self.n_mels = 80
        self.ftype = 1
        self.eps = 1e-5


struct WhisperVocab(Movable):
    """ Whisper BPE Vocabulary and Special / Timestamp Token IDs matching `whisper_vocab` (`whisper.cpp:435`)."""
    var n_vocab: Int
    var id_to_token: List[String]
    var token_to_id: Dict[String, Int32]

    var token_eot: Int32
    var token_sot: Int32
    var token_translate: Int32
    var token_transcribe: Int32
    var token_solm: Int32
    var token_prev: Int32
    var token_nosp: Int32
    var token_not: Int32
    var token_beg: Int32

    def __init__(out self):
        self.n_vocab = 51864
        self.id_to_token = List[String]()
        self.token_to_id = Dict[String, Int32]()
        self.token_eot = 50256
        self.token_sot = 50257
        self.token_translate = 50357
        self.token_transcribe = 50358
        self.token_solm = 50359
        self.token_prev = 50360
        self.token_nosp = 50361
        self.token_not = 50362
        self.token_beg = 50363

    def is_multilingual(self) -> Bool:
        return self.n_vocab >= 51865

    def num_languages(self) -> Int:
        var extra = 1 if self.is_multilingual() else 0
        return self.n_vocab - 51765 - extra

    def configure_special_tokens(mut self, total_vocab: Int):
        """ Configure multilingual offsets and populate synthetic special/timestamp tokens up to `total_vocab`."""
        self.n_vocab = total_vocab
        self.token_eot = 50256
        self.token_sot = 50257
        self.token_translate = 50357
        self.token_transcribe = 50358
        self.token_solm = 50359
        self.token_prev = 50360
        self.token_nosp = 50361
        self.token_not = 50362
        self.token_beg = 50363

        if self.is_multilingual():
            self.token_eot += 1
            self.token_sot += 1
            var dt = Int32(self.num_languages() - 98)
            self.token_translate += dt
            self.token_transcribe += dt
            self.token_solm += dt
            self.token_prev += dt
            self.token_nosp += dt
            self.token_not += dt
            self.token_beg += dt

        var loaded_count = len(self.id_to_token)
        if loaded_count < total_vocab:
            for i in range(loaded_count, total_vocab):
                var idx32 = Int32(i)
                var word: String
                if idx32 > self.token_beg:
                    word = "[_TT_" + String(idx32 - self.token_beg) + "]"
                elif idx32 == self.token_eot:
                    word = "[_EOT_]"
                elif idx32 == self.token_sot:
                    word = "[_SOT_]"
                elif idx32 == self.token_translate:
                    word = "[_TRANSLATE_]"
                elif idx32 == self.token_transcribe:
                    word = "[_TRANSCRIBE_]"
                elif idx32 == self.token_solm:
                    word = "[_SOLM_]"
                elif idx32 == self.token_prev:
                    word = "[_PREV_]"
                elif idx32 == self.token_nosp:
                    word = "[_NOSP_]"
                elif idx32 == self.token_not:
                    word = "[_NOT_]"
                elif idx32 == self.token_beg:
                    word = "[_BEG_]"
                else:
                    word = "[_extra_token_" + String(i) + "]"
                self.id_to_token.append(word)
                self.token_to_id[word] = idx32


# ─── Per-Layer Encoder & Decoder Weight Descriptors ──────────────────────────

struct WhisperEncoderLayer(ImplicitlyCopyable, Copyable, Movable):
    """ Pointers to weight and bias tensors for a single Whisper Audio Encoder layer."""
    var attn_ln_0_w: FloatPtr  # encoder.blocks.%d.attn_ln.weight
    var attn_ln_0_b: FloatPtr  # encoder.blocks.%d.attn_ln.bias

    var attn_q_w: BytePtr      # encoder.blocks.%d.attn.query.weight
    var attn_q_b: FloatPtr     # encoder.blocks.%d.attn.query.bias
    var attn_k_w: BytePtr      # encoder.blocks.%d.attn.key.weight (NO BIAS)
    var attn_v_w: BytePtr      # encoder.blocks.%d.attn.value.weight
    var attn_v_b: FloatPtr     # encoder.blocks.%d.attn.value.bias

    var attn_ln_1_w: BytePtr   # encoder.blocks.%d.attn.out.weight
    var attn_ln_1_b: FloatPtr  # encoder.blocks.%d.attn.out.bias

    var mlp_ln_w: FloatPtr     # encoder.blocks.%d.mlp_ln.weight
    var mlp_ln_b: FloatPtr     # encoder.blocks.%d.mlp_ln.bias
    var mlp_0_w: BytePtr       # encoder.blocks.%d.mlp.0.weight
    var mlp_0_b: FloatPtr      # encoder.blocks.%d.mlp.0.bias
    var mlp_1_w: BytePtr       # encoder.blocks.%d.mlp.2.weight
    var mlp_1_b: FloatPtr      # encoder.blocks.%d.mlp.2.bias

    var dtype_q: UInt32
    var dtype_k: UInt32
    var dtype_v: UInt32
    var dtype_out: UInt32
    var dtype_mlp_0: UInt32
    var dtype_mlp_1: UInt32

    def __init__(out self):
        self.attn_ln_0_w = null_ptr[Float32]()
        self.attn_ln_0_b = null_ptr[Float32]()
        self.attn_q_w = null_ptr[UInt8]()
        self.attn_q_b = null_ptr[Float32]()
        self.attn_k_w = null_ptr[UInt8]()
        self.attn_v_w = null_ptr[UInt8]()
        self.attn_v_b = null_ptr[Float32]()
        self.attn_ln_1_w = null_ptr[UInt8]()
        self.attn_ln_1_b = null_ptr[Float32]()
        self.mlp_ln_w = null_ptr[Float32]()
        self.mlp_ln_b = null_ptr[Float32]()
        self.mlp_0_w = null_ptr[UInt8]()
        self.mlp_0_b = null_ptr[Float32]()
        self.mlp_1_w = null_ptr[UInt8]()
        self.mlp_1_b = null_ptr[Float32]()
        self.dtype_q = GGMLType.F16
        self.dtype_k = GGMLType.F16
        self.dtype_v = GGMLType.F16
        self.dtype_out = GGMLType.F16
        self.dtype_mlp_0 = GGMLType.F16
        self.dtype_mlp_1 = GGMLType.F16


struct WhisperDecoderLayer(ImplicitlyCopyable, Copyable, Movable):
    """ Pointers to weight and bias tensors for a single Whisper Text Decoder layer."""
    # Self-Attention
    var attn_ln_0_w: FloatPtr        # decoder.blocks.%d.attn_ln.weight
    var attn_ln_0_b: FloatPtr        # decoder.blocks.%d.attn_ln.bias
    var attn_q_w: BytePtr            # decoder.blocks.%d.attn.query.weight
    var attn_q_b: FloatPtr           # decoder.blocks.%d.attn.query.bias
    var attn_k_w: BytePtr            # decoder.blocks.%d.attn.key.weight (NO BIAS)
    var attn_v_w: BytePtr            # decoder.blocks.%d.attn.value.weight
    var attn_v_b: FloatPtr           # decoder.blocks.%d.attn.value.bias
    var attn_ln_1_w: BytePtr         # decoder.blocks.%d.attn.out.weight
    var attn_ln_1_b: FloatPtr        # decoder.blocks.%d.attn.out.bias

    # Cross-Attention
    var cross_attn_ln_0_w: FloatPtr  # decoder.blocks.%d.cross_attn_ln.weight
    var cross_attn_ln_0_b: FloatPtr  # decoder.blocks.%d.cross_attn_ln.bias
    var cross_attn_q_w: BytePtr      # decoder.blocks.%d.cross_attn.query.weight
    var cross_attn_q_b: FloatPtr     # decoder.blocks.%d.cross_attn.query.bias
    var cross_attn_k_w: BytePtr      # decoder.blocks.%d.cross_attn.key.weight (NO BIAS)
    var cross_attn_v_w: BytePtr      # decoder.blocks.%d.cross_attn.value.weight
    var cross_attn_v_b: FloatPtr     # decoder.blocks.%d.cross_attn.value.bias
    var cross_attn_ln_1_w: BytePtr   # decoder.blocks.%d.cross_attn.out.weight
    var cross_attn_ln_1_b: FloatPtr  # decoder.blocks.%d.cross_attn.out.bias

    # MLP
    var mlp_ln_w: FloatPtr           # decoder.blocks.%d.mlp_ln.weight
    var mlp_ln_b: FloatPtr           # decoder.blocks.%d.mlp_ln.bias
    var mlp_0_w: BytePtr             # decoder.blocks.%d.mlp.0.weight
    var mlp_0_b: FloatPtr            # decoder.blocks.%d.mlp.0.bias
    var mlp_1_w: BytePtr             # decoder.blocks.%d.mlp.2.weight
    var mlp_1_b: FloatPtr            # decoder.blocks.%d.mlp.2.bias

    var dtype_q: UInt32
    var dtype_k: UInt32
    var dtype_v: UInt32
    var dtype_out: UInt32
    var dtype_cross_q: UInt32
    var dtype_cross_k: UInt32
    var dtype_cross_v: UInt32
    var dtype_cross_out: UInt32
    var dtype_mlp_0: UInt32
    var dtype_mlp_1: UInt32

    def __init__(out self):
        self.attn_ln_0_w = null_ptr[Float32]()
        self.attn_ln_0_b = null_ptr[Float32]()
        self.attn_q_w = null_ptr[UInt8]()
        self.attn_q_b = null_ptr[Float32]()
        self.attn_k_w = null_ptr[UInt8]()
        self.attn_v_w = null_ptr[UInt8]()
        self.attn_v_b = null_ptr[Float32]()
        self.attn_ln_1_w = null_ptr[UInt8]()
        self.attn_ln_1_b = null_ptr[Float32]()

        self.cross_attn_ln_0_w = null_ptr[Float32]()
        self.cross_attn_ln_0_b = null_ptr[Float32]()
        self.cross_attn_q_w = null_ptr[UInt8]()
        self.cross_attn_q_b = null_ptr[Float32]()
        self.cross_attn_k_w = null_ptr[UInt8]()
        self.cross_attn_v_w = null_ptr[UInt8]()
        self.cross_attn_v_b = null_ptr[Float32]()
        self.cross_attn_ln_1_w = null_ptr[UInt8]()
        self.cross_attn_ln_1_b = null_ptr[Float32]()

        self.mlp_ln_w = null_ptr[Float32]()
        self.mlp_ln_b = null_ptr[Float32]()
        self.mlp_0_w = null_ptr[UInt8]()
        self.mlp_0_b = null_ptr[Float32]()
        self.mlp_1_w = null_ptr[UInt8]()
        self.mlp_1_b = null_ptr[Float32]()

        self.dtype_q = GGMLType.F16
        self.dtype_k = GGMLType.F16
        self.dtype_v = GGMLType.F16
        self.dtype_out = GGMLType.F16
        self.dtype_cross_q = GGMLType.F16
        self.dtype_cross_k = GGMLType.F16
        self.dtype_cross_v = GGMLType.F16
        self.dtype_cross_out = GGMLType.F16
        self.dtype_mlp_0 = GGMLType.F16
        self.dtype_mlp_1 = GGMLType.F16


# ─── Unified Whisper Model Handle ────────────────────────────────────────────

struct WhisperTensorEntry(ImplicitlyCopyable, Copyable, Movable):
    var ptr: BytePtr
    var dtype: UInt32
    var n_bytes: Int

    def __init__(out self, ptr: BytePtr, dtype: UInt32, n_bytes: Int):
        self.ptr = ptr
        self.dtype = dtype
        self.n_bytes = n_bytes


struct WhisperModel(Movable):
    """ Loaded Whisper ASR Model supporting both legacy `.bin` and GGUF v2/v3 files."""
    var hparams: WhisperHParams
    var filters: WhisperFilters
    var vocab: WhisperVocab

    # Global Encoder Tensors
    var e_pe: FloatPtr        # encoder.positional_embedding [n_audio_ctx, n_audio_state] (F32)
    var e_conv_1_w: BytePtr   # encoder.conv1.weight [n_audio_state, n_mels, 3]
    var e_conv_1_b: FloatPtr  # encoder.conv1.bias [n_audio_state] (F32)
    var e_conv_2_w: BytePtr   # encoder.conv2.weight [n_audio_state, n_audio_state, 3]
    var e_conv_2_b: FloatPtr  # encoder.conv2.bias [n_audio_state] (F32)
    var e_ln_w: FloatPtr      # encoder.ln_post.weight [n_audio_state] (F32)
    var e_ln_b: FloatPtr      # encoder.ln_post.bias [n_audio_state] (F32)
    var dtype_conv_1: UInt32
    var dtype_conv_2: UInt32

    # Global Decoder Tensors
    var d_pe: FloatPtr        # decoder.positional_embedding [n_text_ctx, n_text_state] (F32)
    var d_te: BytePtr         # decoder.token_embedding.weight [n_vocab, n_text_state]
    var d_ln_w: FloatPtr      # decoder.ln.weight [n_text_state] (F32)
    var d_ln_b: FloatPtr      # decoder.ln.bias [n_text_state] (F32)
    var dtype_d_te: UInt32

    var layers_encoder: List[WhisperEncoderLayer]
    var layers_decoder: List[WhisperDecoderLayer]

    var tensor_Registry: Dict[String, WhisperTensorEntry]
    var n_loaded: Int

    # Memory backing (mmap for .bin or GGUFContext for .gguf)
    var raw_base: BytePtr
    var file_size: Int
    var is_mmapped: Bool
    var fd: Int32
    var gguf_ctx: GGUFContext
    var is_gguf: Bool

    def __init__(out self):
        self.hparams = WhisperHParams()
        self.filters = WhisperFilters()
        self.vocab = WhisperVocab()

        self.e_pe = null_ptr[Float32]()
        self.e_conv_1_w = null_ptr[UInt8]()
        self.e_conv_1_b = null_ptr[Float32]()
        self.e_conv_2_w = null_ptr[UInt8]()
        self.e_conv_2_b = null_ptr[Float32]()
        self.e_ln_w = null_ptr[Float32]()
        self.e_ln_b = null_ptr[Float32]()
        self.dtype_conv_1 = GGMLType.F16
        self.dtype_conv_2 = GGMLType.F16

        self.d_pe = null_ptr[Float32]()
        self.d_te = null_ptr[UInt8]()
        self.d_ln_w = null_ptr[Float32]()
        self.d_ln_b = null_ptr[Float32]()
        self.dtype_d_te = GGMLType.F16

        self.layers_encoder = List[WhisperEncoderLayer]()
        self.layers_decoder = List[WhisperDecoderLayer]()
        self.tensor_Registry = Dict[String, WhisperTensorEntry]()
        self.n_loaded = 0

        self.raw_base = null_ptr[UInt8]()
        self.file_size = 0
        self.is_mmapped = False
        self.fd = -1
        self.gguf_ctx = GGUFContext()
        self.is_gguf = False

    def get_ptr(self, name: String) -> BytePtr:
        var opt = self.tensor_Registry.find(name)
        if opt:
            return opt.value().ptr
        return null_ptr[UInt8]()

    def get_f32_ptr(self, name: String) -> FloatPtr:
        return self.get_ptr(name).unsafe_bitcast[Float32]()

    def get_dtype(self, name: String, default_dtype: UInt32 = GGMLType.F16) -> UInt32:
        var opt = self.tensor_Registry.find(name)
        if opt:
            return opt.value().dtype
        return default_dtype

    def bind_structured_weights(mut self):
        """ Populate `e_*`, `d_*`, `layers_encoder`, and `layers_decoder` from `tensor_Registry`."""
        self.e_pe = self.get_f32_ptr("encoder.positional_embedding")
        self.e_conv_1_w = self.get_ptr("encoder.conv1.weight")
        self.e_conv_1_b = self.get_f32_ptr("encoder.conv1.bias")
        self.e_conv_2_w = self.get_ptr("encoder.conv2.weight")
        self.e_conv_2_b = self.get_f32_ptr("encoder.conv2.bias")
        self.e_ln_w = self.get_f32_ptr("encoder.ln_post.weight")
        self.e_ln_b = self.get_f32_ptr("encoder.ln_post.bias")
        self.dtype_conv_1 = self.get_dtype("encoder.conv1.weight", GGMLType.F16)
        self.dtype_conv_2 = self.get_dtype("encoder.conv2.weight", GGMLType.F16)

        self.d_pe = self.get_f32_ptr("decoder.positional_embedding")
        self.d_te = self.get_ptr("decoder.token_embedding.weight")
        self.d_ln_w = self.get_f32_ptr("decoder.ln.weight")
        self.d_ln_b = self.get_f32_ptr("decoder.ln.bias")
        self.dtype_d_te = self.get_dtype("decoder.token_embedding.weight", GGMLType.F16)

        self.layers_encoder.clear()
        for l in range(self.hparams.n_audio_layer):
            var el = WhisperEncoderLayer()
            var pfx = "encoder.blocks." + String(l) + "."
            el.attn_ln_0_w = self.get_f32_ptr(pfx + "attn_ln.weight")
            el.attn_ln_0_b = self.get_f32_ptr(pfx + "attn_ln.bias")
            el.attn_q_w = self.get_ptr(pfx + "attn.query.weight")
            el.attn_q_b = self.get_f32_ptr(pfx + "attn.query.bias")
            el.attn_k_w = self.get_ptr(pfx + "attn.key.weight")
            el.attn_v_w = self.get_ptr(pfx + "attn.value.weight")
            el.attn_v_b = self.get_f32_ptr(pfx + "attn.value.bias")
            el.attn_ln_1_w = self.get_ptr(pfx + "attn.out.weight")
            el.attn_ln_1_b = self.get_f32_ptr(pfx + "attn.out.bias")
            el.mlp_ln_w = self.get_f32_ptr(pfx + "mlp_ln.weight")
            el.mlp_ln_b = self.get_f32_ptr(pfx + "mlp_ln.bias")
            el.mlp_0_w = self.get_ptr(pfx + "mlp.0.weight")
            el.mlp_0_b = self.get_f32_ptr(pfx + "mlp.0.bias")
            el.mlp_1_w = self.get_ptr(pfx + "mlp.2.weight")
            el.mlp_1_b = self.get_f32_ptr(pfx + "mlp.2.bias")

            el.dtype_q = self.get_dtype(pfx + "attn.query.weight")
            el.dtype_k = self.get_dtype(pfx + "attn.key.weight")
            el.dtype_v = self.get_dtype(pfx + "attn.value.weight")
            el.dtype_out = self.get_dtype(pfx + "attn.out.weight")
            el.dtype_mlp_0 = self.get_dtype(pfx + "mlp.0.weight")
            el.dtype_mlp_1 = self.get_dtype(pfx + "mlp.2.weight")
            self.layers_encoder.append(el^)

        self.layers_decoder.clear()
        for l in range(self.hparams.n_text_layer):
            var dl = WhisperDecoderLayer()
            var pfx = "decoder.blocks." + String(l) + "."
            dl.attn_ln_0_w = self.get_f32_ptr(pfx + "attn_ln.weight")
            dl.attn_ln_0_b = self.get_f32_ptr(pfx + "attn_ln.bias")
            dl.attn_q_w = self.get_ptr(pfx + "attn.query.weight")
            dl.attn_q_b = self.get_f32_ptr(pfx + "attn.query.bias")
            dl.attn_k_w = self.get_ptr(pfx + "attn.key.weight")
            dl.attn_v_w = self.get_ptr(pfx + "attn.value.weight")
            dl.attn_v_b = self.get_f32_ptr(pfx + "attn.value.bias")
            dl.attn_ln_1_w = self.get_ptr(pfx + "attn.out.weight")
            dl.attn_ln_1_b = self.get_f32_ptr(pfx + "attn.out.bias")

            dl.cross_attn_ln_0_w = self.get_f32_ptr(pfx + "cross_attn_ln.weight")
            dl.cross_attn_ln_0_b = self.get_f32_ptr(pfx + "cross_attn_ln.bias")
            dl.cross_attn_q_w = self.get_ptr(pfx + "cross_attn.query.weight")
            dl.cross_attn_q_b = self.get_f32_ptr(pfx + "cross_attn.query.bias")
            dl.cross_attn_k_w = self.get_ptr(pfx + "cross_attn.key.weight")
            dl.cross_attn_v_w = self.get_ptr(pfx + "cross_attn.value.weight")
            dl.cross_attn_v_b = self.get_f32_ptr(pfx + "cross_attn.value.bias")
            dl.cross_attn_ln_1_w = self.get_ptr(pfx + "cross_attn.out.weight")
            dl.cross_attn_ln_1_b = self.get_f32_ptr(pfx + "cross_attn.out.bias")

            dl.mlp_ln_w = self.get_f32_ptr(pfx + "mlp_ln.weight")
            dl.mlp_ln_b = self.get_f32_ptr(pfx + "mlp_ln.bias")
            dl.mlp_0_w = self.get_ptr(pfx + "mlp.0.weight")
            dl.mlp_0_b = self.get_f32_ptr(pfx + "mlp.0.bias")
            dl.mlp_1_w = self.get_ptr(pfx + "mlp.2.weight")
            dl.mlp_1_b = self.get_f32_ptr(pfx + "mlp.2.bias")

            dl.dtype_q = self.get_dtype(pfx + "attn.query.weight")
            dl.dtype_k = self.get_dtype(pfx + "attn.key.weight")
            dl.dtype_v = self.get_dtype(pfx + "attn.value.weight")
            dl.dtype_out = self.get_dtype(pfx + "attn.out.weight")
            dl.dtype_cross_q = self.get_dtype(pfx + "cross_attn.query.weight")
            dl.dtype_cross_k = self.get_dtype(pfx + "cross_attn.key.weight")
            dl.dtype_cross_v = self.get_dtype(pfx + "cross_attn.value.weight")
            dl.dtype_cross_out = self.get_dtype(pfx + "cross_attn.out.weight")
            dl.dtype_mlp_0 = self.get_dtype(pfx + "mlp.0.weight")
            dl.dtype_mlp_1 = self.get_dtype(pfx + "mlp.2.weight")
            self.layers_decoder.append(dl^)

    def free(mut self):
        self.filters.free()
        if self.is_gguf:
            self.gguf_ctx.free()
        else:
            if self.is_mmapped and Int(self.raw_base) != 0:
                _ = external_call["munmap", Int32](self.raw_base, self.file_size)
                self.raw_base = null_ptr[UInt8]()
                self.is_mmapped = False
            if self.fd >= 0:
                _ = external_call["close", Int32](self.fd)
                self.fd = -1


# ─── Unified Loader Entrypoint (`whisper_load_model_from_file`) ──────────────

def _read_bytes_as_string(d: BytePtr, offset: Int, length: Int) -> String:
    if length <= 0:
        return ""
    var buf = List[UInt8](capacity=length + 1)
    for i in range(length):
        buf.append(d[unsafe_offset=offset + i])
    return String(unsafe_from_utf8=buf^)


def whisper_load_model_from_file(path: String) raises -> WhisperModel:
    """ Auto-detect and load a Whisper model from either a legacy `whisper.cpp` `.bin` file
    (`magic == 0x67676d6c`) or a modern `.gguf` file (`magic == 0x46554747`).
    """
    var c_path = _cstr_alloc(path)
    var fd = external_call["open", Int32](c_path, O_RDONLY)
    c_path.unsafe_free()
    if fd < 0:
        raise Error("Failed to open Whisper model file: " + path)

    var fsz = external_call["lseek64", Int64](fd, Int64(0), SEEK_END)
    if fsz < 64:
        _ = external_call["close", Int32](fd)
        raise Error("Whisper model file too small: " + path)

    var magic_buf = unsafe_alloc[UInt8](4)
    _ = external_call["pread64", Int64](fd, magic_buf, Int64(4), Int64(0))
    var magic = rd_u32(magic_buf, 0)
    magic_buf.unsafe_free()

    if magic == GGUF_MAGIC:
        _ = external_call["close", Int32](fd)
        return _load_whisper_from_gguf(path)
    elif magic == WHISPER_BIN_MAGIC:
        return _load_whisper_from_bin(path, fd, Int(fsz))
    else:
        _ = external_call["close", Int32](fd)
        raise Error("Unrecognized Whisper model magic in: " + path)


def _load_whisper_from_bin(path: String, fd: Int32, max_len: Int) raises -> WhisperModel:
    """ Zero-copy POSIX `mmap64` loader for legacy `whisper.cpp` `.bin` models (`0x67676d6c`)."""
    var addr = external_call["mmap64", Int](
        Int(0),
        max_len,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        Int64(0),
    )
    if addr == MAP_FAILED or addr == 0:
        _ = external_call["close", Int32](fd)
        raise Error("Failed to mmap Whisper .bin file: " + path)

    var d = BytePtr(unsafe_from_address=addr)
    var model = WhisperModel()
    model.raw_base = d
    model.file_size = max_len
    model.is_mmapped = True
    model.fd = fd
    model.is_gguf = False

    # 1. Read 11 int32 hparams starting at byte offset 4
    model.hparams.n_vocab = Int(rd_i32(d, 4))
    model.hparams.n_audio_ctx = Int(rd_i32(d, 8))
    model.hparams.n_audio_state = Int(rd_i32(d, 12))
    model.hparams.n_audio_head = Int(rd_i32(d, 16))
    model.hparams.n_audio_layer = Int(rd_i32(d, 20))
    model.hparams.n_text_ctx = Int(rd_i32(d, 24))
    model.hparams.n_text_state = Int(rd_i32(d, 28))
    model.hparams.n_text_head = Int(rd_i32(d, 32))
    model.hparams.n_text_layer = Int(rd_i32(d, 36))
    model.hparams.n_mels = Int(rd_i32(d, 40))
    model.hparams.ftype = Int(rd_i32(d, 44)) % 1000

    # 2. Read Mel Filters at byte offset 48 (zero-copy pointer into mmap!)
    var n_mel = Int(rd_i32(d, 48))
    var n_fft = Int(rd_i32(d, 52))
    var filters_ptr = d.unsafe_offset(56).unsafe_bitcast[Float32]()
    model.filters = WhisperFilters(n_mel, n_fft, filters_ptr, False)

    var p: Int = 56 + n_mel * n_fft * 4

    # 3. Read Vocabulary
    var file_n_vocab = Int(rd_i32(d, p))
    p += 4
    model.vocab.id_to_token.reserve(model.hparams.n_vocab)

    for i in range(file_n_vocab):
        var tlen = Int(rd_u32(d, p))
        p += 4
        var word = _read_bytes_as_string(d, p, tlen)
        p += tlen
        model.vocab.id_to_token.append(word)
        model.vocab.token_to_id[word] = Int32(i)

    model.vocab.configure_special_tokens(model.hparams.n_vocab)

    # 4. Sequentially parse and bind all tensors (zero-copy pointers into mmap)
    var count = 0
    while p + 12 <= max_len:
        var n_dims = Int(rd_i32(d, p))
        var name_len = Int(rd_i32(d, p + 4))
        var ttype = rd_u32(d, p + 8)
        p += 12

        if n_dims <= 0 or n_dims > 4 or p + n_dims * 4 + name_len > max_len:
            break

        var nelements: Int = 1
        for _ in range(n_dims):
            var dim_i = Int(rd_i32(d, p))
            p += 4
            nelements *= dim_i

        var tname = _read_bytes_as_string(d, p, name_len)
        p += name_len

        var blk_sz = Int(ggml_block_size(ttype))
        var type_sz = Int(ggml_type_size(ttype))
        var nbytes = (nelements // blk_sz) * type_sz

        if p + nbytes > max_len:
            break

        var t_ptr = d.unsafe_offset(p)
        model.tensor_Registry[tname] = WhisperTensorEntry(t_ptr, ttype, nbytes)
        p += nbytes
        count += 1

    model.n_loaded = count
    model.bind_structured_weights()
    return model^


def _load_whisper_from_gguf(path: String) raises -> WhisperModel:
    """ Load a Whisper GGUF v2/v3 model via `gguf_load_from_file`."""
    var gguf = gguf_load_from_file(path, True, False, True)
    var model = WhisperModel()
    model.is_gguf = True

    var arch = gguf.get_string("general.architecture", "whisper")
    model.hparams.n_vocab = gguf.get_int(arch + ".vocab_size", 51864)
    model.hparams.n_audio_ctx = gguf.get_int(arch + ".encoder.context_length", 1500)
    model.hparams.n_audio_state = gguf.get_int(arch + ".encoder.embedding_length", 512)
    model.hparams.n_audio_head = gguf.get_int(arch + ".encoder.attention.head_count", 8)
    model.hparams.n_audio_layer = gguf.get_int(arch + ".encoder.block_count", 6)
    model.hparams.n_text_ctx = gguf.get_int(arch + ".decoder.context_length", 448)
    model.hparams.n_text_state = gguf.get_int(arch + ".decoder.embedding_length", 512)
    model.hparams.n_text_head = gguf.get_int(arch + ".decoder.attention.head_count", 8)
    model.hparams.n_text_layer = gguf.get_int(arch + ".decoder.block_count", 6)
    model.hparams.n_mels = gguf.get_int(arch + ".encoder.mels_count", 80)

    # Populate vocabulary if present in GGUF metadata
    var tok_opt = gguf.metadata.find("tokenizer.ggml.tokens")
    if tok_opt:
        var str_arr = tok_opt.value().str_arr.copy()
        for i in range(len(str_arr)):
            var w = str_arr[i]
            model.vocab.id_to_token.append(w)
            model.vocab.token_to_id[w] = Int32(i)
        if len(str_arr) > model.hparams.n_vocab:
            model.hparams.n_vocab = len(str_arr)

    model.vocab.configure_special_tokens(model.hparams.n_vocab)

    # Populate tensor registry from GGUF tensors
    for i in range(len(gguf.tensors)):
        var ti = gguf.tensors[i].copy()
        model.tensor_Registry[ti.name] = WhisperTensorEntry(ti.data_ptr, ti.dtype, Int(ti.size_bytes))

    model.n_loaded = len(gguf.tensors)
    model.gguf_ctx = gguf^
    model.bind_structured_weights()
    return model^
