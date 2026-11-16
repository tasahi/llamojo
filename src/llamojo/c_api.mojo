""" C-ABI Export Bridge for llamojo (`libllamojo.so`).\n
Exposes standard C-compatible symbols conforming to the llama.h ABI:
- Backend lifecycle (llama_backend_init, llama_backend_free)
- Model lifecycle and reflection (llama_model_load_from_file, llama_model_free, llama_model_n_vocab, etc.)
- Context & KV cache lifecycle (llama_init_from_model, llama_free)
- Vocabulary lifecycle & Tokenization (llama_vocab_create, llama_vocab_free, llama_tokenize, llama_token_to_piece)
- Batch creation and inference (llama_batch_init, llama_batch_free, llama_decode, llama_get_logits_ith)
- Sampling pipeline (llama_sampler_chain_init, llama_sampler_sample, etc.)
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc

from llamojo import (
    llama_token,
    llama_pos,
    llama_seq_id,
    llama_backend_init as _backend_init,
    llama_backend_free as _backend_free,
    llama_model_default_params as _model_default_params,
    llama_context_default_params as _context_default_params,
    llama_model_load_from_file as _model_load_from_file,
    llama_model_free as _model_free,
    llama_init_from_model as _init_from_model,
    llama_free as _context_free,
    llama_batch_init as _batch_init,
    llama_batch_free as _batch_free,
    llama_decode as _decode,
    llama_get_logits_ith as _get_logits_ith,
    llama_get_embeddings as _get_embeddings,
    llama_kv_cache_clear as _kv_cache_clear,
    llama_state_get_size as _state_get_size,
    llama_state_get_data as _state_get_data,
    llama_state_set_data as _state_set_data,
    llama_model_n_vocab as _model_n_vocab,
    llama_model_n_ctx_train as _model_n_ctx_train,
    llama_model_n_embd as _model_n_embd,
    llama_model_n_layer as _model_n_layer,
    llama_model_n_head as _model_n_head,
    llama_model_p,
    llama_context,
    llama_context_p,
    llama_batch,
    llama_token_data,
    llama_token_data_array,
    llama_sampler,
    llama_sampler_p,
    llama_sampler_chain_default_params,
    llama_sampler_chain_init as _sampler_chain_init,
    llama_sampler_chain_add as _sampler_chain_add,
    llama_sampler_init_top_k as _sampler_init_top_k,
    llama_sampler_init_top_p as _sampler_init_top_p,
    llama_sampler_init_temp as _sampler_init_temp,
    llama_sampler_init_dist as _sampler_init_dist,
    llama_sampler_init_greedy as _sampler_init_greedy,
    llama_sampler_init_grammar as _sampler_init_grammar,
    llama_sampler_accept as _sampler_accept,
    llama_sampler_sample as _sampler_sample,
    llama_sampler_free as _sampler_free,
    GbnfGrammar,
    llama_grammar_p,
    llama_grammar_init as _grammar_init,
    llama_grammar_free as _grammar_free,
    llama_lora_adapter,
    llama_lora_adapter_p,
    llama_lora_adapter_init as _lora_adapter_init,
    llama_lora_adapter_free as _lora_adapter_free,
    llama_adapter_lora_apply as _adapter_lora_apply,
    llama_vocab,
    llama_vocab_p,
    llama_tokenize as _tokenize,
    llama_token_to_piece as _token_to_piece,
    llama_vocab_bos as _vocab_bos,
    llama_vocab_eos as _vocab_eos,
    llama_vocab_eot as _vocab_eot,
    llama_vocab_is_eog as _vocab_is_eog,
    llama_vocab_n_tokens as _vocab_n_tokens,
    llama_vocab_add_special_token as _vocab_add_special_token,
    llama_vocab_find_token as _vocab_find_token,
    WhisperContext,
    WhisperFullParams,
    whisper_init_from_file as _whisper_init_from_file,
    whisper_full as _whisper_full,
    whisper_full_n_segments as _whisper_full_n_segments,
    whisper_full_get_segment_text as _whisper_full_get_segment_text,
    whisper_full_get_segment_t0 as _whisper_full_get_segment_t0,
    whisper_full_get_segment_t1 as _whisper_full_get_segment_t1,
    load_wav_file as _load_wav_file,
    null_ptr,
)

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime TokenPtr = Pointer[llama_token, MutUntrackedOrigin]
comptime BatchPtr = Pointer[llama_batch, MutUntrackedOrigin]

# ─── Helper string converter ──────────────────────────────────────────────────

@always_inline
def _cstr_to_string(cstr: BytePtr) -> String:
    var length: Int = 0
    while cstr[unsafe_offset=length] != 0:
        length += 1
    var s = String()
    for i in range(length):
        s += chr(Int(cstr[unsafe_offset=i]))
    return s

# ─── Backend Lifecycle ────────────────────────────────────────────────────────

@export
def llama_backend_init() abi("c"):
    """ Initialize the global llamojo backend."""
    _backend_init()

@export
def llama_backend_free() abi("c"):
    """ Release global backend resources."""
    _backend_free()

# ─── Model Lifecycle & Reflection ─────────────────────────────────────────────

@export
def llama_model_load_from_file(
        path_cstr: BytePtr,
        n_gpu_layers: Int32 = 0,
    ) abi("c") -> llama_model_p:
    """ Load a GGUF model from the specified file path."""
    var path = _cstr_to_string(path_cstr)
    var mparams = _model_default_params()
    mparams.n_gpu_layers = n_gpu_layers
    return _model_load_from_file(path, mparams)

@export
def llama_model_free(model: llama_model_p) abi("c"):
    """ Free a loaded model."""
    if Int(model) != 0:
        _model_free(model)

@export
def llama_model_n_vocab(model: llama_model_p) abi("c") -> Int32:
    if Int(model) == 0:
        return 0
    return _model_n_vocab(model)

@export
def llama_model_n_ctx_train(model: llama_model_p) abi("c") -> Int32:
    if Int(model) == 0:
        return 0
    return _model_n_ctx_train(model)

@export
def llama_model_n_embd(model: llama_model_p) abi("c") -> Int32:
    if Int(model) == 0:
        return 0
    return _model_n_embd(model)

@export
def llama_model_n_layer(model: llama_model_p) abi("c") -> Int32:
    if Int(model) == 0:
        return 0
    return _model_n_layer(model)

@export
def llama_model_n_head(model: llama_model_p) abi("c") -> Int32:
    if Int(model) == 0:
        return 0
    return _model_n_head(model)

# ─── Vocabulary Lifecycle & Inspection ────────────────────────────────────────

@export
def llama_model_get_vocab(model: llama_model_p) abi("c") -> llama_vocab_p:
    """ Instantiate a vocabulary handle from the model's GGUF metadata."""
    if Int(model) == 0:
        return null_ptr[llama_vocab]()
    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))
    return vocab

@export
def llama_vocab_free(vocab: llama_vocab_p) abi("c"):
    """ Free a vocabulary handle."""
    if Int(vocab) != 0:
        vocab.unsafe_deinit_pointee()
        vocab.unsafe_free()

@export
def llama_vocab_n_tokens(vocab: llama_vocab_p) abi("c") -> Int32:
    return _vocab_n_tokens(vocab)

@export
def llama_vocab_bos(vocab: llama_vocab_p) abi("c") -> llama_token:
    return _vocab_bos(vocab)

@export
def llama_vocab_eos(vocab: llama_vocab_p) abi("c") -> llama_token:
    return _vocab_eos(vocab)

@export
def llama_vocab_eot(vocab: llama_vocab_p) abi("c") -> llama_token:
    return _vocab_eot(vocab)

@export
def llama_vocab_is_eog(vocab: llama_vocab_p, tok: llama_token) abi("c") -> Bool:
    return _vocab_is_eog(vocab, tok)

@export
def llama_vocab_add_special_token(
        vocab: llama_vocab_p,
        piece_cstr: BytePtr,
        token_type: Int32 = 4,
    ) abi("c") -> llama_token:
    """ Dynamically register a special/control token (e.g., `<opt_0>`) into the vocabulary."""
    if Int(vocab) == 0 or Int(piece_cstr) == 0:
        return -1
    var piece = _cstr_to_string(piece_cstr)
    return _vocab_add_special_token(vocab, piece, token_type)

@export
def llama_vocab_find_token(
        vocab: llama_vocab_p,
        piece_cstr: BytePtr,
    ) abi("c") -> llama_token:
    """ Look up the token ID for an exact piece string."""
    if Int(vocab) == 0 or Int(piece_cstr) == 0:
        return -1
    var piece = _cstr_to_string(piece_cstr)
    return _vocab_find_token(vocab, piece)

@export
def llama_tokenize(
        vocab: llama_vocab_p,
        text_cstr: BytePtr,
        text_len: Int32,
        tokens_out: TokenPtr,
        max_tokens: Int32,
        add_special: Bool,
        parse_special: Bool,
    ) abi("c") -> Int32:
    """ Tokenize raw UTF-8 text into an array of tokens."""
    if Int(vocab) == 0 or Int(text_cstr) == 0 or text_len < 0:
        return 0
    if text_len == 0:
        return _tokenize(vocab, String(""), tokens_out, max_tokens, add_special, parse_special)
    from std.collections import Span
    var s = String(unsafe_from_utf8=Span(unsafe_ptr=text_cstr, length=Int(text_len)))
    return _tokenize(
        vocab,
        s,
        tokens_out,
        max_tokens,
        add_special,
        parse_special,
    )

@export
def llama_token_to_piece(
        vocab: llama_vocab_p,
        token: llama_token,
        buf_out: BytePtr,
        buf_len: Int32,
        lstrip: Int32 = 0,
        special: Bool = True,
    ) abi("c") -> Int32:
    """ Convert a token ID to its string representation into buf_out."""
    if Int(vocab) == 0:
        return 0
    return _token_to_piece(
        vocab,
        token,
        buf_out,
        buf_len,
        lstrip,
        special,
    )

# ─── Context Lifecycle ────────────────────────────────────────────────────────

@export
def llama_init_from_model(
        model: llama_model_p,
        n_ctx: UInt32 = 2048,
        n_batch: UInt32 = 512,
        n_threads: Int32 = 2,
    ) abi("c") -> llama_context_p:
    """ Create an inference context for the model."""
    if Int(model) == 0:
        return null_ptr[llama_context]()
    var cparams = _context_default_params()
    cparams.n_ctx = n_ctx
    cparams.n_batch = n_batch
    cparams.n_threads = n_threads
    return _init_from_model(model, cparams)

@export
def llama_free(ctx: llama_context_p) abi("c"):
    """ Free the inference context."""
    if Int(ctx) != 0:
        _context_free(ctx)

# ─── Batch Management & Inference ─────────────────────────────────────────────

@export
def llama_batch_init(n_tokens: Int32, embd: Int32, n_seq_max: Int32) abi("c") -> BatchPtr:
    """ Allocate a llama_batch on the heap and return its pointer."""
    var batch = _batch_init(n_tokens, embd, n_seq_max)
    var ptr = unsafe_alloc[llama_batch](1)
    ptr[unsafe_offset=0] = batch
    return ptr

@export
def llama_batch_free(batch: BatchPtr) abi("c"):
    """ Free a heap-allocated llama_batch."""
    if Int(batch) != 0:
        _batch_free(batch[unsafe_offset=0])
        batch.unsafe_free()

@export
def llama_batch_set_token(
        batch: BatchPtr,
        index: Int32,
        token: llama_token,
        pos: llama_pos,
        seq_id: llama_seq_id,
        logits: Bool,
    ) abi("c"):
    """ Helper to populate batch elements from C/Python."""
    if Int(batch) == 0:
        return
    var b = batch[unsafe_offset=0]
    b.token[unsafe_offset=Int(index)] = token
    b.pos[unsafe_offset=Int(index)] = pos
    b.n_seq_id[unsafe_offset=Int(index)] = 1
    b.seq_id[unsafe_offset=Int(index)][unsafe_offset=0] = seq_id
    b.logits[unsafe_offset=Int(index)] = Int8(1 if logits else 0)

@export
def llama_batch_set_n_tokens(batch: BatchPtr, n_tokens: Int32) abi("c"):
    if Int(batch) != 0:
        batch[unsafe_offset=0].n_tokens = n_tokens

@export
def llama_decode(ctx: llama_context_p, batch: BatchPtr) abi("c") -> Int32:
    """ Execute model forward pass for the given batch."""
    if Int(ctx) == 0 or Int(batch) == 0:
        return 1
    return _decode(ctx, batch[unsafe_offset=0])

@export
def llama_get_logits_ith(ctx: llama_context_p, i: Int32) abi("c") -> FloatPtr:
    """ Return pointer to logits array for token i in the last decoded batch."""
    if Int(ctx) == 0:
        return null_ptr[Float32]()
    return _get_logits_ith(ctx, i)

@export
def llama_get_embeddings(ctx: llama_context_p) abi("c") -> FloatPtr:
    """ Return pointer to normalized embeddings."""
    if Int(ctx) == 0:
        return null_ptr[Float32]()
    return _get_embeddings(ctx)

@export
def llama_kv_cache_clear(ctx: llama_context_p) abi("c"):
    """ Clear KV cache contents."""
    if Int(ctx) != 0:
        _kv_cache_clear(ctx)

@export
def llama_state_get_size(ctx: llama_context_p) abi("c") -> UInt64:
    """ Return exact size in bytes required for context state serialization."""
    if Int(ctx) == 0:
        return 0
    return _state_get_size(ctx)

@export
def llama_state_get_data(ctx: llama_context_p, dst: BytePtr, size: UInt64) abi("c") -> UInt64:
    """ Serialize context state into dst buffer."""
    if Int(ctx) == 0 or Int(dst) == 0:
        return 0
    return _state_get_data(ctx, dst, size)

@export
def llama_state_set_data(ctx: llama_context_p, src: BytePtr, size: UInt64) abi("c") -> UInt64:
    """ Restore context state from src buffer."""
    if Int(ctx) == 0 or Int(src) == 0:
        return 0
    return _state_set_data(ctx, src, size)

# ─── Sampler Pipeline ─────────────────────────────────────────────────────────

@export
def llama_sampler_chain_init() abi("c") -> llama_sampler_p:
    var sparams = llama_sampler_chain_default_params()
    return _sampler_chain_init(sparams)

@export
def llama_sampler_chain_add(chain: llama_sampler_p, smpl: llama_sampler_p) abi("c"):
    if Int(chain) != 0 and Int(smpl) != 0:
        _sampler_chain_add(chain, smpl)

@export
def llama_sampler_init_top_k(k: Int32) abi("c") -> llama_sampler_p:
    return _sampler_init_top_k(k)

@export
def llama_sampler_init_top_p(p: Float32, min_keep: Int32 = 1) abi("c") -> llama_sampler_p:
    return _sampler_init_top_p(p, Int(min_keep))

@export
def llama_sampler_init_temp(t: Float32) abi("c") -> llama_sampler_p:
    return _sampler_init_temp(t)

@export
def llama_sampler_init_dist(seed: UInt32) abi("c") -> llama_sampler_p:
    return _sampler_init_dist(seed)

@export
def llama_sampler_init_greedy() abi("c") -> llama_sampler_p:
    return _sampler_init_greedy()

@export
def llama_sampler_sample_token(
        smpl: llama_sampler_p,
        logits: FloatPtr,
        n_vocab: Int32,
        temp: Float32,
    ) abi("c") -> llama_token:
    """ Direct high-level sampling helper from logits array."""
    if Int(logits) == 0 or n_vocab <= 0:
        return -1

    if temp <= 0.0:
        # Greedy argmax
        var best_id = 0
        var max_val = logits[unsafe_offset=0]
        for v in range(1, Int(n_vocab)):
            var val = logits[unsafe_offset=v]
            if val > max_val:
                max_val = val
                best_id = v
        return llama_token(best_id)

    # Stochastic sampling via candidates array
    var candidates_buf = unsafe_alloc[llama_token_data](Int(n_vocab))
    for v in range(Int(n_vocab)):
        candidates_buf[unsafe_offset=v] = llama_token_data(Int32(v), logits[unsafe_offset=v], Float32(0.0))

    var cur_p = unsafe_alloc[llama_token_data_array](1)
    cur_p.unsafe_write(llama_token_data_array(candidates_buf, Int(n_vocab), False))

    var next_tok = _sampler_sample(smpl, cur_p)

    cur_p.unsafe_deinit_pointee()
    cur_p.unsafe_free()
    candidates_buf.unsafe_free()
    return next_tok

@export
def llama_sampler_free(smpl: llama_sampler_p) abi("c"):
    if Int(smpl) != 0:
        _sampler_free(smpl)

# ─── GBNF Grammar Sampling ────────────────────────────────────────────────────

@export
def llama_grammar_init(grammar_cstr: BytePtr) abi("c") -> llama_grammar_p:
    """ Initialize a GBNF grammar parser handle."""
    if Int(grammar_cstr) == 0:
        return null_ptr[GbnfGrammar]()
    var g_str = _cstr_to_string(grammar_cstr)
    return _grammar_init(g_str)

@export
def llama_grammar_free(grammar: llama_grammar_p) abi("c"):
    """ Free a GBNF grammar instance."""
    if Int(grammar) != 0:
        _grammar_free(grammar)

@export
def llama_sampler_init_grammar(
    vocab: llama_vocab_p,
    grammar_pattern_cstr: BytePtr,
) abi("c") -> llama_sampler_p:
    """ Initialize a grammar-constrained sampler using model vocabulary tokens."""
    if Int(grammar_pattern_cstr) == 0:
        return null_ptr[llama_sampler]()
    var pat = _cstr_to_string(grammar_pattern_cstr)
    var tokens_list = List[String]()
    if Int(vocab) != 0:
        tokens_list = vocab[].tokens.copy()
    return _sampler_init_grammar(pat, tokens_list)

@export
def llama_sampler_accept_token(smpl: llama_sampler_p, token: llama_token) abi("c"):
    """ Advance sampler state (e.g. grammar pushdown automaton or repetition history)."""
    if Int(smpl) != 0:
        _sampler_accept(smpl, token)

# ─── LoRA Adapter Ingestion & Merging ─────────────────────────────────────────

@export
def llama_lora_adapter_init(path_cstr: BytePtr) abi("c") -> llama_lora_adapter_p:
    """ Load and parse a LoRA adapter GGUF file."""
    if Int(path_cstr) == 0:
        return null_ptr[llama_lora_adapter]()
    var path = _cstr_to_string(path_cstr)
    var dummy = null_ptr[NoneType]()
    return _lora_adapter_init(dummy, path)

@export
def llama_lora_adapter_free(adapter: llama_lora_adapter_p) abi("c"):
    """ Free a loaded LoRA adapter."""
    if Int(adapter) != 0:
        _lora_adapter_free(adapter)

@export
def llama_lora_adapter_apply(
    model: llama_model_p,
    adapter_path_cstr: BytePtr,
    scale: Float32 = 1.0,
) abi("c") -> Int32:
    """ Apply a LoRA adapter GGUF directly to base model weights in-place."""
    if Int(model) == 0 or Int(adapter_path_cstr) == 0:
        return -1
    var path = _cstr_to_string(adapter_path_cstr)
    return _adapter_lora_apply(model, path, scale)

# ─── Zero-Copy Tensor Memory Views (Part 1 Interop) ───────────────────────────

from llamojo.interop import (
    context_logits_view,
    context_embeddings_view,
    context_kv_k_view,
    context_kv_v_view,
    context_hidden_view,
    TensorMemoryView,
)

struct llamojo_view(Copyable, Movable):
    var data: BytePtr
    var dtype: UInt32
    var ndim: Int32
    var shape0: Int64
    var shape1: Int64
    var shape2: Int64
    var shape3: Int64
    var strides0: Int64
    var strides1: Int64
    var strides2: Int64
    var strides3: Int64
    var total_elements: Int64
    var version_tag: UInt64

    def __init__(out self):
        self.data = null_ptr[UInt8]()
        self.dtype = 0
        self.ndim = 0
        self.shape0 = 0
        self.shape1 = 0
        self.shape2 = 0
        self.shape3 = 0
        self.strides0 = 0
        self.strides1 = 0
        self.strides2 = 0
        self.strides3 = 0
        self.total_elements = 0
        self.version_tag = 0

comptime llamojo_view_p = Pointer[llamojo_view, MutUntrackedOrigin]

@always_inline
def _fill_c_view(view: TensorMemoryView, out_view: llamojo_view_p, version: UInt64 = 1):
    if Int(out_view) == 0:
        return
    out_view[].data = view.data
    out_view[].dtype = view.dtype
    out_view[].ndim = Int32(view.ndim)
    out_view[].shape0 = view.shape[0]
    out_view[].shape1 = view.shape[1]
    out_view[].shape2 = view.shape[2]
    out_view[].shape3 = view.shape[3]
    out_view[].strides0 = view.strides[0]
    out_view[].strides1 = view.strides[1]
    out_view[].strides2 = view.strides[2]
    out_view[].strides3 = view.strides[3]
    out_view[].total_elements = view.total_elements
    out_view[].version_tag = version

@export
def llamojo_get_logits_view(ctx: llama_context_p, out_view: llamojo_view_p) abi("c") -> Int32:
    """ Fills out_view with the current logits buffer descriptor."""
    if Int(ctx) == 0 or Int(out_view) == 0:
        return -1
    var v = context_logits_view(ctx[])
    _fill_c_view(v, out_view, 1)
    return 0

@export
def llamojo_get_embeddings_view(ctx: llama_context_p, out_view: llamojo_view_p) abi("c") -> Int32:
    """ Fills out_view with the embeddings buffer descriptor."""
    if Int(ctx) == 0 or Int(out_view) == 0:
        return -1
    if Int(ctx[].embeddings) == 0:
        return -2
    var v = context_embeddings_view(ctx[])
    _fill_c_view(v, out_view, 1)
    return 0

@export
def llamojo_get_hidden_view(ctx: llama_context_p, out_view: llamojo_view_p) abi("c") -> Int32:
    """ Fills out_view with the hidden state buffer descriptor."""
    if Int(ctx) == 0 or Int(out_view) == 0:
        return -1
    var v = context_hidden_view(ctx[])
    _fill_c_view(v, out_view, 1)
    return 0

@export
def llamojo_get_kv_view(ctx: llama_context_p, which_cache: Int32, out_view: llamojo_view_p) abi("c") -> Int32:
    """ Fills out_view with the K (0) or V (1) cache view."""
    if Int(ctx) == 0 or Int(out_view) == 0:
        return -1
    if which_cache == 0:
        var v = context_kv_k_view(ctx[])
        _fill_c_view(v, out_view, 1)
        return 0
    elif which_cache == 1:
        var v = context_kv_v_view(ctx[])
        _fill_c_view(v, out_view, 1)
        return 0
    return -2


# ─── Whisper ASR C-ABI Exports ────────────────────────────────────────────────

comptime whisper_context_p = Pointer[WhisperContext, MutUntrackedOrigin]

@export
def whisper_init_from_file(path_cstr: BytePtr) abi("c") -> whisper_context_p:
    """ Load a Whisper model (.bin or .gguf) and return an opaque WhisperContext pointer."""
    if Int(path_cstr) == 0:
        return null_ptr[WhisperContext]()
    var path = _cstr_to_string(path_cstr)
    try:
        var ctx = _whisper_init_from_file(path)
        var ptr = unsafe_alloc[WhisperContext](1)
        ptr.unsafe_write(ctx^)
        return ptr
    except:
        return null_ptr[WhisperContext]()

@export
def whisper_free(ctx: whisper_context_p) abi("c"):
    """ Free a WhisperContext allocated by `whisper_init_from_file`."""
    if Int(ctx) != 0:
        ctx[].free()
        ctx.unsafe_deinit_pointee()
        ctx.unsafe_free()

@export
def whisper_is_multilingual(ctx: whisper_context_p) abi("c") -> Int32:
    """ Return 1 if the loaded Whisper model is multilingual, 0 if English-only."""
    if Int(ctx) == 0:
        return 0
    return 1 if ctx[].model.vocab.is_multilingual() else 0

@export
def whisper_full_c(
        ctx: whisper_context_p,
        samples: FloatPtr,
        n_samples: Int32,
        no_timestamps: Int32 = 0,
        translate: Int32 = 0,
        n_threads: Int32 = 0,
    ) abi("c") -> Int32:
    """ Run full Whisper transcription on 16 kHz mono F32 PCM samples."""
    if Int(ctx) == 0 or Int(samples) == 0 or n_samples <= 0:
        return -1
    var params = WhisperFullParams()
    params.no_timestamps = (no_timestamps != 0)
    params.translate = (translate != 0)
    params.n_threads = Int(n_threads)
    return Int32(_whisper_full(ctx[], params, samples, Int(n_samples)))

@export
def whisper_full_ex_c(
        ctx: whisper_context_p,
        samples: FloatPtr,
        n_samples: Int32,
        language_cstr: BytePtr,
        no_timestamps: Int32 = 0,
        translate: Int32 = 0,
        n_threads: Int32 = 0,
        offset_ms: Int32 = 0,
        duration_ms: Int32 = 0,
        print_realtime: Int32 = 0,
        no_context: Int32 = 0,
        prompt_tokens: Pointer[Int32, MutUntrackedOrigin] = null_ptr[Int32](),
        n_prompt_tokens: Int32 = 0,
    ) abi("c") -> Int32:
    """ Extended Whisper transcription with language, offset_ms, duration_ms, print_realtime, no_context, and initial_prompt tokens."""
    if Int(ctx) == 0 or Int(samples) == 0 or n_samples <= 0:
        return -1
    var params = WhisperFullParams()
    if Int(language_cstr) != 0:
        params.language = _cstr_to_string(language_cstr)
    params.no_timestamps = (no_timestamps != 0)
    params.translate = (translate != 0)
    params.n_threads = Int(n_threads)
    params.offset_ms = Int(offset_ms)
    params.duration_ms = Int(duration_ms)
    params.print_realtime = (print_realtime != 0)
    params.no_context = (no_context != 0)

    var p_tokens = List[Int32]()
    if Int(prompt_tokens) != 0 and n_prompt_tokens > 0:
        for i in range(Int(n_prompt_tokens)):
            p_tokens.append(prompt_tokens[unsafe_offset=i])

    return Int32(_whisper_full(ctx[], params, samples, Int(n_samples), p_tokens))

@export
def whisper_full_wav_c(
        ctx: whisper_context_p,
        wav_path_cstr: BytePtr,
        no_timestamps: Int32 = 0,
        translate: Int32 = 0,
        n_threads: Int32 = 0,
    ) abi("c") -> Int32:
    """ Load a 16 kHz .wav file and run full Whisper transcription."""
    if Int(ctx) == 0 or Int(wav_path_cstr) == 0:
        return -1
    var wav_path = _cstr_to_string(wav_path_cstr)
    try:
        var pcm = _load_wav_file(wav_path)
        var params = WhisperFullParams()
        params.no_timestamps = (no_timestamps != 0)
        params.translate = (translate != 0)
        params.n_threads = Int(n_threads)
        return Int32(_whisper_full(ctx[], params, pcm))
    except:
        return -2

@export
def whisper_full_n_segments(ctx: whisper_context_p) abi("c") -> Int32:
    """ Return the number of decoded segments in `ctx`."""
    if Int(ctx) == 0:
        return 0
    return Int32(_whisper_full_n_segments(ctx[]))

@export
def whisper_full_get_segment_t0(ctx: whisper_context_p, i_segment: Int32) abi("c") -> Int64:
    """ Return the start timestamp (in 10ms centiseconds) of segment `i_segment`."""
    if Int(ctx) == 0:
        return 0
    return _whisper_full_get_segment_t0(ctx[], Int(i_segment))

@export
def whisper_full_get_segment_t1(ctx: whisper_context_p, i_segment: Int32) abi("c") -> Int64:
    """ Return the end timestamp (in 10ms centiseconds) of segment `i_segment`."""
    if Int(ctx) == 0:
        return 0
    return _whisper_full_get_segment_t1(ctx[], Int(i_segment))

@export
def whisper_full_get_segment_text(ctx: whisper_context_p, i_segment: Int32) abi("c") -> BytePtr:
    """ Return a null-terminated UTF-8 C string for segment `i_segment`."""
    if Int(ctx) == 0:
        return null_ptr[UInt8]()
    var txt = _whisper_full_get_segment_text(ctx[], Int(i_segment))
    var bytes = txt.as_bytes()
    ctx[].last_cstr.clear()
    for b in range(len(bytes)):
        ctx[].last_cstr.append(bytes[b])
    ctx[].last_cstr.append(0)
    return BytePtr(unsafe_from_address=Int(ctx[].last_cstr.unsafe_ptr()))


# ─── Timestral C-ABI Exports ──────────────────────────────────────────────────

from llamojo import TimestralForecaster, ForecastResult

comptime timestral_forecaster_p = Pointer[TimestralForecaster, MutUntrackedOrigin]
comptime timestral_forecast_p = Pointer[ForecastResult, MutUntrackedOrigin]

@export
def llamojo_timestral_load(path_cstr: BytePtr, max_rows: Int32 = 16, max_patches: Int32 = 128) abi("c") -> timestral_forecaster_p:
    """ Load a Timestral model from GGUF file and return an opaque handle."""
    if Int(path_cstr) == 0:
        return null_ptr[TimestralForecaster]()
    var path = _cstr_to_string(path_cstr)
    try:
        var forecaster = TimestralForecaster.from_gguf(path, max_rows=Int(max_rows), max_patches=Int(max_patches))
        var handle = unsafe_alloc[TimestralForecaster](1)
        handle.unsafe_write(forecaster^)
        return handle
    except:
        return null_ptr[TimestralForecaster]()

@export
def llamojo_timestral_free(forecaster: timestral_forecaster_p) abi("c"):
    """ Free a Timestral model handle."""
    if Int(forecaster) != 0:
        forecaster[].free()
        forecaster.unsafe_deinit_pointee()
        forecaster.unsafe_free()

@export
def llamojo_timestral_forecast(
        forecaster: timestral_forecaster_p,
        context: FloatPtr,
        t_context: Int32,
        prediction_length: Int32,
        batch_size: Int32 = 1,
        n_targets: Int32 = 1,
        past_covariates: FloatPtr = null_ptr[Float32](),
        n_past_cov: Int32 = 0,
        known_covariates: FloatPtr = null_ptr[Float32](),
        n_known_cov: Int32 = 0,
    ) abi("c") -> timestral_forecast_p:
    """ Execute probabilistic forecast and return ForecastResult handle."""
    if Int(forecaster) == 0 or Int(context) == 0:
        return null_ptr[ForecastResult]()
    try:
        var res = forecaster[].forecast(
            context=context,
            t_context=Int(t_context),
            prediction_length=Int(prediction_length),
            batch_size=Int(batch_size),
            n_targets=Int(n_targets),
            past_covariates=past_covariates,
            n_past_cov=Int(n_past_cov),
            known_covariates=known_covariates,
            n_known_cov=Int(n_known_cov),
        )
        var handle = unsafe_alloc[ForecastResult](1)
        handle.unsafe_write(res^)
        return handle
    except:
        return null_ptr[ForecastResult]()

@export
def llamojo_timestral_forecast_free(forecast: timestral_forecast_p) abi("c"):
    """ Free a ForecastResult handle."""
    if Int(forecast) != 0:
        forecast[].free()
        forecast.unsafe_deinit_pointee()
        forecast.unsafe_free()

@export
def llamojo_timestral_forecast_get_quantiles(forecast: timestral_forecast_p) abi("c") -> FloatPtr:
    """ Return pointer to contiguous (B, V, H, 21) float32 quantile tensor."""
    if Int(forecast) == 0:
        return null_ptr[Float32]()
    return forecast[].data

@export
def llamojo_timestral_forecast_median(forecast: timestral_forecast_p, b: Int32, v: Int32, h: Int32) abi("c") -> Float32:
    """ Return median forecast (q = 0.50) at step h."""
    if Int(forecast) == 0:
        return 0.0
    return forecast[].median(Int(b), Int(v), Int(h))

@export
def llamojo_timestral_forecast_quantile(forecast: timestral_forecast_p, b: Int32, v: Int32, h: Int32, q: Float32) abi("c") -> Float32:
    """ Return or interpolate quantile level q at step h."""
    if Int(forecast) == 0:
        return 0.0
    return forecast[].quantile(Int(b), Int(v), Int(h), q)

@export
def llamojo_timestral_impute(
        forecaster: timestral_forecaster_p,
        data: FloatPtr,
        t_total: Int32,
        batch_size: Int32 = 1,
        n_targets: Int32 = 1,
    ) abi("c") -> Int32:
    """ In-place causal imputation of NaN gaps. Returns 0 on success, -1 on failure."""
    if Int(forecaster) == 0 or Int(data) == 0:
        return -1
    try:
        forecaster[].impute(data, t_total=Int(t_total), batch_size=Int(batch_size), n_targets=Int(n_targets))
        return 0
    except:
        return -1

