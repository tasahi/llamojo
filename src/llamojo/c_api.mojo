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
    llama_sampler_sample as _sampler_sample,
    llama_sampler_free as _sampler_free,
    llama_vocab,
    llama_vocab_p,
    llama_tokenize as _tokenize,
    llama_token_to_piece as _token_to_piece,
    llama_vocab_bos as _vocab_bos,
    llama_vocab_eos as _vocab_eos,
    llama_vocab_eot as _vocab_eot,
    llama_vocab_is_eog as _vocab_is_eog,
    llama_vocab_n_tokens as _vocab_n_tokens,
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
def llama_tokenize(
        vocab: llama_vocab_p,
        text_cstr: BytePtr,
        text_len: Int32,
        tokens_out: TokenPtr,
        max_tokens: Int32,
        add_special: Bool,
        parse_special: Bool,
    ) abi("c") -> Int32:
    """ Tokenize raw text into an array of tokens."""
    if Int(vocab) == 0:
        return 0
    var s = String()
    for i in range(Int(text_len)):
        s += chr(Int(text_cstr[unsafe_offset=i]))
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
