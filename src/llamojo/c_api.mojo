""" C Foreign Function Interface (ABI) Export Layer for llamojo.

Exports standard `llama.h` C-ABI functions to allow external C, C++, Rust,
and Python (`ctypes`, `cffi`, `llama-cpp-python`) applications to link directly
against `libllama.so`.
"""

from std.ffi import c_char
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from llamojo.types import (
    llama_token,
    llama_pos,
    llama_seq_id,
    llama_model_params,
    llama_context_params,
    llama_model_quantize_params,
    llama_batch,
    llama_token_data,
    llama_token_data_array,
    null_ptr,
)
from llamojo.llama import (
    llama_model,
    llama_model_p,
    llama_context,
    llama_context_p,
    llama_backend_init as _backend_init,
    llama_backend_free as _backend_free,
    llama_numa_init as _numa_init,
    llama_time_us as _time_us,
    llama_max_devices as _max_devices,
    llama_supports_mmap as _supports_mmap,
    llama_supports_mlock as _supports_mlock,
    llama_supports_gpu_offload as _supports_gpu_offload,
    llama_model_default_params as _model_default_params,
    llama_context_default_params as _context_default_params,
    llama_model_quantize_default_params as _model_quantize_default_params,
    llama_model_load_from_file as _model_load_from_file,
    llama_model_free as _model_free,
    llama_init_from_model as _init_from_model,
    llama_free as _free_context,
    llama_batch_init as _batch_init,
    llama_batch_free as _batch_free,
    llama_decode as _decode,
    llama_encode as _encode,
    llama_get_logits as _get_logits,
    llama_get_logits_ith as _get_logits_ith,
    llama_get_embeddings as _get_embeddings,
    llama_kv_cache_clear as _kv_cache_clear,
    llama_kv_cache_seq_rm as _kv_cache_seq_rm,
    llama_kv_cache_seq_cp as _kv_cache_seq_cp,
    llama_kv_cache_seq_keep as _kv_cache_seq_keep,
    llama_kv_cache_seq_add as _kv_cache_seq_add,
    llama_model_n_vocab as _model_n_vocab,
    llama_model_n_embd as _model_n_embd,
    llama_model_n_layer as _model_n_layer,
    llama_model_n_head as _model_n_head,
    llama_model_n_head_kv as _model_n_head_kv,
    llama_model_n_ctx_train as _model_n_ctx_train,
    llama_model_n_params as _model_n_params,
    llama_model_size as _model_size,
    llama_model_desc as _model_desc,
    llama_adapter_lora_apply as _adapter_lora_apply,
)
from llamojo.lora import (
    llama_lora_adapter,
    llama_lora_adapter_p,
    llama_lora_adapter_init as _lora_adapter_init,
    llama_lora_adapter_set as _lora_adapter_set,
    llama_lora_adapter_free as _lora_adapter_free,
)
from llamojo.grammar import (
    GbnfGrammar,
    llama_grammar_p,
    llama_grammar_init as _grammar_init,
    llama_grammar_free as _grammar_free,
)
from llamojo.backend.micro_arch import (
    numa_is_available as _numa_is_available,
    numa_pin_thread as _numa_pin_thread,
    numa_node_count as _numa_node_count,
)
from llamojo.vocab import (
    llama_vocab,
    llama_vocab_p,
    llama_tokenize as _tokenize,
    llama_token_to_piece as _token_to_piece,
    llama_detokenize as _detokenize,
    llama_vocab_bos as _vocab_bos,
    llama_vocab_eos as _vocab_eos,
    llama_vocab_eot as _vocab_eot,
    llama_vocab_is_eog as _vocab_is_eog,
    llama_vocab_n_tokens as _vocab_n_tokens,
    llama_vocab_type as _vocab_type,
    llama_vocab_get_score as _vocab_get_score,
)
from llamojo.sampler import (
    llama_sampler,
    llama_sampler_p,
    llama_sampler_params,
    llama_sampler_chain_default_params as _sampler_chain_default_params,
    llama_sampler_chain_init as _sampler_chain_init,
    llama_sampler_chain_add as _sampler_chain_add,
    llama_sampler_init_top_k as _sampler_init_top_k,
    llama_sampler_init_top_p as _sampler_init_top_p,
    llama_sampler_init_min_p as _sampler_init_min_p,
    llama_sampler_init_temp as _sampler_init_temp,
    llama_sampler_init_penalties as _sampler_init_penalties,
    llama_sampler_init_greedy as _sampler_init_greedy,
    llama_sampler_init_dist as _sampler_init_dist,
    llama_sampler_init_grammar as _sampler_init_grammar,
    llama_sampler_accept as _sampler_accept,
    llama_sampler_apply as _sampler_apply,
    llama_sampler_sample as _sampler_sample,
    llama_sampler_free as _sampler_free,
)
from llamojo.quantize import (
    llama_model_quantize as _model_quantize,
)

# ─── Helper for C-string conversions ─────────────────────────────────────────

@always_inline
def _c_str_to_string(c_str: Pointer[UInt8, MutUntrackedOrigin]) -> String:
    if Int(c_str) == 0:
        return ""
    var len = 0
    while c_str[unsafe_offset=len] != 0:
        len += 1
    return String(unsafe_from_utf8=Span(unsafe_ptr=c_str, length=len))

# ─── Backend & Global Lifecycle Exports ───────────────────────────────────────

@export("llama_backend_init")
def llama_backend_init() abi("C"):
    _backend_init()

@export("llama_backend_free")
def llama_backend_free() abi("C"):
    _backend_free()

@export("llama_numa_init")
def llama_numa_init(numa: Int32) abi("C"):
    _numa_init(numa)

@export("llama_time_us")
def llama_time_us() abi("C") -> Int64:
    return _time_us()

@export("llama_max_devices")
def llama_max_devices() abi("C") -> Int64:
    return Int64(_max_devices())

@export("llama_supports_mmap")
def llama_supports_mmap() abi("C") -> Bool:
    return _supports_mmap()

@export("llama_supports_mlock")
def llama_supports_mlock() abi("C") -> Bool:
    return _supports_mlock()

@export("llama_supports_gpu_offload")
def llama_supports_gpu_offload() abi("C") -> Bool:
    return _supports_gpu_offload()

# ─── Default Parameter Exports ────────────────────────────────────────────────

@export("llama_model_default_params")
def llama_model_default_params() abi("C") -> llama_model_params:
    return _model_default_params()

@export("llama_context_default_params")
def llama_context_default_params() abi("C") -> llama_context_params:
    return _context_default_params()

@export("llama_model_quantize_default_params")
def llama_model_quantize_default_params() abi("C") -> llama_model_quantize_params:
    return _model_quantize_default_params()

# ─── Model Lifecycle & Reflection Exports ─────────────────────────────────────

@export("llama_load_model_from_file")
def llama_load_model_from_file(
    path_model: Pointer[UInt8, MutUntrackedOrigin],
    params: llama_model_params,
) abi("C") -> llama_model_p:
    var path_str = _c_str_to_string(path_model)
    return _model_load_from_file(path_str, params)

@export("llama_model_load_from_file")
def llama_model_load_from_file(
    path_model: Pointer[UInt8, MutUntrackedOrigin],
    params: llama_model_params,
) abi("C") -> llama_model_p:
    return llama_load_model_from_file(path_model, params)

@export("llama_free_model")
def llama_free_model(model: llama_model_p) abi("C"):
    _model_free(model)

@export("llama_model_free")
def llama_model_free(model: llama_model_p) abi("C"):
    _model_free(model)

@export("llama_model_n_vocab")
def llama_model_n_vocab(model: llama_model_p) abi("C") -> Int32:
    return _model_n_vocab(model)

@export("llama_model_n_embd")
def llama_model_n_embd(model: llama_model_p) abi("C") -> Int32:
    return _model_n_embd(model)

@export("llama_model_n_layer")
def llama_model_n_layer(model: llama_model_p) abi("C") -> Int32:
    return _model_n_layer(model)

@export("llama_model_n_head")
def llama_model_n_head(model: llama_model_p) abi("C") -> Int32:
    return _model_n_head(model)

@export("llama_model_n_head_kv")
def llama_model_n_head_kv(model: llama_model_p) abi("C") -> Int32:
    return _model_n_head_kv(model)

@export("llama_model_n_ctx_train")
def llama_model_n_ctx_train(model: llama_model_p) abi("C") -> Int32:
    return _model_n_ctx_train(model)

@export("llama_model_n_params")
def llama_model_n_params(model: llama_model_p) abi("C") -> UInt64:
    return _model_n_params(model)

@export("llama_model_size")
def llama_model_size(model: llama_model_p) abi("C") -> UInt64:
    return _model_size(model)

@export("llama_model_desc")
def llama_model_desc(
    model: llama_model_p,
    buf: Pointer[UInt8, MutUntrackedOrigin],
    buf_size: Int64,
) abi("C") -> Int32:
    if Int(buf) == 0 or buf_size <= 0:
        return 0
    var desc = _model_desc(model)
    var blen = desc.byte_length()
    var write_len = min(blen, Int(buf_size) - 1)
    var ptr = desc.unsafe_ptr()
    for i in range(write_len):
        buf[unsafe_offset=i] = ptr[unsafe_offset=i]
    buf[unsafe_offset=write_len] = 0
    return Int32(write_len)

# ─── Context Lifecycle & Forward Pass Exports ────────────────────────────────

@export("llama_init_from_model")
def llama_init_from_model(
    model: llama_model_p,
    params: llama_context_params,
) abi("C") -> llama_context_p:
    return _init_from_model(model, params)

@export("llama_new_context_with_model")
def llama_new_context_with_model(
    model: llama_model_p,
    params: llama_context_params,
) abi("C") -> llama_context_p:
    return _init_from_model(model, params)

@export("llama_free")
def llama_free(ctx: llama_context_p) abi("C"):
    _free_context(ctx)

@export("llama_decode")
def llama_decode(ctx: llama_context_p, batch: llama_batch) abi("C") -> Int32:
    return _decode(ctx, batch)

@export("llama_encode")
def llama_encode(ctx: llama_context_p, batch: llama_batch) abi("C") -> Int32:
    return _encode(ctx, batch)

@export("llama_get_logits")
def llama_get_logits(ctx: llama_context_p) abi("C") -> Pointer[Float32, MutUntrackedOrigin]:
    return _get_logits(ctx)

@export("llama_get_logits_ith")
def llama_get_logits_ith(ctx: llama_context_p, i: Int32) abi("C") -> Pointer[Float32, MutUntrackedOrigin]:
    return _get_logits_ith(ctx, i)

@export("llama_get_embeddings")
def llama_get_embeddings(ctx: llama_context_p) abi("C") -> Pointer[Float32, MutUntrackedOrigin]:
    return _get_embeddings(ctx)

# ─── KV Cache Operations Exports ─────────────────────────────────────────────

@export("llama_kv_cache_clear")
def llama_kv_cache_clear(ctx: llama_context_p) abi("C"):
    _kv_cache_clear(ctx)

@export("llama_kv_cache_seq_rm")
def llama_kv_cache_seq_rm(
    ctx: llama_context_p,
    seq_id: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos,
) abi("C") -> Bool:
    return _kv_cache_seq_rm(ctx, seq_id, p0, p1)

@export("llama_kv_cache_seq_cp")
def llama_kv_cache_seq_cp(
    ctx: llama_context_p,
    seq_id_src: llama_seq_id,
    seq_id_dst: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos,
) abi("C"):
    _kv_cache_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1)

@export("llama_kv_cache_seq_keep")
def llama_kv_cache_seq_keep(ctx: llama_context_p, seq_id: llama_seq_id) abi("C"):
    _kv_cache_seq_keep(ctx, seq_id)

@export("llama_kv_cache_seq_add")
def llama_kv_cache_seq_add(
    ctx: llama_context_p,
    seq_id: llama_seq_id,
    p0: llama_pos,
    p1: llama_pos,
    delta: llama_pos,
) abi("C"):
    _kv_cache_seq_add(ctx, seq_id, p0, p1, delta)

# ─── Batch Lifecycle Exports ──────────────────────────────────────────────────

@export("llama_batch_init")
def llama_batch_init(n_tokens: Int32, embd: Int32, n_seq_max: Int32) abi("C") -> llama_batch:
    return _batch_init(n_tokens, embd, n_seq_max)

@export("llama_batch_free")
def llama_batch_free(var batch: llama_batch) abi("C"):
    _batch_free(batch)

# ─── Vocabulary & Tokenizer Exports ──────────────────────────────────────────

@export("llama_model_get_vocab")
def llama_model_get_vocab(model: llama_model_p) abi("C") -> llama_vocab_p:
    if Int(model) == 0:
        return null_ptr[llama_vocab]()
    var v_ptr = unsafe_alloc[llama_vocab](1)
    v_ptr.unsafe_write(llama_vocab(model[].gguf))
    return v_ptr

@export("llama_tokenize")
def llama_tokenize(
    vocab: llama_vocab_p,
    text: Pointer[UInt8, MutUntrackedOrigin],
    text_len: Int32,
    tokens: Pointer[llama_token, MutUntrackedOrigin],
    n_tokens_max: Int32,
    add_special: Bool,
    parse_special: Bool,
) abi("C") -> Int32:
    if Int(text) == 0 or text_len <= 0:
        return 0
    var text_str = String(unsafe_from_utf8=Span(unsafe_ptr=text, length=Int(text_len)))
    return _tokenize(vocab, text_str, tokens, n_tokens_max, add_special, parse_special)

@export("llama_token_to_piece")
def llama_token_to_piece(
    vocab: llama_vocab_p,
    token: llama_token,
    buf: Pointer[UInt8, MutUntrackedOrigin],
    length: Int32,
    lstrip: Int32,
    special: Bool,
) abi("C") -> Int32:
    return _token_to_piece(vocab, token, buf, length, lstrip, special)

@export("llama_detokenize")
def llama_detokenize(
    vocab: llama_vocab_p,
    tokens: Pointer[llama_token, MutUntrackedOrigin],
    n_tokens: Int32,
    text_out: Pointer[UInt8, MutUntrackedOrigin],
    text_len_max: Int32,
    remove_special: Bool,
    unparse_special: Bool,
) abi("C") -> Int32:
    return _detokenize(vocab, tokens, n_tokens, text_out, text_len_max, remove_special, unparse_special)

@export("llama_vocab_bos")
def llama_vocab_bos(vocab: llama_vocab_p) abi("C") -> llama_token:
    return _vocab_bos(vocab)

@export("llama_vocab_eos")
def llama_vocab_eos(vocab: llama_vocab_p) abi("C") -> llama_token:
    return _vocab_eos(vocab)

@export("llama_vocab_eot")
def llama_vocab_eot(vocab: llama_vocab_p) abi("C") -> llama_token:
    return _vocab_eot(vocab)

@export("llama_vocab_is_eog")
def llama_vocab_is_eog(vocab: llama_vocab_p, token: llama_token) abi("C") -> Bool:
    return _vocab_is_eog(vocab, token)

@export("llama_vocab_n_tokens")
def llama_vocab_n_tokens(vocab: llama_vocab_p) abi("C") -> Int32:
    return _vocab_n_tokens(vocab)

@export("llama_vocab_type")
def llama_vocab_type(vocab: llama_vocab_p) abi("C") -> Int32:
    return _vocab_type(vocab)

@export("llama_vocab_get_score")
def llama_vocab_get_score(vocab: llama_vocab_p, token: llama_token) abi("C") -> Float32:
    return _vocab_get_score(vocab, token)

# ─── Composable Sampler Exports ──────────────────────────────────────────────

@export("llama_sampler_chain_default_params")
def llama_sampler_chain_default_params() abi("C") -> llama_sampler_params:
    return _sampler_chain_default_params()

@export("llama_sampler_chain_init")
def llama_sampler_chain_init(params: llama_sampler_params) abi("C") -> llama_sampler_p:
    return _sampler_chain_init(params)

@export("llama_sampler_chain_add")
def llama_sampler_chain_add(chain: llama_sampler_p, smpl: llama_sampler_p) abi("C"):
    _sampler_chain_add(chain, smpl)

@export("llama_sampler_init_top_k")
def llama_sampler_init_top_k(k: Int32) abi("C") -> llama_sampler_p:
    return _sampler_init_top_k(k)

@export("llama_sampler_init_top_p")
def llama_sampler_init_top_p(p_val: Float32, min_keep: Int64) abi("C") -> llama_sampler_p:
    return _sampler_init_top_p(p_val, Int(min_keep))

@export("llama_sampler_init_min_p")
def llama_sampler_init_min_p(p_val: Float32, min_keep: Int64) abi("C") -> llama_sampler_p:
    return _sampler_init_min_p(p_val, Int(min_keep))

@export("llama_sampler_init_temp")
def llama_sampler_init_temp(t: Float32) abi("C") -> llama_sampler_p:
    return _sampler_init_temp(t)

@export("llama_sampler_init_penalties")
def llama_sampler_init_penalties(
    n_prev: Int32,
    repeat_penalty: Float32,
    freq_penalty: Float32,
    presence_penalty: Float32,
) abi("C") -> llama_sampler_p:
    return _sampler_init_penalties(n_prev, repeat_penalty, freq_penalty, presence_penalty)

@export("llama_sampler_init_greedy")
def llama_sampler_init_greedy() abi("C") -> llama_sampler_p:
    return _sampler_init_greedy()

@export("llama_sampler_init_dist")
def llama_sampler_init_dist(seed: UInt32) abi("C") -> llama_sampler_p:
    return _sampler_init_dist(seed)

@export("llama_sampler_accept")
def llama_sampler_accept(smpl: llama_sampler_p, token: llama_token) abi("C"):
    _sampler_accept(smpl, token)

@export("llama_sampler_apply")
def llama_sampler_apply(
    smpl: llama_sampler_p,
    cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin],
) abi("C"):
    _sampler_apply(smpl, cur_p)

@export("llama_sampler_sample")
def llama_sampler_sample(
    smpl: llama_sampler_p,
    cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin],
) abi("C") -> llama_token:
    return _sampler_sample(smpl, cur_p)

@export("llama_sampler_free")
def llama_sampler_free(smpl: llama_sampler_p) abi("C"):
    _sampler_free(smpl)

# ─── Model Quantize Export ───────────────────────────────────────────────────

@export("llama_model_quantize")
def llama_model_quantize(
    fname_inp: Pointer[UInt8, MutUntrackedOrigin],
    fname_out: Pointer[UInt8, MutUntrackedOrigin],
    params: llama_model_quantize_params,
) abi("C") -> Int32:
    var inp_str = _c_str_to_string(fname_inp)
    var out_str = _c_str_to_string(fname_out)
    return _model_quantize(inp_str, out_str, params)

# ─── LoRA Adapter Export (Phase 8) ───────────────────────────────────────────

@export("llama_lora_adapter_init")
def llama_lora_adapter_init(
    model: llama_model_p,
    path_lora: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> llama_lora_adapter_p:
    var path_str = _c_str_to_string(path_lora)
    return _lora_adapter_init(model.unsafe_bitcast[NoneType](), path_str)

@export("llama_lora_adapter_set")
def llama_lora_adapter_set(
    adapter: llama_lora_adapter_p,
    scale: Float32,
) abi("C") -> Int32:
    return _lora_adapter_set(adapter, scale)

@export("llama_lora_adapter_free")
def llama_lora_adapter_free(adapter: llama_lora_adapter_p) abi("C"):
    _lora_adapter_free(adapter)

@export("llama_adapter_lora_apply")
def llama_adapter_lora_apply(
    model: llama_model_p,
    path_lora: Pointer[UInt8, MutUntrackedOrigin],
    scale: Float32,
) abi("C") -> Int32:
    var path_str = _c_str_to_string(path_lora)
    return _adapter_lora_apply(model, path_str, scale)

# ─── GBNF Grammar Export (Phase 9) ───────────────────────────────────────────

@export("llama_grammar_init")
def llama_grammar_init(
    grammar_str: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> llama_grammar_p:
    var g_str = _c_str_to_string(grammar_str)
    return _grammar_init(g_str)

@export("llama_grammar_free")
def llama_grammar_free(grammar: llama_grammar_p) abi("C"):
    _grammar_free(grammar)

@export("llama_sampler_init_grammar")
def llama_sampler_init_grammar(
    grammar_str: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> llama_sampler_p:
    var g_str = _c_str_to_string(grammar_str)
    return _sampler_init_grammar(g_str)

# ─── Micro-Architecture & NUMA Exports (Phase 11) ────────────────────────────

@export("llama_numa_is_available")
def llama_numa_is_available() abi("C") -> Bool:
    return _numa_is_available()

@export("llama_numa_node_count")
def llama_numa_node_count() abi("C") -> Int32:
    return Int32(_numa_node_count())

@export("llama_numa_pin_thread")
def llama_numa_pin_thread(core_id: Int32) abi("C") -> Bool:
    return _numa_pin_thread(Int(core_id))