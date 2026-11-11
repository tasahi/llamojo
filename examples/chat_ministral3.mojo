""" Interactive and single-turn chat application for Ministral 3 3B in pure Mojo llamojo.

Supports Mistral format:
  [INST] user query [/INST] model response

Usage:
    # Run test prompt:
    mojo run -I src examples/chat_ministral3.mojo models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf

    # Pass custom prompt:
    mojo run -I src examples/chat_ministral3.mojo models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf "Explain quantum computing in one sentence."
"""

from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

from llamojo import (
    null_ptr,
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_context_default_params,
    llama_init_from_model,
    llama_free,
    llama_batch_init,
    llama_batch_free,
    llama_decode,
    llama_get_logits_ith,
    llama_token,
    llama_token_data,
    llama_token_data_array,
    llama_vocab,
    llama_vocab_p,
    llama_tokenize,
    llama_token_to_piece,
    llama_vocab_is_eog,
    llama_sampler_chain_default_params,
    llama_sampler_chain_init,
    llama_sampler_chain_add,
    llama_sampler_init_top_k,
    llama_sampler_init_top_p,
    llama_sampler_init_temp,
    llama_sampler_init_dist,
    llama_sampler_sample,
    llama_sampler_free,
    llama_context_p,
    llama_model_p,
)

def format_mistral_prompt(user_msg: String) -> String:
    """ Format user message using standard Mistral instruction format."""
    return "[INST] " + user_msg + " [/INST]"

def run_chat_turn(
        ctx: llama_context_p,
        model: llama_model_p,
        vocab: llama_vocab_p,
        formatted_prompt: String,
        max_tokens: Int = 128,
        temperature: Float32 = 0.15,
    ) raises:
    """ Ingest user message into KV cache and generate streaming response."""
    var cfg = model[].config
    var n_vocab = Int(cfg.vocab_size)
    var cur_pos = Int(ctx[].n_tokens)

    # Tokenize input prompt
    var prompt_tokens_buf = unsafe_alloc[llama_token](4096)
    var n_tokens = llama_tokenize(vocab, formatted_prompt, prompt_tokens_buf, 4096, add_special=True, parse_special=True)
    if n_tokens < 0:
        n_tokens = -n_tokens

    print("Ingesting prompt (", n_tokens, " tokens): ", end="", flush=True)

    # Ingest prompt tokens into KV cache
    var t_prompt_start = perf_counter_ns()
    for i in range(Int(n_tokens)):
        var tok = prompt_tokens_buf[unsafe_offset=i]
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tok
        b.pos[unsafe_offset=0] = Int32(cur_pos + i)
        b.logits[unsafe_offset=0] = 1 if i == Int(n_tokens) - 1 else 0
        _ = llama_decode(ctx, b)
        llama_batch_free(b)
        print(".", end="", flush=True)
    var t_prompt_end = perf_counter_ns()
    print(" done.")

    prompt_tokens_buf.unsafe_free()

    var prompt_ms = Float64(t_prompt_end - t_prompt_start) / 1_000_000.0
    var prompt_tps = (Float64(n_tokens) / (prompt_ms / 1000.0)) if prompt_ms > 0.0 else 0.0

    print("\nMinistral 3: ", end="")

    # Initialize sampling chain
    var sparams = llama_sampler_chain_default_params()
    sparams.no_perf = True
    var smpl = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature))
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(Float32(0.9)))
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42))

    var piece_buf = unsafe_alloc[UInt8](256)
    var candidates_buf = unsafe_alloc[llama_token_data](n_vocab)

    var gen_count = 0
    var t_gen_start = perf_counter_ns()

    for step in range(max_tokens):
        var logits = llama_get_logits_ith(ctx, 0)
        if Int(logits) == 0:
            break

        var next_tok: llama_token
        if temperature <= 0.0:
            var best_id = 0
            var max_val = logits[unsafe_offset=0]
            for v in range(1, n_vocab):
                var val = logits[unsafe_offset=v]
                if val > max_val:
                    max_val = val
                    best_id = v
            next_tok = llama_token(best_id)
        else:
            for v in range(n_vocab):
                candidates_buf[unsafe_offset=v] = llama_token_data(Int32(v), logits[unsafe_offset=v], Float32(0.0))

            var cur_p = unsafe_alloc[llama_token_data_array](1)
            cur_p.unsafe_write(llama_token_data_array(candidates_buf, n_vocab, False))
            next_tok = llama_sampler_sample(smpl, cur_p)
            cur_p.unsafe_deinit_pointee()
            cur_p.unsafe_free()

        # Stop condition: End-Of-Generation token (EOS / EOT)
        if next_tok < 0 or llama_vocab_is_eog(vocab, next_tok):
            break

        gen_count += 1
        var plen = llama_token_to_piece(vocab, next_tok, piece_buf, 256)
        if plen > 0:
            var piece_str = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
            print(piece_str, end="", flush=True)

        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = next_tok
        b.pos[unsafe_offset=0] = Int32(cur_pos + Int(n_tokens) + step)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    var t_gen_end = perf_counter_ns()
    print("\n")

    piece_buf.unsafe_free()
    candidates_buf.unsafe_free()
    llama_sampler_free(smpl)

    var gen_ms = Float64(t_gen_end - t_gen_start) / 1_000_000.0
    var gen_tps = (Float64(gen_count) / (gen_ms / 1000.0)) if gen_ms > 0.0 else 0.0

    print("------------------------------------------------------------")
    print("Metrics: Prompt ", n_tokens, " tokens @ ", prompt_tps, " tok/s | Generation ", gen_count, " tokens @ ", gen_tps, " tok/s")
    print("------------------------------------------------------------")

def main() raises:
    var args = argv()
    var model_path = "models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
    var user_msg = "Explain quantum computing in simple terms for high school students."

    if len(args) > 1:
        model_path = args[1]
    if len(args) > 2:
        user_msg = args[2]

    print("============================================================")
    print("llamojo Ministral 3 3B Chat Application")
    print("============================================================")
    print("Model:  ", model_path)
    print("User:   ", user_msg)
    print("============================================================")

    llama_backend_init()

    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    if Int(model) == 0:
        print("Error: Could not load model from:", model_path)
        print("Run tools/download_ministral3.sh first to download the weights.")
        llama_backend_free()
        return

    var cfg = model[].config
    print("Model Architecture: ", cfg.arch)
    print("Layers:             ", cfg.n_layers)
    print("Hidden Dim:         ", cfg.hidden_dim)
    print("FFN Dim:            ", cfg.intermediate_dim)
    print("Heads (Q / KV):     ", cfg.n_heads, "/", cfg.n_kv_heads)
    print("Sliding Window:     ", cfg.sliding_window)
    print("RoPE Scaling Type:  ", cfg.rope_scaling_type)
    print("Tied Embeddings:    ", cfg.is_tied_embeddings)

    var cparams = llama_context_default_params()
    cparams.n_ctx = 2048
    cparams.n_batch = 512

    var ctx = llama_init_from_model(model, cparams)
    if Int(ctx) == 0:
        print("Error: Could not allocate llama context.")
        llama_model_free(model)
        llama_backend_free()
        return

    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))

    var formatted = format_mistral_prompt(user_msg)
    run_chat_turn(ctx, model, vocab, formatted, max_tokens=128, temperature=0.15)

    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
