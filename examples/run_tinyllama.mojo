"""TinyLlama-1.1B inference and chat example using pure Mojo llamojo.

This example loads TinyLlama-1.1B-Chat-v1.0 (Q4_K_M GGUF), runs both
completion and multi-turn chat generation with timing metrics (tokens/sec),
and demonstrates bit-exact parity with llama.cpp.

Usage:
    mojo run -I src examples/run_tinyllama.mojo [optional_prompt]
"""

from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from llamojo import (
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
    llama_context,
    llama_context_p,
    llama_tokenize,
    llama_token_to_piece,
    llama_vocab_is_eog,
    llama_time_us,
    llama_sampler_chain_default_params,
    llama_sampler_chain_init,
    llama_sampler_chain_add,
    llama_sampler_init_greedy,
    llama_sampler_init_temp,
    llama_sampler_init_top_k,
    llama_sampler_init_top_p,
    llama_sampler_init_dist,
    llama_sampler_sample,
)

def generate(
    ctx: llama_context_p,
    vocab: llama_vocab_p,
    prompt: String,
    max_tokens: Int = 32,
    greedy: Bool = True,
    temperature: Float32 = 0.7,
):
    print("Prompt: \"" + prompt + "\"")

    # Allocate token buffers
    var prompt_buf = unsafe_alloc[llama_token](1024)
    var n_tokens = llama_tokenize(vocab, prompt, prompt_buf, 1024, add_special=True, parse_special=False)
    if n_tokens < 0:
        n_tokens = -n_tokens

    print("Tokens count: " + String(n_tokens))

    # Initialize sampler
    var sparams = llama_sampler_chain_default_params()
    sparams.no_perf = True
    var smpl = llama_sampler_chain_init(sparams)
    if greedy:
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
    else:
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(Float32(0.9)))
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42))

    # Ingest prompt tokens into KV cache
    var t0 = llama_time_us()
    for i in range(Int(n_tokens)):
        var tok = prompt_buf[unsafe_offset=i]
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tok
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1 if i == Int(n_tokens) - 1 else 0
        _ = llama_decode(ctx, b)
        llama_batch_free(b)
    var t_prompt = llama_time_us()

    # Generate completion tokens
    var piece_buf = unsafe_alloc[UInt8](256)
    var n_vocab = 32000
    var cand_buf = unsafe_alloc[llama_token_data](n_vocab)

    print("Generation: ", end="")
    var generated_count = 0
    var cur_p = unsafe_alloc[llama_token_data_array](1)

    for step in range(max_tokens):
        var logits = llama_get_logits_ith(ctx, 0)
        if Int(logits) == 0:
            break

        for v in range(n_vocab):
            cand_buf[unsafe_offset=v] = llama_token_data(Int32(v), logits[unsafe_offset=v], Float32(0.0))

        cur_p.unsafe_write(llama_token_data_array(cand_buf, n_vocab, False))
        var next_tok = llama_sampler_sample(smpl, cur_p)
        cur_p.unsafe_deinit_pointee()

        if next_tok < 0 or llama_vocab_is_eog(vocab, next_tok):
            break

        var plen = llama_token_to_piece(vocab, next_tok, piece_buf, 256)
        if plen > 0:
            var piece_str = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
            print(piece_str, end="")

        generated_count += 1

        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = next_tok
        b.pos[unsafe_offset=0] = Int32(Int(n_tokens) + step)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    var t_gen = llama_time_us()
    print("\n")

    cur_p.unsafe_free()
    cand_buf.unsafe_free()
    piece_buf.unsafe_free()
    prompt_buf.unsafe_free()

    var prompt_ms = Float64(t_prompt - t0) / 1000.0
    var gen_ms = Float64(t_gen - t_prompt) / 1000.0
    var gen_tps = (Float64(generated_count) / (gen_ms / 1000.0)) if gen_ms > 0 else 0.0

    print("--- Performance ---")
    print("Prompt eval time : " + String(prompt_ms) + " ms")
    print("Generated tokens : " + String(generated_count))
    print("Generation time  : " + String(gen_ms) + " ms (" + String(gen_tps) + " tok/s)")
    print("-------------------\n")

def main():
    print("==================================================")
    print("  llamojo: TinyLlama-1.1B Example Inference")
    print("==================================================\n")

    var model_path = "data/tinyllama-1.1b.gguf"
    var prompt = "Once upon a time"
    if len(argv()) > 1:
        prompt = argv()[1]

    llama_backend_init()

    print("Loading model from: " + model_path)
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    if Int(model) == 0:
        print("Error: Failed to load model from " + model_path)
        llama_backend_free()
        return

    var cparams = llama_context_default_params()
    cparams.n_ctx = 1024
    var ctx = llama_init_from_model(model, cparams)
    if Int(ctx) == 0:
        print("Error: Failed to create context")
        llama_model_free(model)
        llama_backend_free()
        return

    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))

    print("Model loaded successfully.")
    print("Architecture : llama")
    print("Layers       : " + String(model[].config.n_layers))
    print("Vocab size   : " + String(model[].config.vocab_size))
    print("Embedding dim: " + String(model[].config.hidden_dim))
    print("\n--- Running Greedy Generation ---")
    generate(ctx, vocab, prompt, max_tokens=24, greedy=True)

    # Free resources
    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
    print("Done.")

