"""Benchmark tool measuring prompt evaluation and token generation speed.

Usage:
    mojo run -I src -I vendor/llama2.mojo tools/llama_bench.mojo [model.gguf] [n_prompt] [n_gen]
"""

from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

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
    llama_token_data,
    llama_token_data_array,
    llama_sampler_init_greedy,
    llama_sampler_sample,
    llama_sampler_free,
)

def main():
    var args = argv()
    var model_path = "test_model.gguf"
    var n_prompt = 64
    var n_gen = 32

    var idx = 1
    while idx < len(args):
        var a = args[idx]
        if a == "-m" and idx + 1 < len(args):
            model_path = args[idx + 1]
            idx += 2
        elif a == "-p" and idx + 1 < len(args):
            try:
                n_prompt = Int(args[idx + 1])
            except:
                n_prompt = 64
            idx += 2
        elif a == "-n" and idx + 1 < len(args):
            try:
                n_gen = Int(args[idx + 1])
            except:
                n_gen = 32
            idx += 2
        else:
            if idx == 1:
                model_path = a
            elif idx == 2:
                try:
                    n_prompt = Int(a)
                except:
                    n_prompt = 64
            elif idx == 3:
                try:
                    n_gen = Int(a)
                except:
                    n_gen = 32
            idx += 1

    print("=== llamojo Benchmark ===")
    print("Model:", model_path)
    print("Prompt tokens:", n_prompt)
    print("Generation tokens:", n_gen)

    llama_backend_init()

    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    if Int(model) == 0:
        print("Failed to load model from:", model_path)
        llama_backend_free()
        return

    var cparams = llama_context_default_params()
    cparams.n_ctx = 1024
    var ctx = llama_init_from_model(model, cparams)
    if Int(ctx) == 0:
        print("Failed to initialize context")
        llama_model_free(model)
        llama_backend_free()
        return

    var n_vocab = Int(model[].config.vocab_size)
    var smpl = llama_sampler_init_greedy()
    var candidates_buf = unsafe_alloc[llama_token_data](n_vocab)

    # 1. Benchmark Prompt Processing
    var t0 = perf_counter_ns()
    for i in range(n_prompt):
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = 1
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1 if i == n_prompt - 1 else 0
        _ = llama_decode(ctx, b)
        llama_batch_free(b)
    var t1 = perf_counter_ns()

    var prompt_sec = Float64(t1 - t0) / 1e9
    var prompt_tps = Float64(n_prompt) / prompt_sec

    # 2. Benchmark Token Generation
    var cur_tok: Int32 = 1
    var t2 = perf_counter_ns()
    for i in range(n_gen):
        var logits = llama_get_logits_ith(ctx, 0)
        for v in range(n_vocab):
            candidates_buf[unsafe_offset=v] = llama_token_data(Int32(v), logits[unsafe_offset=v], Float32(0.0))

        var cur_p = unsafe_alloc[llama_token_data_array](1)
        cur_p.unsafe_write(llama_token_data_array(candidates_buf, n_vocab, False))
        cur_tok = llama_sampler_sample(smpl, cur_p)
        cur_p.unsafe_deinit_pointee()
        cur_p.unsafe_free()

        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = cur_tok
        b.pos[unsafe_offset=0] = Int32(n_prompt + i)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)
    var t3 = perf_counter_ns()

    var gen_sec = Float64(t3 - t2) / 1e9
    var gen_tps = Float64(n_gen) / gen_sec

    print("\n--- Benchmark Results ---")
    print("Prompt processing: ", prompt_tps, "tok/s (", prompt_sec, "s total)")
    print("Token generation:  ", gen_tps, "tok/s (", gen_sec, "s total)")

    candidates_buf.unsafe_free()
    llama_sampler_free(smpl)
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
