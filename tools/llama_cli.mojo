"""CLI tool for text completion and inference using llamojo.

Usage:
    mojo run -I src tools/llama_cli.mojo [model.gguf] [prompt] [n_predict]
"""

from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

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
    llama_vocab_eos,
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
    llama_sampler,
    llama_sampler_p,
    llama_sampler_params,
)

def main():
    var args = argv()
    var model_path = "test_model.gguf"
    var prompt = "Once upon a time"
    var n_predict = 32
    var temp: Float32 = 0.8

    # Parse arguments (supports positional or flags -m, -p, -n, --temp)
    var idx = 1
    while idx < len(args):
        var a = args[idx]
        if a == "-m" and idx + 1 < len(args):
            model_path = args[idx + 1]
            idx += 2
        elif a == "-p" and idx + 1 < len(args):
            prompt = args[idx + 1]
            idx += 2
        elif a == "-n" and idx + 1 < len(args):
            try:
                n_predict = Int(args[idx + 1])
            except:
                n_predict = 32
            idx += 2
        elif (a == "--temp" or a == "-t") and idx + 1 < len(args):
            try:
                temp = Float32(Float64(args[idx + 1]))
            except:
                temp = 0.8
            idx += 2
        else:
            if idx == 1:
                model_path = a
            elif idx == 2:
                prompt = a
            elif idx == 3:
                try:
                    n_predict = Int(a)
                except:
                    n_predict = 32
            idx += 1

    print("=== llamojo CLI ===")
    print("Model:", model_path)
    print("Prompt:", prompt)
    print("Max tokens:", n_predict)
    print("Temperature:", temp)

    llama_backend_init()

    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    if Int(model) == 0:
        print("Failed to load model from:", model_path)
        llama_backend_free()
        return

    var cparams = llama_context_default_params()
    cparams.n_ctx = 512
    var ctx = llama_init_from_model(model, cparams)
    if Int(ctx) == 0:
        print("Failed to initialize context from model")
        llama_model_free(model)
        llama_backend_free()
        return

    # Setup vocabulary from model GGUF context
    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))

    # Tokenize input prompt
    var prompt_tokens_buf = unsafe_alloc[llama_token](512)
    var n_tokens = llama_tokenize(vocab, prompt, prompt_tokens_buf, 512, True, False)
    if n_tokens < 0:
        n_tokens = -n_tokens

    print("Prompt tokens count:", n_tokens)

    # Initialize standard sampler chain or greedy
    var smpl: llama_sampler_p = null_ptr[llama_sampler]()
    if temp > 0.0:
        var sparams = llama_sampler_chain_default_params()
        smpl = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp))
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(Float32(0.9)))
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(1337))

    # Decode prompt tokens
    var n_vocab = Int(model[].config.vocab_size)
    var piece_buf = unsafe_alloc[UInt8](256)
    var candidates_buf = unsafe_alloc[llama_token_data](n_vocab)

    for i in range(Int(n_tokens)):
        var tok = prompt_tokens_buf[unsafe_offset=i]
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tok
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1 if i == Int(n_tokens) - 1 else 0
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    # Generation loop
    print("\n--- Output ---")
    for step in range(n_predict):
        var logits = llama_get_logits_ith(ctx, 0)
        if Int(logits) == 0:
            break

        var next_tok: llama_token = -1
        if temp <= 0.0:
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

        if next_tok < 0 or llama_vocab_is_eog(vocab, next_tok):
            break

        var plen = llama_token_to_piece(vocab, next_tok, piece_buf, 256)
        if plen > 0:
            var piece_str = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
            print(piece_str, end="")

        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = next_tok
        b.pos[unsafe_offset=0] = Int32(Int(n_tokens) + step)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    print("\n--- End of Generation ---")

    candidates_buf.unsafe_free()
    piece_buf.unsafe_free()
    prompt_tokens_buf.unsafe_free()
    if Int(smpl) != 0:
        llama_sampler_free(smpl)
    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
