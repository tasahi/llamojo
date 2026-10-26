"""CLI tool for text completion and inference using llamojo.

Usage:
    mojo run -I src -I vendor/llama2.mojo tools/llama_cli.mojo [model.gguf] [prompt] [n_predict]
"""

from std.env import argv
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
    llama_batch_get_one,
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
)

def main():
    var args = argv()
    var model_path = "test_model.gguf"
    var prompt = "Hello, world!"
    var n_predict = 32

    if len(args) > 1:
        model_path = args[1]
    if len(args) > 2:
        prompt = args[2]
    if len(args) > 3:
        try:
            n_predict = Int(args[3])
        except:
            n_predict = 32

    print("=== llamojo CLI ===")
    print("Model:", model_path)
    print("Prompt:", prompt)
    print("Max tokens:", n_predict)

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
    vocab.init_pointee_move(llama_vocab(model[].gguf))

    # Tokenize input prompt
    var prompt_tokens_buf = unsafe_alloc[llama_token](512)
    var n_tokens = llama_tokenize(vocab, prompt, prompt_tokens_buf, 512, True, False)
    if n_tokens < 0:
        n_tokens = -n_tokens

    print("Prompt tokens count:", n_tokens)

    # Initialize standard sampler chain: Temp(0.8) -> Top-K(40) -> Top-P(0.9) -> Dist(1337)
    var sparams = llama_sampler_chain_default_params()
    var smpl = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(Float32(0.8)))
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40))
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(Float32(0.9)))
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1337))

    # Decode prompt tokens
    var n_vocab = Int(model[].cfg.vocab_size)
    var piece_buf = unsafe_alloc[UInt8](256)
    var candidates_buf = unsafe_alloc[llama_token_data](n_vocab)

    var cur_token: llama_token = -1
    for i in range(Int(n_tokens)):
        var tok = prompt_tokens_buf[unsafe_offset=i]
        var b = llama_batch_get_one(tok, Int32(i))
        _ = llama_decode(ctx, b)
        cur_token = tok

    # Generation loop
    print("\n--- Output ---")
    for step in range(n_predict):
        var logits = llama_get_logits_ith(ctx, 0)
        if Int(logits) == 0:
            break

        for v in range(n_vocab):
            candidates_buf[unsafe_offset=v] = llama_token_data(Int32(v), logits[unsafe_offset=v], Float32(0.0))

        var cur_p = unsafe_alloc[llama_token_data_array](1)
        cur_p.init_pointee_move(llama_token_data_array(candidates_buf, n_vocab, False))

        var next_tok = llama_sampler_sample(smpl, cur_p)
        cur_p.free()

        if next_tok < 0 or llama_vocab_is_eog(vocab, next_tok):
            break

        var plen = llama_token_to_piece(vocab, next_tok, piece_buf, 256)
        if plen > 0:
            var piece_str = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
            print(piece_str, end="")

        var b = llama_batch_get_one(next_tok, Int32(Int(n_tokens) + step))
        _ = llama_decode(ctx, b)

    print("\n--- End of Generation ---")

    candidates_buf.free()
    piece_buf.free()
    prompt_tokens_buf.free()
    llama_sampler_free(smpl)
    vocab.destroy_pointee()
    vocab.free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

