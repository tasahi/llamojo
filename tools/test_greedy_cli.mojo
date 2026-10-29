from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.collections import Span

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
    llama_vocab,
    llama_vocab_p,
    llama_tokenize,
    llama_token_to_piece,
    llama_vocab_is_eog,
)

def main():
    var model_path = "$MODEL_PATH"
    var prompt = "Once upon a time"
    var n_predict = 16

    llama_backend_init()
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    var cparams = llama_context_default_params()
    cparams.n_ctx = 512
    var ctx = llama_init_from_model(model, cparams)

    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))

    var prompt_tokens_buf = unsafe_alloc[llama_token](512)
    var n_tokens = llama_tokenize(vocab, prompt, prompt_tokens_buf, 512, True, False)
    if n_tokens < 0:
        n_tokens = -n_tokens

    var n_vocab = Int(model[].config.vocab_size)
    var piece_buf = unsafe_alloc[UInt8](256)

    # Ingest prompt tokens
    for i in range(Int(n_tokens)):
        var tok = prompt_tokens_buf[unsafe_offset=i]
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tok
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1 if i == Int(n_tokens) - 1 else 0
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    print("--- Greedy Output ---")
    for step in range(n_predict):
        var logits = llama_get_logits_ith(ctx, 0)
        var best_id: Int32 = 0
        var best_val: Float32 = -1e9
        for v in range(n_vocab):
            var val = logits[unsafe_offset=v]
            if val > best_val:
                best_val = val
                best_id = Int32(v)

        if best_id < 0 or llama_vocab_is_eog(vocab, best_id):
            break

        var plen = llama_token_to_piece(vocab, best_id, piece_buf, 256)
        if plen > 0:
            var piece_str = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
            print(piece_str, end="")

        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = best_id
        b.pos[unsafe_offset=0] = Int32(Int(n_tokens) + step)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    print("\n--- End of Generation ---")
    piece_buf.unsafe_free()
    prompt_tokens_buf.unsafe_free()
    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

