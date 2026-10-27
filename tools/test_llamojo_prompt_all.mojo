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
    llama_vocab_n_tokens,
    llama_token_to_piece,
)

def main() raises:
    llama_backend_init()
    var model_path = "/home/tasahi/.cache/huggingface/hub/blobs/a1/a156058e62f4541ff0b9226ed3bc57fdb30ddc9d266f63cc1716a3eb32ab955c"
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    var cparams = llama_context_default_params()
    cparams.n_ctx = 128
    var ctx = llama_init_from_model(model, cparams)
    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))
    var n_vocab = llama_vocab_n_tokens(vocab)
    var piece_buf = unsafe_alloc[UInt8](256)

    var prompt_tokens = List[llama_token]()
    prompt_tokens.append(2)
    prompt_tokens.append(14946)
    prompt_tokens.append(3324)
    prompt_tokens.append(496)
    prompt_tokens.append(990)

    for i in range(len(prompt_tokens)):
        var tok = prompt_tokens[i]
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tok
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1 if i == len(prompt_tokens) - 1 else 0
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

    var logits = llama_get_logits_ith(ctx, 0)
    print("Top 10 tokens at pos 4:")
    # Find top 10
    for rank in range(10):
        var best_id: Int32 = -1
        var best_val: Float32 = -1e9
        for v in range(n_vocab):
            var val = logits[unsafe_offset=v]
            if val > best_val:
                best_val = val
                best_id = Int32(v)
        var plen = llama_token_to_piece(vocab, best_id, piece_buf, 256)
        var p = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
        print("  #", rank, ":", best_id, "(", best_val, ") piece=", repr(p))
        logits[unsafe_offset=Int(best_id)] = -1e10

    piece_buf.unsafe_free()
    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

