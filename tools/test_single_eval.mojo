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

def eval_tokens(tokens: List[llama_token]) raises:
    llama_backend_init()
    var model_path = "$MODEL_PATH"
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    var cparams = llama_context_default_params()
    cparams.n_ctx = 128
    var ctx = llama_init_from_model(model, cparams)
    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))
    var n_vocab = llama_vocab_n_tokens(vocab)
    var piece_buf = unsafe_alloc[UInt8](256)

    for i in range(len(tokens)):
        var tok = tokens[i]
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tok
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)

        var logits = llama_get_logits_ith(ctx, 0)
        var best_id: Int32 = 0
        var best_val: Float32 = -1e9
        for v in range(n_vocab):
            var val = logits[unsafe_offset=v]
            if val > best_val:
                best_val = val
                best_id = Int32(v)
        var plen = llama_token_to_piece(vocab, best_id, piece_buf, 256)
        var piece_str = String(unsafe_from_utf8=Span(unsafe_ptr=piece_buf, length=Int(plen)))
        print("Step", i, "tok", tok, ": top=", best_id, "(", best_val, ") piece=", repr(piece_str))

    piece_buf.unsafe_free()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

def main() raises:
    var toks = List[llama_token]()
    toks.append(2)
    toks.append(14946)
    toks.append(3324)
    toks.append(496)
    toks.append(990)
    eval_tokens(toks)

