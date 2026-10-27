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
    var piece_buf = unsafe_alloc[UInt8](256)

    # Step 0: token 2
    var b0 = llama_batch_init(1, embd=0, n_seq_max=1)
    b0.token[unsafe_offset=0] = 2
    b0.pos[unsafe_offset=0] = 0
    b0.logits[unsafe_offset=0] = 1
    _ = llama_decode(ctx, b0)
    llama_batch_free(b0)

    # Step 1: token 14946
    var b1 = llama_batch_init(1, embd=0, n_seq_max=1)
    b1.token[unsafe_offset=0] = 14946
    b1.pos[unsafe_offset=0] = 1
    b1.logits[unsafe_offset=0] = 1
    _ = llama_decode(ctx, b1)
    llama_batch_free(b1)

    var logits1 = llama_get_logits_ith(ctx, 0)
    print("Step 1 (tok 14946) logit for 107 ('\\n'):", logits1[107])
    print("Step 1 (tok 14946) logit for 506 (' the'):", logits1[506])

    piece_buf.unsafe_free()
    vocab.unsafe_free()
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

