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
)
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

def main():
    llama_backend_init()
    var mp = llama_model_default_params()
    var model = llama_model_load_from_file("$MODEL_PATH", mp)
    var cp = llama_context_default_params()
    cp.n_ctx = 64
    var ctx = llama_init_from_model(model, cp)
    var tokens = List[Int32]()
    tokens.append(2)
    tokens.append(14946)
    tokens.append(3324)
    tokens.append(496)
    tokens.append(990)
    for i in range(len(tokens)):
        var b = llama_batch_init(1, embd=0, n_seq_max=1)
        b.token[unsafe_offset=0] = tokens[i]
        b.pos[unsafe_offset=0] = Int32(i)
        b.logits[unsafe_offset=0] = 1
        _ = llama_decode(ctx, b)
        llama_batch_free(b)
        var logits = llama_get_logits_ith(ctx, 0)
        var max_v = Float32(-1e9)
        var max_id = -1
        for v in range(262144):
            var l = logits[unsafe_offset=v]
            if l > max_v:
                max_v = l
                max_id = v
        print("After token", i, "(", tokens[i], "): max_id =", max_id, "max_val =", max_v, "logit[236764] =", logits[unsafe_offset=236764])
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
