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
)
from llamojo.backend.kernels_norm_rope import rms_norm

def main():
    llama_backend_init()
    var mp = llama_model_default_params()
    var model = llama_model_load_from_file("/home/tasahi/.cache/huggingface/hub/blobs/a1/a156058e62f4541ff0b9226ed3bc57fdb30ddc9d266f63cc1716a3eb32ab955c", mp)
    var cp = llama_context_default_params()
    cp.n_ctx = 64
    var ctx = llama_init_from_model(model, cp)
    var b = llama_batch_init(1, embd=0, n_seq_max=1)
    b.token[unsafe_offset=0] = 2
    b.pos[unsafe_offset=0] = 0
    b.logits[unsafe_offset=0] = 1
    _ = llama_decode(ctx, b)
    llama_batch_free(b)
    
    # Compute output norm on ctx[].x into ctx[].xb
    rms_norm(ctx[].xb, ctx[].x, model[].output_norm, 1536, model[].config.norm_eps)
    print("llamojo result_norm after token 2:")
    for i in range(8):
        print(" ", ctx[].xb[unsafe_offset=i], end="")
    print("")

    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

