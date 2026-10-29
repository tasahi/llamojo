from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.math import sqrt
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
    llama_token,
    dot_quant_row,
    rms_norm,
    ggml_block_size,
    ggml_type_size,
)

def main() raises:
    llama_backend_init()
    var model_path = "$MODEL_PATH"
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(model_path, mparams)
    var cparams = llama_context_default_params()
    cparams.n_ctx = 128
    var ctx = llama_init_from_model(model, cparams)

    # Step 0: token 2
    var b0 = llama_batch_init(1, embd=0, n_seq_max=1)
    b0.token[unsafe_offset=0] = 2
    b0.pos[unsafe_offset=0] = 0
    b0.logits[unsafe_offset=0] = 1
    _ = llama_decode(ctx, b0)
    llama_batch_free(b0)

    # Let's inspect K cache and V cache at layer 0, pos 0
    var k_ptr = ctx[].kv_cache.k_f32
    var v_ptr = ctx[].kv_cache.v_f32
    print("Layer 0, pos 0 K[0..4]:", k_ptr[0], k_ptr[1], k_ptr[2], k_ptr[3])
    print("Layer 0, pos 0 V[0..4]:", v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3])

    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()

