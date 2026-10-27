from llamojo import (
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_model_load_from_file,
    llama_model_free,
)

def main():
    llama_backend_init()
    var mp = llama_model_default_params()
    var model = llama_model_load_from_file("/home/tasahi/.cache/huggingface/hub/blobs/a1/a156058e62f4541ff0b9226ed3bc57fdb30ddc9d266f63cc1716a3eb32ab955c", mp)
    for i in range(5):
        var s = model[].layers[i].layer_output_scale
        print("Layer", i, "scale:", s)
    llama_model_free(model)
    llama_backend_free()

