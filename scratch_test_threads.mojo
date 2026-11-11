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
)
from std.time import perf_counter_ns

def main() raises:
    llama_backend_init()
    var model_path = "models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
    var model = llama_model_load_from_file(model_path, llama_model_default_params())
    var cparams = llama_context_default_params()
    print("Default cparams.n_threads:", cparams.n_threads)
    var ctx = llama_init_from_model(model, cparams)

    # Warmup 1 decode
    var b = llama_batch_init(1, embd=0, n_seq_max=1)
    b.token[unsafe_offset=0] = llama_token(1)
    b.pos[unsafe_offset=0] = 0
    b.logits[unsafe_offset=0] = 1
    var t0 = perf_counter_ns()
    _ = llama_decode(ctx, b)
    var t1 = perf_counter_ns()
    print("1 token decode time:", Float64(t1 - t0) / 1_000_000.0, "ms")
    llama_batch_free(b)

    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
