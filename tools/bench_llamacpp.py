import sys
import time
import llama_cpp

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
prompt_text = sys.argv[2] if len(sys.argv) > 2 else "Why is the sky blue?"
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64

print("============================================================")
print("llama.cpp Benchmark (Ministral 3 3B)")
print("============================================================")
print("Model:      ", model_path)
print("Prompt:     ", prompt_text)
print("Max Tokens: ", max_tokens)
print("============================================================")

# 1. Model Loading Benchmark
t0 = time.perf_counter()
llm = llama_cpp.Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,
    verbose=False
)
t_load = time.perf_counter() - t0
print(f"Model Load Time: {t_load:.2f} s")

formatted = f"[INST] {prompt_text} [/INST]"
tokens = llm.tokenize(formatted.encode("utf-8"), special=False)
n_prompt_tokens = len(tokens)

# 2. Prompt Ingestion Benchmark
t_prompt_start = time.perf_counter()
llm.reset()
llm.eval(tokens)
t_prompt_end = time.perf_counter()
prompt_time = t_prompt_end - t_prompt_start
prompt_tps = n_prompt_tokens / prompt_time if prompt_time > 0 else 0

print(f"\nResponse: ", end="", flush=True)

# 3. Generation Benchmark
gen_count = 0
t_gen_start = time.perf_counter()
for tok in llm.generate(tokens, temp=0.15, top_k=40, top_p=0.9):
    gen_count += 1
    piece = llm.detokenize([tok]).decode("utf-8", errors="ignore")
    print(piece, end="", flush=True)
    if tok == llm.token_eos() or gen_count >= max_tokens:
        break
t_gen_end = time.perf_counter()

gen_time = t_gen_end - t_gen_start
gen_tps = gen_count / gen_time if gen_time > 0 else 0

print("\n\n------------------------------------------------------------")
print(f"Metrics: Prompt {n_prompt_tokens} tokens @ {prompt_tps:.2f} tok/s | Generation {gen_count} tokens @ {gen_tps:.2f} tok/s")
print("------------------------------------------------------------")
