#!/usr/bin/env bash
# Script to compare llamojo inference and tokenization against upstream llama.cpp
#
# Usage:
#   ./tools/compare_llama_cpp.sh [model.gguf] [prompt] [n_tokens]

set -e

PYTHON_BIN="bin/python3"
MOJO_BIN="bin/mojo"

MODEL="${1:-data/smollm2-135m-q4_k_m.gguf}"
PROMPT="${2:-Once upon a time}"
N_TOKENS="${3:-24}"

echo "============================================================"
echo "          llamojo vs llama.cpp Comparison Benchmark         "
echo "============================================================"
echo "Model:    $MODEL"
echo "Prompt:   \"$PROMPT\""
echo "Tokens:   $N_TOKENS"
echo "------------------------------------------------------------"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file '$MODEL' not found!"
    exit 1
fi

echo ""
echo ">>> [1/2] Running Reference llama.cpp (upstream C++ engine)..."
$PYTHON_BIN - << EOF
import sys
import time
from llama_cpp import Llama

model_path = "$MODEL"
prompt = """$PROMPT"""
n_predict = int("$N_TOKENS")

t0 = time.perf_counter()
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    verbose=False,
)
tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False)
load_time = time.perf_counter() - t0

t1 = time.perf_counter()
output = llm(
    prompt,
    max_tokens=n_predict,
    temperature=0.0, # Greedy
    top_k=1,
    top_p=1.0,
    echo=False,
)
gen_time = time.perf_counter() - t1

gen_text = output["choices"][0]["text"]
usage = output.get("usage", {})
out_tokens = usage.get("completion_tokens", n_predict)
speed = out_tokens / gen_time if gen_time > 0 else 0.0

print(f"Tokenized prompt ({len(tokens)} tokens): {tokens}")
print(f"Generation throughput: {speed:.2f} tok/s ({gen_time:.3f} s)")
print("Generated text:")
print("------------------------------------------------------------")
print(prompt + gen_text)
print("------------------------------------------------------------")
EOF

echo ""
echo ">>> [2/2] Running llamojo (Pure Mojo 1.1 CPU engine)..."
$MOJO_BIN run -I src -I vendor/llama2.mojo tools/llama_cli.mojo \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$N_TOKENS"

echo ""
echo "============================================================"
echo "Comparison completed."
echo "============================================================"

