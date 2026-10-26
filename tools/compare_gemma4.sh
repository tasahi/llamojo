#!/usr/bin/env bash
# Compare Gemma 4 inference between llama.cpp and llamojo
set -e

MODEL_PATH=$(python -c '
import glob, os
m = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mradermacher--gemma-4-E2B-it-GGUF/snapshots/*/*.gguf"))
if m: print(os.path.realpath(m[0]))
')

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Gemma 4 GGUF model not found in HuggingFace cache."
    exit 1
fi

PROMPT="${1:-Once upon a time}"
MAX_TOKENS="${2:-16}"

echo "=================================================="
echo "  Comparing Gemma 4 on llama.cpp vs llamojo"
echo "=================================================="
echo "Model  : $MODEL_PATH"
echo "Prompt : \"$PROMPT\""
echo "Tokens : $MAX_TOKENS"
echo ""

echo ">>> [1/2] Running llama.cpp Reference Test..."
python tools/test_gemma4.py --model "$MODEL_PATH" -p "$PROMPT" -n "$MAX_TOKENS"

echo ""
echo ">>> [2/2] Running llamojo CLI Loader..."
mojo run -I src -I vendor/llama2.mojo tools/llama_cli.mojo -m "$MODEL_PATH" -p "$PROMPT" -n "$MAX_TOKENS" || true

