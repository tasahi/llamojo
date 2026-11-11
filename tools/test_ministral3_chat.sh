#!/usr/bin/env bash
set -e

# ==============================================================================
# test_ministral3_chat.sh: Download & Test Ministral 3 3B Chat in llamojo
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MOJO_BIN="${MOJO_BIN:-/home/tasahi/miniconda/envs/moj/bin/mojo}"
if ! command -v "$MOJO_BIN" >/dev/null 2>&1; then
    MOJO_BIN="mojo"
fi

QUANT="${1:-Q4_K_M}"
PROMPT="${2:-Explain quantum computing in simple terms for high school students.}"
MODEL_PATH="${REPO_ROOT}/models/Ministral-3-3B-Instruct-2512-${QUANT}.gguf"

# Performance optimization for WSL2: if a fast local Linux ext4 copy exists in /tmp, prefer it
if [ -f "/tmp/Ministral-3-3B-Instruct-2512-${QUANT}.gguf" ]; then
    echo "⚡ Found fast Linux local SSD copy at /tmp/Ministral-3-3B-Instruct-2512-${QUANT}.gguf (bypassing slow 9p Windows mount)"
    MODEL_PATH="/tmp/Ministral-3-3B-Instruct-2512-${QUANT}.gguf"
fi

echo "============================================================"
echo "Ministral 3 3B Model Download & Chat Test"
echo "============================================================"
echo "Model Quantization: ${QUANT}"
echo "Model File:         ${MODEL_PATH}"
echo "Mojo Compiler:      ${MOJO_BIN}"
echo "Prompt:             ${PROMPT}"
echo "============================================================"

# Step 1: Download model if not present
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo ">>> Step 1: Downloading model weights via tools/download_ministral3.sh..."
    bash "${REPO_ROOT}/tools/download_ministral3.sh" "$QUANT"
else
    echo "✓ Model file already exists: ${MODEL_PATH}"
    ls -lh "$MODEL_PATH"
fi

# Step 2: Run Chat Application
echo ""
echo ">>> Step 2: Launching llamojo Ministral 3 Chat..."
"$MOJO_BIN" -I src -I vendor/llama2.mojo -I vendor/MojoLlama/src "${REPO_ROOT}/examples/chat_ministral3.mojo" "$MODEL_PATH" "$PROMPT"
