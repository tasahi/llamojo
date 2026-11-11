#!/usr/bin/env bash
set -e

# ==============================================================================
# download_ministral3.sh: Download Ministral 3 3B GGUF model for llamojo
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${REPO_ROOT}/models"
mkdir -p "$MODELS_DIR"

QUANT="${1:-Q4_K_M}"
OUT_FILE="${MODELS_DIR}/Ministral-3-3B-Instruct-2512-${QUANT}.gguf"

# Verify supported quantizations
if [ "$QUANT" != "Q4_K_M" ] && [ "$QUANT" != "Q2_K" ] && [ "$QUANT" != "Q3_K_M" ] && [ "$QUANT" != "Q8_0" ]; then
    echo "Unknown quantization: $QUANT"
    echo "Available options: Q4_K_M (default ~2.15GB), Q2_K (~1.46GB), Q3_K_M (~1.79GB), Q8_0 (~3.5GB)"
    exit 1
fi

URL="https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-${QUANT}.gguf"

echo "============================================================"
echo "Ministral 3 3B GGUF Model Downloader"
echo "============================================================"
echo "Target Model: Ministral-3-3B-Instruct-2512"
echo "Quantization: ${QUANT}"
echo "Output Path:  ${OUT_FILE}"
echo "Source URL:   ${URL}"
echo "============================================================"

if [ -f "$OUT_FILE" ]; then
    echo "Model already exists at: ${OUT_FILE}"
    ls -lh "$OUT_FILE"
    exit 0
fi

echo "Checking available disk space..."
df -h "$MODELS_DIR"

echo "Downloading ${QUANT} weights via curl..."
curl -L --fail --retry 3 -C - -o "$OUT_FILE" "$URL"

echo ""
echo "✓ Download complete!"
ls -lh "$OUT_FILE"
