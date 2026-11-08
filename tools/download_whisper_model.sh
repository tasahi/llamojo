#!/usr/bin/env bash
# Download pre-converted ggml Whisper models from Hugging Face (ggerganov/whisper.cpp)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${1:-}"
OUT_DIR="${2:-$REPO_ROOT/models}"

AVAILABLE_MODELS="tiny tiny.en tiny-q5_1 tiny.en-q5_1 tiny-q8_0
base base.en base-q5_1 base.en-q5_1 base-q8_0
small small.en small-q5_1 small.en-q5_1 small-q8_0
medium medium.en medium-q5_0 medium.en-q5_0 medium-q8_0
large-v1 large-v2 large-v2-q5_0 large-v2-q8_0
large-v3 large-v3-q5_0 large-v3-turbo large-v3-turbo-q5_0 large-v3-turbo-q8_0"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_name> [output_dir]"
    echo ""
    echo "Available models:"
    echo "$AVAILABLE_MODELS" | sed 's/^/  /'
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/ggml-${MODEL}.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"

if [ -f "$OUT_FILE" ]; then
    echo "Model already exists at: $OUT_FILE"
    ls -lh "$OUT_FILE"
    exit 0
fi

echo "Downloading ggml-${MODEL}.bin from $URL ..."
curl -L --fail --retry 3 -o "$OUT_FILE" "$URL"
echo "✓ Saved to $OUT_FILE"
ls -lh "$OUT_FILE"

