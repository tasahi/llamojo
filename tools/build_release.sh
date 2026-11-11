#!/usr/bin/env bash
set -e

# ==============================================================================
# build_release.sh: Build standalone release binary for llamojo Ministral 3
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MOJO_BIN="${MOJO_BIN:-/home/tasahi/miniconda/envs/moj/bin/mojo}"
if ! command -v "$MOJO_BIN" >/dev/null 2>&1; then
    MOJO_BIN="mojo"
fi

mkdir -p bin

MODULAR_LIB="$(dirname "$(dirname "$MOJO_BIN")")/lib/python3.14/site-packages/modular/lib"
if [ ! -d "$MODULAR_LIB" ]; then
    # Fallback search if python version varies
    MODULAR_LIB="$(find "$(dirname "$(dirname "$MOJO_BIN")")/lib" -type d -name "modular" -exec test -d "{}/lib" \; -print -quit)/lib"
fi

echo "============================================================"
echo "Compiling Release Executable: bin/chat_ministral3"
echo "============================================================"
echo "Compiler:        ${MOJO_BIN}"
echo "Modular Lib:     ${MODULAR_LIB}"
echo "Optimization:    -O3"
echo "Target Triple:   x86_64-unknown-linux-gnu"
echo "Target CPU:      x86-64-v3 (AVX2, FMA, BMI2)"
echo "FP Mode:         contract=fast"
echo "============================================================"

"$MOJO_BIN" build \
    -O3 \
    --target-triple x86_64-unknown-linux-gnu \
    --mcpu=x86-64-v3 \
    --fp-mode=contract=fast \
    -Xlinker -rpath -Xlinker "${MODULAR_LIB}" \
    -I src \
    -I vendor/llama2.mojo \
    -I vendor/MojoLlama/src \
    -o bin/chat_ministral3 \
    examples/chat_ministral3.mojo

echo "✓ Release executable built successfully:"
ls -lh bin/chat_ministral3
