#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MOJO_BIN="${MOJO_BIN:-/home/tasahi/miniconda/envs/moj/bin/mojo}"
INCLUDE_FLAGS="-I src -I vendor/llama2.mojo -I vendor/MojoLlama/src"

TESTS=(
    "tests/test_gguf.mojo"
    "tests/test_llama_api.mojo"
    "tests/test_kernels.mojo"
    "tests/test_context_decode.mojo"
)

echo "============================================================"
echo "Running llamojo Test Suite (Phases 1, 2 & 3)"
echo "============================================================"

for test in "${TESTS[@]}"; do
    echo ""
    echo ">>> Running $test..."
    "$MOJO_BIN" $INCLUDE_FLAGS "$test"
done

echo ""
echo "============================================================"
echo "All llamojo tests passed! ✓"
echo "============================================================"

