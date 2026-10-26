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
    "tests/test_vocab_sampler.mojo"
    "tests/test_lora.mojo"
)

echo "============================================================"
echo "Running llamojo Test Suite (Phases 1 - 6)"
echo "============================================================"

for test in "${TESTS[@]}"; do
    echo ""
    echo ">>> Running $test..."
    "$MOJO_BIN" $INCLUDE_FLAGS "$test"
done

if command -v gcc >/dev/null 2>&1 && [ -f "build/libllama.so" ]; then
    echo ""
    echo ">>> Running tests/test_c_abi.c against build/libllama.so..."
    gcc tests/test_c_abi.c -Lbuild -lllama -Wl,-rpath,"$SCRIPT_DIR/build" -o tests/test_c_abi_bin
    ./tests/test_c_abi_bin
    rm -f tests/test_c_abi_bin
fi

echo ""
echo "============================================================"
echo "All llamojo tests passed! ✓"
echo "============================================================"

