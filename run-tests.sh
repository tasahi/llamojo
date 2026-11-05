#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MOJO_BIN="${MOJO_BIN:-bin/mojo}"
INCLUDE_FLAGS="-I src -I vendor/llama2.mojo -I vendor/MojoLlama/src"

TESTS=(
    "tests/test_gguf.mojo"
    "tests/test_llama_api.mojo"
    "tests/test_kernels.mojo"
    "tests/test_context_decode.mojo"
    "tests/test_context_views.mojo"
    "tests/test_vocab_sampler.mojo"
    "tests/test_lora.mojo"
    "tests/test_grammar.mojo"
    "tests/test_moe.mojo"
    "tests/test_micro_arch.mojo"
    "tests/test_whisper_phase1.mojo"
    "tests/test_whisper_phase2.mojo"
    "tests/test_whisper_phase3.mojo"
    "tests/test_whisper_phase4.mojo"
)

echo "============================================================"
echo "Running llamojo Test Suite (Phases 1 - 6)"
echo "============================================================"

for test in "${TESTS[@]}"; do
    echo ""
    echo ">>> Running $test..."
    "$MOJO_BIN" $INCLUDE_FLAGS "$test"
done

if command -v gcc >/dev/null 2>&1 && [ -f "build/libllamojo.so" ]; then
    echo ""
    echo ">>> Running tests/test_c_abi.c against build/libllamojo.so..."
    gcc tests/test_c_abi.c -Lbuild -lllamojo -Wl,-rpath,"$SCRIPT_DIR/build" -o tests/test_c_abi_bin
    ./tests/test_c_abi_bin
    rm -f tests/test_c_abi_bin
fi

if command -v python >/dev/null 2>&1; then
    echo ""
    echo ">>> Running Python Zero-Copy Views, Hugging Face Compat & Safetensors Interop Test Suites..."
    PYTHONPATH=python python -m unittest tests.test_python_views tests.test_python_hf_compat tests.test_safetensors_interop -v
fi

echo ""
echo "============================================================"
echo "All llamojo tests passed! ✓"
echo "============================================================"
