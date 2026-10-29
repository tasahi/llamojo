#!/usr/bin/env bash
# Build libllamojo.so shared library
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MOJO_BIN="${MOJO_BIN:-bin/mojo}"
OUTPUT_LIB="${1:-build/libllamojo.so}"

mkdir -p "$(dirname "$OUTPUT_LIB")"

echo "Building llamojo shared library ($OUTPUT_LIB)..."
$MOJO_BIN build --emit shared-lib -O3 --target-cpu alderlake \
  -I src -I vendor/llama2.mojo \
  src/llamojo/c_api.mojo -o "$OUTPUT_LIB"

echo "✓ Successfully built $OUTPUT_LIB"
ls -lh "$OUTPUT_LIB"

