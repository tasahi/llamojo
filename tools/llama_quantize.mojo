"""Model quantization utility converting full-precision GGUF files to quantized representations.

Usage:
    mojo run -I src -I vendor/llama2.mojo tools/llama_quantize.mojo <input.gguf> <output.gguf> [ftype]
    where ftype is:
      2 = Q4_0 (default)
      6 = Q5_0
      8 = Q8_0
      12 = Q4_K
      14 = Q6_K
"""

from std.sys import argv
from llamojo import (
    GGMLType,
    llama_model_quantize_params,
    llama_model_quantize_default_params,
    llama_model_quantize,
)

def main():
    var args = argv()
    if len(args) < 3:
        print("Usage: mojo run tools/llama_quantize.mojo <input.gguf> <output.gguf> [ftype]")
        print("  Supported ftypes: 2 (Q4_0), 6 (Q5_0), 8 (Q8_0), 12 (Q4_K), 14 (Q6_K)")
        return

    var fname_in = args[1]
    var fname_out = args[2]
    var ftype: Int32 = GGMLType.Q4_0

    if len(args) > 3:
        try:
            ftype = Int32(args[3])
        except:
            ftype = GGMLType.Q4_0

    var qparams = llama_model_quantize_default_params()
    qparams.ftype = ftype

    print("=== llamojo Quantize Tool ===")
    print("Input: ", fname_in)
    print("Output:", fname_out)
    print("Target type ID:", ftype)

    var res = llama_model_quantize(fname_in, fname_out, qparams)
    if res == 0:
        print("Quantization completed successfully! ✓")
    else:
        print("Quantization failed with error code:", res)

