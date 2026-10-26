#!/usr/bin/env python3
"""
Side-by-side test and inspection harness for Gemma 4 E2B-it GGUF model:
`~/.cache/huggingface/hub/models--mradermacher--gemma-4-E2B-it-GGUF/`

Tests:
1. GGUF metadata & architecture inspection (tensors, block count, SWA, heads).
2. Tokenizer test (prompt tokenization & piece analysis).
3. Text completion generation.
4. Instruction/Chat generation using the model's native chat template.
5. Performance profiling (prompt eval tok/s and generation tok/s).
"""

import os
import sys
import glob
import time
import argparse

def find_gemma4_path():
    search_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mradermacher--gemma-4-E2B-it-GGUF/snapshots/*/*.gguf"
    )
    matches = glob.glob(search_pattern)
    if matches:
        return os.path.realpath(matches[0])
    # Check default blob location if snapshot symlink is broken
    blob_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mradermacher--gemma-4-E2B-it-GGUF/blobs/*"
    )
    blobs = glob.glob(blob_pattern)
    for b in blobs:
        if os.path.getsize(b) > 1_000_000_000:
            return os.path.realpath(b)
    return None

def inspect_metadata(model_path):
    import gguf
    print("=" * 60)
    print("  Gemma 4 Metadata & Architecture Inspection")
    print("=" * 60)
    reader = gguf.GGUFReader(model_path)
    arch = reader.get_field("general.architecture").parts[-1].tobytes().decode("utf-8")
    print(f"Architecture         : {arch}")
    print(f"Tensors count        : {len(reader.tensors)}")
    
    fields_to_show = [
        "general.name",
        "general.size_label",
        "gemma4.block_count",
        "gemma4.context_length",
        "gemma4.embedding_length",
        "gemma4.feed_forward_length",
        "gemma4.attention.head_count",
        "gemma4.attention.head_count_kv",
        "gemma4.attention.key_length",
        "gemma4.attention.value_length",
        "gemma4.attention.sliding_window",
        "gemma4.attention.shared_kv_layers",
        "gemma4.embedding_length_per_layer_input",
        "gemma4.final_logit_softcapping",
        "gemma4.rope.freq_base",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    ]
    for k in fields_to_show:
        f = reader.get_field(k)
        if f is not None:
            val = f.parts[-1].tolist() if hasattr(f.parts[-1], "tolist") else f.parts[-1]
            if isinstance(val, (bytes, bytearray)):
                val = val.decode("utf-8", errors="replace")
            print(f"  {k:<38}: {val}")
    print("-" * 60)

def run_inference(model_path, prompt, chat_mode=False, max_tokens=32, temp=0.0):
    from llama_cpp import Llama
    print("=" * 60)
    print(f"  Testing Inference with llama.cpp Engine")
    print("=" * 60)
    print(f"Loading: {model_path}")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        verbose=False,
    )
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.2f} s.\n")

    # 1. Tokenizer check
    prompt_bytes = prompt.encode("utf-8")
    tokens = llm.tokenize(prompt_bytes, add_bos=False)
    print(f"Prompt        : {prompt!r}")
    print(f"Tokens        : {tokens}")
    pieces = [llm.detokenize([t]).decode("utf-8", errors="replace") for t in tokens]
    print(f"Token pieces  : {pieces}")
    print("-" * 60)

    # 2. Text completion or Chat completion
    if chat_mode:
        print("Running Chat Completion (instruct template)...")
        messages = [
            {"role": "user", "content": prompt}
        ]
        t1 = time.time()
        res = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        t_gen = time.time() - t1
        output_text = res["choices"][0]["message"]["content"]
    else:
        print("Running Raw Completion (greedy)...")
        t1 = time.time()
        res = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
        )
        t_gen = time.time() - t1
        output_text = res["choices"][0]["text"]

    gen_tokens = len(llm.tokenize(output_text.encode("utf-8"), add_bos=False))
    tps = gen_tokens / t_gen if t_gen > 0 else 0.0

    print("\n--- Output ---")
    print(output_text)
    print("--- End of Output ---\n")
    print(f"Generated tokens : {gen_tokens}")
    print(f"Generation time  : {t_gen*1000:.1f} ms ({tps:.2f} tok/s)")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Test Gemma 4 model against llama.cpp")
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF model")
    parser.add_argument("-p", "--prompt", type=str, default="Once upon a time", help="Test prompt")
    parser.add_argument("-n", "--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--chat", action="store_true", help="Run multi-turn chat/instruct completion")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        model_path = find_gemma4_path()

    if not model_path or not os.path.exists(model_path):
        print("Error: Could not locate Gemma 4 GGUF file in cache or provided path.")
        sys.exit(1)

    print(f"Target model: {model_path}")
    inspect_metadata(model_path)
    run_inference(model_path, args.prompt, chat_mode=args.chat, max_tokens=args.max_tokens, temp=args.temp)

if __name__ == "__main__":
    main()
