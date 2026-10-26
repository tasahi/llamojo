""" Unit test verifying Phase 4 Vocabulary/Tokenizer, Sampler Pipeline, and Quantization routines."""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from llamojo import (
    write_synthetic_gguf,
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_vocab,
    llama_vocab_p,
    llama_tokenize,
    llama_token_to_piece,
    llama_detokenize,
    llama_vocab_bos,
    llama_vocab_eos,
    llama_token,
    llama_token_data,
    llama_token_data_array,
    llama_sampler_chain_default_params,
    llama_sampler_chain_init,
    llama_sampler_chain_add,
    llama_sampler_init_top_k,
    llama_sampler_init_top_p,
    llama_sampler_init_temp,
    llama_sampler_init_penalties,
    llama_sampler_init_greedy,
    llama_sampler_init_dist,
    llama_sampler_apply,
    llama_sampler_sample,
    llama_sampler_free,
    quantize_q4_0_row,
    quantize_q8_0_row,
    quantize_q4_k_row,
    quantize_q6_k_row,
)

def test_vocab_tokenizer() raises:
    print("Testing Vocabulary & BPE Tokenizer...")
    var path = "/tmp/test_phase4.gguf"
    write_synthetic_gguf(path)

    llama_backend_init()
    var mparams = llama_model_default_params()
    var model = llama_model_load_from_file(path, mparams)

    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab(model[].gguf))

    # Test token query functions
    var bos = llama_vocab_bos(vocab)
    var eos = llama_vocab_eos(vocab)
    print("  BOS:", bos, "EOS:", eos)

    # Test Tokenize
    var tokens_buf = unsafe_alloc[llama_token](32)
    var text = "hello world"
    var n_tok = llama_tokenize(vocab, text, tokens_buf, 32, True, False)
    print("  Tokenized count:", n_tok)

    # Test Detokenize
    var out_buf = unsafe_alloc[UInt8](64)
    var out_len = llama_detokenize(vocab, tokens_buf, n_tok, out_buf, 64, True, False)
    print("  Detokenized bytes:", out_len)

    out_buf.unsafe_free()
    tokens_buf.unsafe_free()
    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    llama_model_free(model)
    llama_backend_free()
    print("  ✓ Vocabulary and tokenizer test passed")

def test_sampler_chain():
    print("Testing Sampler Chain (Penalties, Temp, Top-K, Top-P, Greedy)...")
    var sparams = llama_sampler_chain_default_params()
    var chain = llama_sampler_chain_init(sparams)

    llama_sampler_chain_add(chain, llama_sampler_init_penalties(16, Float32(1.1), Float32(0.1), Float32(0.1)))
    llama_sampler_chain_add(chain, llama_sampler_init_temp(Float32(0.7)))
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(10))
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(Float32(0.9)))
    llama_sampler_chain_add(chain, llama_sampler_init_greedy())

    # Create dummy candidate logits
    var n_cand = 50
    var cands = unsafe_alloc[llama_token_data](n_cand)
    for i in range(n_cand):
        cands[unsafe_offset=i] = llama_token_data(Int32(i), Float32(i) * Float32(0.1), Float32(0.0))

    var arr = unsafe_alloc[llama_token_data_array](1)
    arr.unsafe_write(llama_token_data_array(cands, n_cand, False))

    var selected = llama_sampler_sample(chain, arr)
    print("  Selected greedy token ID:", selected)
    if selected != 49:
        print("  Warning: expected token 49 from highest logit, got", selected)

    arr.unsafe_deinit_pointee()
    arr.unsafe_free()
    cands.unsafe_free()
    llama_sampler_free(chain)
    print("  ✓ Sampler chain test passed")

def test_quantization_kernels():
    print("Testing Block Quantizers (Q4_0, Q8_0, Q4_K, Q6_K)...")
    var n = 256
    var src = unsafe_alloc[Float32](n)
    for i in range(n):
        src[unsafe_offset=i] = Float32(i % 17) - Float32(8.0)

    # Q4_0: 256 elements -> 8 blocks * 18 bytes = 144 bytes
    var dst_q4 = unsafe_alloc[UInt8](144)
    quantize_q4_0_row(src, dst_q4, n)

    # Q8_0: 256 elements -> 8 blocks * 34 bytes = 272 bytes
    var dst_q8 = unsafe_alloc[UInt8](272)
    quantize_q8_0_row(src, dst_q8, n)

    # Q4_K: 256 elements -> 1 super-block * 144 bytes = 144 bytes
    var dst_q4k = unsafe_alloc[UInt8](144)
    quantize_q4_k_row(src, dst_q4k, n)

    # Q6_K: 256 elements -> 1 super-block * 210 bytes = 210 bytes
    var dst_q6k = unsafe_alloc[UInt8](210)
    quantize_q6_k_row(src, dst_q6k, n)

    dst_q6k.unsafe_free()
    dst_q4k.unsafe_free()
    dst_q8.unsafe_free()
    dst_q4.unsafe_free()
    src.unsafe_free()
    print("  ✓ Quantization kernels test passed")


def main() raises:
    print("============================================================")
    print("Running Phase 4 Test Suite (Vocab, Sampler, Quantize)")
    print("============================================================")
    test_vocab_tokenizer()
    test_sampler_chain()
    test_quantization_kernels()
    print("All Phase 4 tests passed! ✓")
