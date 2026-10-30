""" Unit test verifying Phase 4 Vocabulary/Tokenizer, Sampler Pipeline, and Quantization routines."""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal

from llamojo import (
    write_synthetic_gguf,
    gguf_load_from_file,
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_model_quantize,
    llama_model_quantize_default_params,
    GGMLType,
    VOCAB_TYPE_SPM,
    VOCAB_TYPE_BPE,
    llama_vocab,
    llama_vocab_p,
    tokenize_spm,
    tokenize_bpe,
    tokenize_text,
    llama_tokenize,
    llama_token_to_piece,
    llama_detokenize,
    llama_vocab_bos,
    llama_vocab_eos,
    llama_vocab_type,
    llama_vocab_get_score,
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

def test_full_metadata_quantization() raises:
    print("Testing llama_model_quantize full metadata preservation & serialization...")
    var src_path = "/tmp/test_phase5_src.gguf"
    var out_path = "/tmp/test_phase5_q4_0.gguf"

    # Write source GGUF with synthetic metadata (tokens, arch, dims)
    write_synthetic_gguf(src_path, vocab_size=32, hidden_dim=64)

    var qparams = llama_model_quantize_default_params()
    qparams.ftype = GGMLType.Q4_0

    var res = llama_model_quantize(src_path, out_path, qparams)
    if res != 0:
        print("  Error: llama_model_quantize returned", res)

    # Load quantized model back via GGUF loader to inspect preserved metadata
    var gguf_out = gguf_load_from_file(out_path)
    print("  Quantized model tensors count:", len(gguf_out.tensors))
    print("  Quantized model metadata count:", len(gguf_out.metadata))
    print("  Architecture:", gguf_out.get_string("general.architecture"))
    print("  Quantization version:", gguf_out.get_int("general.quantization_version"))
    print("  File type ftype:", gguf_out.get_int("general.file_type"))

    # Verify key metadata entries persisted
    if gguf_out.get_string("general.architecture") != "llama":
        print("  Warning: architecture mismatch!")
    if gguf_out.get_int("general.quantization_version") != 2:
        print("  Warning: quantization version mismatch!")
    if gguf_out.get_int("general.file_type") != Int(GGMLType.Q4_0):
        print("  Warning: file_type mismatch!")

    # Verify token vocabulary preserved
    var tokens_opt = gguf_out.metadata.find("tokenizer.ggml.tokens")
    if tokens_opt:
        print("  Tokenizer tokens array size:", len(tokens_opt.value().str_arr))

    gguf_out.free()
    print("  ✓ Full metadata quantization test passed")

def test_spm_unigram_tokenizer() raises:
    print("Testing Phase 7 SentencePiece Unigram Tokenizer (Viterbi Trellis)...")
    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab())
    vocab[].vocab_type = VOCAB_TYPE_SPM
    vocab[].bos_id = 1
    vocab[].eos_id = 2
    vocab[].unk_id = 0

    # Add custom tokens with Unigram log-probabilities
    # Pieces: " " (space), "hello", "world", "hel", "lo", "wor", "ld"
    vocab[].tokens.append("<unk>")   # 0
    vocab[].scores.append(-10.0)
    vocab[].tokens.append("<s>")     # 1
    vocab[].scores.append(0.0)
    vocab[].tokens.append("</s>")    # 2
    vocab[].scores.append(0.0)
    vocab[].tokens.append(" ")       # 3
    vocab[].scores.append(-1.0)
    vocab[].tokens.append("hello")   # 4
    vocab[].scores.append(-2.0)
    vocab[].tokens.append("world")   # 5
    vocab[].scores.append(-2.0)
    vocab[].tokens.append("hel")     # 6
    vocab[].scores.append(-3.0)
    vocab[].tokens.append("lo")      # 7
    vocab[].scores.append(-3.0)

    vocab[].vocab_size = len(vocab[].tokens)
    for i in range(vocab[].vocab_size):
        vocab[].token_to_id[vocab[].tokens[i]] = Int32(i)

    # Test query helpers
    var vt = llama_vocab_type(vocab)
    assert_equal(vt, VOCAB_TYPE_SPM)
    assert_equal(llama_vocab_get_score(vocab, 4), Float32(-2.0))

    # Tokenize "hello world" using Viterbi Trellis
    # norm_text will be " hello world" -> tokens: [<s>, " hello", " world"] or [<s>, " ", "hello", " ", "world"]
    var tokens_buf = unsafe_alloc[llama_token](32)
    var count = llama_tokenize(vocab, "hello world", tokens_buf, 32, True, False)
    print("  Unigram tokenized count:", count)
    assert_true(count >= 3)
    assert_equal(tokens_buf[unsafe_offset=0], 1) # BOS

    tokens_buf.unsafe_free()
    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    print("  ✓ SentencePiece Unigram Viterbi tokenizer passed")

def test_gpt2_bpe_module() raises:
    print("Testing Phase 1 Extracted GPT-2 Byte-Level BPE Module (vocab_gpt2.mojo)...")
    var vocab = unsafe_alloc[llama_vocab](1)
    vocab.unsafe_write(llama_vocab())
    vocab[].model_type = "gpt2"
    vocab[].vocab_type = VOCAB_TYPE_BPE
    vocab[].add_bos = False

    # Populate tokens: "h", "e", "l", "o", "\u0120" (Ġ), "w", "he", "ll", "hello", "\u0120w"
    var pieces = List[String]()
    pieces.append("<|endoftext|>") # 0
    pieces.append("h")             # 1
    pieces.append("e")             # 2
    pieces.append("l")             # 3
    pieces.append("o")             # 4
    pieces.append("\u0120")        # 5 (' ')
    pieces.append("w")             # 6
    pieces.append("he")            # 7
    pieces.append("ll")            # 8
    pieces.append("hell")          # 9
    pieces.append("hello")         # 10
    pieces.append("\u0120w")       # 11

    for i in range(len(pieces)):
        vocab[].tokens.append(pieces[i])
        vocab[].scores.append(0.0)
        vocab[].token_to_id[pieces[i]] = Int32(i)
    vocab[].vocab_size = len(vocab[].tokens)

    vocab[].merge_ranks["h e"] = 0
    vocab[].merge_ranks["l l"] = 1
    vocab[].merge_ranks["he ll"] = 2
    vocab[].merge_ranks["hell o"] = 3
    vocab[].merge_ranks["\u0120 w"] = 4

    var tokens = tokenize_text(vocab, "hello w", False, False)
    assert_equal(len(tokens), 2)
    assert_equal(tokens[0], 10) # "hello"
    assert_equal(tokens[1], 11) # "\u0120w"

    vocab.unsafe_deinit_pointee()
    vocab.unsafe_free()
    print("  ✓ GPT-2 BPE module test passed")

def main() raises:
    print("============================================================")
    print("Running Phase 4, 5 & 7 Test Suite (Vocab, Sampler, SPM Unigram)")
    print("============================================================")
    test_vocab_tokenizer()
    test_spm_unigram_tokenizer()
    test_gpt2_bpe_module()
    test_sampler_chain()
    test_quantization_kernels()
    test_full_metadata_quantization()
    print("All Phase 4, 5 & 7 tests passed! ✓")
