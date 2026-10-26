""" Unit tests for Phase 9: GBNF Grammar Engine & JSON-Schema Parsing Filter."""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.collections import List
from std.memory.alloc import unsafe_alloc

from llamojo import (
    parse_gbnf,
    grammar_create_json,
    grammar_create_integer,
    grammar_create_float,
    is_prefix_valid_json,
    is_prefix_valid_integer,
    is_prefix_valid_pattern,
    apply_grammar_bitmask,
    llama_sampler_init_grammar,
    llama_sampler_apply,
    llama_sampler_sample,
    llama_sampler_accept,
    llama_sampler_free,
    llama_token_data,
    llama_token_data_array,
)

def test_gbnf_parser_and_presets() raises:
    print("Testing GBNF parser and preset generators...")
    var g_spec = """
    root ::= expr
    expr ::= term ("+" term)*
    term ::= [0-9]+
    """
    var g = parse_gbnf(g_spec)
    assert_true(g.is_active)
    assert_true(len(g.rules) >= 3)
    assert_true(g.rule_names.find("root"))
    assert_true(g.rule_names.find("expr"))
    assert_true(g.rule_names.find("term"))

    var g_json = grammar_create_json()
    assert_true(g_json.is_active)

    var g_int = grammar_create_integer()
    assert_true(g_int.is_active)

    var g_float = grammar_create_float()
    assert_true(g_float.is_active)
    print("  ✓ GBNF parser and presets initialized cleanly")

def test_json_and_integer_prefix_validation() raises:
    print("Testing JSON and Integer prefix validity checks...")
    # JSON prefix validity
    assert_true(is_prefix_valid_json(""))
    assert_true(is_prefix_valid_json("{\"key\":"))
    assert_true(is_prefix_valid_json("{\"key\": \"val\"}"))
    assert_true(is_prefix_valid_json("[1, 2, 3]"))
    assert_true(is_prefix_valid_json("{\"a\": [1, 2, {\"b\": true}]}"))

    # Unbalanced closing braces should fail
    assert_true(not is_prefix_valid_json("}"))
    assert_true(not is_prefix_valid_json("]"))
    assert_true(not is_prefix_valid_json("{\"key\": [1, 2}}"))

    # Integer prefix validity
    assert_true(is_prefix_valid_integer(""))
    assert_true(is_prefix_valid_integer("-"))
    assert_true(is_prefix_valid_integer("12345"))
    assert_true(is_prefix_valid_integer("-98765"))
    assert_true(not is_prefix_valid_integer("12a45"))
    assert_true(not is_prefix_valid_integer("12.45"))
    print("  ✓ JSON and Integer prefix validation verified")

def test_grammar_sampler_bitmasking() raises:
    print("Testing grammar logit bitmasking in sampler pipeline...")
    # Vocabulary:
    # 0: "{"
    # 1: "\"name\""
    # 2: ":"
    # 3: "invalid_token_!!!"
    # 4: "}"
    var vocab = List[String]()
    vocab.append("{")
    vocab.append("\"name\"")
    vocab.append(":")
    vocab.append("invalid_token_!!!")
    vocab.append("}")

    var smpl = llama_sampler_init_grammar("json", vocab)

    # Candidate array with 3 options:
    # candidate 0: token 0 ("{") -> valid start of JSON
    # candidate 1: token 4 ("}") -> invalid start of JSON (unbalanced)
    # candidate 2: token 1 ("\"name\"") -> valid JSON string
    var data = unsafe_alloc[llama_token_data](3)
    data[unsafe_offset=0] = llama_token_data(id=0, logit=2.0, p=0.0)
    data[unsafe_offset=1] = llama_token_data(id=4, logit=5.0, p=0.0) # higher initial logit!
    data[unsafe_offset=2] = llama_token_data(id=1, logit=1.0, p=0.0)

    var arr = unsafe_alloc[llama_token_data_array](1)
    arr.unsafe_write(llama_token_data_array(data, 3, False))

    # Apply grammar filter
    llama_sampler_apply(smpl, arr)

    # Token 4 ("}") should be masked with large negative logit (-1e30)
    assert_true(data[unsafe_offset=1].logit < -1e20)
    # Token 0 ("{") should retain its logit
    assert_almost_equal(data[unsafe_offset=0].logit, Float32(2.0), atol=1e-5)

    # Sampling greedily now must choose token 0 instead of token 4
    var selected = llama_sampler_sample(smpl, arr)
    assert_equal(selected, 0)

    # Check grammar state buffer updated on accept
    assert_equal(smpl[].grammar_buffer, "{")

    arr.unsafe_deinit_pointee()
    arr.unsafe_free()
    data.unsafe_free()
    llama_sampler_free(smpl)
    print("  ✓ Grammar sampler logit bitmasking and state transition verified")

def main() raises:
    print("============================================================")
    print("Running Phase 9 GBNF Grammar Engine Test Suite")
    print("============================================================")
    test_gbnf_parser_and_presets()
    test_json_and_integer_prefix_validation()
    test_grammar_sampler_bitmasking()
    print("All Phase 9 GBNF Grammar tests passed! ✓")