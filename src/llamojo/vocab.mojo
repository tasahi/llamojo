"""Vocabulary & Tokenizer Implementation for llamojo.

Integrates:
- llama2.mojo/bpe_encode & Tokenizer
- GGUF vocabulary metadata (tokenizer.ggml.tokens, scores, token_type, merges)
- SentencePiece (' ' prefix) & Byte-level BPE (<0xXX> byte fallback)
- Public llama.h API:
  - llama_vocab (opaque struct)
  - llama_model_get_vocab
  - llama_tokenize
  - llama_token_to_piece
  - llama_detokenize
  - llama_token_is_eog / llama_token_bos / llama_token_eos / llama_token_eot
"""

from std.collections import List, Dict
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from .types import llama_token, null_ptr
from .gguf import GGUFContext

comptime CharList = List[UInt8]

struct llama_vocab(Movable):
    """Vocabulary and BPE merger dictionary populated from GGUF metadata."""
    var tokens: List[String]
    var scores: List[Float32]
    var token_types: List[Int32]
    var token_to_id: Dict[String, Int32]
    var bos_id: llama_token
    var eos_id: llama_token
    var eot_id: llama_token
    var pad_id: llama_token
    var vocab_size: Int

    def __init__(out self):
        self.tokens = List[String]()
        self.scores = List[Float32]()
        self.token_types = List[Int32]()
        self.token_to_id = Dict[String, Int32]()
        self.bos_id = 1
        self.eos_id = 2
        self.eot_id = -1
        self.pad_id = -1
        self.vocab_size = 0

    def __init__(out self, ref gguf: GGUFContext):
        self.tokens = List[String]()
        self.scores = List[Float32]()
        self.token_types = List[Int32]()
        self.token_to_id = Dict[String, Int32]()

        var tok_opt = gguf.metadata.find("tokenizer.ggml.tokens")
        if tok_opt:
            self.tokens = tok_opt.value().str_arr.copy()

        var score_opt = gguf.metadata.find("tokenizer.ggml.scores")
        if score_opt:
            self.scores = score_opt.value().float_arr.copy()

        var type_opt = gguf.metadata.find("tokenizer.ggml.token_type")
        if type_opt:
            var arr = type_opt.value().int_arr.copy()
            for i in range(len(arr)):
                self.token_types.append(Int32(arr[i]))

        self.vocab_size = len(self.tokens)
        for i in range(self.vocab_size):
            self.token_to_id[self.tokens[i]] = Int32(i)

        self.bos_id = Int32(gguf.get_int("tokenizer.ggml.bos_token_id", 1))
        self.eos_id = Int32(gguf.get_int("tokenizer.ggml.eos_token_id", 2))
        self.eot_id = Int32(gguf.get_int("tokenizer.ggml.eot_token_id", -1))
        self.pad_id = Int32(gguf.get_int("tokenizer.ggml.padding_token_id", -1))

    def find_token(self, s: String) -> Int32:
        """Find token ID for an exact piece string."""
        var opt = self.token_to_id.find(s)
        if opt:
            return opt.value()
        return -1

    def token_to_piece_str(self, tok: llama_token) -> String:
        """Decode a token ID into its raw text representation."""
        if tok < 0 or Int(tok) >= self.vocab_size:
            return ""
        var s = self.tokens[Int(tok)]

        # Handle hex byte fallback: <0xXX>
        if s.byte_length() == 6:
            if s[byte=0:3] == "<0x" and s[byte=5:6] == ">":
                var h0 = s[byte=3:4]
                var h1 = s[byte=4:5]
                var b0: Int = 0
                var b1: Int = 0
                if h0 >= "0" and h0 <= "9":
                    b0 = ord(h0) - ord("0")
                elif h0 >= "A" and h0 <= "F":
                    b0 = ord(h0) - ord("A") + 10
                elif h0 >= "a" and h0 <= "f":
                    b0 = ord(h0) - ord("a") + 10

                if h1 >= "0" and h1 <= "9":
                    b1 = ord(h1) - ord("0")
                elif h1 >= "A" and h1 <= "F":
                    b1 = ord(h1) - ord("A") + 10
                elif h1 >= "a" and h1 <= "f":
                    b1 = ord(h1) - ord("a") + 10

                var byte_val = UInt8((b0 << 4) | b1)
                var bytes = List[UInt8]()
                bytes.append(byte_val)
                return String(unsafe_from_utf8=Span(unsafe_ptr=bytes.unsafe_ptr(), length=1))

        # Handle SentencePiece space prefix (U+2581: \xe2\x96\x81)
        if s.byte_length() >= 3 and s[byte=0:3] == " ":
            return " " + s[byte=3:s.byte_length()]

        return s

comptime llama_vocab_p = Pointer[llama_vocab, MutUntrackedOrigin]

# ─── BPE Tokenizer Engine ────────────────────────────────────────────────────

def tokenize_bpe(
    vocab: llama_vocab_p,
    text: String,
    add_special: Bool = True,
    parse_special: Bool = False,
) -> List[llama_token]:
    """Tokenize UTF-8 text string into a list of llama_tokens using BPE merge priority."""
    var tokens = List[llama_token]()
    if Int(vocab) == 0:
        return tokens^

    if add_special and vocab[].bos_id >= 0:
        tokens.append(vocab[].bos_id)

    if text.byte_length() == 0:
        return tokens^

    # 1. Break text into initial single-byte or character tokens
    for pos in range(text.byte_length()):
        var ch = String(text[byte=pos:pos + 1])
        var id = vocab[].find_token(ch)
        if id == -1:
            # Try hex byte representation <0xXX>
            var b = text.unsafe_ptr()[unsafe_offset=pos]
            var hex_digits = "0123456789ABCDEF"
            var h0 = String(hex_digits[byte=Int(b >> 4):Int((b >> 4) + 1)])
            var h1 = String(hex_digits[byte=Int(b & 0x0F):Int((b & 0x0F) + 1)])
            var hex_token = "<0x" + h0 + h1 + ">"
            id = vocab[].find_token(hex_token)
        if id != -1:
            tokens.append(id)

    # 2. Iteratively merge adjacent pairs with highest merge score
    while True:
        var best_score = Float32(-1e10)
        var best_id: llama_token = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            var id0 = Int(tokens[i])
            var id1 = Int(tokens[i + 1])
            if id0 < vocab[].vocab_size and id1 < vocab[].vocab_size:
                var pair = vocab[].tokens[id0] + vocab[].tokens[id1]
                var merged_id = vocab[].find_token(pair)
                if merged_id != -1:
                    var score = vocab[].scores[Int(merged_id)] if len(vocab[].scores) > Int(merged_id) else Float32(0.0)
                    if score > best_score:
                        best_score = score
                        best_id = merged_id
                        best_idx = i

        if best_idx == -1:
            break

        tokens[best_idx] = best_id
        var new_tokens = List[llama_token]()
        for i in range(best_idx + 1):
            new_tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            new_tokens.append(tokens[i])
        tokens = new_tokens^

    return tokens^

def detokenize_text(
    vocab: llama_vocab_p,
    tokens: Pointer[llama_token, MutUntrackedOrigin],
    n_tokens: Int,
    remove_special: Bool = True,
) -> String:
    """Reconstruct UTF-8 text from an array of tokens."""
    if Int(vocab) == 0 or n_tokens <= 0:
        return ""
    var result = String("")
    for i in range(n_tokens):
        var tok = tokens[unsafe_offset=i]
        if remove_special and (tok == vocab[].bos_id or tok == vocab[].eos_id or tok == vocab[].eot_id):
            continue
        result += vocab[].token_to_piece_str(tok)
    return result

# ─── Public llama.h Vocab & Tokenizer API ────────────────────────────────────

def llama_tokenize(
    vocab: llama_vocab_p,
    text: String,
    tokens: Pointer[llama_token, MutUntrackedOrigin],
    n_tokens_max: Int32,
    add_special: Bool = True,
    parse_special: Bool = False,
) -> Int32:
    """Tokenize `text` into output buffer `tokens`. Returns number of tokens written or required."""
    var tok_list = tokenize_bpe(vocab, text, add_special, parse_special)
    var count = len(tok_list)
    if count > Int(n_tokens_max):
        return -Int32(count)
    for i in range(count):
        tokens[unsafe_offset=i] = tok_list[i]
    return Int32(count)

def llama_token_to_piece(
    vocab: llama_vocab_p,
    token: llama_token,
    buf: Pointer[UInt8, MutUntrackedOrigin],
    length: Int32,
    lstrip: Int32 = 0,
    special: Bool = False,
) -> Int32:
    """Convert `token` to its piece string and write into `buf`."""
    if Int(vocab) == 0:
        return 0
    var piece = vocab[].token_to_piece_str(token)
    var plen = piece.byte_length()
    if plen > Int(length):
        return -Int32(plen)
    var p_ptr = piece.unsafe_ptr()
    for i in range(plen):
        buf[unsafe_offset=i] = p_ptr[unsafe_offset=i]
    return Int32(plen)

def llama_detokenize(
    vocab: llama_vocab_p,
    tokens: Pointer[llama_token, MutUntrackedOrigin],
    n_tokens: Int32,
    text_out: Pointer[UInt8, MutUntrackedOrigin],
    text_len_max: Int32,
    remove_special: Bool = True,
    unparse_special: Bool = False,
) -> Int32:
    """Detokenize tokens array into byte buffer `text_out`."""
    var res = detokenize_text(vocab, tokens, Int(n_tokens), remove_special)
    var blen = res.byte_length()
    if blen > Int(text_len_max):
        return -Int32(blen)
    var res_ptr = res.unsafe_ptr()
    for i in range(blen):
        text_out[unsafe_offset=i] = res_ptr[unsafe_offset=i]
    return Int32(blen)

@always_inline
def llama_vocab_bos(vocab: llama_vocab_p) -> llama_token:
    if Int(vocab) == 0:
        return 1
    return vocab[].bos_id

@always_inline
def llama_vocab_eos(vocab: llama_vocab_p) -> llama_token:
    if Int(vocab) == 0:
        return 2
    return vocab[].eos_id

@always_inline
def llama_vocab_eot(vocab: llama_vocab_p) -> llama_token:
    if Int(vocab) == 0:
        return -1
    return vocab[].eot_id

@always_inline
def llama_vocab_is_eog(vocab: llama_vocab_p, token: llama_token) -> Bool:
    if Int(vocab) == 0:
        return False
    return token == vocab[].eos_id or token == vocab[].eot_id

@always_inline
def llama_vocab_n_tokens(vocab: llama_vocab_p) -> Int32:
    if Int(vocab) == 0:
        return 0
    return Int32(vocab[].vocab_size)

