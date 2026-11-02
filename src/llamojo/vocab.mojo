""" Vocabulary & Tokenizer Implementation for llamojo.

Integrates:
- llama2.mojo/bpe_encode & Tokenizer
- GGUF vocabulary metadata (tokenizer.ggml.tokens, scores, token_type, merges)
- SentencePiece (' ' prefix) & Byte-level BPE (<0xXX> byte fallback, GPT-2 byte mapping)
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
from std.collections import Span
from std.memory.alloc import unsafe_alloc

from .types import llama_token, null_ptr
from .gguf import GGUFContext
from .vocab_gpt2 import get_gpt2_byte_table, split_gpt2_chunks, tokenize_gpt2_bpe
from .vocab_gemma import (
    _SpmSymbol,
    _SpmBigram,
    normalize_spm_spaces,
    encode_hex_byte_fallback,
    tokenize_gemma_bpe,
    tokenize_spm,
)

comptime CharList = List[UInt8]

comptime VOCAB_TYPE_NONE: Int32 = 0
comptime VOCAB_TYPE_SPM: Int32 = 1   # LLaMA / SentencePiece Unigram
comptime VOCAB_TYPE_BPE: Int32 = 2   # GPT-2 / LLaMA 3 BPE
comptime VOCAB_TYPE_WPM: Int32 = 3   # WordPiece

struct llama_vocab(Movable):
    """ Vocabulary and BPE/Unigram tokenizer dictionary populated from GGUF metadata."""
    var model_type: String
    var vocab_type: Int32
    var tokens: List[String]
    var scores: List[Float32]
    var token_types: List[Int32]
    var token_to_id: Dict[String, Int32]
    var merge_ranks: Dict[String, Int]
    var special_tokens: List[String]
    var add_bos: Bool
    var add_eos: Bool
    var add_space_prefix: Bool
    var bos_id: llama_token
    var eos_id: llama_token
    var eot_id: llama_token
    var pad_id: llama_token
    var unk_id: llama_token
    var mask_id: llama_token
    var vocab_size: Int

    def __init__(out self):
        self.model_type = "llama"
        self.vocab_type = VOCAB_TYPE_BPE
        self.tokens = List[String]()
        self.scores = List[Float32]()
        self.token_types = List[Int32]()
        self.token_to_id = Dict[String, Int32]()
        self.merge_ranks = Dict[String, Int]()
        self.special_tokens = List[String]()
        self.add_bos = True
        self.add_eos = False
        self.add_space_prefix = True
        self.bos_id = 1
        self.eos_id = 2
        self.eot_id = -1
        self.pad_id = -1
        self.unk_id = 0
        self.mask_id = -1
        self.vocab_size = 0

    def __init__(out self, gguf: GGUFContext):
        self.model_type = gguf.get_string("tokenizer.ggml.model", "llama")
        if self.model_type == "llama":
            self.vocab_type = VOCAB_TYPE_SPM
        elif self.model_type == "gpt2":
            self.vocab_type = VOCAB_TYPE_BPE
        elif self.model_type == "bert":
            self.vocab_type = VOCAB_TYPE_WPM
        else:
            var merges_opt = gguf.metadata.find("tokenizer.ggml.merges")
            if not merges_opt:
                self.vocab_type = VOCAB_TYPE_SPM
            else:
                self.vocab_type = VOCAB_TYPE_BPE

        self.tokens = List[String]()
        self.scores = List[Float32]()
        self.token_types = List[Int32]()
        self.token_to_id = Dict[String, Int32]()
        self.merge_ranks = Dict[String, Int]()
        self.special_tokens = List[String]()

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

        var merges_opt = gguf.metadata.find("tokenizer.ggml.merges")
        if merges_opt:
            var m_arr = merges_opt.value().str_arr.copy()
            for m_i in range(len(m_arr)):
                self.merge_ranks[m_arr[m_i]] = m_i

        self.vocab_size = len(self.tokens)
        for i in range(self.vocab_size):
            self.token_to_id[self.tokens[i]] = Int32(i)
            if i < len(self.token_types):
                var t_type = self.token_types[i]
                # 3 = CONTROL, 4 = USER_DEFINED
                if (t_type == 3 or t_type == 4) and self.tokens[i].byte_length() > 0:
                    self.special_tokens.append(self.tokens[i])

        var arch = gguf.get_string("general.architecture", "llama")
        var default_space_prefix = True
        if (
            self.model_type == "gemma"
            or self.model_type == "gemma2"
            or self.model_type == "gemma-2"
            or self.model_type == "gemma4"
            or self.model_type == "gemma-4"
            or self.model_type == "mmbert"
            or self.model_type == "modernbert"
            or arch == "gemma"
            or arch == "gemma2"
            or arch == "gemma4"
            or arch == "modernbert"
            or arch == "mmbert"
        ):
            default_space_prefix = False
        self.add_space_prefix = gguf.get_bool("tokenizer.ggml.add_space_prefix", default_space_prefix)

        var default_add_eos = False
        if (
            self.model_type == "mmbert"
            or self.model_type == "modernbert"
            or self.model_type == "bert"
            or arch == "modernbert"
            or arch == "mmbert"
            or arch == "bert"
            or arch == "laya"
        ):
            default_add_eos = True
        self.add_eos = gguf.get_bool("tokenizer.ggml.add_eos_token", default_add_eos)

        self.add_bos = gguf.get_bool("tokenizer.ggml.add_bos_token", True)
        if self.model_type == "gemma4" or self.model_type == "gemma-4":
            # Workaround for Gemma 4 (matches upstream llama.cpp PR #21500):
            # Gemma 4 requires BOS=2 at the start of sequence even if add_bos_token is set to false in GGUF.
            self.add_bos = True
        self.bos_id = Int32(gguf.get_int("tokenizer.ggml.bos_token_id", 1))
        self.eos_id = Int32(gguf.get_int("tokenizer.ggml.eos_token_id", 2))
        self.eot_id = Int32(gguf.get_int("tokenizer.ggml.eot_token_id", -1))
        self.pad_id = Int32(gguf.get_int("tokenizer.ggml.padding_token_id", -1))
        self.unk_id = Int32(gguf.get_int("tokenizer.ggml.unknown_token_id", 0))
        self.mask_id = Int32(gguf.get_int("tokenizer.ggml.mask_token_id", -1))
        if self.mask_id == -1:
            var m_id = self.find_token("<mask>")
            if m_id == -1:
                m_id = self.find_token("[MASK]")
            self.mask_id = m_id

        # Ensure known special IDs have their piece strings in special_tokens
        var known_ids = List[llama_token]()
        known_ids.append(self.bos_id)
        known_ids.append(self.eos_id)
        known_ids.append(self.eot_id)
        known_ids.append(self.pad_id)
        known_ids.append(self.mask_id)
        for k_i in range(len(known_ids)):
            var kid = known_ids[k_i]
            if kid >= 0 and Int(kid) < self.vocab_size:
                var k_str = self.tokens[Int(kid)]
                if k_str.byte_length() > 0:
                    var already = False
                    for s_i in range(len(self.special_tokens)):
                        if self.special_tokens[s_i] == k_str:
                            already = True
                            break
                    if not already:
                        self.special_tokens.append(k_str)

    def add_special_token(mut self, piece: String, token_type: Int32 = 4) -> llama_token:
        """ Dynamically register a special/control token (e.g., `<opt_0>`, `<mask>`) into the vocabulary."""
        var existing = self.find_token(piece)
        if existing != -1:
            if Int(existing) < len(self.token_types):
                self.token_types[Int(existing)] = token_type
            var already = False
            for i in range(len(self.special_tokens)):
                if self.special_tokens[i] == piece:
                    already = True
                    break
            if not already and piece.byte_length() > 0:
                self.special_tokens.append(piece)
            return existing

        var new_id = Int32(self.vocab_size)
        self.tokens.append(piece)
        self.scores.append(0.0)
        while len(self.token_types) < self.vocab_size:
            self.token_types.append(1)
        self.token_types.append(token_type)
        self.token_to_id[piece] = new_id
        if piece.byte_length() > 0:
            self.special_tokens.append(piece)
        self.vocab_size += 1
        return new_id

    def find_token(self, s: String) -> Int32:
        """ Find token ID for an exact piece string."""
        var opt = self.token_to_id.find(s)
        if opt:
            return opt.value()
        return -1

    def token_to_piece_str(self, tok: llama_token) -> String:
        """ Decode a token ID into its raw text representation."""
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
        if s.byte_length() >= 3 and s.unsafe_ptr()[unsafe_offset=0] == 0xE2 and s.unsafe_ptr()[unsafe_offset=1] == 0x96 and s.unsafe_ptr()[unsafe_offset=2] == 0x81:
            var rem_len = s.byte_length() - 3
            if rem_len == 0:
                return " "
            var rem_ptr = s.unsafe_ptr().unsafe_offset(3)
            return " " + String(unsafe_from_utf8=Span(unsafe_ptr=rem_ptr, length=rem_len))

        # Translate GPT-2 space (\u0120: 0xc4, 0xa0) to ASCII space
        # Translate GPT-2 newline (\u010a: 0xc4, 0x8a) to ASCII newline
        # Translate GPT-2 tab (\u0109: 0xc4, 0x89) to ASCII tab
        var slen = s.byte_length()
        var s_ptr = s.unsafe_ptr()
        var has_special_bytes = False
        for i in range(slen - 1):
            if s_ptr[unsafe_offset=i] == 0xC4:
                var b2 = s_ptr[unsafe_offset=i + 1]
                if b2 == 0xA0 or b2 == 0x8A or b2 == 0x89:
                    has_special_bytes = True
                    break

        if has_special_bytes:
            var out_bytes = List[UInt8]()
            var idx = 0
            while idx < slen:
                if idx < slen - 1 and s_ptr[unsafe_offset=idx] == 0xC4:
                    var b2 = s_ptr[unsafe_offset=idx + 1]
                    if b2 == 0xA0: # Ġ -> ' '
                        out_bytes.append(UInt8(32))
                        idx += 2
                        continue
                    elif b2 == 0x8A: # Ċ -> '\n'
                        out_bytes.append(UInt8(10))
                        idx += 2
                        continue
                    elif b2 == 0x89: # ĉ -> '\t'
                        out_bytes.append(UInt8(9))
                        idx += 2
                        continue
                out_bytes.append(s_ptr[unsafe_offset=idx])
                idx += 1
            return String(unsafe_from_utf8=Span(unsafe_ptr=out_bytes.unsafe_ptr(), length=len(out_bytes)))

        return s

comptime llama_vocab_p = Pointer[llama_vocab, MutUntrackedOrigin]

# ─── BPE Tokenizer Engine ────────────────────────────────────────────────────

def tokenize_bpe(
        vocab: llama_vocab_p,
        text: String,
        add_special: Bool = True,
        parse_special: Bool = False,
    ) -> List[llama_token]:
    """ Tokenize UTF-8 text string into a list of llama_tokens using BPE merge priority."""
    var tokens = List[llama_token]()
    if Int(vocab) == 0:
        return tokens^

    # Check additive BOS flag from GGUF metadata
    if add_special and vocab[].add_bos and vocab[].bos_id >= 0:
        tokens.append(vocab[].bos_id)

    if text.byte_length() == 0:
        return tokens^

    if (
        vocab[].model_type == "gemma"
        or vocab[].model_type == "gemma2"
        or vocab[].model_type == "gemma-2"
        or vocab[].model_type == "gemma4"
        or vocab[].model_type == "gemma-4"
        or vocab[].model_type == "mmbert"
        or vocab[].model_type == "modernbert"
    ):
        var gemma_tokens = tokenize_gemma_bpe(vocab, text)
        for i in range(len(gemma_tokens)):
            tokens.append(gemma_tokens[i])
        return tokens^

    if vocab[].model_type == "gpt2" and len(vocab[].merge_ranks) > 0:
        var gpt2_tokens = tokenize_gpt2_bpe(vocab, text)
        for i in range(len(gpt2_tokens)):
            tokens.append(gpt2_tokens[i])
        return tokens^

    # Standard BPE fallback for SentencePiece / LLaMA-1/2 models
    for pos in range(text.byte_length()):
        var ch = String(text[byte=pos:pos + 1])
        var id = vocab[].find_token(ch)
        if id == -1:
            var b = text.unsafe_ptr()[unsafe_offset=pos]
            var hex_digits = "0123456789ABCDEF"
            var h0 = String(hex_digits[byte=Int(b >> 4):Int((b >> 4) + 1)])
            var h1 = String(hex_digits[byte=Int(b & 0x0F):Int((b & 0x0F) + 1)])
            var hex_token = "<0x" + h0 + h1 + ">"
            id = vocab[].find_token(hex_token)
        if id != -1:
            tokens.append(id)

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


def _encode_raw_span(
        vocab: llama_vocab_p,
        span_text: String,
        mut tokens: List[llama_token],
    ):
    """ Dispatch a single non-special text span to the active tokenization engine."""
    if span_text.byte_length() == 0:
        return
    if vocab[].vocab_type == VOCAB_TYPE_SPM:
        var spm_toks = tokenize_spm(vocab, span_text, False, False)
        for i in range(len(spm_toks)):
            tokens.append(spm_toks[i])
    else:
        var bpe_toks = tokenize_bpe(vocab, span_text, False, False)
        for i in range(len(bpe_toks)):
            tokens.append(bpe_toks[i])


def tokenize_text(
        vocab: llama_vocab_p,
        text: String,
        add_special: Bool = True,
        parse_special: Bool = False,
    ) -> List[llama_token]:
    """ Unified tokenization router with special-token pre-splitting and BOS/EOS encoder framing."""
    var tokens = List[llama_token]()
    if Int(vocab) == 0:
        return tokens^

    if add_special and vocab[].add_bos and vocab[].bos_id >= 0:
        tokens.append(vocab[].bos_id)

    var n_bytes = text.byte_length()
    if n_bytes > 0:
        if not parse_special or len(vocab[].special_tokens) == 0:
            _encode_raw_span(vocab, text, tokens)
        else:
            var t_ptr = text.unsafe_ptr()
            var span_start = 0
            var pos = 0

            while pos < n_bytes:
                var best_match_idx = -1
                var best_match_len = 0

                for s_i in range(len(vocab[].special_tokens)):
                    var sp_str = vocab[].special_tokens[s_i]
                    var sp_len = sp_str.byte_length()
                    if sp_len > best_match_len and pos + sp_len <= n_bytes:
                        var sp_ptr = sp_str.unsafe_ptr()
                        var match_ok = True
                        for k in range(sp_len):
                            if t_ptr[unsafe_offset=pos + k] != sp_ptr[unsafe_offset=k]:
                                match_ok = False
                                break
                        if match_ok:
                            best_match_idx = s_i
                            best_match_len = sp_len

                if best_match_idx != -1:
                    if pos > span_start:
                        var sub_len = pos - span_start
                        var sub_ptr = t_ptr.unsafe_offset(span_start)
                        var raw_span = String(unsafe_from_utf8=Span(unsafe_ptr=sub_ptr, length=sub_len))
                        _encode_raw_span(vocab, raw_span, tokens)

                    var sp_piece = vocab[].special_tokens[best_match_idx]
                    var sp_id = vocab[].find_token(sp_piece)
                    if sp_id != -1:
                        tokens.append(sp_id)

                    pos += best_match_len
                    span_start = pos
                else:
                    pos += 1

            if span_start < n_bytes:
                var rem_len = n_bytes - span_start
                var rem_ptr = t_ptr.unsafe_offset(span_start)
                var rem_span = String(unsafe_from_utf8=Span(unsafe_ptr=rem_ptr, length=rem_len))
                _encode_raw_span(vocab, rem_span, tokens)

    if add_special and vocab[].add_eos and vocab[].eos_id >= 0:
        tokens.append(vocab[].eos_id)

    return tokens^


def detokenize_text(
        vocab: llama_vocab_p,
        tokens: Pointer[llama_token, MutUntrackedOrigin],
        n_tokens: Int,
        remove_special: Bool = True,
    ) -> String:
    """ Reconstruct UTF-8 text from an array of tokens."""
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
    """ Tokenize `text` into output buffer `tokens`. Returns number of tokens written or required."""
    var tok_list = tokenize_text(vocab, text, add_special, parse_special)
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
    """ Convert `token` to its piece string and write into `buf`."""
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
    """ Detokenize tokens array into byte buffer `text_out`."""
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

def llama_vocab_type(vocab: llama_vocab_p) -> Int32:
    if Int(vocab) == 0:
        return VOCAB_TYPE_NONE
    return vocab[].vocab_type

def llama_vocab_get_score(vocab: llama_vocab_p, token: llama_token) -> Float32:
    if Int(vocab) == 0 or token < 0 or Int(token) >= len(vocab[].scores):
        return Float32(0.0)
    return vocab[].scores[Int(token)]

def llama_vocab_add_special_token(
        vocab: llama_vocab_p,
        piece: String,
        token_type: Int32 = 4,
    ) -> llama_token:
    """ Register a dynamic special/control token on `vocab` and return its token ID."""
    if Int(vocab) == 0:
        return -1
    return vocab[].add_special_token(piece, token_type)

def llama_vocab_find_token(vocab: llama_vocab_p, piece: String) -> llama_token:
    """ Look up the token ID for an exact piece string."""
    if Int(vocab) == 0:
        return -1
    return vocab[].find_token(piece)
