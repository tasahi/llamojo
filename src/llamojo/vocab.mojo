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

comptime CharList = List[UInt8]

comptime VOCAB_TYPE_NONE: Int32 = 0
comptime VOCAB_TYPE_SPM: Int32 = 1   # LLaMA / SentencePiece Unigram
comptime VOCAB_TYPE_BPE: Int32 = 2   # GPT-2 / LLaMA 3 BPE
comptime VOCAB_TYPE_WPM: Int32 = 3   # WordPiece

def get_gpt2_byte_table() -> List[String]:
    var t = List[String]()
    t.append("\u0100")
    t.append("\u0101")
    t.append("\u0102")
    t.append("\u0103")
    t.append("\u0104")
    t.append("\u0105")
    t.append("\u0106")
    t.append("\u0107")
    t.append("\u0108")
    t.append("\u0109")
    t.append("\u010a")
    t.append("\u010b")
    t.append("\u010c")
    t.append("\u010d")
    t.append("\u010e")
    t.append("\u010f")
    t.append("\u0110")
    t.append("\u0111")
    t.append("\u0112")
    t.append("\u0113")
    t.append("\u0114")
    t.append("\u0115")
    t.append("\u0116")
    t.append("\u0117")
    t.append("\u0118")
    t.append("\u0119")
    t.append("\u011a")
    t.append("\u011b")
    t.append("\u011c")
    t.append("\u011d")
    t.append("\u011e")
    t.append("\u011f")
    t.append("\u0120")
    t.append("!")
    t.append("\"")
    t.append("#")
    t.append("$")
    t.append("%")
    t.append("&")
    t.append("'")
    t.append("(")
    t.append(")")
    t.append("*")
    t.append("+")
    t.append(",")
    t.append("-")
    t.append(".")
    t.append("/")
    t.append("0")
    t.append("1")
    t.append("2")
    t.append("3")
    t.append("4")
    t.append("5")
    t.append("6")
    t.append("7")
    t.append("8")
    t.append("9")
    t.append(":")
    t.append(";")
    t.append("<")
    t.append("=")
    t.append(">")
    t.append("?")
    t.append("@")
    t.append("A")
    t.append("B")
    t.append("C")
    t.append("D")
    t.append("E")
    t.append("F")
    t.append("G")
    t.append("H")
    t.append("I")
    t.append("J")
    t.append("K")
    t.append("L")
    t.append("M")
    t.append("N")
    t.append("O")
    t.append("P")
    t.append("Q")
    t.append("R")
    t.append("S")
    t.append("T")
    t.append("U")
    t.append("V")
    t.append("W")
    t.append("X")
    t.append("Y")
    t.append("Z")
    t.append("[")
    t.append("\\")
    t.append("]")
    t.append("^")
    t.append("_")
    t.append("`")
    t.append("a")
    t.append("b")
    t.append("c")
    t.append("d")
    t.append("e")
    t.append("f")
    t.append("g")
    t.append("h")
    t.append("i")
    t.append("j")
    t.append("k")
    t.append("l")
    t.append("m")
    t.append("n")
    t.append("o")
    t.append("p")
    t.append("q")
    t.append("r")
    t.append("s")
    t.append("t")
    t.append("u")
    t.append("v")
    t.append("w")
    t.append("x")
    t.append("y")
    t.append("z")
    t.append("{")
    t.append("|")
    t.append("}")
    t.append("~")
    t.append("\u0121")
    t.append("\u0122")
    t.append("\u0123")
    t.append("\u0124")
    t.append("\u0125")
    t.append("\u0126")
    t.append("\u0127")
    t.append("\u0128")
    t.append("\u0129")
    t.append("\u012a")
    t.append("\u012b")
    t.append("\u012c")
    t.append("\u012d")
    t.append("\u012e")
    t.append("\u012f")
    t.append("\u0130")
    t.append("\u0131")
    t.append("\u0132")
    t.append("\u0133")
    t.append("\u0134")
    t.append("\u0135")
    t.append("\u0136")
    t.append("\u0137")
    t.append("\u0138")
    t.append("\u0139")
    t.append("\u013a")
    t.append("\u013b")
    t.append("\u013c")
    t.append("\u013d")
    t.append("\u013e")
    t.append("\u013f")
    t.append("\u0140")
    t.append("\u0141")
    t.append("\u0142")
    t.append("\u00a1")
    t.append("\u00a2")
    t.append("\u00a3")
    t.append("\u00a4")
    t.append("\u00a5")
    t.append("\u00a6")
    t.append("\u00a7")
    t.append("\u00a8")
    t.append("\u00a9")
    t.append("\u00aa")
    t.append("\u00ab")
    t.append("\u00ac")
    t.append("\u0143")
    t.append("\u00ae")
    t.append("\u00af")
    t.append("\u00b0")
    t.append("\u00b1")
    t.append("\u00b2")
    t.append("\u00b3")
    t.append("\u00b4")
    t.append("\u00b5")
    t.append("\u00b6")
    t.append("\u00b7")
    t.append("\u00b8")
    t.append("\u00b9")
    t.append("\u00ba")
    t.append("\u00bb")
    t.append("\u00bc")
    t.append("\u00bd")
    t.append("\u00be")
    t.append("\u00bf")
    t.append("\u00c0")
    t.append("\u00c1")
    t.append("\u00c2")
    t.append("\u00c3")
    t.append("\u00c4")
    t.append("\u00c5")
    t.append("\u00c6")
    t.append("\u00c7")
    t.append("\u00c8")
    t.append("\u00c9")
    t.append("\u00ca")
    t.append("\u00cb")
    t.append("\u00cc")
    t.append("\u00cd")
    t.append("\u00ce")
    t.append("\u00cf")
    t.append("\u00d0")
    t.append("\u00d1")
    t.append("\u00d2")
    t.append("\u00d3")
    t.append("\u00d4")
    t.append("\u00d5")
    t.append("\u00d6")
    t.append("\u00d7")
    t.append("\u00d8")
    t.append("\u00d9")
    t.append("\u00da")
    t.append("\u00db")
    t.append("\u00dc")
    t.append("\u00dd")
    t.append("\u00de")
    t.append("\u00df")
    t.append("\u00e0")
    t.append("\u00e1")
    t.append("\u00e2")
    t.append("\u00e3")
    t.append("\u00e4")
    t.append("\u00e5")
    t.append("\u00e6")
    t.append("\u00e7")
    t.append("\u00e8")
    t.append("\u00e9")
    t.append("\u00ea")
    t.append("\u00eb")
    t.append("\u00ec")
    t.append("\u00ed")
    t.append("\u00ee")
    t.append("\u00ef")
    t.append("\u00f0")
    t.append("\u00f1")
    t.append("\u00f2")
    t.append("\u00f3")
    t.append("\u00f4")
    t.append("\u00f5")
    t.append("\u00f6")
    t.append("\u00f7")
    t.append("\u00f8")
    t.append("\u00f9")
    t.append("\u00fa")
    t.append("\u00fb")
    t.append("\u00fc")
    t.append("\u00fd")
    t.append("\u00fe")
    t.append("\u00ff")
    return t^


def split_gpt2_chunks(text: String) -> List[String]:
    var chunks = List[String]()
    var n = text.byte_length()
    var i = 0
    var ptr = text.unsafe_ptr()

    while i < n:
        var b = ptr[unsafe_offset=i]
        # Carriage return or newline
        if b == 13 or b == 10:
            var start = i
            while i < n and (ptr[unsafe_offset=i] == 13 or ptr[unsafe_offset=i] == 10):
                i += 1
            var sp_start = i
            while i < n and (ptr[unsafe_offset=i] == 32 or ptr[unsafe_offset=i] == 9):
                i += 1
            var sp_count = i - sp_start
            if i < n and sp_count > 0:
                var next_b = ptr[unsafe_offset=i]
                var is_alpha = (next_b >= 65 and next_b <= 90) or (next_b >= 97 and next_b <= 122)
                var is_digit = (next_b >= 48 and next_b <= 57)
                if is_alpha or is_digit:
                    i -= 1
            chunks.append(String(text[byte=start:i]))
        elif b == 32 or b == 9:
            if i + 1 < n:
                var next_b = ptr[unsafe_offset=i + 1]
                var is_alpha = (next_b >= 65 and next_b <= 90) or (next_b >= 97 and next_b <= 122)
                var is_digit = (next_b >= 48 and next_b <= 57)
                if is_alpha or is_digit:
                    var start = i
                    i += 1
                    if is_alpha:
                        while i < n:
                            var cb = ptr[unsafe_offset=i]
                            if (cb >= 65 and cb <= 90) or (cb >= 97 and cb <= 122):
                                i += 1
                            else:
                                break
                    elif is_digit:
                        while i < n and (i - start) <= 3:
                            var cb = ptr[unsafe_offset=i]
                            if cb >= 48 and cb <= 57:
                                i += 1
                            else:
                                break
                    chunks.append(String(text[byte=start:i]))
                else:
                    var start = i
                    while i < n and (ptr[unsafe_offset=i] == 32 or ptr[unsafe_offset=i] == 9):
                        i += 1
                    chunks.append(String(text[byte=start:i]))
            else:
                var start = i
                while i < n and (ptr[unsafe_offset=i] == 32 or ptr[unsafe_offset=i] == 9):
                    i += 1
                chunks.append(String(text[byte=start:i]))
        elif (b >= 65 and b <= 90) or (b >= 97 and b <= 122):
            var start = i
            while i < n:
                var cb = ptr[unsafe_offset=i]
                if (cb >= 65 and cb <= 90) or (cb >= 97 and cb <= 122):
                    i += 1
                else:
                    break
            chunks.append(String(text[byte=start:i]))
        elif b >= 48 and b <= 57:
            var start = i
            while i < n and (i - start) < 3:
                var cb = ptr[unsafe_offset=i]
                if cb >= 48 and cb <= 57:
                    i += 1
                else:
                    break
            chunks.append(String(text[byte=start:i]))
        else:
            var start = i
            while i < n:
                var cb = ptr[unsafe_offset=i]
                var is_alnum = (cb >= 65 and cb <= 90) or (cb >= 97 and cb <= 122) or (cb >= 48 and cb <= 57)
                var is_space = cb == 32 or cb == 9 or cb == 10 or cb == 13
                if is_alnum or is_space:
                    break
                i += 1
            chunks.append(String(text[byte=start:i]))

    return chunks^

struct llama_vocab(Movable):
    """ Vocabulary and BPE/Unigram tokenizer dictionary populated from GGUF metadata."""
    var model_type: String
    var vocab_type: Int32
    var tokens: List[String]
    var scores: List[Float32]
    var token_types: List[Int32]
    var token_to_id: Dict[String, Int32]
    var merge_ranks: Dict[String, Int]
    var add_bos: Bool
    var bos_id: llama_token
    var eos_id: llama_token
    var eot_id: llama_token
    var pad_id: llama_token
    var unk_id: llama_token
    var vocab_size: Int

    def __init__(out self):
        self.model_type = "llama"
        self.vocab_type = VOCAB_TYPE_BPE
        self.tokens = List[String]()
        self.scores = List[Float32]()
        self.token_types = List[Int32]()
        self.token_to_id = Dict[String, Int32]()
        self.merge_ranks = Dict[String, Int]()
        self.add_bos = True
        self.bos_id = 1
        self.eos_id = 2
        self.eot_id = -1
        self.pad_id = -1
        self.unk_id = 0
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

    if (vocab[].model_type == "gemma4" or vocab[].model_type == "gemma-4") and len(vocab[].merge_ranks) > 0:
        # Gemma 4 SPM-style BPE:
        # 1. Replace spaces with SentencePiece marker \u2581 (0xE2 0x96 0x81)
        var sp_marker = "\u2581"
        var norm_text = String("")
        var n_bytes = text.byte_length()
        var t_ptr = text.unsafe_ptr()
        for i in range(n_bytes):
            if t_ptr[unsafe_offset=i] == 32:
                norm_text += sp_marker
            else:
                norm_text += String(text[byte=i:i + 1])

        # 2. Split into UTF-8 codepoints
        var blen = norm_text.byte_length()
        var norm_ptr = norm_text.unsafe_ptr()
        var u_chars = List[String]()
        var offs = 0
        while offs < blen:
            var b = norm_ptr[unsafe_offset=offs]
            var clen = 1
            if (b & 0x80) == 0:
                clen = 1
            elif (b & 0xE0) == 0xC0:
                clen = 2
            elif (b & 0xF0) == 0xE0:
                clen = 3
            elif (b & 0xF8) == 0xF0:
                clen = 4
            clen = min(clen, blen - offs)
            u_chars.append(String(norm_text[byte=offs:offs + clen]))
            offs += clen

        # 3. Iteratively merge highest-priority (lowest rank) bigrams from merge_ranks
        while len(u_chars) > 1:
            var best_rank = 1000000000
            var best_idx = -1
            var best_merged = String("")

            for i in range(len(u_chars) - 1):
                var c0 = u_chars[i]
                var c1 = u_chars[i + 1]
                var pair_str = c0 + " " + c1
                var rank_opt = vocab[].merge_ranks.find(pair_str)
                if rank_opt:
                    var rank = rank_opt.value()
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
                        best_merged = c0 + c1

            if best_idx == -1:
                break

            var next_chars = List[String]()
            for i in range(best_idx):
                next_chars.append(u_chars[i])
            next_chars.append(best_merged)
            for i in range(best_idx + 2, len(u_chars)):
                next_chars.append(u_chars[i])
            u_chars = next_chars^

        for p_i in range(len(u_chars)):
            var tid = vocab[].find_token(u_chars[p_i])
            if tid != -1:
                tokens.append(tid)

        return tokens^

    if vocab[].model_type == "gpt2" and len(vocab[].merge_ranks) > 0:
        var byte_table = get_gpt2_byte_table()
        var chunks = split_gpt2_chunks(text)

        for c_idx in range(len(chunks)):
            var chunk = chunks[c_idx]
            var c_len = chunk.byte_length()
            var c_ptr = chunk.unsafe_ptr()

            var u_chars = List[String]()
            for b_i in range(c_len):
                var byte_val = Int(c_ptr[unsafe_offset=b_i])
                u_chars.append(byte_table[byte_val])

            while len(u_chars) > 1:
                var best_rank = 1000000000
                var best_idx = -1
                var best_merged = String("")

                for i in range(len(u_chars) - 1):
                    var c0 = u_chars[i]
                    var c1 = u_chars[i + 1]
                    var pair_str = c0 + " " + c1
                    var rank_opt = vocab[].merge_ranks.find(pair_str)
                    if rank_opt:
                        var rank = rank_opt.value()
                        if rank < best_rank:
                            best_rank = rank
                            best_idx = i
                            best_merged = c0 + c1

                if best_idx == -1:
                    break

                var next_chars = List[String]()
                for i in range(best_idx):
                    next_chars.append(u_chars[i])
                next_chars.append(best_merged)
                for i in range(best_idx + 2, len(u_chars)):
                    next_chars.append(u_chars[i])
                u_chars = next_chars^

            for p_i in range(len(u_chars)):
                var tid = vocab[].find_token(u_chars[p_i])
                if tid != -1:
                    tokens.append(tid)

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

# ─── SentencePiece Unigram Tokenizer Engine (Viterbi Trellis) ───────────────

# ─── SentencePiece (SPM) Bigram BPE Tokenizer ──────────────────────────────

# ─── SentencePiece (SPM) Bigram BPE Tokenizer ──────────────────────────────

struct _SpmSymbol(Copyable, Movable):
    var text: String
    var prev: Int
    var next: Int
    var n: Int

    def __init__(out self, text: String, prev: Int, next: Int):
        self.text = text
        self.prev = prev
        self.next = next
        self.n = text.byte_length()

struct _SpmBigram(ImplicitlyCopyable, Copyable, Movable):
    var left: Int
    var right: Int
    var score: Float32
    var size: Int

    def __init__(out self, left: Int, right: Int, score: Float32, size: Int):
        self.left = left
        self.right = right
        self.score = score
        self.size = size

def tokenize_spm(
        vocab: llama_vocab_p,
        text: String,
        add_special: Bool = True,
        parse_special: Bool = False,
    ) -> List[llama_token]:
    """ Tokenize UTF-8 string using SentencePiece SPM bigram merging matching llama.cpp."""
    var tokens = List[llama_token]()
    if Int(vocab) == 0:
        return tokens^

    if add_special and vocab[].add_bos and vocab[].bos_id >= 0:
        tokens.append(vocab[].bos_id)

    var n_bytes = text.byte_length()
    if n_bytes == 0:
        return tokens^

    # Normalization: replace all spaces with SentencePiece marker \u2581 (0xE2 0x96 0x81) and prefix
    var sp_marker = "\u2581"
    var norm_text = sp_marker
    var t_ptr = text.unsafe_ptr()
    for i in range(n_bytes):
        if t_ptr[unsafe_offset=i] == 32:
            norm_text += sp_marker
        else:
            norm_text += String(text[byte=i:i + 1])

    var blen = norm_text.byte_length()
    if blen == 0:
        return tokens^

    # Split string into UTF-8 codepoints
    var norm_ptr = norm_text.unsafe_ptr()
    var symbols = List[_SpmSymbol]()
    var offs = 0
    var s_idx = 0

    while offs < blen:
        var b = norm_ptr[unsafe_offset=offs]
        var clen = 1
        if (b & 0x80) == 0:
            clen = 1
        elif (b & 0xE0) == 0xC0:
            clen = 2
        elif (b & 0xF0) == 0xE0:
            clen = 3
        elif (b & 0xF8) == 0xF0:
            clen = 4
        clen = min(clen, blen - offs)

        var sym_str = String(norm_text[byte=offs:offs + clen])
        var next_i = s_idx + 1 if (offs + clen < blen) else -1
        symbols.append(_SpmSymbol(sym_str, s_idx - 1, next_i))
        offs += clen
        s_idx += 1

    var n_syms = len(symbols)
    if n_syms == 0:
        return tokens^

    # Queue of bigrams
    var queue = List[_SpmBigram]()

    for i in range(1, n_syms):
        var left_text = symbols[i - 1].text
        var right_text = symbols[i].text
        var comb = left_text + right_text
        var tok_id = vocab[].find_token(comb)
        if tok_id != -1 and Int(tok_id) < vocab[].vocab_size:
            var score = vocab[].scores[Int(tok_id)] if len(vocab[].scores) > Int(tok_id) else Float32(0.0)
            queue.append(_SpmBigram(i - 1, i, score, comb.byte_length()))

    # Iteratively merge highest-scoring bigram (tie-break on smaller left index)
    while len(queue) > 0:
        # Find best bigram in queue
        var best_idx = 0
        var best_score = queue[0].score
        var best_left = queue[0].left

        for q_i in range(1, len(queue)):
            var item_ref = queue[q_i]
            if (item_ref.score > best_score) or (item_ref.score == best_score and item_ref.left < best_left):
                best_score = item_ref.score
                best_left = item_ref.left
                best_idx = q_i

        var bg = queue[best_idx]
        # Fast remove by swap-with-last (unordered queue)
        queue[best_idx] = queue[len(queue) - 1]
        _ = queue.pop()

        var left_i = bg.left
        var right_i = bg.right

        # Check if already merged or invalidated
        if symbols[left_i].n == 0 or symbols[right_i].n == 0:
            continue
        if symbols[left_i].n + symbols[right_i].n != bg.size:
            continue

        # Merge right into left
        var r_text = symbols[right_i].text
        symbols[left_i].text += r_text
        symbols[left_i].n += symbols[right_i].n
        symbols[right_i].n = 0

        # Remove right from chain
        var next_of_right = symbols[right_i].next
        symbols[left_i].next = next_of_right
        if next_of_right >= 0:
            symbols[next_of_right].prev = left_i

        # Try new bigrams with neighbors: (prev, left_i) and (left_i, next_of_right)
        var prev_i = symbols[left_i].prev
        if prev_i != -1:
            var prev_text = symbols[prev_i].text
            var curr_left_text = symbols[left_i].text
            var comb_prev = prev_text + curr_left_text
            var tok_prev = vocab[].find_token(comb_prev)
            if tok_prev != -1 and Int(tok_prev) < vocab[].vocab_size:
                var score = vocab[].scores[Int(tok_prev)] if len(vocab[].scores) > Int(tok_prev) else Float32(0.0)
                queue.append(_SpmBigram(prev_i, left_i, score, comb_prev.byte_length()))

        if next_of_right != -1:
            var curr_left_text = symbols[left_i].text
            var next_text = symbols[next_of_right].text
            var comb_next = curr_left_text + next_text
            var tok_next = vocab[].find_token(comb_next)
            if tok_next != -1 and Int(tok_next) < vocab[].vocab_size:
                var score = vocab[].scores[Int(tok_next)] if len(vocab[].scores) > Int(tok_next) else Float32(0.0)
                queue.append(_SpmBigram(left_i, next_of_right, score, comb_next.byte_length()))

    # Collect final tokens
    var curr = 0
    while curr != -1:
        var sym_text = symbols[curr].text
        var sym_n = symbols[curr].n
        var tok_id = vocab[].find_token(sym_text)
        if tok_id != -1:
            tokens.append(tok_id)
        else:
            # Fallback to byte tokens or unk
            var s_bytes = sym_text.unsafe_ptr()
            for b_idx in range(sym_n):
                var b = s_bytes[unsafe_offset=b_idx]
                var hex_digits = "0123456789ABCDEF"
                var h0 = String(hex_digits[byte=Int(b >> 4):Int((b >> 4) + 1)])
                var h1 = String(hex_digits[byte=Int(b & 0x0F):Int((b & 0x0F) + 1)])
                var hex_token = "<0x" + h0 + h1 + ">"
                var byte_id = vocab[].find_token(hex_token)
                if byte_id == -1:
                    byte_id = vocab[].unk_id
                tokens.append(byte_id)
        curr = symbols[curr].next

    return tokens^

def tokenize_text(
        vocab: llama_vocab_p,
        text: String,
        add_special: Bool = True,
        parse_special: Bool = False,
    ) -> List[llama_token]:
    """ Unified tokenization router dispatching to BPE or SentencePiece Unigram based on vocab_type."""
    if Int(vocab) == 0:
        return List[llama_token]()
    if vocab[].vocab_type == VOCAB_TYPE_SPM:
        return tokenize_spm(vocab, text, add_special, parse_special)
    else:
        return tokenize_bpe(vocab, text, add_special, parse_special)

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
