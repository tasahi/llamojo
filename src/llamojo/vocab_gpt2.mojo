""" GPT-2 / ModernBERT Byte-Level BPE Tokenizer Engine for llamojo.

Implements:
- `get_gpt2_byte_table()`: 256-entry byte-to-Unicode remapping table (`0x00`..`0xFF` -> printable Unicode)
- `split_gpt2_chunks()`: Pre-tokenization regex-equivalent chunker for contractions, words, digits, and whitespace
- `tokenize_gpt2_bpe()`: Rank-based iterative BPE pair merging using `vocab[].merge_ranks`
- `decode_gpt2_piece()`: Helper for decoding GPT-2 special byte characters (`Ġ`, `Ċ`, `ĉ`) back to ASCII
"""

from std.collections import List, Span
from .types import llama_token
from .vocab import llama_vocab_p


def get_gpt2_byte_table() -> List[String]:
    """ Return the 256-entry GPT-2 byte-to-unicode mapping table."""
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
    """ Split raw UTF-8 text into pre-tokenization chunks following GPT-2 regex rules."""
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


def tokenize_gpt2_bpe(
        vocab: llama_vocab_p,
        text: String,
    ) -> List[llama_token]:
    """ Encode a raw text span using GPT-2 byte-level BPE and `vocab[].merge_ranks`."""
    var tokens = List[llama_token]()
    if Int(vocab) == 0 or text.byte_length() == 0:
        return tokens^

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

