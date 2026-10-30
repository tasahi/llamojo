""" SentencePiece (SPM) and Gemma 2 / Gemma 4 / mmBERT BPE Tokenizer Engine for llamojo.

Implements:
- `_SpmSymbol` & `_SpmBigram`: Doubly-linked UTF-8 symbol list and priority-queue items
- `normalize_spm_spaces()`: Converts ASCII `' '` (`0x20`) to SentencePiece `\u2581` (`0xE2 0x96 0x81`), with configurable `add_space_prefix` (`False` for Gemma 2 / Gemma 4 / mmBERT, `True` for classic LLaMA SPM)
- `encode_hex_byte_fallback()`: `<0xXX>` hexadecimal byte fallback for unmapped UTF-8 bytes
- `tokenize_gemma_bpe()`: Doubly-linked priority-queue BPE tokenizer supporting both `merge_ranks` and `scores` without a dummy leading space prefix
- `tokenize_spm()`: SentencePiece bigram tokenizer matching `llama.cpp`
"""

from std.collections import List, Span
from .types import llama_token
from .vocab import llama_vocab_p


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


def normalize_spm_spaces(text: String, add_space_prefix: Bool) -> String:
    """ Replace ASCII space (0x20) with SentencePiece marker \u2581, optionally prepending a dummy prefix."""
    var out_bytes = List[UInt8]()
    if add_space_prefix:
        out_bytes.append(0xE2)
        out_bytes.append(0x96)
        out_bytes.append(0x81)
    var n_bytes = text.byte_length()
    var t_ptr = text.unsafe_ptr()
    for i in range(n_bytes):
        var b = t_ptr[unsafe_offset=i]
        if b == 32:
            out_bytes.append(0xE2)
            out_bytes.append(0x96)
            out_bytes.append(0x81)
        else:
            out_bytes.append(b)
    if len(out_bytes) == 0:
        return String("")
    return String(unsafe_from_utf8=Span(unsafe_ptr=out_bytes.unsafe_ptr(), length=len(out_bytes)))


def encode_hex_byte_fallback(
        vocab: llama_vocab_p,
        sym_text: String,
        mut tokens: List[llama_token],
    ):
    """ Decompose an unmapped UTF-8 symbol into `<0xXX>` byte fallback tokens (or `unk_id`)."""
    var sym_n = sym_text.byte_length()
    var s_bytes = sym_text.unsafe_ptr()
    var hex_digits = "0123456789ABCDEF"
    for b_idx in range(sym_n):
        var b = s_bytes[unsafe_offset=b_idx]
        var h0 = String(hex_digits[byte=Int(b >> 4):Int((b >> 4) + 1)])
        var h1 = String(hex_digits[byte=Int(b & 0x0F):Int((b & 0x0F) + 1)])
        var hex_token = "<0x" + h0 + h1 + ">"
        var byte_id = vocab[].find_token(hex_token)
        if byte_id == -1:
            byte_id = vocab[].unk_id
        if byte_id >= 0:
            tokens.append(byte_id)


def _split_utf8_symbols(norm_text: String) -> List[_SpmSymbol]:
    """ Split normalized UTF-8 string into doubly-linked codepoint symbols."""
    var blen = norm_text.byte_length()
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

    return symbols^


@always_inline
def _get_pair_priority(
        vocab: llama_vocab_p,
        left_text: String,
        right_text: String,
        use_merge_ranks: Bool,
    ) -> Tuple[Bool, Float32]:
    """ Check if `(left_text, right_text)` can merge and return `(can_merge, priority_score)`."""
    if use_merge_ranks:
        var pair_str = left_text + " " + right_text
        var rank_opt = vocab[].merge_ranks.find(pair_str)
        if rank_opt:
            # Lower merge rank = higher priority; invert to positive score so higher score wins
            return (True, Float32(-rank_opt.value()))
        return (False, Float32(0.0))
    else:
        var comb = left_text + right_text
        var tok_id = vocab[].find_token(comb)
        if tok_id != -1 and Int(tok_id) < vocab[].vocab_size:
            var score = vocab[].scores[Int(tok_id)] if len(vocab[].scores) > Int(tok_id) else Float32(0.0)
            return (True, score)
        return (False, Float32(0.0))


def _merge_spm_symbols(
        vocab: llama_vocab_p,
        mut symbols: List[_SpmSymbol],
        use_merge_ranks: Bool,
        mut tokens: List[llama_token],
    ):
    """ Run the doubly-linked priority-queue bigram merger and append resulting token IDs."""
    var n_syms = len(symbols)
    if n_syms == 0:
        return

    var queue = List[_SpmBigram]()

    for i in range(1, n_syms):
        var left_text = symbols[i - 1].text
        var right_text = symbols[i].text
        var prio = _get_pair_priority(vocab, left_text, right_text, use_merge_ranks)
        if prio[0]:
            var comb_len = symbols[i - 1].n + symbols[i].n
            queue.append(_SpmBigram(i - 1, i, prio[1], comb_len))

    # Iteratively merge highest-priority bigram (tie-break on smaller left index)
    while len(queue) > 0:
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
        queue[best_idx] = queue[len(queue) - 1]
        _ = queue.pop()

        var left_i = bg.left
        var right_i = bg.right

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

        # Enqueue new neighbor bigrams: (prev, left_i) and (left_i, next_of_right)
        var prev_i = symbols[left_i].prev
        if prev_i != -1:
            var prio_prev = _get_pair_priority(vocab, symbols[prev_i].text, symbols[left_i].text, use_merge_ranks)
            if prio_prev[0]:
                var len_prev = symbols[prev_i].n + symbols[left_i].n
                queue.append(_SpmBigram(prev_i, left_i, prio_prev[1], len_prev))

        if next_of_right != -1:
            var prio_next = _get_pair_priority(vocab, symbols[left_i].text, symbols[next_of_right].text, use_merge_ranks)
            if prio_next[0]:
                var len_next = symbols[left_i].n + symbols[next_of_right].n
                queue.append(_SpmBigram(left_i, next_of_right, prio_next[1], len_next))

    # Collect final tokens with <0xXX> hex byte fallback
    var curr = 0
    while curr != -1:
        var sym_text = symbols[curr].text
        var tok_id = vocab[].find_token(sym_text)
        if tok_id != -1:
            tokens.append(tok_id)
        else:
            encode_hex_byte_fallback(vocab, sym_text, tokens)
        curr = symbols[curr].next


def tokenize_gemma_bpe(
        vocab: llama_vocab_p,
        text: String,
    ) -> List[llama_token]:
    """ Encode a raw text span using mmBERT / Gemma 2 / Gemma 4 SentencePiece BPE (no dummy prefix space)."""
    var tokens = List[llama_token]()
    if Int(vocab) == 0 or text.byte_length() == 0:
        return tokens^

    var norm_text = normalize_spm_spaces(text, vocab[].add_space_prefix)
    if norm_text.byte_length() == 0:
        return tokens^

    var symbols = _split_utf8_symbols(norm_text)
    var use_merge_ranks = len(vocab[].merge_ranks) > 0
    _merge_spm_symbols(vocab, symbols, use_merge_ranks, tokens)
    return tokens^


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

    if text.byte_length() == 0:
        return tokens^

    var norm_text = normalize_spm_spaces(text, vocab[].add_space_prefix)
    if norm_text.byte_length() == 0:
        return tokens^

    var symbols = _split_utf8_symbols(norm_text)
    var use_merge_ranks = len(vocab[].merge_ranks) > 0 and len(vocab[].scores) == 0
    _merge_spm_symbols(vocab, symbols, use_merge_ranks, tokens)
    return tokens^