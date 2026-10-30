# Feasibility Study: Native mmBERT / Gemma 2 Tokenizer Management in `llamojo`

## 1. Executive Summary

This document evaluates the feasibility and modular architectural design for natively managing the **mmBERT / Gemma 2 tokenizer** within the **`llamojo`** library (`/mnt/c/Documents/Mojo/llamojo`).

### Key Findings
- **High Feasibility (100% Implemented ✅)**: `llamojo` natively implements the full SentencePiece BPE, Gemma 2 / mmBERT normalization, special-token pre-splitting, and bidirectional encoder framing across [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo), [`src/llamojo/vocab_gpt2.mojo`](../src/llamojo/vocab_gpt2.mojo), and [`src/llamojo/vocab_gemma.mojo`](../src/llamojo/vocab_gemma.mojo).
- **Modular `vocab_*.mojo` File Separation (All Phases Completed ✅)**: Rather than keeping all tokenization engines inside a monolithic `vocab.mojo`, the vocabulary subsystem is cleanly partitioned into three focused files:
  1. **[`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo) (Completed ✅)**: Core vocabulary struct (`llama_vocab`), GGUF metadata loading, dynamic special-token registration (`add_special_token`), longest-match special-token pre-splitting (`parse_special=True`), BOS/EOS bidirectional framing, detokenization, top-level routing (`tokenize_text`), and the public `llama.h` C-compatible API.
  2. **[`src/llamojo/vocab_gpt2.mojo`](../src/llamojo/vocab_gpt2.mojo) (Completed ✅)**: GPT-2 / English ModernBERT byte-level BPE encoding functions (`get_gpt2_byte_table`, `split_gpt2_chunks`, `tokenize_gpt2_bpe`).
  3. **[`src/llamojo/vocab_gemma.mojo`](../src/llamojo/vocab_gemma.mojo) (Completed ✅)**: SentencePiece / Gemma 2 / Gemma 4 / mmBERT BPE encoding functions (`_SpmSymbol`, `_SpmBigram`, `normalize_spm_spaces`, `encode_hex_byte_fallback`, `tokenize_gemma_bpe`, `tokenize_spm`).
- **Clean Architectural Boundary**: Keeping all text normalization, special token splitting (`<opt_0>`...`<opt_15>`), and BPE/SentencePiece merging inside `llamojo` preserves the strict boundary where `ggojo` acts purely as the mathematical tensor/autodiff engine (`ggml` equivalent) and `llamojo` acts as the runtime, vocabulary, and model lifecycle manager (`llama.cpp` equivalent).
- **Targeted Enhancements & Implementation Status**:
  1. ✅ Modular extraction of GPT-2 BPE into `vocab_gpt2.mojo` and SentencePiece/Gemma BPE into `vocab_gemma.mojo`.
  2. ✅ Model type routing in `vocab.mojo` for `gemma2` / `modernbert` / `mmbert` (respecting `add_space_prefix == False` without dummy prefix space injection).
  3. ✅ Bidirectional encoder special-token framing (`[<bos>, ..., <eos>]`).
  4. ✅ Pre-tokenization splitting of special/control tokens (`parse_special=True` for `<opt_k>` and `<mask>`).
  5. ✅ Byte-fallback (`<0xXX>`) and doubly-linked symbol priority-queue merge optimization for large $256,000$-token vocabularies in `vocab_gemma.mojo`.

---

## 2. Characteristics of the mmBERT / Gemma 2 Tokenizer

Unlike classic BERT models (which use WordPiece with `[CLS]` and `[SEP]`) or original English ModernBERT (which uses a 50k GPT-2/OLMo-style byte-level BPE), **mmBERT (`jhu-clsp/mmBERT-base`)** uses the **Gemma 2 tokenizer**:

| Property | Value / Behavior in mmBERT (Gemma 2 Tokenizer) |
| :--- | :--- |
| **Tokenizer Algorithm** | SentencePiece Byte-Pair Encoding (SPM-BPE) with byte fallback |
| **Vocabulary Size** | `256,000` base tokens (+ Laya `<opt_k>` added tokens) |
| **Whitespace Normalization** | Replaces ASCII space `' '` (`0x20`) with SentencePiece `▁` (`\u2581`, bytes `0xE2 0x96 0x81`). **Does not** prepend a leading dummy `▁` at the start of text. |
| **`bos_token` (`<bos>`)** | ID `2` (acts as the `[CLS]` / sequence start token) |
| **`eos_token` / `sep_token` (`<eos>`)** | ID `1` (acts as `[SEP]` and sequence termination token) |
| **`pad_token` (`<pad>`)** | ID `0` |
| **`unk_token` (`<unk>`)** | ID `3` |
| **`mask_token` (`<mask>`)** | ID `4` (or mapped unused token in GGUF metadata) |
| **Byte Fallback** | Unknown UTF-8 characters decompose into raw byte tokens `<0x00>` through `<0xFF>` |
| **Sequence Framing** | Bidirectional encoder inputs are framed as `<bos> + tokens + <eos>` |

---

## 3. Current State Analysis & Proposed Modular Split

Currently, [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo) (~1,053 lines) combines GGUF loading, detokenization, public C-API wrappers, and three distinct tokenization engines in a single file. We will refactor it into a clean multi-file `vocab*.mojo` architecture:

### 3.1 `src/llamojo/vocab.mojo` (Vocabulary Loader, Router & Public API)
Acts as the primary vocabulary management and routing module:
- **Vocabulary Constants & Struct**: `VOCAB_TYPE_*`, `llama_vocab`, `llama_vocab_p`.
- **GGUF Metadata Loading (`llama_vocab.__init__`)**: Reads `tokenizer.ggml.model`, `tokens`, `scores`, `token_type`, `merges`, `add_bos_token`, `add_eos_token`, `add_space_prefix`, and special token IDs (`bos_id`, `eos_id`, `eot_id`, `pad_id`, `unk_id`, `mask_id`).
- **Dynamic Special Token Registration**: `add_special_token(mut self, piece: String, token_type: Int32 = 4) -> llama_token` for registering `<opt_0>`..`<opt_15>` and `<mask>`.
- **Special Token Pre-Splitter**: Splits input strings around registered special/control tokens when `parse_special=True` before dispatching raw text spans to the underlying engine.
- **Top-Level Dispatchers & Framing**: `tokenize_text`, `detokenize_text`, `token_to_piece_str`, and public `llama.h` functions (`llama_tokenize`, `llama_detokenize`, `llama_token_to_piece`, `llama_vocab_bos`, `llama_vocab_eos`, etc.).

### 3.2 `src/llamojo/vocab_gpt2.mojo` (GPT-2 Byte-Level BPE Engine)
Houses all GPT-2 / RoBERTa / English ModernBERT byte-level BPE encoding functions extracted from `vocab.mojo`:
- **`get_gpt2_byte_table() -> List[String]`**: Maps raw bytes (`0x00`..`0xFF`) to printable GPT-2 Unicode characters (`\u0100`..`\u0143`, `Ġ` for space, `Ċ` for newline).
- **`split_gpt2_chunks(text: String) -> List[String]`**: Pre-tokenizes text into contractions, words with leading spaces, digit groups, punctuation, and whitespace runs.
- **`tokenize_gpt2_bpe(vocab: llama_vocab_p, text: String) -> List[llama_token]`**: Encodes text spans by converting bytes via `get_gpt2_byte_table()` and iteratively merging pairs according to `vocab[].merge_ranks`.

### 3.3 `src/llamojo/vocab_gemma.mojo` (mmBERT / Gemma 2 / SentencePiece BPE Engine)
Houses the SentencePiece and Gemma-family (Gemma 2, Gemma 4, mmBERT, LLaMA SPM) BPE encoding functions:
- **`_SpmSymbol` & `_SpmBigram`**: Doubly-linked UTF-8 codepoint list and priority queue items for fast $O(N \log N)$ bigram merging on long contexts (up to 8,192 tokens).
- **`normalize_spm_spaces(text: String, add_space_prefix: Bool) -> String`**: Converts ASCII `' '` (`0x20`) to SentencePiece `\u2581` (`0xE2 0x96 0x81`), conditionally prepending `\u2581` only for classic LLaMA SPM (`add_space_prefix == True`) and omitting the dummy prefix for Gemma 2 / Gemma 4 / mmBERT (`add_space_prefix == False`).
- **`encode_hex_byte_fallback(vocab: llama_vocab_p, sym_text: String, mut tokens: List[llama_token])`**: Shared `<0xXX>` hexadecimal byte fallback for unmapped UTF-8 bytes.
- **`tokenize_gemma_bpe(vocab: llama_vocab_p, text: String) -> List[llama_token]`**: Tokenizes mmBERT / Gemma 2 / Gemma 4 text spans using `\u2581` space normalization (no dummy prefix), linked-list merge-rank / score priority merging, and `<0xXX>` byte fallback.
- **`tokenize_spm(vocab: llama_vocab_p, text: String, add_special: Bool, parse_special: Bool) -> List[llama_token]`**: Score-based SentencePiece bigram tokenizer supporting both classic LLaMA (`add_space_prefix=True`) and score-based Gemma 2 / mmBERT GGUF files (`add_space_prefix=False`).

---

## 4. Gap Analysis for Full mmBERT & Laya Support in `llamojo`

While existing code is very close, four specific gaps must be addressed during the modular split to make `llamojo` the universal tokenizer for mmBERT and Laya:

### Gap 1: Tokenizer Model-Type Routing & Leading Space Rules (Resolved in Phase 2 ✅)
* **Previous Behavior**:
  - In `llama_vocab.__init__`, if `tokenizer.ggml.model` was `"llama"`, it routed to `tokenize_spm`, which unconditionally prepended a leading `▁`.
  - Conversely, `tokenize_bpe` only activated the Gemma `\u2581` path if `model_type == "gemma4"`.
* **Implemented Solution**:
  - In [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo), `llama_vocab` now reads GGUF metadata `tokenizer.ggml.add_space_prefix` (defaulting to `False` for `"gemma"`, `"gemma2"`, `"gemma4"`, `"mmbert"`, `"modernbert"`).
  - Routes GPT-2 models (`model_type == "gpt2"`) to `vocab_gpt2.tokenize_gpt2_bpe` and Gemma / mmBERT / SPM models to `vocab_gemma.tokenize_gemma_bpe` / `tokenize_spm`.
  - Converts ASCII spaces to `\u2581` via `normalize_spm_spaces` **without** injecting a leading `\u2581` at index 0 when `add_space_prefix == False`.

### Gap 2: Special & Atomic Option Token Splitting (`parse_special`) (Resolved in Phase 3 ✅)
* **Previous Behavior**:
  - `tokenize_text` accepted `parse_special: Bool = False`, but neither `tokenize_bpe` nor `tokenize_spm` inspected `parse_special` to split out atomic special strings before BPE character decomposition.
  - If a Laya prompt contained `"Options: <opt_0> Refund <opt_1> Escalate"`, the BPE engine split `<opt_0>` into individual characters (`<`, `o`, `p`, `t`, `_`, `0`, `>`) instead of matching the single atomic token ID for `<opt_0>`.
* **Implemented Solution**:
  - Implemented a longest-match pre-tokenization pass in [`tokenize_text`](../src/llamojo/vocab.mojo) when `parse_special == True` (using tokens marked as `TOKEN_TYPE_CONTROL` / `TOKEN_TYPE_USER_DEFINED` in `tokenizer.ggml.token_type` or registered via `add_special_token`).
  - Scans the input text for registered special tokens (`<bos>`, `<eos>`, `<mask>`, `<opt_0>` .. `<opt_15>`) and partitions the string into `[RawTextSegment, SpecialTokenID, RawTextSegment, ...]`, dispatching only the `RawTextSegment` spans to `vocab_gemma.mojo` or `vocab_gpt2.mojo`.

### Gap 3: Encoder Framing (`add_eos` / `[SEP]` Appending) (Resolved in Phase 3 ✅)
* **Previous Behavior**:
  - Causal decoders (LLaMA, Gemma) only prepended `bos_id` when `add_special == True`.
  - Bidirectional encoders (ModernBERT, mmBERT) require **both** a leading `<bos>` (`2`) and a trailing `<eos>` (`1`) when `add_special == True`.
* **Implemented Solution**:
  - In [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo), `llama_vocab` reads `tokenizer.ggml.add_eos_token` (defaulting `add_eos = True` when `general.architecture` or `model_type` is `"modernbert"`, `"mmbert"`, or `"bert"`).
  - Appends `vocab[].eos_id` at the end of the token list in `tokenize_text` when `add_special == True` and `vocab[].add_eos == True`.

### Gap 4: Byte Fallback & Linked-List Merging in the Gemma BPE Path (Resolved in Phase 2 ✅)
* **Previous Behavior**:
  - `tokenize_spm` had `<0xXX>` byte fallback, but the Gemma `merge_ranks` loop simply dropped tokens if `vocab[].find_token(u_chars[p_i]) == -1` and used an $O(N^2)$ list reallocation loop.
* **Implemented Solution**:
  - In [`src/llamojo/vocab_gemma.mojo`](../src/llamojo/vocab_gemma.mojo), unified the Gemma merge-rank and SPM score-based BPE engines around `_SpmSymbol` doubly-linked lists (`_merge_spm_symbols`) and `encode_hex_byte_fallback` so rare emojis or multi-byte UTF-8 characters in multilingual inputs decompose into `<0xXX>` tokens rather than being skipped.

---

## 5. Proposed Implementation Architecture in `llamojo`

### 5.1 Module Organization & Data Flow
```
src/llamojo/
├── vocab.mojo         # llama_vocab struct, GGUF loader, special token splitter,
│                      # BOS/EOS framing, detokenizer, and public llama.h API
├── vocab_gpt2.mojo    # GPT-2 byte table, regex chunk splitter, tokenize_gpt2_bpe()
└── vocab_gemma.mojo   # mmBERT / Gemma 2 / Gemma 4 / SPM BPE engine (_SpmSymbol,
                       # \u2581 normalization, rank/score priority queue, <0xXX> fallback)
```

```
Input String: "State: Cliente solicita reembolso. Options: <opt_0> Aprobar <opt_1> Escalar"
      │
      ▼
1. Special Token Partitioner in `vocab.mojo` (parse_special=True)
   Splits on registered control/user tokens (<opt_0>, <opt_1>, <mask>, etc.)
      │
      ├──► Span 0 (Text): "State: Cliente solicita reembolso. Options: "
      ├──► Token: <opt_0> (ID: 256000)
      ├──► Span 1 (Text): " Aprobar "
      ├──► Token: <opt_1> (ID: 256001)
      └──► Span 2 (Text): " Escalar"
      │
      ▼
2. Engine Dispatch (`vocab.mojo` ──► `vocab_gemma.mojo` or `vocab_gpt2.mojo`)
   For mmBERT / Gemma 2 (`vocab_gemma.mojo`):
   - Replace ' ' -> '▁' (\u2581) (add_space_prefix=False -> no dummy prefix)
   - Merge bigrams via _SpmSymbol linked-list / priority queue
   - Byte fallback (<0xXX>) for any unmapped bytes
      │
      ▼
3. Sequence Framing in `vocab.mojo` (add_special=True)
   - Prepend <bos> (ID: 2, if add_bos=True)
   - Append <eos> (ID: 1, if add_eos=True)
      │
      ▼
Output: List[llama_token] + Option Marker Positions
```

### 5.2 Dynamic Registration of Laya Option Tokens (`vocab.mojo`)
When loading a base mmBERT GGUF model or preparing a dataset in Mojo, `llama_vocab` in `vocab.mojo` exposes a helper method:
```mojo
def add_special_token(mut self, piece: String, token_type: Int32 = 4) -> llama_token:
    var existing = self.find_token(piece)
    if existing != -1:
        if Int(existing) < len(self.token_types):
            self.token_types[Int(existing)] = token_type
        self.special_tokens.append(piece)
        return existing
    var new_id = Int32(self.vocab_size)
    self.tokens.append(piece)
    self.scores.append(0.0)
    self.token_types.append(token_type) # 4 = USER_DEFINED / SPECIAL
    self.token_to_id[piece] = new_id
    self.special_tokens.append(piece)
    self.vocab_size += 1
    return new_id
```
This mirrors Hugging Face's `tokenizer.add_special_tokens({"additional_special_tokens": ["<opt_0>", ...]})` in pure Mojo.

---

## 6. Performance & Memory Considerations for 256k Vocab

1. **Initialization Time & Memory Footprint**:
   - `256,000` token strings + `token_to_id: Dict[String, Int32]` + `merge_ranks: Dict[String, Int]`.
   - Total RAM overhead in `llama_vocab`: **~18–25 MB** (negligible on modern CPU workstations).
   - Dictionary population time at model load: **~35–60 ms** in compiled Mojo.
2. **Tokenization Complexity**:
   - The naive $O(N^2)$ array-reslicing loop in `tokenize_bpe` is fast for short strings ($N < 100$ chars), taking $< 0.2\text{ ms}$.
   - For 8,192-token long-context documents, the doubly-linked symbol list (`_SpmSymbol`) in `vocab_gemma.mojo` is much more scalable ($O(N \log N)$) and avoids repeated `List[String]` allocations. Sharing the `_SpmSymbol` linked-list engine in `vocab_gemma.mojo` for both score-based (`tokenizer.ggml.scores`) and rank-based (`tokenizer.ggml.merges`) BPE guarantees sub-millisecond tokenization even on long multilingual documents.

---

## 7. Phased Roadmap

| Phase | Target File(s) | Description | Status | Effort |
| :--- | :--- | :--- | :---: | :--- |
| **Phase 1: Extract GPT-2 BPE Module** | [`src/llamojo/vocab_gpt2.mojo`](../src/llamojo/vocab_gpt2.mojo), [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo) | Move `get_gpt2_byte_table`, `split_gpt2_chunks`, and GPT-2 byte-level BPE merge logic into `vocab_gpt2.mojo` (`tokenize_gpt2_bpe`). Keep `vocab.mojo` as the loader and router. | **Completed ✅** | ~150 LOC moved |
| **Phase 2: Extract & Enhance mmBERT / Gemma 2 BPE Module** | [`src/llamojo/vocab_gemma.mojo`](../src/llamojo/vocab_gemma.mojo), [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo) | Move SentencePiece and Gemma BPE encoding into `vocab_gemma.mojo` (`tokenize_gemma_bpe`, `tokenize_spm`). Support `add_space_prefix=False` (no leading dummy `▁`), unify `<0xXX>` byte fallback, and support both merge-ranks and SPM scores via `_SpmSymbol`. | **Completed ✅** | ~220 LOC |
| **Phase 3: Special Token Pre-Splitter & Encoder Framing** | [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo), [`src/llamojo/__init__.mojo`](../src/llamojo/__init__.mojo) | Add `add_eos` flag to `llama_vocab`; implement `add_special_token()` and `parse_special=True` pre-splitter in `vocab.mojo` to isolate `<opt_k>`, `<mask>`, `<bos>`, `<eos>` before dispatching spans to `vocab_gemma` or `vocab_gpt2`. Re-export symbols in `__init__.mojo`. | **Completed ✅** | ~110 LOC |
| **Phase 4: Parity Verification Suite & C-API Integration** | [`tests/test_vocab_sampler.mojo`](../tests/test_vocab_sampler.mojo), [`src/llamojo/c_api.mojo`](../src/llamojo/c_api.mojo) | Verify 100% token-ID parity between `llamojo` native tokenization and mmBERT/Gemma 2 conventions on English, Spanish (`es-ES`), UTF-8 multi-byte characters, and Laya `<opt_k>` prompts. Expose special token registration in `c_api.mojo`. | **Completed ✅** | ~100 LOC |

---

## 8. Conclusion

Refactoring `llamojo`'s vocabulary subsystem into:
- [`src/llamojo/vocab.mojo`](../src/llamojo/vocab.mojo) (GGUF vocabulary loading, special-token partitioning, sequence framing, and public API),
- [`src/llamojo/vocab_gpt2.mojo`](../src/llamojo/vocab_gpt2.mojo) (GPT-2 byte-level BPE encoding), and
- [`src/llamojo/vocab_gemma.mojo`](../src/llamojo/vocab_gemma.mojo) (mmBERT / Gemma 2 / Gemma 4 / SentencePiece BPE encoding)

provides a clean, modular, and maintainable architecture. Because `llamojo` already implements the core SentencePiece linked-list queue and UTF-8 handling, completing full native support for mmBERT and Laya requires only **~150–200 net new lines of Mojo code** across the modularized `vocab*.mojo` files.
