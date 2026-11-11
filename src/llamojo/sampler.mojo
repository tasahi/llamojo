""" Composable Sampling Pipeline for llamojo.

Integrates:
- MojoLlama/sampling.mojo & kernels/sampling.mojo
- llama.h ABI & APIs:
  - llama_sampler (opaque handle)
  - llama_sampler_chain
  - llama_sampler_chain_init
  - llama_sampler_chain_add
  - llama_sampler_init_top_k
  - llama_sampler_init_top_p
  - llama_sampler_init_min_p
  - llama_sampler_init_temp
  - llama_sampler_init_penalties
  - llama_sampler_init_greedy
  - llama_sampler_init_dist
  - llama_sampler_init_grammar (GBNF parser/filter)
  - llama_sampler_sample
  - llama_sampler_accept
  - llama_sampler_free
"""

from std.collections import List
from std.math import exp, log, sqrt
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from .types import (
    llama_token,
    llama_token_data,
    llama_token_data_array,
    null_ptr,
)

# ─── Sampler Implementation Types ────────────────────────────────────────────

comptime SAMPLER_TYPE_CHAIN: Int32 = 0
comptime SAMPLER_TYPE_TOP_K: Int32 = 1
comptime SAMPLER_TYPE_TOP_P: Int32 = 2
comptime SAMPLER_TYPE_MIN_P: Int32 = 3
comptime SAMPLER_TYPE_TEMP: Int32 = 4
comptime SAMPLER_TYPE_PENALTIES: Int32 = 5
comptime SAMPLER_TYPE_GREEDY: Int32 = 6
comptime SAMPLER_TYPE_DIST: Int32 = 7
comptime SAMPLER_TYPE_GRAMMAR: Int32 = 8


@fieldwise_init
struct llama_sampler_params(ImplicitlyCopyable, Copyable, Movable):
    var no_perf: Bool

    def __init__(out self):
        self.no_perf = False

@always_inline
def llama_sampler_chain_default_params() -> llama_sampler_params:
    return llama_sampler_params()

struct llama_sampler(Movable):
    var sampler_type: Int32
    # Chain
    var chain_items: List[Pointer[llama_sampler, MutUntrackedOrigin]]
    # Top-K
    var k: Int32
    # Top-P / Min-P
    var p_val: Float32
    var min_keep: Int
    # Temperature
    var temp: Float32
    # Penalties
    var penalty_repeat: Float32
    var penalty_freq: Float32
    var penalty_present: Float32
    var n_prev: Int32
    var prev_tokens: List[llama_token]
    # Dist (LCG RNG state)
    var rng_state: UInt64
    # Grammar pattern (GBNF literal prefix / rule filter)
    var grammar_pattern: String
    var grammar_buffer: String
    var vocab_tokens: List[String]

    def __init__(out self, sampler_type: Int32):
        self.sampler_type = sampler_type
        self.chain_items = List[Pointer[llama_sampler, MutUntrackedOrigin]]()
        self.k = 40
        self.p_val = 0.95
        self.min_keep = 1
        self.temp = 0.8
        self.penalty_repeat = 1.0
        self.penalty_freq = 0.0
        self.penalty_present = 0.0
        self.n_prev = 64
        self.prev_tokens = List[llama_token]()
        self.rng_state = 1337
        self.grammar_pattern = ""
        self.grammar_buffer = ""
        self.vocab_tokens = List[String]()

comptime llama_sampler_p = Pointer[llama_sampler, MutUntrackedOrigin]

# ─── Low-level Candidate Sorting & Normalization ─────────────────────────────

def _softmax_token_data(cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    var n = cur_p[].size
    if n <= 0:
        return
    var data = cur_p[].data

    # 1. Max logit
    var max_val = data[unsafe_offset=0].logit
    for i in range(1, n):
        if data[unsafe_offset=i].logit > max_val:
            max_val = data[unsafe_offset=i].logit

    # 2. Exp and sum
    var sum_val = Float32(0.0)
    for i in range(n):
        var p = exp(data[unsafe_offset=i].logit - max_val)
        data[unsafe_offset=i].p = p
        sum_val += p

    # 3. Normalize
    var inv_sum = Float32(1.0) / (sum_val if sum_val > Float32(0.0) else Float32(1.0))
    for i in range(n):
        data[unsafe_offset=i].p = data[unsafe_offset=i].p * inv_sum

def _sort_token_data_descending(cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin], top_k: Int = -1):
    """Sort candidates by logit/prob descending (in-place partial or full selection sort)."""
    var n = cur_p[].size
    var data = cur_p[].data
    var max_i = n if top_k <= 0 or top_k > n else top_k
    for i in range(max_i):
        var best_idx = i
        var best_val = data[unsafe_offset=i].logit
        for j in range(i + 1, n):
            if data[unsafe_offset=j].logit > best_val:
                best_val = data[unsafe_offset=j].logit
                best_idx = j
        if best_idx != i:
            var tmp = data[unsafe_offset=i]
            data[unsafe_offset=i] = data[unsafe_offset=best_idx]
            data[unsafe_offset=best_idx] = tmp
    cur_p[].sorted = True

# ─── Sampler Stage Operations ────────────────────────────────────────────────

def _apply_penalties(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    var r_pen = smpl[].penalty_repeat
    var f_pen = smpl[].penalty_freq
    var p_pen = smpl[].penalty_present
    var n_tokens = cur_p[].size
    var n_prev = len(smpl[].prev_tokens)
    if n_prev == 0 or (r_pen == Float32(1.0) and f_pen == Float32(0.0) and p_pen == Float32(0.0)):
        return

    # Count occurrences of tokens in history window
    var window_start = 0 if n_prev <= Int(smpl[].n_prev) else n_prev - Int(smpl[].n_prev)
    var data = cur_p[].data

    for i in range(n_tokens):
        var tok_id = data[unsafe_offset=i].id
        var count = 0
        for p in range(window_start, n_prev):
            if smpl[].prev_tokens[p] == tok_id:
                count += 1
        if count > 0:
            var logit = data[unsafe_offset=i].logit
            # Repeat penalty
            if r_pen != Float32(1.0):
                if logit > Float32(0.0):
                    logit = logit / r_pen
                else:
                    logit = logit * r_pen
            # Frequency & presence penalties
            logit = logit - (Float32(count) * f_pen + p_pen)
            data[unsafe_offset=i].logit = logit

def _apply_temperature(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    var t = smpl[].temp
    if t <= Float32(1e-6):
        return
    var inv_t = Float32(1.0) / t
    var n = cur_p[].size
    var data = cur_p[].data
    for i in range(n):
        data[unsafe_offset=i].logit = data[unsafe_offset=i].logit * inv_t

def _apply_top_k(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    var k = Int(smpl[].k)
    var n = cur_p[].size
    if k <= 0 or k >= n:
        return
    _sort_token_data_descending(cur_p, top_k=k)
    cur_p[].size = k

def _apply_top_p(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    var p = smpl[].p_val
    var n = cur_p[].size
    if p >= Float32(1.0) or n <= smpl[].min_keep:
        return

    _sort_token_data_descending(cur_p)
    _softmax_token_data(cur_p)

    var cumsum = Float32(0.0)
    var cutoff = n
    var data = cur_p[].data
    for i in range(n):
        cumsum += data[unsafe_offset=i].p
        if cumsum >= p and i + 1 >= smpl[].min_keep:
            cutoff = i + 1
            break
    cur_p[].size = cutoff

def _apply_min_p(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    var min_p = smpl[].p_val
    var n = cur_p[].size
    if min_p <= Float32(0.0) or n <= smpl[].min_keep:
        return

    _sort_token_data_descending(cur_p)
    _softmax_token_data(cur_p)

    var data = cur_p[].data
    var max_p = data[unsafe_offset=0].p
    var thresh = max_p * min_p
    var count = 0
    for i in range(n):
        if data[unsafe_offset=i].p >= thresh:
            count += 1
        else:
            break
    if count < smpl[].min_keep:
        count = smpl[].min_keep
    if count > n:
        count = n
    cur_p[].size = count

def _apply_grammar(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    """ GBNF Grammar filtering: when a grammar rule/pattern is active,
    filter out candidate tokens that cannot legally extend the sequence.
    """
    if smpl[].grammar_pattern.byte_length() == 0 or cur_p[].size <= 0:
        return
    if len(smpl[].vocab_tokens) == 0:
        return

    from .grammar import apply_grammar_bitmask
    apply_grammar_bitmask(
        cur_p,
        smpl[].grammar_pattern,
        smpl[].grammar_buffer,
        smpl[].vocab_tokens,
    )

# ─── Public llama.h Sampler API ──────────────────────────────────────────────

def llama_sampler_chain_init(params: llama_sampler_params) -> llama_sampler_p:
    """ Create a new root sampler chain container."""
    var p = unsafe_alloc[llama_sampler](1)
    p.unsafe_write(llama_sampler(SAMPLER_TYPE_CHAIN))
    return p

def llama_sampler_chain_add(chain: llama_sampler_p, smpl: llama_sampler_p):
    """ Append a sub-sampler stage to the chain."""
    if Int(chain) == 0 or Int(smpl) == 0:
        return
    chain[].chain_items.append(smpl)

def llama_sampler_init_top_k(k: Int32) -> llama_sampler_p:
    """ Initialize Top-K sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_TOP_K)
    s.k = k
    p.unsafe_write(s^)
    return p

def llama_sampler_init_top_p(p_val: Float32, min_keep: Int = 1) -> llama_sampler_p:
    """Initialize Nucleus Top-P sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_TOP_P)
    s.p_val = p_val
    s.min_keep = min_keep
    p.unsafe_write(s^)
    return p

def llama_sampler_init_min_p(p_val: Float32, min_keep: Int = 1) -> llama_sampler_p:
    """Initialize Min-P sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_MIN_P)
    s.p_val = p_val
    s.min_keep = min_keep
    p.unsafe_write(s^)
    return p

def llama_sampler_init_temp(t: Float32) -> llama_sampler_p:
    """ Initialize Temperature scaling sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_TEMP)
    s.temp = t
    p.unsafe_write(s^)
    return p

def llama_sampler_init_penalties(
        n_prev: Int32,
        repeat_penalty: Float32,
        freq_penalty: Float32,
        presence_penalty: Float32,
    ) -> llama_sampler_p:
    """ Initialize Repetition, Frequency, and Presence penalties sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_PENALTIES)
    s.n_prev = n_prev
    s.penalty_repeat = repeat_penalty
    s.penalty_freq = freq_penalty
    s.penalty_present = presence_penalty
    p.unsafe_write(s^)
    return p

def llama_sampler_init_greedy() -> llama_sampler_p:
    """ Initialize Greedy argmax sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    p.unsafe_write(llama_sampler(SAMPLER_TYPE_GREEDY))
    return p

def llama_sampler_init_dist(seed: UInt32) -> llama_sampler_p:
    """ Initialize Categorical distribution random sampler with PRNG seed."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_DIST)
    s.rng_state = UInt64(seed) if seed != 0 else 42
    p.unsafe_write(s^)
    return p

def llama_sampler_init_grammar(
        grammar_str: String,
        vocab_tokens: List[String] = List[String](),
    ) -> llama_sampler_p:
    """ Initialize GBNF grammar parser filter sampler."""
    var p = unsafe_alloc[llama_sampler](1)
    var s = llama_sampler(SAMPLER_TYPE_GRAMMAR)
    s.grammar_pattern = grammar_str
    s.vocab_tokens = vocab_tokens.copy()
    p.unsafe_write(s^)
    return p

def llama_sampler_accept(smpl: llama_sampler_p, token: llama_token):
    """Inform sampler of selected token to update repetition history and grammar state."""
    if Int(smpl) == 0:
        return
    if smpl[].sampler_type == SAMPLER_TYPE_CHAIN:
        for i in range(len(smpl[].chain_items)):
            llama_sampler_accept(smpl[].chain_items[i], token)
    elif smpl[].sampler_type == SAMPLER_TYPE_PENALTIES:
        smpl[].prev_tokens.append(token)
    elif smpl[].sampler_type == SAMPLER_TYPE_GRAMMAR:
        var tok_id = Int(token)
        if tok_id >= 0 and tok_id < len(smpl[].vocab_tokens):
            smpl[].grammar_buffer += smpl[].vocab_tokens[tok_id]

def llama_sampler_apply(smpl: llama_sampler_p, cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin]):
    """Apply transformation of sampler stage to candidate array."""
    if Int(smpl) == 0 or Int(cur_p) == 0 or cur_p[].size <= 0:
        return

    var t = smpl[].sampler_type
    if t == SAMPLER_TYPE_CHAIN:
        for i in range(len(smpl[].chain_items)):
            llama_sampler_apply(smpl[].chain_items[i], cur_p)
    elif t == SAMPLER_TYPE_PENALTIES:
        _apply_penalties(smpl, cur_p)
    elif t == SAMPLER_TYPE_TEMP:
        _apply_temperature(smpl, cur_p)
    elif t == SAMPLER_TYPE_TOP_K:
        _apply_top_k(smpl, cur_p)
    elif t == SAMPLER_TYPE_TOP_P:
        _apply_top_p(smpl, cur_p)
    elif t == SAMPLER_TYPE_MIN_P:
        _apply_min_p(smpl, cur_p)
    elif t == SAMPLER_TYPE_GRAMMAR:
        _apply_grammar(smpl, cur_p)

def llama_sampler_sample(
    smpl: llama_sampler_p,
    cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin],
) -> llama_token:
    """Execute the full sampler chain and select a final token ID."""
    if Int(smpl) == 0 or Int(cur_p) == 0 or cur_p[].size <= 0:
        return -1

    # First pass: filter & reshape candidate logits
    llama_sampler_apply(smpl, cur_p)

    var n = cur_p[].size
    if n == 1:
        var tok = cur_p[].data[unsafe_offset=0].id
        llama_sampler_accept(smpl, tok)
        return tok

    # Find effective terminating selection policy
    var is_greedy = False
    var dist_sampler: llama_sampler_p = null_ptr[llama_sampler]()

    if smpl[].sampler_type == SAMPLER_TYPE_CHAIN:
        for i in range(len(smpl[].chain_items)):
            var item = smpl[].chain_items[i]
            if item[].sampler_type == SAMPLER_TYPE_GREEDY:
                is_greedy = True
            elif item[].sampler_type == SAMPLER_TYPE_DIST:
                dist_sampler = item
    elif smpl[].sampler_type == SAMPLER_TYPE_GREEDY:
        is_greedy = True
    elif smpl[].sampler_type == SAMPLER_TYPE_DIST:
        dist_sampler = smpl

    # Greedy selection (highest logit)
    if is_greedy or Int(dist_sampler) == 0:
        var best_id = cur_p[].data[unsafe_offset=0].id
        var best_logit = cur_p[].data[unsafe_offset=0].logit
        for i in range(1, n):
            if cur_p[].data[unsafe_offset=i].logit > best_logit:
                best_logit = cur_p[].data[unsafe_offset=i].logit
                best_id = cur_p[].data[unsafe_offset=i].id
        llama_sampler_accept(smpl, best_id)
        return best_id

    # Categorical distribution sampling (LCG PRNG over softmax probabilities)
    _softmax_token_data(cur_p)

    var state = dist_sampler[].rng_state
    state = state * 6364136223846793005 + 1442695040888963407
    dist_sampler[].rng_state = state

    var rand_val = Float32((state >> 11) & 0x1FFFFF) / Float32(0x1FFFFF)
    var cumsum = Float32(0.0)
    var selected_tok = cur_p[].data[unsafe_offset=n - 1].id

    for i in range(n):
        cumsum += cur_p[].data[unsafe_offset=i].p
        if cumsum >= rand_val:
            selected_tok = cur_p[].data[unsafe_offset=i].id
            break

    llama_sampler_accept(smpl, selected_tok)
    return selected_tok

def llama_sampler_free(smpl: llama_sampler_p):
    """Deallocate sampler and all nested stages in a chain."""
    if Int(smpl) == 0:
        return
    if smpl[].sampler_type == SAMPLER_TYPE_CHAIN:
        for i in range(len(smpl[].chain_items)):
            llama_sampler_free(smpl[].chain_items[i])
    smpl.unsafe_deinit_pointee()
    smpl.unsafe_free()
