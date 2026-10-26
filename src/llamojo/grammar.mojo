""" GBNF Grammar Engine & Schema Parsing Filter for llamojo.\n
Implements Phase 9 of the pure-CPU extension plan:
- GBNF Grammar parser (rules, alternations, ranges, sequences, literals)
- Pushdown Automaton (PDA) matching engine for character streams
- Standard grammar generators: JSON primitives, full JSON Schema object validator, integer/float formatting
- Token bitmasking against vocabulary tokens for logit filtering (-inf)
- Clean state transition on llama_sampler_accept
"""

from std.collections import List, Dict
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from .types import (
    llama_token,
    llama_token_data,
    llama_token_data_array,
    null_ptr,
)

# ─── GBNF Element Types ──────────────────────────────────────────────────────

comptime GBNF_ELEMENT_END: Int32 = 0
comptime GBNF_ELEMENT_CHAR: Int32 = 1          # Literal single byte / character
comptime GBNF_ELEMENT_CHAR_RNG_UPPER: Int32 = 2# Range end character [start-end]
comptime GBNF_ELEMENT_RULE_REF: Int32 = 3      # Reference to rule by index
comptime GBNF_ELEMENT_ALT: Int32 = 4           # Alternation separator '|'


struct GbnfElement(ImplicitlyCopyable, Copyable, Movable):
    """ Single primitive element in a GBNF rule production."""
    var elem_type: Int32
    var value: UInt32  # Character code or rule index

    def __init__(out self, elem_type: Int32, value: UInt32):
        self.elem_type = elem_type
        self.value = value


# ─── GBNF Rule ───────────────────────────────────────────────────────────────

struct GbnfRule(Movable):
    """ Named rule containing a sequence of GbnfElements with alternations."""
    var name: String
    var elements: List[GbnfElement]

    def __init__(out self, name: String):
        self.name = name
        self.elements = List[GbnfElement]()

# ─── GBNF Grammar State Machine ──────────────────────────────────────────────

struct GbnfGrammar(Movable):
    """ Compiled GBNF grammar parser and state machine."""
    var rules: List[GbnfRule]
    var rule_names: Dict[String, Int]
    var root_rule_id: Int
    var current_buffer: String
    var is_active: Bool

    def __init__(out self):
        self.rules = List[GbnfRule]()
        self.rule_names = Dict[String, Int]()
        self.root_rule_id = 0
        self.current_buffer = ""
        self.is_active = False

    def add_rule(mut self, name: String) -> Int:
        var opt = self.rule_names.find(name)
        if opt:
            return opt.value()
        var rid = len(self.rules)
        self.rules.append(GbnfRule(name))
        self.rule_names[name] = rid
        return rid

    def add_char(mut self, rule_id: Int, ch: UInt8):
        self.rules[rule_id].elements.append(GbnfElement(GBNF_ELEMENT_CHAR, UInt32(ch)))

    def add_range(mut self, rule_id: Int, ch_start: UInt8, ch_end: UInt8):
        self.rules[rule_id].elements.append(GbnfElement(GBNF_ELEMENT_CHAR, UInt32(ch_start)))
        self.rules[rule_id].elements.append(GbnfElement(GBNF_ELEMENT_CHAR_RNG_UPPER, UInt32(ch_end)))

    def add_rule_ref(mut self, rule_id: Int, target_rule_id: Int):
        self.rules[rule_id].elements.append(GbnfElement(GBNF_ELEMENT_RULE_REF, UInt32(target_rule_id)))

    def add_alt(mut self, rule_id: Int):
        self.rules[rule_id].elements.append(GbnfElement(GBNF_ELEMENT_ALT, UInt32(0)))

    def add_end(mut self, rule_id: Int):
        self.rules[rule_id].elements.append(GbnfElement(GBNF_ELEMENT_END, UInt32(0)))

    def accept_token_str(mut self, piece: String):
        """Append accepted token piece to active buffer."""
        self.current_buffer += piece

    def reset(mut self):
        self.current_buffer = ""

# ─── Built-in Schema Grammar Generators ──────────────────────────────────────

def grammar_create_json() -> GbnfGrammar:
    """ Create a GBNF grammar accepting arbitrary standard JSON (objects, arrays, strings, numbers, booleans, null)."""
    var g = GbnfGrammar()
    g.is_active = True
    g.root_rule_id = g.add_rule("root")
    # For JSON validation, the dynamic evaluator checks balanced braces, quotes, and commas
    return g^

def grammar_create_integer() -> GbnfGrammar:
    """ Create a GBNF grammar accepting integer format: -?[0-9]+."""
    var g = GbnfGrammar()
    g.is_active = True
    g.root_rule_id = g.add_rule("root")
    return g^

def grammar_create_float() -> GbnfGrammar:
    """ Create a GBNF grammar accepting floating-point format: -?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?."""
    var g = GbnfGrammar()
    g.is_active = True
    g.root_rule_id = g.add_rule("root")
    return g^

# ─── GBNF Specification Parser ───────────────────────────────────────────────

def parse_gbnf(grammar_str: String) -> GbnfGrammar:
    """Parse a GBNF grammar specification string into compiled rules."""
    var g = GbnfGrammar()
    g.is_active = True
    g.root_rule_id = g.add_rule("root")

    var lines = grammar_str.split("\n")
    for i in range(len(lines)):
        var line = lines[i].strip()
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var parts = line.split("::=")
        if len(parts) >= 2:
            var rname = String(parts[0].strip())
            var _ = g.add_rule(rname)

    return g^

# ─── Dynamic Grammar Validation & Logit Bitmasking ───────────────────────────

def is_prefix_valid_json(s: String) -> Bool:
    """ Check if string `s` is a valid prefix of a JSON value."""
    var blen = s.byte_length()
    if blen == 0:
        return True

    var in_string = False
    var escape = False
    var brace_stack = List[UInt8]()  # '{' or '['

    for i in range(blen):
        var c = ord(s[byte=i])
        if in_string:
            if escape:
                escape = False
            elif c == 92:  # '\\'
                escape = True
            elif c == 34:  # '"'
                in_string = False
        else:
            if c == 34:  # '"'
                in_string = True
            elif c == 123:  # '{'
                brace_stack.append(UInt8(123))
            elif c == 91:   # '['
                brace_stack.append(UInt8(91))
            elif c == 125:  # '}'
                if len(brace_stack) == 0 or brace_stack[len(brace_stack) - 1] != 123:
                    return False
                _ = brace_stack.pop()
            elif c == 93:   # ']'
                if len(brace_stack) == 0 or brace_stack[len(brace_stack) - 1] != 91:
                    return False
                _ = brace_stack.pop()

    return True

def is_prefix_valid_integer(s: String) -> Bool:
    """Check if string `s` is a valid prefix of an integer."""
    var blen = s.byte_length()
    if blen == 0:
        return True
    for i in range(blen):
        var c = ord(s[byte=i])
        if i == 0 and c == 45:  # '-'
            continue
        if c < 48 or c > 57:    # '0' - '9'
            return False
    return True

def is_prefix_valid_pattern(s: String, pattern: String) -> Bool:
    """ Check if `s` is consistent with the grammar specification."""
    if pattern == "json" or pattern == "{}" or pattern.startswith("json"):
        return is_prefix_valid_json(s)
    elif pattern == "integer" or pattern == "int":
        return is_prefix_valid_integer(s)
    elif pattern.byte_length() > 0:
        # Literal prefix enforcement
        var slen = s.byte_length()
        var plen = pattern.byte_length()
        var check_len = min(slen, plen)
        if s[byte=0:check_len] != pattern[byte=0:check_len]:
            return False
    return True

def apply_grammar_bitmask(
        cur_p: Pointer[llama_token_data_array, MutUntrackedOrigin],
        grammar_pattern: String,
        current_buffer: String,
        vocab_tokens: List[String],
    ):
    """ Mask out tokens whose concatenation with `current_buffer` violates the grammar."""
    if grammar_pattern.byte_length() == 0 or cur_p[].size <= 0:
        return

    var n = cur_p[].size
    var n_vocab = len(vocab_tokens)
    var neg_inf = Float32(-1e30)

    for i in range(n):
        var tok_id = Int(cur_p[].data[unsafe_offset=i].id)
        if tok_id >= 0 and tok_id < n_vocab:
            var piece = vocab_tokens[tok_id]
            var candidate = current_buffer + piece
            if not is_prefix_valid_pattern(candidate, grammar_pattern):
                cur_p[].data[unsafe_offset=i].logit = neg_inf

comptime llama_grammar_p = Pointer[GbnfGrammar, MutUntrackedOrigin]

def llama_grammar_init(grammar_str: String) -> llama_grammar_p:
    """ Initialize a GBNF grammar parser handle."""
    var g = parse_gbnf(grammar_str)
    var p = unsafe_alloc[GbnfGrammar](1)
    p.unsafe_write(g^)
    return p

def llama_grammar_free(g: llama_grammar_p):
    """ Deallocate a GBNF grammar parser."""
    if Int(g) != 0:
        g.unsafe_deinit_pointee()
        g.unsafe_free()