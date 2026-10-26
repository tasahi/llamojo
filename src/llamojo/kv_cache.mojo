""" KV Cache Management for llamojo: Unified Flat Cache & PagedAttention PageTable.

Ports and extends:
- MojoLlama/src/mojollama/kernels/cengine_batch_instr.c (PageTable, KVBlock)
- MojoLlama/src/mojollama/kernels/engine_v3.mojo (KVCache, KVCacheConfig)
- llama.cpp llama_kv_cache_unified / llama_kv_cache_clear / llama_kv_cache_seq_*

Features:
- Pure Mojo 1.1.0 with zero external dependencies
- Flat contiguous unified KV cache supporting F32 and Q8_0 quant
- Sequence ID and position cell metadata for prefix caching, sliding windows, and multi-sequence routing
- PagedAttention PageTable for non-contiguous block allocation
- Exact llama.h API: llama_kv_cache_clear, llama_kv_cache_seq_rm, seq_cp, seq_keep, seq_add
"""

from std.memory import Pointer, unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc

from .types import (
    llama_pos,
    llama_seq_id,
    GGMLType,
    ggml_type_size,
    ggml_block_size,
    null_ptr,
)
from .backend.isa import SIMD_WIDTH, f16_to_f32, load_f16_scale
from .backend.kernels_quant import quantize_row_q8_0

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime Int32Ptr = Pointer[Int32, MutUntrackedOrigin]

# ─── PagedAttention PageTable ────────────────────────────────────────────────

struct PageTable(Movable):
    """ PagedAttention PageTable ported from cengine_batch_instr.c.\n
    Translates logical page indices to physical memory blocks to prevent memory
    fragmentation during variable-length batch generation.
    """
    var page_size: Int          # Number of tokens/slots per physical page (e.g. 16 or 64)
    var max_pages: Int          # Total allocated physical pages
    var n_layers: Int
    var n_kv_heads: Int
    var head_dim: Int
    var pages_k: FloatPtr       # [n_layers, max_pages, page_size, n_kv_heads * head_dim]
    var pages_v: FloatPtr       # [n_layers, max_pages, page_size, n_kv_heads * head_dim]
    var logical_map: Int32Ptr   # Array of size max_pages mapping logical -> physical ID
    var free_pages: Int32Ptr    # Free physical page stack
    var free_count: Int
    var allocated: Bool

    def __init__(out self, max_pages: Int, page_size: Int, n_layers: Int, n_kv_heads: Int, head_dim: Int):
        self.max_pages = max(1, max_pages)
        self.page_size = max(1, page_size)
        self.n_layers = max(1, n_layers)
        self.n_kv_heads = max(1, n_kv_heads)
        self.head_dim = max(1, head_dim)
        self.free_count = self.max_pages

        var page_elements = self.page_size * self.n_kv_heads * self.head_dim
        var total_elements = self.n_layers * self.max_pages * page_elements

        self.pages_k = unsafe_alloc[Float32](total_elements)
        self.pages_v = unsafe_alloc[Float32](total_elements)
        unsafe_memset_zero(self.pages_k, total_elements)
        unsafe_memset_zero(self.pages_v, total_elements)

        self.logical_map = unsafe_alloc[Int32](self.max_pages)
        for i in range(self.max_pages):
            self.logical_map[unsafe_offset=i] = -1

        self.free_pages = unsafe_alloc[Int32](self.max_pages)
        for i in range(self.max_pages):
            self.free_pages[unsafe_offset=i] = Int32(self.max_pages - 1 - i)

        self.allocated = True

    def alloc_page(mut self) -> Int:
        """ Pop a free physical page from the pool, or return -1 if exhausted."""
        if self.free_count <= 0:
            return -1
        self.free_count -= 1
        return Int(self.free_pages[unsafe_offset=self.free_count])

    def free_page(mut self, page_id: Int):
        """ Return a physical page back to the free pool."""
        if self.free_count < self.max_pages and page_id >= 0:
            self.free_pages[unsafe_offset=self.free_count] = Int32(page_id)
            self.free_count += 1

    def map_logical(mut self, logical_page: Int, physical_page: Int):
        """ Map logical page index to physical page ID."""
        if logical_page >= 0 and logical_page < self.max_pages:
            self.logical_map[unsafe_offset=logical_page] = Int32(physical_page)

    def deinit_storage(mut self):
        if self.allocated:
            self.pages_k.unsafe_free()
            self.pages_v.unsafe_free()
            self.logical_map.unsafe_free()
            self.free_pages.unsafe_free()
            self.allocated = False

# ─── Cell Metadata (Matching llama.cpp llama_kv_cache_unified) ───────────────

@fieldwise_init
struct LlamaKVCell(Copyable, Movable):
    var pos: llama_pos       # Token sequence position
    var seq_id: llama_seq_id # Sequence ID (e.g. 0 for primary seq, -1 if free)

    def __init__(out self):
        self.pos = -1
        self.seq_id = -1

# ─── Unified KV Cache (F32 and Q8_0) ─────────────────────────────────────────

struct LlamaKVCache(Movable):
    """Unified KV Cache supporting both F32 and quantized Q8_0 buffers.
    
    Layout: [n_layers, max_ctx, n_kv_heads * head_dim]
    """
    var max_ctx: Int
    var n_layers: Int
    var n_kv_heads: Int
    var head_dim: Int
    var kv_dim: Int
    var type_k: UInt32
    var type_v: UInt32
    var k_f32: FloatPtr
    var v_f32: FloatPtr
    var k_q8: BytePtr
    var v_q8: BytePtr
    var cells: Pointer[LlamaKVCell, MutUntrackedOrigin]
    var page_table: PageTable
    var has_paged: Bool
    var allocated: Bool

    # Gemma 4 hybrid & shared KV parameters
    var n_layer_kv: Int         # Number of layers that physically store KV (e.g. 15 for Gemma 4)
    var shared_kv_layers: Int   # Number of shared layers (e.g. 20 for Gemma 4)

    def __init__(
            out self,
            max_ctx: Int,
            n_layers: Int,
            n_kv_heads: Int,
            head_dim: Int,
            type_k: UInt32 = GGMLType.F16,
            type_v: UInt32 = GGMLType.F16,
            enable_paged: Bool = False,
            shared_kv_layers: Int = 0,
        ):
        self.max_ctx = max(1, max_ctx)
        self.n_layers = max(1, n_layers)
        self.n_kv_heads = max(1, n_kv_heads)
        self.head_dim = max(1, head_dim)
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.type_k = type_k
        self.type_v = type_v
        self.has_paged = enable_paged
        self.shared_kv_layers = shared_kv_layers
        self.n_layer_kv = self.n_layers - shared_kv_layers if (shared_kv_layers > 0 and shared_kv_layers < self.n_layers) else self.n_layers

        var total_tokens = self.max_ctx
        var total_layer_tokens = self.n_layer_kv * total_tokens

        # Allocate per-token metadata cells
        self.cells = unsafe_alloc[LlamaKVCell](self.max_ctx)
        for i in range(self.max_ctx):
            self.cells[unsafe_offset=i] = LlamaKVCell()

        # Allocate KV buffers based on quantization type (only for n_layer_kv active storage layers)
        var total_floats = total_layer_tokens * self.kv_dim
        self.k_f32 = unsafe_alloc[Float32](total_floats)
        self.v_f32 = unsafe_alloc[Float32](total_floats)
        unsafe_memset_zero(self.k_f32, total_floats)
        unsafe_memset_zero(self.v_f32, total_floats)

        if type_k == GGMLType.Q8_0 or type_v == GGMLType.Q8_0:
            var bytes_per_token = (self.kv_dim // 32) * 34
            var total_bytes = total_layer_tokens * bytes_per_token
            self.k_q8 = unsafe_alloc[UInt8](total_bytes)
            self.v_q8 = unsafe_alloc[UInt8](total_bytes)
            unsafe_memset_zero(self.k_q8, total_bytes)
            unsafe_memset_zero(self.v_q8, total_bytes)
        else:
            self.k_q8 = null_ptr[UInt8]()
            self.v_q8 = null_ptr[UInt8]()

        if enable_paged:
            var page_size = 16
            var max_pages = (self.max_ctx + page_size - 1) // page_size
            self.page_table = PageTable(max_pages, page_size, self.n_layer_kv, self.n_kv_heads, self.head_dim)
        else:
            self.page_table = PageTable(1, 16, 1, 1, 1)

        self.allocated = True

    @always_inline
    def get_reused_layer(self, layer: Int, is_swa: Bool) -> Int:
        """ Map a layer index to its backing physical KV storage layer.
        For Gemma 4: layers 0..14 store their own KV.
        Layers >= 15 reuse layer 13 (if SWA) or 14 (if global attention).
        """
        if self.shared_kv_layers <= 0 or layer < self.n_layer_kv:
            return layer
        return self.n_layer_kv - (2 if is_swa else 1)

    @always_inline
    def has_own_kv(self, layer: Int) -> Bool:
        """ Returns true if this layer computes and stores its own KV projection."""
        return self.shared_kv_layers <= 0 or layer < self.n_layer_kv

    @always_inline
    def get_k_ptr(self, layer: Int, slot: Int, is_swa: Bool = False) -> FloatPtr:
        """ Return pointer to K cache for (layer, slot), taking layer reuse into account."""
        var phys_layer = self.get_reused_layer(layer, is_swa)
        var offset = phys_layer * self.max_ctx * self.kv_dim + slot * self.kv_dim
        return self.k_f32.unsafe_offset(offset)

    @always_inline
    def get_v_ptr(self, layer: Int, slot: Int, is_swa: Bool = False) -> FloatPtr:
        """ Return pointer to V cache for (layer, slot), taking layer reuse into account."""
        var phys_layer = self.get_reused_layer(layer, is_swa)
        var offset = phys_layer * self.max_ctx * self.kv_dim + slot * self.kv_dim
        return self.v_f32.unsafe_offset(offset)

    def write_kv(self, layer: Int, slot: Int, k_in: FloatPtr, v_in: FloatPtr, pos: llama_pos, seq_id: llama_seq_id):
        """ Store key and value vectors into cache at specified slot and update cell metadata."""
        if slot < 0 or slot >= self.max_ctx:
            return

        if layer == 0:
            self.cells[unsafe_offset=slot].pos = pos
            self.cells[unsafe_offset=slot].seq_id = seq_id

        var k_dest = self.get_k_ptr(layer, slot)
        var v_dest = self.get_v_ptr(layer, slot)
        unsafe_memcpy(dest=k_dest, src=k_in, count=self.kv_dim)
        unsafe_memcpy(dest=v_dest, src=v_in, count=self.kv_dim)

        # If quantized KV cache, also quantize into Q8_0 buffer
        if self.type_k == GGMLType.Q8_0:
            var bytes_per_token = (self.kv_dim // 32) * 34
            var off_bytes = layer * self.max_ctx * bytes_per_token + slot * bytes_per_token
            quantize_row_q8_0(k_in, self.k_q8.unsafe_offset(off_bytes), self.kv_dim)
        if self.type_v == GGMLType.Q8_0:
            var bytes_per_token = (self.kv_dim // 32) * 34
            var off_bytes = layer * self.max_ctx * bytes_per_token + slot * bytes_per_token
            quantize_row_q8_0(v_in, self.v_q8.unsafe_offset(off_bytes), self.kv_dim)

    def clear(mut self):
        """Clear all slots in KV cache."""
        for i in range(self.max_ctx):
            self.cells[unsafe_offset=i].pos = -1
            self.cells[unsafe_offset=i].seq_id = -1

    def seq_rm(mut self, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) -> Bool:
        """Remove slots for sequence `seq_id` in range [p0, p1). If p0 < 0 or p1 < 0, removes all matching seq_id."""
        var count = 0
        for i in range(self.max_ctx):
            var cell_seq = self.cells[unsafe_offset=i].seq_id
            var cell_pos = self.cells[unsafe_offset=i].pos
            if seq_id < 0 or cell_seq == seq_id:
                if (p0 < 0 or p1 < 0) or (cell_pos >= p0 and cell_pos < p1):
                    self.cells[unsafe_offset=i].pos = -1
                    self.cells[unsafe_offset=i].seq_id = -1
                    count += 1
        return count > 0

    def seq_cp(mut self, seq_id_src: llama_seq_id, seq_id_dst: llama_seq_id, p0: llama_pos, p1: llama_pos):
        """Copy sequence slots from `seq_id_src` to `seq_id_dst` for prefix sharing."""
        for i in range(self.max_ctx):
            if self.cells[unsafe_offset=i].seq_id == seq_id_src:
                var pos = self.cells[unsafe_offset=i].pos
                if (p0 < 0 or p1 < 0) or (pos >= p0 and pos < p1):
                    # In unified cache, alias sequence or assign to next available slot
                    self.cells[unsafe_offset=i].seq_id = seq_id_dst

    def seq_keep(mut self, seq_id: llama_seq_id):
        """Keep only slots belonging to `seq_id`, removing all other sequences."""
        for i in range(self.max_ctx):
            if self.cells[unsafe_offset=i].seq_id != seq_id:
                self.cells[unsafe_offset=i].pos = -1
                self.cells[unsafe_offset=i].seq_id = -1

    def seq_add(mut self, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos, delta: llama_pos):
        """Shift positions in sequence `seq_id` by `delta` (for context shifting / RoPE rotation)."""
        for i in range(self.max_ctx):
            if self.cells[unsafe_offset=i].seq_id == seq_id:
                var pos = self.cells[unsafe_offset=i].pos
                if (p0 < 0 or p1 < 0) or (pos >= p0 and pos < p1):
                    self.cells[unsafe_offset=i].pos = pos + delta

    def deinit_storage(mut self):
        if self.allocated:
            self.k_f32.unsafe_free()
            self.v_f32.unsafe_free()
            if Int(self.k_q8) != 0:
                self.k_q8.unsafe_free()
            if Int(self.v_q8) != 0:
                self.v_q8.unsafe_free()
            self.cells.unsafe_free()
            if self.has_paged:
                self.page_table.deinit_storage()
            self.allocated = False
