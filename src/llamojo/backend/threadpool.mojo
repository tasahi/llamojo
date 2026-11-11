""" Persistent Spin-Barrier Thread Pool & Matrix Tensor Wrapper for llamojo.

Integrates `/mnt/c/Documents/Mojo/llama2.mojo` (`vendor/llama2.mojo`) as a direct
sublibrary, importing `Matrix`, `BufferPtrFloat32`, `nelts`, `chunk_start`,
`chunk_end`, `spin_wait`, `ForwardArgs`, `PoolShared`, `PoolSharedPtr`, and
`barrier`, while extending the sense-reversing `Atomic[Int]` spin-barrier pool
into `LlamaPoolShared` and `LlamaWorkerPool` for arbitrary GGUF models.
"""

from std.atomic import Atomic
from std.memory import Pointer, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys.info import num_performance_cores, num_logical_cores
from max.algorithm import parallelize

# Direct sublibrary import from /mnt/c/Documents/Mojo/llama2.mojo
from llama2 import (
    Matrix,
    BufferPtrFloat32,
    nelts,
    chunk_start,
    chunk_end,
    spin_wait,
    ForwardArgs,
    PoolShared,
    PoolSharedPtr,
    barrier,
)

@always_inline
def free_matrix(mut m: Matrix):
    """Explicitly free heap memory owned by a `llama2.Matrix`."""
    if m.allocated == 1:
        m.data.unsafe_free()
        m.allocated = 0

# ─── Model-Agnostic Sense-Reversing Spin-Barrier Pool for GGUF Models ────────

struct LlamaPoolShared:
    """Shared synchronization and scratchpad state across worker threads.

    Generalizes `llama2.PoolShared` so it can be initialized directly from runtime
    `workers` and `dim` without requiring a `llama2.c` `ForwardArgs` struct.
    """
    var workers: Int
    var bar_count: Atomic[Int]       # Spin barrier: arrival counter
    var bar_gen: Atomic[Int]         # Spin barrier: sense-reversing generation
    var xbp: BufferPtrFloat32        # Per-worker private scratch buffer (workers * xbp_dim)
    var xbp_dim: Int
    var q8_act: Pointer[UInt8, MutUntrackedOrigin] # Shared Q8_K quantized activation buffer
    var q8_dim: Int

    def __init__(out self, workers: Int, dim: Int):
        self.workers = max(1, workers)
        self.bar_count = Atomic[Int](0)
        self.bar_gen = Atomic[Int](0)
        self.xbp_dim = max(1, dim)
        self.xbp = unsafe_alloc[Float32](self.workers * self.xbp_dim)
        unsafe_memset_zero(self.xbp, self.workers * self.xbp_dim)
        var max_cols = max(16384, dim)
        self.q8_dim = (max_cols // 256) * 292
        self.q8_act = unsafe_alloc[UInt8](self.q8_dim)
        unsafe_memset_zero(self.q8_act, self.q8_dim)

comptime LlamaPoolSharedPtr = Pointer[LlamaPoolShared, MutUntrackedOrigin]

@always_inline
def llama_barrier(sh: LlamaPoolSharedPtr):
    """Sense-reversing atomic spin barrier across all workers (~4 us latency).

    Identical algorithm to `llama2.barrier(sh: PoolSharedPtr)`, operating on
    `LlamaPoolSharedPtr`. No-op when `workers == 1`.
    """
    if sh[].workers <= 1:
        return
    var gen = sh[].bar_gen.load()
    if sh[].bar_count.fetch_add(1) + 1 == sh[].workers:
        sh[].bar_count.store(0)
        sh[].bar_gen.store(gen + 1)
    else:
        var spins = 0
        while sh[].bar_gen.load() == gen:
            spin_wait(spins)
            spins += 1

struct LlamaWorkerPool(Movable):
    """Persistent spin-barrier worker pool controller for `llamojo` contexts."""
    var workers: Int
    var sh: LlamaPoolSharedPtr
    var initialized: Bool

    def __init__(out self, workers: Int = 0, dim: Int = 4096):
        var hw_cores = max(1, num_logical_cores())
        var actual_workers = workers if workers > 0 else hw_cores
        self.workers = max(1, actual_workers)
        self.sh = unsafe_alloc[LlamaPoolShared](1)
        self.sh.unsafe_write(LlamaPoolShared(self.workers, dim))
        self.initialized = True

    @always_inline
    def worker_scratch(self, wid: Int) -> BufferPtrFloat32:
        """Return pointer to worker `wid`'s private `xbp_dim` Float32 scratch slice."""
        return self.sh[].xbp.unsafe_offset(wid * self.sh[].xbp_dim)

    @always_inline
    def sync(self):
        """Synchronize all workers at a sense-reversing barrier."""
        llama_barrier(self.sh)

    def resize_scratch(mut self, dim: Int):
        """Ensure per-worker scratchpad has at least `dim` Float32 elements."""
        if not self.initialized or dim <= self.sh[].xbp_dim:
            return
        self.sh[].xbp.unsafe_free()
        self.sh[].xbp_dim = dim
        self.sh[].xbp = unsafe_alloc[Float32](self.workers * dim)
        unsafe_memset_zero(self.sh[].xbp, self.workers * dim)

    def free(mut self):
        """Release heap allocations owned by this worker pool."""
        if self.initialized:
            self.sh[].xbp.unsafe_free()
            if Int(self.sh[].q8_act) != 0:
                self.sh[].q8_act.unsafe_free()
            self.sh.unsafe_deinit_pointee()
            self.sh.unsafe_free()
            self.initialized = False