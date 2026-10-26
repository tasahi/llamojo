""" Sparse Mixture-of-Experts (MoE) Routing and Forward Execution for llamojo.\n
Implements Phase 10 of pure-CPU extension plan:
- MoE router gating over hidden states (ffn_gate_inp.weight)
- Softmax over expert routing logits
- Top-k expert selection and routing weight normalization
- Selective expert execution (SwiGLU gate/up/down projections)
- Weighted accumulation into the residual stream
- Multi-threaded expert execution scheduling
"""

from std.collections import List
from std.math import exp
from std.memory import Pointer, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.utils import StaticTuple

from ..types import ModelConfig, GGMLType, ggml_block_size, ggml_type_size
from .threadpool import (
    LlamaPoolShared,
    LlamaPoolSharedPtr,
    llama_barrier,
    chunk_start,
    chunk_end,
)
from .kernels_norm_rope import rms_norm, fused_silu_mul
from .kernels_quant import BytePtr, FloatPtr, dot_quant_row, matmul_quant_fused

# ─── MoE Expert Weights Container ────────────────────────────────────────────

struct MoEExpertWeights(ImplicitlyCopyable, Copyable, Movable):
    """ Pointers to the FFN weights of a single expert."""
    var w_gate: BytePtr
    var w_up: BytePtr
    var w_down: BytePtr
    var dtype: UInt32

    def __init__(out self):
        var z: Int = 0
        self.w_gate = BytePtr(unsafe_from_address=z)
        self.w_up = BytePtr(unsafe_from_address=z)
        self.w_down = BytePtr(unsafe_from_address=z)
        self.dtype = GGMLType.F32

    def __init__(out self, w_gate: BytePtr, w_up: BytePtr, w_down: BytePtr, dtype: UInt32):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.dtype = dtype

# ─── MoE Layer Descriptor ────────────────────────────────────────────────────

struct MoELayerWeights(Movable):
    """ Router and expert weights for a single MoE layer."""
    var w_router: FloatPtr              # [n_experts, dim] router projection (ffn_gate_inp.weight)
    var n_experts: Int                  # Total number of available experts (e.g. 8)
    var n_experts_used: Int             # Top-k experts activated per token (e.g. 2)
    var experts: List[MoEExpertWeights] # List of individual expert FFN weights

    def __init__(out self, n_experts: Int, n_experts_used: Int):
        var z: Int = 0
        self.w_router = FloatPtr(unsafe_from_address=z)
        self.n_experts = n_experts
        self.n_experts_used = n_experts_used
        self.experts = List[MoEExpertWeights]()

# ─── Router & Top-K Softmax Selection ────────────────────────────────────────

struct SelectedExpert(ImplicitlyCopyable, Copyable, Movable):
    var index: Int
    var weight: Float32

    def __init__(out self, index: Int, weight: Float32):
        self.index = index
        self.weight = weight

def route_experts(
        x_norm: FloatPtr,
        w_router: FloatPtr,
        dim: Int,
        n_experts: Int,
        n_used: Int,
    ) -> List[SelectedExpert]:
    """ Compute router logits, softmax, and select top-k experts."""
    var selected = List[SelectedExpert]()
    if n_experts <= 0 or n_used <= 0 or Int(w_router) == 0:
        return selected^

    # 1. Compute router logits: logit_e = dot(x_norm, w_router[e])
    var logits = unsafe_alloc[Float32](n_experts)
    var max_logit = Float32(-1e30)

    for e in range(n_experts):
        var r_row = w_router.unsafe_offset(e * dim)
        var sum_val: Float32 = 0.0
        for d in range(dim):
            sum_val += x_norm[unsafe_offset=d] * r_row[unsafe_offset=d]
        logits[unsafe_offset=e] = sum_val
        if sum_val > max_logit:
            max_logit = sum_val

    # 2. Softmax over expert logits
    var sum_exp: Float32 = 0.0
    for e in range(n_experts):
        var p = exp(logits[unsafe_offset=e] - max_logit)
        logits[unsafe_offset=e] = p
        sum_exp += p

    var inv_sum = Float32(1.0) / (sum_exp if sum_exp > Float32(0.0) else Float32(1.0))
    for e in range(n_experts):
        logits[unsafe_offset=e] *= inv_sum

    # 3. Select top-k experts (greedy selection)
    var k = min(n_used, n_experts)
    var used_mask = unsafe_alloc[Bool](n_experts)
    for e in range(n_experts):
        used_mask[unsafe_offset=e] = False

    for _ in range(k):
        var best_idx = -1
        var best_weight = Float32(-1.0)
        for e in range(n_experts):
            if not used_mask[unsafe_offset=e] and logits[unsafe_offset=e] > best_weight:
                best_weight = logits[unsafe_offset=e]
                best_idx = e
        if best_idx >= 0:
            used_mask[unsafe_offset=best_idx] = True
            selected.append(SelectedExpert(best_idx, best_weight))

    # 4. Renormalize selected expert weights so they sum to 1.0
    var norm_sum: Float32 = 0.0
    for i in range(len(selected)):
        norm_sum += selected[i].weight
    if norm_sum > Float32(0.0):
        var inv_norm = Float32(1.0) / norm_sum
        for i in range(len(selected)):
            selected[i].weight *= inv_norm

    logits.unsafe_free()
    used_mask.unsafe_free()
    return selected^

# ─── MoE FFN Forward Execution ───────────────────────────────────────────────

def forward_moe_ffn(
        x: FloatPtr,
        xbp: FloatPtr,
        hb: FloatPtr,
        hb2: FloatPtr,
        moe_layer: MoELayerWeights,
        dim: Int,
        hidden_dim: Int,
        wid: Int,
        sh: LlamaPoolSharedPtr,
    ):
    """ Execute sparse MoE FFN routing, evaluating only top-k active experts."""
    var workers = sh[].workers
    var d0 = chunk_start(dim, wid, workers)
    var d1 = chunk_end(dim, wid, workers)
    var f0 = chunk_start(hidden_dim, wid, workers)
    var f1 = chunk_end(hidden_dim, wid, workers)

    # Worker 0 computes routing selection
    var active_experts = route_experts(
        xbp,
        moe_layer.w_router,
        dim,
        moe_layer.n_experts,
        moe_layer.n_experts_used,
    )

    llama_barrier(sh)

    # If no router or no experts found, return
    if len(active_experts) == 0:
        return

    # Buffer to accumulate expert output into residual: out_acc [dim]
    # We can use worker 0 or distribute across workers
    for exp_idx in range(len(active_experts)):
        var expert_meta = active_experts[exp_idx]
        var e_id = expert_meta.index
        var e_weight = expert_meta.weight
        if e_id < 0 or e_id >= len(moe_layer.experts):
            continue

        var exp_w = moe_layer.experts[e_id]
        if Int(exp_w.w_gate) == 0 or Int(exp_w.w_up) == 0 or Int(exp_w.w_down) == 0:
            continue

        # Step A: Expert SwiGLU: compute gate (hb) and up (hb2) projections
        var c_gate_up = StaticTuple[FloatPtr, 2](hb, hb2)
        var w_gate_up = StaticTuple[BytePtr, 2](exp_w.w_gate, exp_w.w_up)
        matmul_quant_fused[2](exp_w.dtype, c_gate_up, xbp, w_gate_up, f0, f1, dim)

        # Apply SiLU to worker's slice of hidden dimension: hb = silu(hb) * hb2
        for i in range(f0, f1):
            fused_silu_mul(hb.unsafe_offset(i), hb2.unsafe_offset(i), hb.unsafe_offset(i), 1)

        llama_barrier(sh)

        # Step B: Expert Down projection scaled by e_weight and accumulated into x
        var blk_sz_down = Int(ggml_block_size(exp_w.dtype))
        var type_sz_down = Int(ggml_type_size(exp_w.dtype))
        var bpr_down = (hidden_dim // blk_sz_down) * type_sz_down

        for r in range(d0, d1):
            var w_row = exp_w.w_down.unsafe_offset(r * bpr_down)
            var res_down = dot_quant_row(exp_w.dtype, w_row, hb, hidden_dim)
            x[unsafe_offset=r] += res_down * e_weight

        llama_barrier(sh)
