""" Zero-copy TensorMemoryView exporters for llamojo contexts, KV caches and batches (Phase 2)."""

from std.memory import Pointer
from std.utils import StaticTuple
from llamojo.types import GGMLType, ggml_type_size, llama_batch
from llamojo.llama import llama_context
from .view_protocol import TensorMemoryView, LLAMOJO_MAX_DIMS, BytePtr, FloatPtr


def f32_buffer_view(
    ptr: FloatPtr, cols: Int, rows: Int = 1, planes: Int = 1
) -> TensorMemoryView:
    """ Describe a contiguous F32 buffer as [cols (fastest), rows, planes]."""
    var shape = StaticTuple[Int64, LLAMOJO_MAX_DIMS]()
    var strides = StaticTuple[Int64, LLAMOJO_MAX_DIMS]()
    shape[0] = Int64(cols)
    shape[1] = Int64(rows)
    shape[2] = Int64(planes)
    shape[3] = 1
    strides[0] = 4
    strides[1] = strides[0] * shape[0]
    strides[2] = strides[1] * shape[1]
    strides[3] = strides[2] * shape[2]
    var ndim = 1
    if planes > 1:
        ndim = 3
    elif rows > 1:
        ndim = 2
    return TensorMemoryView(
        data=ptr.unsafe_bitcast[UInt8](),
        dtype=GGMLType.F32,
        ndim=ndim,
        shape=shape,
        strides=strides,
        total_elements=Int64(cols) * Int64(rows) * Int64(planes),
        is_contiguous_layout=True,
    )


def context_hidden_view(ctx: llama_context) -> TensorMemoryView:
    """ View of the current hidden-state buffer `x` ([dim])."""
    return f32_buffer_view(ctx.x, ctx.dim)


def context_logits_view(ctx: llama_context) -> TensorMemoryView:
    """ View of the output logits buffer ([vocab_size])."""
    return f32_buffer_view(ctx.logits, ctx.vocab_size)


def context_embeddings_view(ctx: llama_context) -> TensorMemoryView:
    """ View of the embeddings output buffer ([dim])."""
    return f32_buffer_view(ctx.embeddings, ctx.dim)


def context_kv_k_view(ctx: llama_context) -> TensorMemoryView:
    """ View of the F32 K cache as [kv_dim, max_ctx, n_layers]."""
    var kv = ctx.kv_cache.kv_dim
    return f32_buffer_view(
        ctx.kv_cache.k_f32, kv, ctx.kv_cache.max_ctx, ctx.kv_cache.n_layers
    )


def context_kv_v_view(ctx: llama_context) -> TensorMemoryView:
    """ View of the F32 V cache as [kv_dim, max_ctx, n_layers]."""
    var kv = ctx.kv_cache.kv_dim
    return f32_buffer_view(
        ctx.kv_cache.v_f32, kv, ctx.kv_cache.max_ctx, ctx.kv_cache.n_layers
    )


def batch_embd_view(batch: llama_batch, embd_dim: Int) -> TensorMemoryView:
    """ View of a batch's input embeddings as [embd_dim, n_tokens]."""
    return f32_buffer_view(batch.embd, embd_dim, Int(batch.n_tokens))


def batch_tokens_view(batch: llama_batch) -> TensorMemoryView:
    """ View of a batch's token ids (Int32) as [n_tokens]."""
    var view = f32_buffer_view(
        batch.token.unsafe_bitcast[Float32](), Int(batch.n_tokens)
    )
    view.dtype = GGMLType.I32
    return view^

