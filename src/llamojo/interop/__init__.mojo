""" Interoperability module for llamojo providing zero-copy view adapters."""

from .view_protocol import (
    TensorMemoryView,
    LLAMOJO_MAX_DIMS,
    matrix_to_memory_view,
    memory_view_to_matrix,
)
from .numojo_bridge import (
    NuMojoNDArrayView,
    memory_view_to_numojo_2d,
    matrix_to_numojo_2d,
    numojo_to_memory_view,
    numojo_to_matrix_view,
)


from .context_views import (
    f32_buffer_view,
    context_hidden_view,
    context_logits_view,
    context_embeddings_view,
    context_kv_k_view,
    context_kv_v_view,
    batch_embd_view,
    batch_tokens_view,
)
