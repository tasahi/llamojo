""" Backend package for llamojo: ISA, thread pool, quantized matmuls, fused norm/RoPE,
attention & multi-arch forward passes."""

from .isa import (
    has_avx512f,
    has_avx512vnni,
    has_avx2,
    has_fma,
    get_simd_width,
    SIMD_WIDTH,
    F32xW,
    I8xW,
    U8xW,
    I32xW,
    isa_info,
    simd_info,
    bswap16,
    f16_to_f32,
    bf16_to_f32,
    load_f16_scale,
    saturating_cast_i8,
    saturating_cast_u8,
)
from .threadpool import (
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
    free_matrix,
    LlamaPoolShared,
    LlamaPoolSharedPtr,
    llama_barrier,
    LlamaWorkerPool,
)
from .kernels_norm_rope import (
    fused_rms_norm_residual,
    rms_norm,
    fused_silu_mul,
    silu,
    rope_head,
    apply_rope_neox,
)
from .kernels_quant import (
    BytePtr,
    FloatPtr,
    k4_scale,
    quantize_row_q8_0,
    dot_f32_row,
    dot_f16_row,
    dot_q4_0_row,
    dot_q4_1_row,
    dot_q5_0_row,
    dot_q8_0_row,
    dot_q4_k_row,
    dot_q5_k_row,
    dot_q6_k_row,
    dot_mxfp4_row,
    dot_quant_row,
    matmul_quant_fused,
)
from .kernels_attn import (
    dot_vectors,
    axpy_vector,
    gqa_decode_head,
    gqa_decode,
)
from .models import (
    LayerWeights,
    forward_layer_dense,
)
from .moe import (
    MoEExpertWeights,
    MoELayerWeights,
    SelectedExpert,
    route_experts,
    forward_moe_ffn,
)

