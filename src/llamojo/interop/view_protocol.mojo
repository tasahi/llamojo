""" External Memory View and Tensor Interoperability Protocol for llamojo.

Defines:
- `TensorMemoryView`: Universal, zero-cost, non-owning descriptor of a contiguous or strided tensor buffer.
- Bi-directional zero-copy conversion routines for `llama2.Matrix` and raw buffers.
- Interoperability bridge to external numerical engines (NuMojo, MAX Graph, C BLAS).
"""

from std.memory import Pointer
from std.utils import StaticTuple
from llamojo.types import GGMLType, ggml_type_size, ggml_type_name
from llama2 import Matrix

comptime LLAMOJO_MAX_DIMS: Int = 4
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]

struct TensorMemoryView(Copyable, Movable):
    """ Lightweight, non-owning tensor descriptor exposing raw pointers, shapes, and byte strides."""
    var data: BytePtr
    var dtype: UInt32
    var ndim: Int
    var shape: StaticTuple[Int64, LLAMOJO_MAX_DIMS]
    var strides: StaticTuple[Int64, LLAMOJO_MAX_DIMS]
    var total_elements: Int64
    var is_contiguous_layout: Bool

    def __init__(
        out self,
        data: BytePtr,
        dtype: UInt32,
        ndim: Int,
        shape: StaticTuple[Int64, LLAMOJO_MAX_DIMS],
        strides: StaticTuple[Int64, LLAMOJO_MAX_DIMS],
        total_elements: Int64,
        is_contiguous_layout: Bool,
    ):
        self.data = data
        self.dtype = dtype
        self.ndim = ndim
        self.shape = shape
        self.strides = strides
        self.total_elements = total_elements
        self.is_contiguous_layout = is_contiguous_layout

    @always_inline
    def ptr_f32(self) -> FloatPtr:
        """ Returns the buffer bitcasted to Float32 pointer."""
        return self.data.unsafe_bitcast[Float32]()

    @always_inline
    def ptr_u8(self) -> BytePtr:
        """ Returns the buffer as a BytePtr."""
        return self.data

    @always_inline
    def rows(self) -> Int:
        """ Returns number of rows for 2D matrices (shape[1])."""
        return Int(self.shape[1]) if self.ndim >= 2 else 1

    @always_inline
    def cols(self) -> Int:
        """ Returns number of columns for 2D matrices (shape[0])."""
        return Int(self.shape[0])

# ─── Conversion to / from llama2.Matrix ───────────────────────────────────────

def matrix_to_memory_view(m: Matrix, dtype: UInt32 = GGMLType.F32) -> TensorMemoryView:
    """ Export a llama2.Matrix into a TensorMemoryView without allocation."""
    var ndim = len(m.dims)
    var shape = StaticTuple[Int64, LLAMOJO_MAX_DIMS]()
    shape[0] = 1
    shape[1] = 1
    shape[2] = 1
    shape[3] = 1

    var strides = StaticTuple[Int64, LLAMOJO_MAX_DIMS]()
    var elem_sz = Int64(ggml_type_size(dtype))
    strides[0] = elem_sz

    if ndim == 1:
        shape[0] = Int64(m.dims[0])
        strides[1] = strides[0] * shape[0]
        strides[2] = strides[1]
        strides[3] = strides[2]
    elif ndim == 2:
        # In llama2.Matrix, dims[0] is rows, dims[1] is cols (row-major [rows, cols]).
        # In GGML / GojoTensor, ne[0] is cols, ne[1] is rows.
        shape[0] = Int64(m.dims[1]) # cols (fastest stride)
        shape[1] = Int64(m.dims[0]) # rows
        strides[1] = strides[0] * shape[0]
        strides[2] = strides[1] * shape[1]
        strides[3] = strides[2]
    elif ndim == 3:
        shape[0] = Int64(m.dims[2])
        shape[1] = Int64(m.dims[1])
        shape[2] = Int64(m.dims[0])
        strides[1] = strides[0] * shape[0]
        strides[2] = strides[1] * shape[1]
        strides[3] = strides[2] * shape[2]

    var total_elem = Int64(m.size())
    return TensorMemoryView(
        data=m.data.unsafe_bitcast[UInt8](),
        dtype=dtype,
        ndim=ndim,
        shape=shape,
        strides=strides,
        total_elements=total_elem,
        is_contiguous_layout=True,
    )

def memory_view_to_matrix(view: TensorMemoryView) raises -> Matrix:
    """ Wrap a TensorMemoryView into a non-owning llama2.Matrix view."""
    if not view.is_contiguous_layout:
        raise Error("Cannot wrap non-contiguous TensorMemoryView into Matrix")

    var dims_list = List[Int]()
    if view.ndim == 1:
        dims_list.append(Int(view.shape[0]))
    elif view.ndim == 2:
        # Row-major Matrix dims: [rows, cols] = [shape[1], shape[0]]
        dims_list.append(Int(view.shape[1]))
        dims_list.append(Int(view.shape[0]))
    elif view.ndim == 3:
        dims_list.append(Int(view.shape[2]))
        dims_list.append(Int(view.shape[1]))
        dims_list.append(Int(view.shape[0]))
    else:
        dims_list.append(Int(view.total_elements))

    return Matrix(view.ptr_f32(), dims_list^)

