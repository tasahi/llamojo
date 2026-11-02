""" NuMojo Zero-Copy Buffer & Array Layout Adapter for llamojo.

Defines the zero-copy layout and adapter structures compatible with NuMojo's
NDArray and DataContainer memory representations. This allows seamless zero-copy
interoperability between llamojo Matrices and external numerical engines.
"""

from std.memory import Pointer
from std.utils import StaticTuple
from llamojo.types import GGMLType
from llama2 import Matrix
from llamojo.interop.view_protocol import (
    TensorMemoryView,
    LLAMOJO_MAX_DIMS,
    matrix_to_memory_view,
    memory_view_to_matrix,
)

comptime FloatPtr = Pointer[Float32, MutUntrackedOrigin]
comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

struct NuMojoNDArrayView(Copyable, Movable):
    """ Non-owning 2D NDArray view matching NuMojo's internal layout.
    
    Exposes raw pointer, dimensions, strides, and element access with zero allocation.
    """
    var data_ptr: FloatPtr
    var rows: Int
    var cols: Int
    var stride_row: Int
    var stride_col: Int
    var size: Int
    var is_view: Bool

    def __init__(out self, ptr: FloatPtr, rows: Int, cols: Int):
        self.data_ptr = ptr
        self.rows = rows
        self.cols = cols
        self.stride_row = cols
        self.stride_col = 1
        self.size = rows * cols
        self.is_view = True

    @always_inline
    def load(self, r: Int, c: Int) -> Float32:
        """ Load element at (r, c) with zero overhead."""
        return self.data_ptr[unsafe_offset=r * self.stride_row + c * self.stride_col]

    @always_inline
    def store(self, r: Int, c: Int, val: Float32):
        """ Store element at (r, c) mutating the underlying buffer."""
        self.data_ptr[unsafe_offset=r * self.stride_row + c * self.stride_col] = val

    @always_inline
    def raw_byte_ptr(self) -> BytePtr:
        return self.data_ptr.unsafe_bitcast[UInt8]()

def memory_view_to_numojo_2d(view: TensorMemoryView) raises -> NuMojoNDArrayView:
    """ Wrap a 2D TensorMemoryView into a non-owning NuMojo NDArray view."""
    if not view.is_contiguous_layout:
        raise Error("Cannot wrap non-contiguous TensorMemoryView into NuMojo NDArray")

    var rows = view.rows()
    var cols = view.cols()
    return NuMojoNDArrayView(view.ptr_f32(), rows, cols)

def matrix_to_numojo_2d(m: Matrix) raises -> NuMojoNDArrayView:
    """ Wrap a 2D llama2.Matrix into a non-owning NuMojo NDArray view."""
    var view = matrix_to_memory_view(m)
    return memory_view_to_numojo_2d(view)

def numojo_to_memory_view(arr: NuMojoNDArrayView) -> TensorMemoryView:
    """ Export a NuMojo NDArray view as a TensorMemoryView."""
    var shape = StaticTuple[Int64, LLAMOJO_MAX_DIMS]()
    shape[0] = Int64(arr.cols)
    shape[1] = Int64(arr.rows)
    shape[2] = 1
    shape[3] = 1

    var strides = StaticTuple[Int64, LLAMOJO_MAX_DIMS]()
    strides[0] = 4
    strides[1] = strides[0] * shape[0]
    strides[2] = strides[1]
    strides[3] = strides[2]

    return TensorMemoryView(
        data=arr.raw_byte_ptr(),
        dtype=GGMLType.F32,
        ndim=2,
        shape=shape,
        strides=strides,
        total_elements=Int64(arr.size),
        is_contiguous_layout=True,
    )

def numojo_to_matrix_view(arr: NuMojoNDArrayView) raises -> Matrix:
    """ Export a NuMojo NDArray view as a non-owning llama2.Matrix view."""
    var view = numojo_to_memory_view(arr)
    return memory_view_to_matrix(view)

