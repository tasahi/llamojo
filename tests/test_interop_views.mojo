""" Unit tests verifying Phase 1 TensorMemoryView protocol and NuMojo NDArray bridge in llamojo. """

from std.memory import Pointer
from std.testing import assert_equal, assert_true
from llama2 import Matrix
from llamojo.interop import (
    TensorMemoryView,
    matrix_to_memory_view,
    memory_view_to_matrix,
    matrix_to_numojo_2d,
    numojo_to_matrix_view,
    NuMojoNDArrayView,
)

def test_matrix_view_bridge() raises:
    print("1. Testing llama2.Matrix <-> TensorMemoryView Zero-Copy Bridge...")
    var m = Matrix(3, 4)
    for r in range(3):
        for c in range(4):
            m.data[unsafe_offset=r * 4 + c] = Float32(r * 10 + c)

    var view = matrix_to_memory_view(m)
    assert_equal(view.rows(), 3)
    assert_equal(view.cols(), 4)
    assert_equal(view.total_elements, 12)
    assert_equal(view.ptr_f32(), m.data)
    assert_true(view.is_contiguous_layout)

    # Re-wrap into Matrix view
    var m_wrapped = memory_view_to_matrix(view)
    assert_equal(len(m_wrapped.dims), 2)
    assert_equal(m_wrapped.dims[0], 3)
    assert_equal(m_wrapped.dims[1], 4)
    assert_equal(m_wrapped.allocated, 0) # Non-owning!
    assert_equal(m_wrapped.data[unsafe_offset=5], Float32(11.0))

    m.data.unsafe_free()
    print("   ✓ llama2.Matrix <-> TensorMemoryView zero-copy verified!")

def test_numojo_bridge() raises:
    print("2. Testing NuMojo NDArray Zero-Copy Bridge on Matrix...")
    var m = Matrix(2, 3)
    for r in range(2):
        for c in range(3):
            m.data[unsafe_offset=r * 3 + c] = Float32((r + 1) * (c + 1))

    # Wrap into NuMojo NDArray View
    var ndarr = matrix_to_numojo_2d(m)
    assert_equal(ndarr.rows, 2)
    assert_equal(ndarr.cols, 3)
    assert_equal(ndarr.size, 6)
    assert_equal(ndarr.load(1, 2), Float32(6.0))

    # Mutate through NuMojo view and verify it changes Matrix directly
    ndarr.store(0, 0, Float32(777.0))
    assert_equal(m.data[unsafe_offset=0], Float32(777.0))

    # Wrap NuMojo view back into Matrix
    var m_back = numojo_to_matrix_view(ndarr)
    assert_equal(m_back.allocated, 0)
    assert_equal(m_back.data[unsafe_offset=0], Float32(777.0))

    m.data.unsafe_free()
    print("   ✓ NuMojo NDArray <-> Matrix zero-copy verified!")

def main() raises:
    print("═══════════════════════════════════════════════════════════════════")
    print("   Running llamojo Phase 1 Interoperability & View Tests           ")
    print("═══════════════════════════════════════════════════════════════════")
    test_matrix_view_bridge()
    test_numojo_bridge()
    print("═══════════════════════════════════════════════════════════════════")
    print("   All llamojo Phase 1 Tests PASSED Successfully!                  ")
    print("═══════════════════════════════════════════════════════════════════")

