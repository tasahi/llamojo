""" Zero-copy tensor views, NumPy wrappers, and DLPack exporters (Part 1)."""

from __future__ import annotations

import ctypes
from typing import Any, Optional, Sequence, Tuple

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


class LlamojoViewCtypes(ctypes.Structure):
    """ C layout matching `struct llamojo_view` in c_api.mojo."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_uint32),
        ("ndim", ctypes.c_int32),
        ("shape0", ctypes.c_int64),
        ("shape1", ctypes.c_int64),
        ("shape2", ctypes.c_int64),
        ("shape3", ctypes.c_int64),
        ("strides0", ctypes.c_int64),
        ("strides1", ctypes.c_int64),
        ("strides2", ctypes.c_int64),
        ("strides3", ctypes.c_int64),
        ("total_elements", ctypes.c_int64),
        ("version_tag", ctypes.c_uint64),
    ]


class TensorView:
    """ Non-owning descriptor referencing active buffer memory in `llama_context`.\n
    Includes lifetime invalidation via parent generation tag to prevent reading
    stale or deallocated memory after decode/reset operations.
    """

    def __init__(
        self,
        parent: Any,
        data_ptr: int,
        shape: Tuple[int, ...],
        strides: Tuple[int, ...],
        dtype: int,
        generation: int,
        dtype_name: str = "float32",
    ):
        self._parent = parent
        self._data_ptr = data_ptr
        self._shape = shape
        self._strides = strides
        self._dtype = dtype
        self._dtype_name = dtype_name
        self._generation = generation

    def _check_valid(self):
        if self._parent is None:
            raise RuntimeError("View cannot be accessed: parent context was freed.")
        if hasattr(self._parent, "_generation") and self._parent._generation != self._generation:
            raise RuntimeError(
                f"View is stale: buffer version {self._generation} superseded by generation "
                f"{self._parent._generation} after context advancement or reset."
            )

    @property
    def data_ptr(self) -> int:
        self._check_valid()
        return self._data_ptr

    @property
    def shape(self) -> Tuple[int, ...]:
        self._check_valid()
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        self._check_valid()
        return self._strides

    @property
    def total_elements(self) -> int:
        self._check_valid()
        tot = 1
        for s in self._shape:
            tot *= s
        return tot

    def as_numpy(self, copy: bool = False, writable: bool = False) -> Any:
        """ Convert descriptor to a NumPy ndarray without copy (unless copy=True).\n
        Args:
            copy: When True, produces a standalone copy detached from the native buffer.
            writable: When False (default), marks the array non-writable to safeguard
                internal engine buffers against unintentional overwrite.
        """
        self._check_valid()
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is not installed; install numpy to use as_numpy().")

        if self._dtype_name == "float32":
            c_type = ctypes.c_float
            np_dtype = np.float32
        elif self._dtype_name in ("int32", "i32"):
            c_type = ctypes.c_int32
            np_dtype = np.int32
        elif self._dtype_name in ("uint8", "u8"):
            c_type = ctypes.c_uint8
            np_dtype = np.uint8
        else:
            c_type = ctypes.c_float
            np_dtype = np.float32

        addr = self._data_ptr
        total = self.total_elements
        flat_arr = np.ctypeslib.as_array((c_type * total).from_address(addr))

        if len(self._shape) == 1:
            arr = flat_arr.reshape(self._shape)
        else:
            arr = np.lib.stride_tricks.as_strided(
                flat_arr,
                shape=self._shape,
                strides=self._strides,
            )

        if copy:
            return arr.copy()

        if not writable:
            arr.flags.writeable = False
        return arr

    def as_dlpack(self) -> Any:
        """ Export as a DLPack tensor capsule consumable by PyTorch or NumPy."""
        self._check_valid()
        arr = self.as_numpy(copy=False, writable=True)
        if hasattr(arr, "__dlpack__"):
            return arr.__dlpack__()
        if _TORCH_AVAILABLE:
            t = torch.from_numpy(arr)
            return t.__dlpack__()
        raise RuntimeError("DLPack export requires NumPy >= 1.22.0 or PyTorch.")

    def as_torch(self, copy: bool = False) -> Any:
        """ Export directly to a torch.Tensor."""
        self._check_valid()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; install torch to use as_torch().")
        arr = self.as_numpy(copy=copy, writable=True)
        t = torch.from_numpy(arr)
        return t if not copy else t.clone()
