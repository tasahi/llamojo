""" Zero-dependency Safetensors reader, writer, and GGUF converter (Part 2)."""

from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .weight_maps import hf_to_gguf_name, gguf_to_hf_name, permute_rope_hf_to_ggml, permute_rope_ggml_to_hf


DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

INV_DTYPE_MAP = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int32"): "I32",
    np.dtype("int16"): "I16",
    np.dtype("int8"): "I8",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}


class SafetensorsFile:
    """ Memory-mapped reader for a single .safetensors file."""

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        self.file_size = os.path.getsize(self.filepath)
        if self.file_size < 8:
            raise ValueError(f"File {filepath} is too small to be a valid safetensors file.")

        self._f = open(self.filepath, "rb")
        self._mmap = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        # 1. Read 8-byte little endian header length
        header_len = struct.unpack("<Q", self._mmap[:8])[0]
        if header_len <= 0 or 8 + header_len > self.file_size:
            raise ValueError(f"Invalid safetensors header length: {header_len}")

        # 2. Parse JSON header
        header_bytes = self._mmap[8 : 8 + header_len]
        try:
            self.header: Dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Corrupt safetensors JSON header: {e}")

        self.data_base = 8 + header_len
        self.metadata = self.header.get("__metadata__", {})

        # Validate offsets
        for name, info in self.header.items():
            if name == "__metadata__":
                continue
            offsets = info.get("data_offsets")
            if not offsets or len(offsets) != 2:
                raise ValueError(f"Invalid data_offsets for tensor {name}")
            start, end = offsets
            if start < 0 or end < start or self.data_base + end > self.file_size:
                raise ValueError(f"Tensor {name} offsets [{start}, {end}] exceed file bounds.")

    def tensor_names(self) -> List[str]:
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, name: str) -> np.ndarray:
        """Return memory-mapped zero-copy numpy array for tensor `name`."""
        if name not in self.header or name == "__metadata__":
            raise KeyError(f"Tensor {name} not found in {self.filepath}")

        info = self.header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]

        # Handle bfloat16: upcast to float32
        if dtype_str == "BF16":
            raw_u16 = np.frombuffer(
                self._mmap,
                dtype=np.uint16,
                count=(end - start) // 2,
                offset=self.data_base + start,
            )
            # BF16 to F32: shift 16 bits
            u32 = raw_u16.astype(np.uint32) << 16
            return u32.view(np.float32).reshape(shape)

        np_dtype = DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")

        elem_size = np.dtype(np_dtype).itemsize
        count = (end - start) // elem_size
        arr = np.frombuffer(
            self._mmap,
            dtype=np_dtype,
            count=count,
            offset=self.data_base + start,
        )
        return arr.reshape(shape)

    def close(self):
        if hasattr(self, "_mmap") and self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # Active buffer views still open; allow garbage collection to close
                pass
            self._mmap = None
        if hasattr(self, "_f") and self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def __del__(self):
        self.close()


def save_safetensors(
    tensors: Dict[str, np.ndarray],
    filepath: str,
    metadata: Optional[Dict[str, str]] = None,
):
    """Serialize a dictionary of numpy arrays to a standard .safetensors file."""
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    data_bytes_list: List[bytes] = []
    current_offset = 0

    # Ensure tensors are contiguous and collect offsets
    for name, arr in tensors.items():
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        dtype_str = INV_DTYPE_MAP.get(arr.dtype, "F32")
        raw = arr.tobytes()
        length = len(raw)

        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + length],
        }
        current_offset += length
        data_bytes_list.append(raw)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Align header to 8 bytes for clean alignment
    extra = (8 - (len(header_json) % 8)) % 8
    if extra > 0:
        header_json += b" " * extra

    header_len = len(header_json)

    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for chunk in data_bytes_list:
            f.write(chunk)


def load_safetensors_dir(dir_path: str) -> Dict[str, np.ndarray]:
    """Load all tensors from a directory containing `model.safetensors` or sharded files."""
    dir_path = os.path.abspath(dir_path)
    single_file = os.path.join(dir_path, "model.safetensors")

    weights: Dict[str, np.ndarray] = {}

    if os.path.isfile(single_file):
        sf = SafetensorsFile(single_file)
        try:
            for name in sf.tensor_names():
                weights[name] = sf.get_tensor(name)
        finally:
            sf.close()
        return weights

    # Check for index file
    index_file = os.path.join(dir_path, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map: Dict[str, str] = idx_data.get("weight_map", {})
        unique_files = sorted(list(set(weight_map.values())))
        for fn in unique_files:
            fp = os.path.join(dir_path, fn)
            sf = SafetensorsFile(fp)
            try:
                for name in sf.tensor_names():
                    weights[name] = sf.get_tensor(name)
            finally:
                sf.close()
        return weights

    # Scan for any .safetensors files
    st_files = [f for f in os.listdir(dir_path) if f.endswith(".safetensors")]
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {dir_path}")

    for fn in sorted(st_files):
        fp = os.path.join(dir_path, fn)
        sf = SafetensorsFile(fp)
        try:
            for name in sf.tensor_names():
                weights[name] = sf.get_tensor(name)
        finally:
            sf.close()

    return weights