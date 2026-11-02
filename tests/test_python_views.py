""" Unit tests for Part 1: Zero-Copy Views, NumPy/DLPack/PyTorch exporters,
and Batched Full Logits."""

import ctypes
import unittest
from typing import List

import numpy as np

from llamojo import Llama, LlamojoForCausalLM
from llamojo._views import TensorView


class FakeLlamaContext:
    def __init__(self, vocab_size: int = 16, n_embd: int = 32):
        self._n_vocab = vocab_size
        self._n_embd = n_embd
        self._generation = 1
        self.input_ids = []
        self._buf_logits = (ctypes.c_float * (vocab_size * 4))()
        self._buf_embd = (ctypes.c_float * n_embd)()
        self.scores = None

    def reset(self):
        self._generation += 1
        self.input_ids = []

    def close(self):
        self._generation += 1

    def eval(self, tokens: List[int], all_logits: bool = False):
        self._generation += 1
        self.input_ids.extend(tokens)
        n = len(tokens)
        for i in range(n):
            for v in range(self._n_vocab):
                self._buf_logits[i * self._n_vocab + v] = float(i * 100 + v)
        self.scores = [self._buf_logits[(n - 1) * self._n_vocab + v] for v in range(self._n_vocab)]

    def logits_all(self, n_tokens: int = None):
        n = n_tokens or len(self.input_ids)
        out = []
        for i in range(n):
            row = [self._buf_logits[i * self._n_vocab + v] for v in range(self._n_vocab)]
            out.append(row)
        return np.array(out, dtype=np.float32)

    def get_logits_view(self):
        addr = ctypes.addressof(self._buf_logits)
        return TensorView(
            parent=self,
            data_ptr=addr,
            shape=(self._n_vocab,),
            strides=(4,),
            dtype=0,
            generation=self._generation,
            dtype_name="float32",
        )

    def get_embeddings_view(self):
        addr = ctypes.addressof(self._buf_embd)
        return TensorView(
            parent=self,
            data_ptr=addr,
            shape=(self._n_embd,),
            strides=(4,),
            dtype=0,
            generation=self._generation,
            dtype_name="float32",
        )


class TestPart1ZeroCopyViews(unittest.TestCase):
    def setUp(self):
        self.fake = FakeLlamaContext(vocab_size=8, n_embd=16)

    def test_numpy_zero_copy_and_stale_invalidation(self):
        self.fake.eval([1, 2, 3])
        view = self.fake.get_logits_view()
        self.assertEqual(view.shape, (8,))
        self.assertEqual(view.strides, (4,))

        arr1 = view.as_numpy()
        self.assertEqual(arr1.shape, (8,))
        self.assertFalse(arr1.flags.writeable)
        self.assertEqual(arr1[0], 0.0)

        # Mutating underlying C buffer reflects in read-only numpy view
        self.fake._buf_logits[0] = 999.0
        self.assertEqual(arr1[0], 999.0)

        # Evaluating new tokens increments generation and invalidates existing view
        self.fake.eval([4])
        with self.assertRaises(RuntimeError) as ctx:
            _ = view.as_numpy()
        self.assertIn("View is stale", str(ctx.exception))

    def test_dlpack_and_torch_export(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in this environment.")

        self.fake.eval([5])
        view = self.fake.get_logits_view()
        t = view.as_torch()
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, torch.Size([8]))

        # DLPack capsule verification
        cap = view.as_dlpack()
        self.assertIsNotNone(cap)
        from_dlp = torch.from_dlpack(cap)
        self.assertEqual(from_dlp.shape, torch.Size([8]))

    def test_hf_compat_full_sequence_logits(self):
        model = LlamojoForCausalLM(self.fake)
        # 1. Default last-token logits [batch, vocab]
        out_last = model([[1, 2], [3, 4, 5]], output_all_logits=False)
        self.assertEqual(out_last.logits.shape, (2, 8))

        # 2. Full-sequence logits [batch, seq, vocab]
        out_all = model([[1, 2, 3]], output_all_logits=True)
        self.assertEqual(out_all.logits.shape, (1, 3, 8))
        # Sequence step 0, 1, 2 logits match step calculation
        self.assertEqual(out_all.logits[0, 0, 1], 1.0)
        self.assertEqual(out_all.logits[0, 1, 1], 101.0)
        self.assertEqual(out_all.logits[0, 2, 1], 201.0)


if __name__ == "__main__":
    unittest.main()
