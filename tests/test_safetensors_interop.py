"""Unit tests for Part 2: Safetensors reader/writer, weight mapping, and direct loading."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from llamojo import (
    SafetensorsFile,
    save_safetensors,
    load_safetensors_dir,
    convert_safetensors_to_gguf,
    hf_to_gguf_name,
    gguf_to_hf_name,
    permute_rope_hf_to_ggml,
    permute_rope_ggml_to_hf,
    Llama,
    LlamojoForCausalLM,
)


class TestSafetensorsAndWeightMaps(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_safetensors_io_and_bounds_validation(self):
        tensors = {
            "weight_f32": np.arange(12, dtype=np.float32).reshape(3, 4),
            "weight_i32": np.array([10, 20, 30], dtype=np.int32),
        }
        file_path = os.path.join(self.test_dir, "test.safetensors")
        save_safetensors(tensors, file_path, metadata={"format": "test"})

        sf = SafetensorsFile(file_path)
        self.assertEqual(sorted(sf.tensor_names()), ["weight_f32", "weight_i32"])
        self.assertEqual(sf.metadata.get("format"), "test")

        t_f32 = sf.get_tensor("weight_f32")
        self.assertTrue(np.allclose(t_f32, tensors["weight_f32"]))
        t_i32 = sf.get_tensor("weight_i32")
        self.assertTrue(np.array_equal(t_i32, tensors["weight_i32"]))
        sf.close()

        # Corrupt file test: too small
        bad_path = os.path.join(self.test_dir, "bad.safetensors")
        with open(bad_path, "wb") as f:
            f.write(b"123")
        with self.assertRaises(ValueError):
            _ = SafetensorsFile(bad_path)

    def test_sharded_and_directory_loading(self):
        t1 = {"model.embed_tokens.weight": np.ones((4, 4), dtype=np.float32)}
        t2 = {"model.norm.weight": np.ones((4,), dtype=np.float32) * 2.0}

        f1 = os.path.join(self.test_dir, "shard1.safetensors")
        f2 = os.path.join(self.test_dir, "shard2.safetensors")
        save_safetensors(t1, f1)
        save_safetensors(t2, f2)

        idx = {
            "weight_map": {
                "model.embed_tokens.weight": "shard1.safetensors",
                "model.norm.weight": "shard2.safetensors",
            }
        }
        with open(os.path.join(self.test_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f)

        loaded = load_safetensors_dir(self.test_dir)
        self.assertIn("model.embed_tokens.weight", loaded)
        self.assertIn("model.norm.weight", loaded)
        self.assertEqual(loaded["model.norm.weight"][0], 2.0)

    def test_weight_naming_taxonomy_and_rope(self):
        # Bijective mapping checks
        self.assertEqual(hf_to_gguf_name("model.embed_tokens.weight"), "token_embd.weight")
        self.assertEqual(hf_to_gguf_name("model.layers.3.self_attn.q_proj.weight"), "blk.3.attn_q.weight")
        self.assertEqual(hf_to_gguf_name("model.layers.3.mlp.down_proj.weight"), "blk.3.attn_down.weight" if False else "blk.3.ffn_down.weight")
        self.assertEqual(gguf_to_hf_name("blk.3.ffn_down.weight"), "model.layers.3.mlp.down_proj.weight")

        # RoPE rotation bijection
        q_orig = np.random.randn(8, 16).astype(np.float32)
        q_ggml = permute_rope_hf_to_ggml(q_orig, n_heads=2, head_dim=4)
        q_hf = permute_rope_ggml_to_hf(q_ggml, n_heads=2, head_dim=4)
        self.assertTrue(np.allclose(q_orig, q_hf))

    def test_direct_safetensors_model_loading_and_inference(self):
        # Construct synthetic HF-format model directory
        model_dir = os.path.join(self.test_dir, "hf_synthetic")
        os.makedirs(model_dir, exist_ok=True)

        cfg = {
            "model_type": "llama",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(cfg, f)

        weights = {
            "model.embed_tokens.weight": np.random.randn(8, 16).astype(np.float32),
            "model.norm.weight": np.ones((16,), dtype=np.float32),
            "lm_head.weight": np.random.randn(8, 16).astype(np.float32),
            "model.layers.0.input_layernorm.weight": np.ones((16,), dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones((16,), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(8, 16).astype(np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(8, 16).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(32, 16).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(32, 16).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(16, 32).astype(np.float32),
        }
        save_safetensors(weights, os.path.join(model_dir, "model.safetensors"))

        # 1. Load directly via Llama.from_safetensors
        llm = Llama.from_safetensors(model_dir, n_ctx=32)
        self.assertEqual(int(llm.n_vocab()), 8)
        self.assertEqual(int(llm.n_embd()), 16)
        llm.eval([1, 2], all_logits=True)
        logits = llm.logits_all()
        self.assertEqual(logits.shape, (2, 8))
        llm.close()

        # 2. Load directly via HuggingFace-compat facade LlamojoForCausalLM
        hf_model = LlamojoForCausalLM.from_pretrained(model_dir, n_ctx=32)
        out = hf_model([[1, 2]], output_all_logits=True)
        self.assertEqual(out.logits.shape, (1, 2, 8))
        hf_model.close()


if __name__ == "__main__":
    unittest.main()

