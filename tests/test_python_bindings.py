import os
import unittest
from llamojo import Llama

MODEL_PATH = "data/SmolLM2-135M-Instruct-Q4_K_M.gguf"


class TestLlamojoPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Model {MODEL_PATH} not found.")
        cls.llm = Llama(model_path=MODEL_PATH, n_ctx=256)

    @classmethod
    def tearDownClass(cls):
        cls.llm.close()

    def test_tokenization(self):
        text = "Hello world"
        tokens = self.llm.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        detok = self.llm.detokenize(tokens).decode("utf-8")
        self.assertIn("Hello world", detok)

    def test_completion_schema(self):
        res = self.llm.create_completion("Once upon a time", max_tokens=8, temperature=0.0)
        self.assertIn("id", res)
        self.assertIn("choices", res)
        self.assertIn("usage", res)
        self.assertEqual(res["object"], "text_completion")
        self.assertEqual(len(res["choices"]), 1)
        self.assertIn("text", res["choices"][0])
        self.assertIn("finish_reason", res["choices"][0])

    def test_callable_and_stop(self):
        res = self.llm("Once upon a time", max_tokens=32, temperature=0.0, stop=["dreams"])
        text = res["choices"][0]["text"]
        self.assertNotIn("dreams", text)
        self.assertEqual(res["choices"][0]["finish_reason"], "stop")

    def test_streaming(self):
        chunks = list(self.llm("Once upon a time", max_tokens=5, temperature=0.0, stream=True))
        self.assertGreaterEqual(len(chunks), 5)
        for c in chunks:
            self.assertIn("choices", c)
            self.assertIn("text", c["choices"][0])

    def test_chat_completion(self):
        res = self.llm.create_chat_completion(
            [{"role": "user", "content": "Say hello"}],
            max_tokens=8,
            temperature=0.0,
        )
        self.assertEqual(res["object"], "chat.completion")
        self.assertEqual(res["choices"][0]["message"]["role"], "assistant")
        self.assertIsInstance(res["choices"][0]["message"]["content"], str)

    def test_from_pretrained(self):
        from unittest.mock import patch
        with patch("huggingface_hub.list_repo_files") as mock_list, \
             patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_list.return_value = ["SmolLM2-135M-Instruct-Q4_K_M.gguf"]
            mock_download.return_value = MODEL_PATH
            llm = Llama.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF", filename="*Q4_K_M.gguf", n_ctx=128)
            try:
                self.assertIsNotNone(llm)
                self.assertEqual(llm.model_path, os.path.abspath(MODEL_PATH))
                mock_list.assert_called_once()
                mock_download.assert_called_once()
            finally:
                llm.close()

if __name__ == "__main__":
    unittest.main()
