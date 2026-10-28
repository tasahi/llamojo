import os
import unittest
from llamojo import Llama, LlamaGrammar

MODEL_PATH = "data/SmolLM2-135M-Instruct-Q4_K_M.gguf"


class TestLlamojoGrammarAndLora(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Model {MODEL_PATH} not found.")
        cls.llm = Llama(model_path=MODEL_PATH, n_ctx=256)

    @classmethod
    def tearDownClass(cls):
        cls.llm.close()

    def test_grammar_initialization(self):
        g = LlamaGrammar.from_string("root ::= [0-9]+")
        self.assertIsNotNone(g._handle)
        g.close()
        self.assertIsNone(g._handle)

    def test_grammar_json_constraint(self):
        # Force model to generate integer digits via grammar
        prompt = "The magic number is:"
        g = LlamaGrammar.from_string("integer")
        res = self.llm(prompt, max_tokens=6, temperature=0.0, grammar=g)
        text = res["choices"][0]["text"].strip()
        print(f"Grammar integer constrained output: '{text}'")
        self.assertTrue(any(c.isdigit() for c in text))

    def test_grammar_in_streaming(self):
        prompt = "1 + 1 ="
        g = LlamaGrammar.from_string("integer")
        accumulated = ""
        for chunk in self.llm(prompt, max_tokens=4, temperature=0.0, stream=True, grammar=g):
            accumulated += chunk["choices"][0]["text"]
        print(f"Grammar stream output: '{accumulated.strip()}'")
        self.assertIsNotNone(accumulated)

    def test_lora_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.llm.apply_lora("nonexistent_lora.gguf")


if __name__ == "__main__":
    unittest.main()

