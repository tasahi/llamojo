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

    def test_chat_completion_streaming(self):
        chunks = list(
            self.llm.create_chat_completion(
                [{"role": "user", "content": "Count to 3"}],
                max_tokens=6,
                temperature=0.0,
                stream=True,
            )
        )
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["choices"][0]["delta"].get("role"), "assistant")

    def test_chat_template_formats(self):
        from llamojo import get_chat_formatter
        messages = [{"role": "user", "content": "Hi"}]

        # Test llama-3 formatter
        llama3_fmt = get_chat_formatter("llama-3")
        self.assertIsNotNone(llama3_fmt)
        resp = llama3_fmt(messages)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", resp.prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", resp.prompt)

        # Test gemma formatter
        gemma_fmt = get_chat_formatter("gemma")
        self.assertIsNotNone(gemma_fmt)
        resp = gemma_fmt(messages)
        self.assertIn("<start_of_turn>user\nHi<end_of_turn>", resp.prompt)
        self.assertIn("<start_of_turn>model\n", resp.prompt)

    def test_context_and_model_properties(self):
        # Attribute access
        self.assertEqual(self.llm.n_ctx, 256)
        self.assertGreater(self.llm.n_embd, 0)
        self.assertGreater(self.llm.n_vocab, 0)
        # Callable method access (parity with llama_cpp.Llama)
        self.assertEqual(self.llm.n_ctx(), 256)
        self.assertGreater(self.llm.n_embd(), 0)
        self.assertGreater(self.llm.n_vocab(), 0)
        self.assertIsNotNone(self.llm.model)
        self.assertIsNotNone(self.llm.ctx)
        self.llm.set_seed(4242)
        self.assertEqual(self.llm.seed, 4242)

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

    def test_logits_processor_and_logit_bias(self):
        from llamojo import LogitsProcessorList

        banana_toks = self.llm.tokenize("banana", add_bos=False)
        b_tok = banana_toks[0]

        # 1. Direct logit_bias in create_completion
        res = self.llm("Once upon a", max_tokens=1, temperature=0.0, logit_bias={b_tok: 100.0})
        self.assertEqual(res["choices"][0]["text"], self.llm.token_to_piece(b_tok))

        # 2. Negative logit_bias to forbid most likely token
        res_normal = self.llm("1 2 3", max_tokens=1, temperature=0.0)
        tok_normal = self.llm.tokenize(res_normal["choices"][0]["text"], add_bos=False)[0]
        res_forbidden = self.llm("1 2 3", max_tokens=1, temperature=0.0, logit_bias={tok_normal: -100.0})
        self.assertNotEqual(res_forbidden["choices"][0]["text"], res_normal["choices"][0]["text"])

        # 3. Custom LogitsProcessorList
        def boost_processor(input_ids, scores):
            scores[b_tok] += 500.0
            return scores

        procs = LogitsProcessorList([boost_processor])
        res_proc = self.llm("The secret word is", max_tokens=1, temperature=0.0, logits_processor=procs)
        self.assertEqual(res_proc["choices"][0]["text"], self.llm.token_to_piece(b_tok))

        # 4. Streaming with logit_bias
        chunks = list(self.llm("Once upon a", max_tokens=1, temperature=0.0, logit_bias={b_tok: 100.0}, stream=True))
        self.assertEqual(chunks[0]["choices"][0]["text"], self.llm.token_to_piece(b_tok))

    def test_logprob_computation(self):
        # 1. Non-streaming logprobs
        res = self.llm("The sun rises in the", max_tokens=3, temperature=0.0, logprobs=5)
        lp = res["choices"][0]["logprobs"]
        self.assertIsNotNone(lp)
        self.assertEqual(len(lp["tokens"]), 3)
        self.assertEqual(len(lp["token_logprobs"]), 3)
        self.assertEqual(len(lp["top_logprobs"]), 3)
        self.assertGreaterEqual(len(lp["top_logprobs"][0]), 5)
        for val in lp["token_logprobs"]:
            self.assertLessEqual(val, 0.0)

        # 2. Streaming logprobs
        chunks = list(self.llm("The sun rises in the", max_tokens=2, temperature=0.0, logprobs=3, stream=True))
        first_lp = chunks[0]["choices"][0]["logprobs"]
        self.assertIsNotNone(first_lp)
        self.assertEqual(len(first_lp["tokens"]), 1)
        self.assertEqual(len(first_lp["token_logprobs"]), 1)
        self.assertGreaterEqual(len(first_lp["top_logprobs"][0]), 3)


if __name__ == "__main__":
    unittest.main()
