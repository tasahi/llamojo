""" Tests for the Hugging Face-style facade, using a fake Llama (no model needed)."""
import unittest

from llamojo.hf_compat import LlamojoForCausalLM, LlamojoForSpeechSeq2Seq


class FakeLlama:
    """ Deterministic stand-in: next token = (last token + 1) % 10; eos = 9."""

    n_vocab = 10

    def __init__(self):
        self.input_ids = []
        self.scores = None
        self.resets = 0

    def token_eos(self):
        return 9

    def token_bos(self):
        return 1

    def reset(self):
        self.input_ids = []
        self.resets += 1

    def eval(self, tokens):
        self.input_ids.extend(tokens)
        nxt = (self.input_ids[-1] + 1) % 10
        self.scores = [0.0] * 10
        self.scores[nxt] = 5.0

    def sample(self, top_k=40, top_p=0.95, temp=0.8, **kw):
        s = self.scores
        return max(range(len(s)), key=lambda i: s[i])

    def tokenize(self, text, add_bos=True, special=False):
        return ([1] if add_bos else []) + [b % 8 for b in text]

    def detokenize(self, tokens, special=False):
        return "".join(chr(97 + t) for t in tokens).encode()

    def set_seed(self, seed):
        self.seed = seed

    def close(self):
        pass


class TestHFCompat(unittest.TestCase):
    def setUp(self):
        self.fake = FakeLlama()
        self.model = LlamojoForCausalLM(self.fake)

    def test_greedy_generate_flat_list(self):
        out = self.model.generate([1, 2], max_new_tokens=3, do_sample=False)
        self.assertEqual(out, [1, 2, 3, 4, 5])

    def test_stops_at_eos_and_includes_it(self):
        out = self.model.generate([6], max_new_tokens=10, do_sample=False)
        self.assertEqual(out, [6, 7, 8, 9])

    def test_batch_rows_are_independent(self):
        out = self.model.generate([[1], [4]], max_new_tokens=2, do_sample=False)
        self.assertEqual(out, [[1, 2, 3], [4, 5, 6]])
        self.assertGreaterEqual(self.fake.resets, 2)

    def test_forward_returns_last_token_logits(self):
        res = self.model([[1, 2], [5]])
        logits = res.logits
        self.assertEqual(len(logits), 2)
        self.assertEqual(len(logits[0]), 10)
        self.assertEqual(int(max(range(10), key=lambda i: logits[0][i])), 3)
        self.assertEqual(int(max(range(10), key=lambda i: logits[1][i])), 6)
        self.assertIs(res[0], res.logits)

    def test_unsupported_args_raise(self):
        with self.assertRaises(NotImplementedError):
            self.model.generate([1], num_beams=2)
        with self.assertRaises(NotImplementedError):
            self.model.generate([1], repetition_penalty=1.2)
        with self.assertRaises(ValueError):
            self.model.generate([])

    def test_tokenizer_roundtrip(self):
        tok = self.model.tokenizer
        batch = tok(["ab", "c"])
        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(batch["attention_mask"][1], [1] * len(batch["input_ids"][1]))
        self.assertEqual(tok.decode([0, 1, 2]), "abc")
        self.assertEqual(tok.eos_token_id, 9)

    def test_torch_roundtrip_if_available(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")
        ids = torch.tensor([[1, 2]])
        out = self.model.generate(ids, max_new_tokens=2, do_sample=False)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.tolist(), [[1, 2, 3, 4]])


class FakeWhisper:
    """ Deterministic stand-in for llamojo.Whisper."""

    def __init__(self):
        self.closed = False

    def transcribe(self, audio, no_timestamps=False, translate=False):
        return {
            "text": " And so my fellow Americans.",
            "segments": [
                {
                    "t0": 0,
                    "t1": 350,
                    "start": 0.0,
                    "end": 3.5,
                    "text": " And so my fellow Americans.",
                }
            ],
        }

    def close(self):
        self.closed = True


class TestSpeechSeq2SeqCompat(unittest.TestCase):
    def test_speech_seq2seq_transcribe_and_chunks(self):
        fake_w = FakeWhisper()
        asr = LlamojoForSpeechSeq2Seq(fake_w)
        res = asr("jfk.wav", return_timestamps=True)
        self.assertEqual(res["text"], " And so my fellow Americans.")
        self.assertEqual(len(res["chunks"]), 1)
        self.assertEqual(res["chunks"][0]["timestamp"], (0.0, 3.5))
        asr.close()
        self.assertTrue(fake_w.closed)

    def test_resolve_path_missing_leading_slash_and_load_audio_16k(self):
        from llamojo.whisper import _resolve_path, load_audio_16k

        # Missing leading slash on 'mnt/c/...' should resolve to '/mnt/c/...'
        resolved = _resolve_path("mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav")
        self.assertEqual(resolved, "/mnt/c/Documents/Programming/whisper.cpp/samples/jfk.wav")

        pcm = load_audio_16k(resolved)
        self.assertEqual(len(pcm), 176000)


if __name__ == "__main__":
    unittest.main()
