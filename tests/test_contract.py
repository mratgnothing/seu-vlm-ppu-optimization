from __future__ import annotations

import unittest

from evaluation_wrapper import GenerationConfig, VLMModel


class WrapperContractTest(unittest.TestCase):
    def test_dummy_backend_returns_valid_metrics(self) -> None:
        model = VLMModel("unused", backend="dummy")
        result = model.generate_with_metrics(
            image=None,
            prompt="Choose one option.",
            choices={"A": "one", "B": "two", "C": "three", "D": "four"},
            generation_config=GenerationConfig(max_new_tokens=32),
            sample_id="smoke-1",
        )

        self.assertEqual(model.backend_name, "dummy")
        self.assertTrue(result.text)
        self.assertGreater(result.token_count, 0)
        self.assertGreater(result.ttft_seconds, 0)
        self.assertGreaterEqual(result.elapsed_seconds, result.ttft_seconds)
        self.assertEqual(result.meta["backend"], "dummy")


if __name__ == "__main__":
    unittest.main()

