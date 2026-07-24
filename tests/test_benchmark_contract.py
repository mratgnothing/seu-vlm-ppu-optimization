from __future__ import annotations

import unittest

from benchmark_public import compute_throughput, extract_answer, validate_public_result


class BenchmarkContractTest(unittest.TestCase):
    def test_extracts_supported_answer_forms(self) -> None:
        cases = {
            "Answer: A": "A",
            "Final answer is (B).": "B",
            "答案为：C": "C",
            "D. short reason": "D",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text), expected)

    def test_throughput_matches_official_formula(self) -> None:
        self.assertAlmostEqual(
            compute_throughput(token_count=11, ttft_seconds=0.5, elapsed_seconds=2.5),
            5.0,
        )

    def test_validation_rejects_ambiguous_output(self) -> None:
        errors = validate_public_result(
            text="I am not sure.",
            parsed_answer=None,
            token_count=4,
            max_new_tokens=256,
        )
        self.assertIn("missing_choice_answer", errors)


if __name__ == "__main__":
    unittest.main()

