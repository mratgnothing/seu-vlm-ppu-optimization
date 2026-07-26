from __future__ import annotations

import unittest

from benchmark_public import compute_throughput, extract_answer, validate_public_result
from evaluation_wrapper import (
    normalize_choice_markup,
    normalize_explicit_choice_conclusion,
)


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

    def test_normalizes_markdown_wrapped_choice_without_changing_semantics(self) -> None:
        cases = {
            "正确答案是：**B**\n\n理由：实验比较有蜡和无蜡。": "B",
            "正确答案是：**B. 使用白面粉制作的松饼体积更大。**": "B",
            "Answer: _C. The third option._": "C",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                normalized = normalize_choice_markup(raw)
                self.assertEqual(extract_answer(normalized), expected)

    def test_normalizes_one_explicit_bolded_conclusion(self) -> None:
        raw = (
            "- A: skips required values, so it doesn't match.\n"
            "- B: includes an extra value, so it doesn't match.\n"
            "- C: covers exactly 0 through 8 — **matches exactly**.\n"
            "- D: starts from the wrong value."
        )
        normalized = normalize_explicit_choice_conclusion(raw)

        self.assertTrue(normalized.endswith("Answer: C"))
        self.assertEqual(extract_answer(normalized), "C")

    def test_does_not_normalize_ambiguous_or_unbolded_reasoning(self) -> None:
        cases = [
            "- A: **is correct**.\n- B: **matches exactly**.",
            "- C: matches exactly.",
            "The code in option C appears suitable.",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_explicit_choice_conclusion(raw), raw)

    def test_preserves_existing_canonical_answer(self) -> None:
        raw = "Answer: B\n- C: the value **matches exactly**."
        self.assertEqual(normalize_explicit_choice_conclusion(raw), raw)

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
