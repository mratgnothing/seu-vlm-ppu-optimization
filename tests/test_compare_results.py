from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_results import compare_payloads


def _payload(ttft: float, throughput: float, answer: str = "B") -> dict:
    return {
        "performance": {
            "avg_ttft_ms": ttft,
            "avg_throughput_tokens_per_sec": throughput,
        },
        "accuracy": {"score": 1.0},
        "public_validation": {"passed": True},
        "answers": [
            {
                "question_id": "1",
                "parsed_answer": answer,
                "token_count": 12,
                "ttft_ms": ttft,
                "throughput_tokens_per_sec": throughput,
            }
        ],
    }


class CompareResultsTest(unittest.TestCase):
    def test_computes_official_improvement_rates(self) -> None:
        result = compare_payloads(
            _payload(ttft=400.0, throughput=20.0),
            _payload(ttft=300.0, throughput=25.0),
        )
        self.assertAlmostEqual(result["ttft_ms"]["improvement_rate"], 0.25)
        self.assertAlmostEqual(
            result["throughput_tokens_per_sec"]["improvement_rate"],
            0.25,
        )
        self.assertTrue(result["sample_contract"]["same_question_ids"])
        self.assertEqual(result["sample_contract"]["changed_parsed_answers"], [])

    def test_reports_answer_drift(self) -> None:
        result = compare_payloads(
            _payload(ttft=400.0, throughput=20.0, answer="B"),
            _payload(ttft=300.0, throughput=25.0, answer="C"),
        )
        self.assertEqual(len(result["sample_contract"]["changed_parsed_answers"]), 1)


if __name__ == "__main__":
    unittest.main()
