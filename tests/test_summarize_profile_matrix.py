from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_profile_matrix import summarize_payloads


def _payload(profile: str, ttft: float, throughput: float) -> dict:
    return {
        "dataset_path": "/data/mmbench_dev_cn.tsv",
        "sample_count": 1,
        "performance": {
            "avg_ttft_ms": ttft,
            "avg_throughput_tokens_per_sec": throughput,
        },
        "accuracy": {"score": 1.0},
        "public_validation": {"passed": True},
        "answers": [{
            "question_id": "1",
            "parsed_answer": "B",
            "token_count": 8,
            "meta": {"optimization_profile": profile},
        }],
    }


class SummarizeProfileMatrixTest(unittest.TestCase):
    def test_uses_run_medians_and_official_improvement_formulas(self) -> None:
        payloads = [
            ("o0-r1.json", _payload("o0_no_grad", 400.0, 20.0)),
            ("o0-r2.json", _payload("o0_no_grad", 420.0, 19.0)),
            (
                "o1-r1.json",
                _payload("o1_inference_mode", 300.0, 25.0),
            ),
            (
                "o1-r2.json",
                _payload("o1_inference_mode", 320.0, 24.0),
            ),
        ]
        result = summarize_payloads(payloads)
        comparison = result["comparisons"][0]
        self.assertAlmostEqual(
            comparison["ttft_improvement_rate"],
            1.0 - 310.0 / 410.0,
        )
        self.assertAlmostEqual(
            comparison["throughput_improvement_rate"],
            24.5 / 19.5 - 1.0,
        )
        self.assertTrue(
            all(
                group["stable_sample_answers_and_tokens"]
                for group in result["groups"]
            )
        )


if __name__ == "__main__":
    unittest.main()
