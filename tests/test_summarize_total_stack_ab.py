from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_total_stack_ab import summarize


def write_run(
    path: Path,
    *,
    ttft: float,
    throughput: float,
    sample_throughputs: tuple[float, float],
    token_counts: tuple[int, int] = (10, 12),
) -> None:
    answers = []
    for index, (sample_tps, token_count) in enumerate(
        zip(sample_throughputs, token_counts)
    ):
        answers.append(
            {
                "question_id": str(index),
                "parsed_answer": "A",
                "correct": True,
                "ttft_ms": ttft,
                "throughput_tokens_per_sec": sample_tps,
                "token_count": token_count,
                "meta": {"ppu_gdn_patched_modules": 18},
            }
        )
    payload = {
        "backend": "transformers",
        "sample_count": 2,
        "performance": {
            "avg_ttft_ms": ttft,
            "avg_throughput_tokens_per_sec": throughput,
        },
        "accuracy": {"score": 1.0, "correct": 2, "total": 2},
        "public_validation": {"passed": True},
        "answers": answers,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class SummarizeTotalStackABTest(unittest.TestCase):
    def test_repeated_cross_process_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("e1.json", "e2.json", "c1.json", "c2.json")]
            write_run(paths[0], ttft=10.0, throughput=50.0, sample_throughputs=(40, 60))
            write_run(paths[1], ttft=12.0, throughput=52.0, sample_throughputs=(42, 62))
            write_run(paths[2], ttft=9.0, throughput=100.0, sample_throughputs=(80, 120))
            write_run(paths[3], ttft=11.0, throughput=104.0, sample_throughputs=(84, 124))

            payload = summarize(paths[:2], paths[2:])

            self.assertAlmostEqual(payload["aggregate"]["throughput_speedup"], 2.0)
            self.assertAlmostEqual(
                payload["aggregate"]["throughput_improvement_percent"], 100.0
            )
            self.assertEqual(payload["aggregate"]["per_sample_wins"], 2)
            self.assertEqual(payload["consistency"]["same_parsed_answer_all_runs"], 2)
            self.assertEqual(payload["consistency"]["same_token_count_all_runs"], 2)
            self.assertTrue(payload["passed"])

    def test_token_drift_is_reported_without_faking_text_exactness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("e1.json", "e2.json", "c1.json", "c2.json")]
            write_run(paths[0], ttft=10, throughput=50, sample_throughputs=(40, 60))
            write_run(paths[1], ttft=10, throughput=50, sample_throughputs=(40, 60))
            write_run(
                paths[2],
                ttft=10,
                throughput=75,
                sample_throughputs=(60, 90),
                token_counts=(10, 13),
            )
            write_run(
                paths[3],
                ttft=10,
                throughput=75,
                sample_throughputs=(60, 90),
                token_counts=(10, 13),
            )

            payload = summarize(paths[:2], paths[2:])

            self.assertEqual(payload["consistency"]["same_token_count_all_runs"], 1)
            self.assertEqual(
                payload["consistency"]["strict_full_text_comparison"],
                "unavailable_in_benchmark_public_output",
            )

    def test_labels_repeated_abba_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eager = root / "eager.json"
            candidate = root / "candidate.json"
            write_run(eager, ttft=10, throughput=50, sample_throughputs=(40, 60))
            write_run(
                candidate,
                ttft=9,
                throughput=100,
                sample_throughputs=(80, 120),
            )

            payload = summarize(
                [eager, eager, eager, eager],
                [candidate, candidate, candidate, candidate],
            )

            self.assertEqual(
                payload["method"]["process_order"],
                "2 independent ABBA blocks",
            )


if __name__ == "__main__":
    unittest.main()
