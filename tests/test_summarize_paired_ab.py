import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_paired_ab.py"


class SummarizePairedABTest(unittest.TestCase):
    def test_recomputes_accuracy_speed_and_pair_consistency(self) -> None:
        baseline_records = [
            {
                "throughput_tokens_per_sec": 100.0,
                "correct": True,
                "text_sha256": "same-a",
                "answer": "A",
                "token_count": 8,
            },
            {
                "throughput_tokens_per_sec": 80.0,
                "correct": False,
                "text_sha256": "same-b",
                "answer": "B",
                "token_count": 9,
            },
        ]
        candidate_records = [
            {
                "throughput_tokens_per_sec": 110.0,
                "correct": True,
                "text_sha256": "same-a",
                "answer": "A",
                "token_count": 8,
            },
            {
                "throughput_tokens_per_sec": 72.0,
                "correct": False,
                "text_sha256": "same-b",
                "answer": "B",
                "token_count": 9,
            },
        ]
        payload = {
            "sample_offset": 0,
            "sample_count": 2,
            "max_new_tokens": 64,
            "ab_target": "candidate",
            "projection_backend": "test",
            "packed_gdn_modules": 18,
            "residual_rmsnorm_modules": 24,
            "gdn_gate_prep_modules": 18,
            "baseline": {
                "avg_ttft_ms": 10.0,
                "avg_throughput_tokens_per_sec": 90.0,
                "accuracy": 0.5,
                "records": baseline_records,
            },
            "candidate": {
                "avg_ttft_ms": 11.0,
                "avg_throughput_tokens_per_sec": 91.0,
                "accuracy": 0.5,
                "records": candidate_records,
            },
            "passed": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "summary.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertAlmostEqual(summary["paired_decode"]["median_speedup"], 1.0)
        self.assertEqual(summary["paired_decode"]["wins"], 1)
        self.assertEqual(summary["paired_decode"]["losses"], 1)
        self.assertEqual(summary["baseline"]["correct"], 1)
        self.assertEqual(summary["candidate"]["correct"], 1)
        self.assertEqual(summary["pair_consistency"]["exact_text"], 2)
        self.assertEqual(summary["pair_consistency"]["same_answer"], 2)
        self.assertEqual(summary["pair_consistency"]["same_token_count"], 2)


if __name__ == "__main__":
    unittest.main()
