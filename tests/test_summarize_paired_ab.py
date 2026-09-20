import json
import hashlib
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
            "raw_stream_query_ab_enabled": True,
            "packed_gdn_modules": 18,
            "residual_rmsnorm_modules": 24,
            "gdn_gate_prep_modules": 18,
            "acblas_packed_mlp_modules": 24,
            "acblas_attention_prep_modules": 6,
            "performance_gate_required": True,
            "performance_passed": False,
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
            expected_source_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
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
        self.assertEqual(summary["module_counts"]["acblas_packed_mlp"], 24)
        self.assertEqual(summary["module_counts"]["acblas_attention_prep"], 6)
        self.assertTrue(summary["raw_stream_query_ab_enabled"])
        self.assertTrue(summary["performance_gate_required"])
        self.assertFalse(summary["performance_passed"])
        self.assertEqual(
            summary["source_sha256"],
            expected_source_sha256,
        )

    def test_classifies_answer_equivalent_non_exact_candidate(self) -> None:
        def record(speed: float, text: str) -> dict[str, object]:
            return {
                "throughput_tokens_per_sec": speed,
                "correct": True,
                "text_sha256": text,
                "answer": "A",
                "token_count": 8,
            }

        payload = {
            "sample_offset": 0,
            "sample_count": 2,
            "max_new_tokens": 64,
            "ab_target": "acblas_gdn_single_gemv",
            "projection_backend": "acblas-grouped",
            "raw_stream_query_ab_enabled": False,
            "packed_gdn_modules": 18,
            "residual_rmsnorm_modules": 24,
            "gdn_gate_prep_modules": 18,
            "acblas_packed_mlp_modules": 24,
            "acblas_attention_prep_modules": 0,
            "performance_gate_required": True,
            "performance_passed": True,
            "baseline": {
                "avg_ttft_ms": 10.0,
                "avg_throughput_tokens_per_sec": 100.0,
                "accuracy": 1.0,
                "records": [record(100.0, "a"), record(100.0, "b")],
            },
            "acblas_gdn_single_gemv": {
                "avg_ttft_ms": 10.0,
                "avg_throughput_tokens_per_sec": 103.0,
                "accuracy": 1.0,
                "records": [record(103.0, "a"), record(103.0, "changed")],
            },
            "passed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "summary.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 1)
        self.assertFalse(summary["decision_gates"]["strict_bit_exact_passed"])
        self.assertTrue(
            summary["decision_gates"]["answer_accuracy_budget_passed"]
        )
        self.assertEqual(
            summary["decision_gates"]["recommended_profile"],
            "performance_accuracy_budget",
        )
        self.assertEqual(
            summary["decision_gates"]["raw_stream_query_mode"],
            "enabled_both_arms",
        )


if __name__ == "__main__":
    unittest.main()
