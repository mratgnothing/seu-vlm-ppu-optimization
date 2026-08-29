from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs" / "work-summary" / "README.md"


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class WorkSummaryEvidenceTest(unittest.TestCase):
    def test_eager_baseline_matches_source(self) -> None:
        result = load_json("results/ppu-eager-cn20-baseline-summary-20260826.json")
        self.assertEqual(result["sample_count"], 20)
        self.assertEqual(result["performance"]["avg_ttft_ms"], 118.493)
        self.assertEqual(
            result["performance"]["avg_throughput_tokens_per_sec"], 49.737
        )
        self.assertEqual(result["accuracy"]["correct"], 17)
        self.assertTrue(result["public_validation"]["passed"])

    def assert_full_gate(
        self,
        path: str,
        candidate_key: str,
        expected_median: float,
        expected_correct: int,
    ) -> None:
        result = load_json(path)
        self.assertEqual(result["sample_count"], 4029)
        self.assertTrue(result["passed"])
        self.assertEqual(result["baseline"]["correct"], expected_correct)
        self.assertEqual(result[candidate_key]["correct"], expected_correct)
        self.assertEqual(result["pair_consistency"]["exact_text"], 4029)
        self.assertEqual(result["pair_consistency"]["same_answer"], 4029)
        self.assertEqual(result["pair_consistency"]["same_token_count"], 4029)
        self.assertTrue(
            math.isclose(
                result["paired_decode"]["median_speedup"],
                expected_median,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_full_dataset_incremental_claims(self) -> None:
        self.assert_full_gate(
            "results/gate-prep-scratch-cn-full4029-summary.json",
            "gdn_gate_prep",
            1.086228841490808,
            3374,
        )
        self.assert_full_gate(
            "results/acblas-packed-mlp-cn-full4029-summary.json",
            "acblas_packed_mlp",
            1.112471881143032,
            3374,
        )
        self.assert_full_gate(
            "results/acblas-packed-mlp-en-full4029-summary-20260828.json",
            "acblas_packed_mlp",
            1.109344697983514,
            3214,
        )
        self.assert_full_gate(
            "results/raw-stream-query-cn4029-summary-20260828.json",
            "raw_stream_query",
            1.090628923196299,
            3374,
        )
        self.assert_full_gate(
            "results/raw-stream-query-en4029-summary-20260828.json",
            "raw_stream_query",
            1.0900797570311287,
            3214,
        )

    def test_negative_results_are_not_presented_as_wins(self) -> None:
        swiglu = load_json("results/ppu-swiglu-thread-sweep-negative-20260827.json")
        acblas = load_json("results/ppu-acblas-ab128-final-20260827.json")
        acblaslt = load_json("results/acblaslt-square-ab128-20260828.json")
        attention_r2 = load_json(
            "results/acblas-attention-prep-cn20-r2-20260828.json"
        )
        scratch = load_json("results/residual-rmsnorm-scratch-ab128-20260828.json")
        graph = load_json("results/acblas-packed-mlp-graph-smoke-20260828.json")

        self.assertLess(swiglu["best_speedup"], 1.0)
        self.assertFalse(swiglu["integrated_into_model"])
        self.assertLess(acblas["speedup"]["paired_decode_median"], 1.0)
        self.assertLess(acblaslt["paired_decode"]["median_speedup"], 1.0)
        self.assertLess(attention_r2["paired_decode"]["median_speedup"], 1.0)
        self.assertFalse(attention_r2["performance_passed"])
        self.assertLess(scratch["paired_decode"]["median_speedup"], 1.0)
        self.assertLess(graph["graph_with_input_copy_speedup"], 1.0)

    def test_summary_links_every_primary_evidence_file(self) -> None:
        text = SUMMARY.read_text(encoding="utf-8")
        for filename in (
            "ppu-eager-cn20-baseline-summary-20260826.json",
            "gate-prep-scratch-cn-full4029-summary.json",
            "acblas-packed-mlp-cn-full4029-summary.json",
            "acblas-packed-mlp-en-full4029-summary-20260828.json",
            "raw-stream-query-cn4029-summary-20260828.json",
            "raw-stream-query-en4029-summary-20260828.json",
            "ppu-swiglu-thread-sweep-negative-20260827.json",
            "acblas-attention-prep-cn20-r2-20260828.json",
            "residual-rmsnorm-scratch-ab128-20260828.json",
        ):
            self.assertIn(filename, text)

    def test_all_relative_markdown_links_resolve(self) -> None:
        text = SUMMARY.read_text(encoding="utf-8")
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        self.assertGreater(len(targets), 20)
        for target in targets:
            if "://" in target or target.startswith("#"):
                continue
            path_text = target.split("#", 1)[0]
            resolved = (SUMMARY.parent / path_text).resolve()
            self.assertTrue(resolved.exists(), f"broken Markdown link: {target}")


if __name__ == "__main__":
    unittest.main()
