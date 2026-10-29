import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_policy_suite import point_in_box, summarize  # noqa: E402


class EvaluatePolicySuiteTest(unittest.TestCase):
    def test_point_in_box_includes_edges(self):
        self.assertTrue(point_in_box([20, 30], [20, 30, 40, 50]))
        self.assertFalse(point_in_box([41, 30], [20, 30, 40, 50]))

    def test_summary_counts_metrics(self):
        cases = [
            {
                "parsed": True,
                "gate_match": True,
                "raw_hit": False,
                "expected_gate": "ACT",
                "effective_grounding": True,
                "handled_as_expected": True,
                "execution_ready": True,
            },
            {
                "parsed": True,
                "gate_match": True,
                "raw_hit": None,
                "expected_gate": "HANDOFF",
                "effective_grounding": True,
                "handled_as_expected": True,
                "execution_ready": False,
            },
        ]
        result = summarize(cases)
        self.assertEqual(result["gate_matches"], 2)
        self.assertEqual(result["raw_grounding_total"], 1)
        self.assertEqual(result["handled_as_expected"], 2)
        self.assertEqual(result["execution_ready"], 1)


if __name__ == "__main__":
    unittest.main()
