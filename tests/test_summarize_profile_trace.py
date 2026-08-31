from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_profile_trace import summarize_trace


class SummarizeProfileTraceTest(unittest.TestCase):
    def test_groups_selected_operator_input_shapes(self) -> None:
        events = [
            {
                "ph": "X",
                "name": "aten::mul",
                "cat": "cpu_op",
                "dur": 4.0,
                "args": {"Input Dims": [[1, 2048], [1, 2048]]},
            },
            {
                "ph": "X",
                "name": "aten::mul",
                "cat": "cpu_op",
                "dur": 6.0,
                "args": {"Input Dims": [[1, 2048], [1, 2048]]},
            },
            {
                "ph": "X",
                "name": "aten::mm",
                "cat": "cpu_op",
                "dur": 8.0,
                "args": {"Input Dims": [[1, 2048], [2048, 2048]]},
            },
        ]

        summary = summarize_trace(
            events,
            top=10,
            shape_ops={"aten::mm", "aten::mul"},
        )

        mul = summary["operator_shapes"]["aten::mul"][0]
        self.assertEqual(mul["input_dims"], "[[1,2048],[1,2048]]")
        self.assertEqual(mul["count"], 2)
        self.assertEqual(mul["duration_ms"], 0.01)
        self.assertEqual(summary["aten_mm_shapes"][0]["count"], 1)

    def test_tracks_runtime_overhead_events(self) -> None:
        events = [
            {
                "ph": "X",
                "name": "cudaGetDeviceProperties_v2",
                "cat": "cuda_runtime",
                "dur": 3.0,
            },
            {
                "ph": "X",
                "name": "cudaFree",
                "cat": "cuda_runtime",
                "dur": 5.0,
            },
        ]

        summary = summarize_trace(events, top=10, shape_ops=set())

        self.assertEqual(
            summary["tracked_events"]["cudaGetDeviceProperties_v2"]["count"], 1
        )
        self.assertEqual(summary["tracked_events"]["cudaFree"]["duration_ms"], 0.005)


if __name__ == "__main__":
    unittest.main()
