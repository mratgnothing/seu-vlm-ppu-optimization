from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_ppu_gemv import parse_records, render_markdown, summarize


class SummarizePpuGemvTest(unittest.TestCase):
    def test_computes_median_and_reference_improvement(self) -> None:
        lines = [
            'RESULT {"kernel":"bf16_gemv_reference","n":8,"k":4,"threads":256,"matrix_copies":16,"average_ms":2.0,"passed":true}',
            'RESULT {"kernel":"bf16_gemv_reference","n":8,"k":4,"threads":256,"matrix_copies":16,"average_ms":2.2,"passed":true}',
            'RESULT {"kernel":"bf16_gemv_warp_vec2","n":8,"k":4,"threads":64,"matrix_copies":16,"average_ms":1.0,"passed":true}',
            'RESULT {"kernel":"bf16_gemv_warp_vec2","n":8,"k":4,"threads":64,"matrix_copies":16,"average_ms":1.2,"passed":true}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text("\n".join(lines), encoding="utf-8")
            rows = summarize(parse_records([path]))

        optimized = next(row for row in rows if row["threads"] == 64)
        self.assertAlmostEqual(optimized["median_ms"], 1.1)
        self.assertAlmostEqual(optimized["reference_ms"], 2.1)
        self.assertAlmostEqual(optimized["speedup"], 2.1 / 1.1)
        self.assertIn("47.62%", render_markdown(rows))


if __name__ == "__main__":
    unittest.main()
