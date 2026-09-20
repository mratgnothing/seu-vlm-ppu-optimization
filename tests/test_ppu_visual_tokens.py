from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "benchmark_ppu_visual_tokens_ab.py"
WRAPPER = ROOT / "evaluation_wrapper.py"


class PPUVisualTokenContractTest(unittest.TestCase):
    def test_visual_cap_is_opt_in_and_recorded(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('os.getenv("SEU_VISION_MAX_PIXELS")', text)
        self.assertIn('"vision_max_pixels": self._vision_max_pixels', text)
        self.assertIn('"visual_tokens": visual_tokens', text)

    def test_benchmark_prefers_checkout_wrapper(self) -> None:
        text = BENCHMARK.read_text(encoding="utf-8")
        self.assertIn("for path in (custom_ops, repo_root)", text)
        self.assertIn("from evaluation_wrapper import VLMModel", text)

    def test_sparse_reduction_uses_mean_and_changed_count(self) -> None:
        text = BENCHMARK.read_text(encoding="utf-8")
        self.assertIn("changed_visual_pairs > 0", text)
        self.assertIn("statistics.fmean(visual_ratios) < 1.0", text)
        self.assertIn('"changed_visual_pairs": changed_visual_pairs', text)


if __name__ == "__main__":
    unittest.main()
