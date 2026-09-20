from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "evaluation_wrapper.py"
GDN = ROOT / "ppu" / "custom_ops" / "ppu_gdn.py"
PROFILE = ROOT / "scripts" / "profile_ppu_first_token.py"


class PPUPrefillRowFusionContractTest(unittest.TestCase):
    def test_candidate_is_opt_in(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('SEU_PPU_PREFILL_ROW_FUSIONS_ENABLE", "0"', text)
        self.assertIn('"ppu_prefill_row_fusions_enabled"', text)

    def test_residual_fusion_accepts_multirow_only_when_enabled(self) -> None:
        text = GDN.read_text(encoding="utf-8")
        self.assertIn("_seu_prefill_residual_rmsnorm_enabled", text)
        self.assertIn("hidden_states.shape[0] == 1", text)

    def test_profiler_generates_exactly_one_token(self) -> None:
        text = PROFILE.read_text(encoding="utf-8")
        self.assertIn('"max_new_tokens": 1', text)
        self.assertIn('"profile_scope": "one warm multimodal prefill producing first token"', text)


if __name__ == "__main__":
    unittest.main()
