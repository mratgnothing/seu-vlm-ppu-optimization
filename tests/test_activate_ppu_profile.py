from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "activate_ppu_profile.sh"


class ActivatePPUProfileContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_requires_sourcing_and_known_profile(self) -> None:
        self.assertIn('${BASH_SOURCE[0]}" == "$0', self.text)
        self.assertIn("precision|performance|experimental-single", self.text)
        self.assertIn("Unknown PPU profile", self.text)

    def test_enables_every_validated_precision_component(self) -> None:
        for variable in (
            "SEU_PPU_GDN_LIBRARY",
            "SEU_PPU_CONV_ENABLE",
            "SEU_PPU_RMSNORM_ENABLE",
            "SEU_PPU_GATED_RMSNORM_ENABLE",
            "SEU_PPU_QK_ROPE_ENABLE",
            "SEU_PPU_PACK_MLP_ENABLE",
            "SEU_PPU_RESIDUAL_RMSNORM_ENABLE",
            "SEU_PPU_GDN_GATE_PREP_ENABLE",
            "SEU_PPU_RAW_STREAM_QUERY_ENABLE",
            "SEU_PPU_ACBLAS_GDN_BUILD_DIR",
            "SEU_PPU_ACBLAS_PACKED_MLP_BUILD_DIR",
        ):
            self.assertIn(f"export {variable}=", self.text)

    def test_experimental_profile_only_adds_single_gemv(self) -> None:
        self.assertIn('if [[ "${_seu_profile}" == "experimental-single" ]]', self.text)
        self.assertIn("export SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE=1", self.text)
        self.assertIn("export SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE=0", self.text)
        self.assertIn("unset SEU_PPU_ACBLAS_ATTENTION_PREP_BUILD_DIR", self.text)
        self.assertIn("lost one correct answer", self.text)

    def test_performance_profile_only_adds_ba_gemv(self) -> None:
        self.assertIn('if [[ "${_seu_profile}" == "performance" ]]', self.text)
        self.assertIn("export SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE=1", self.text)
        self.assertIn("Bilingual MMBench 4029/4029 exact", self.text)


if __name__ == "__main__":
    unittest.main()
