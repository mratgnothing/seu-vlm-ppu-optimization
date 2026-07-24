import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MICROBENCH = ROOT / "ppu" / "microbench"


class PPUMicrobenchContractTest(unittest.TestCase):
    def test_qwen35_suite_contains_the_three_profiled_decode_shapes(self) -> None:
        script = (MICROBENCH / "run_qwen35_suite.sh").read_text(encoding="utf-8")

        self.assertIn("run_shape 6144 2048", script)
        self.assertIn("run_shape 2048 6144", script)
        self.assertIn("run_shape 2048 2048", script)

    def test_microbench_preserves_timing_correctness_and_machine_output_contract(
        self,
    ) -> None:
        source = (MICROBENCH / "qwen35_bf16_gemv.hg").read_text(encoding="utf-8")

        self.assertIn("hggcEventElapsedTime", source)
        self.assertIn("__ppu_bfloat16", source)
        self.assertIn("__bfloat162float", source)
        self.assertIn("failed_values", source)
        self.assertIn('RESULT {\\"kernel\\":\\"bf16_gemv_reference\\"', source)

    def test_build_script_uses_the_verified_linker_rpath_form(self) -> None:
        script = (MICROBENCH / "build.sh").read_text(encoding="utf-8")

        self.assertIn('-Wl,-rpath,"${PPU_SDK_ROOT}/lib"', script)
        self.assertNotIn("\n  -rpath", script)


if __name__ == "__main__":
    unittest.main()
