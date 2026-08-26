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
        self.assertIn('RESULT {\\"kernel\\":\\"', source)
        self.assertIn('return "bf16_gemv_reference"', source)
        self.assertIn('return "bf16_gemv_warp"', source)
        self.assertIn('return "bf16_gemv_warp_vec2"', source)
        self.assertIn("--matrix-copies", source)
        self.assertIn('\\"matrix_copies\\"', source)

    def test_sweep_covers_warp_reduction_vectorization_and_thread_counts(self) -> None:
        script = (MICROBENCH / "sweep_qwen35_gemv.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("warp warp_vec2", script)
        self.assertIn("64 128 256 512", script)
        for shape in ("6144 2048", "2048 6144", "2048 2048"):
            self.assertIn(shape, script)

    def test_build_script_uses_the_verified_linker_rpath_form(self) -> None:
        script = (MICROBENCH / "build.sh").read_text(encoding="utf-8")

        self.assertIn('-Wl,-rpath,"${PPU_SDK_ROOT}/lib"', script)
        self.assertIn('HGGC_RUNTIME_LIBRARY="${HGGC_RUNTIME_LIBRARY:-hggcrt1}"', script)
        self.assertIn('-l"${HGGC_RUNTIME_LIBRARY}"', script)
        self.assertNotIn("\n  -rpath", script)

    def test_repeated_benchmark_compares_reference_candidate_and_torch(self) -> None:
        script = (MICROBENCH / "run_repeated_benchmark.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('REPEATS="${REPEATS:-3}"', script)
        self.assertIn('MATRIX_COPIES="${MATRIX_COPIES:-16}"', script)
        self.assertIn("run_hggc reference 256", script)
        self.assertIn("run_hggc warp_vec2", script)
        self.assertIn("torch_gemv_baseline.py", script)


if __name__ == "__main__":
    unittest.main()
