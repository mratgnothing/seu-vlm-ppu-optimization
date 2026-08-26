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

    def test_gdn_recurrent_kernel_matches_locked_qwen35_decode_contract(self) -> None:
        source = (MICROBENCH / "qwen35_gdn_recurrent.hg").read_text(
            encoding="utf-8"
        )
        baseline = (MICROBENCH / "torch_gdn_recurrent_baseline.py").read_text(
            encoding="utf-8"
        )

        for contract in (
            "constexpr int kHeads = 16",
            "constexpr int kKeyDim = 128",
            "constexpr int kValueDim = 128",
            "gdn_recurrent_fused",
            "state[state_offset] = updated",
            "__float2bfloat16_rn(result)",
        ):
            self.assertIn(contract, source)
        self.assertIn("HEADS = 16", baseline)
        self.assertIn("state = state * g.exp()", baseline)
        self.assertIn("memory_projection", baseline)
        self.assertIn("delta", baseline)

    def test_gdn_shared_operator_preserves_stream_and_fixed_shape_contract(self) -> None:
        kernel = (ROOT / "ppu" / "custom_ops" / "gdn_recurrent_ppu.hg").read_text(
            encoding="utf-8"
        )
        wrapper = (ROOT / "ppu" / "custom_ops" / "ppu_gdn.py").read_text(
            encoding="utf-8"
        )
        benchmark = (ROOT / "scripts" / "benchmark_ppu_gdn_ab.py").read_text(
            encoding="utf-8"
        )
        integration = (ROOT / "evaluation_wrapper.py").read_text(encoding="utf-8")
        self.assertIn("stream_handle", kernel)
        self.assertIn("round_to_bf16", kernel)
        for symbol in (
            "seu_ppu_gdn_recurrent_decode_bf16",
            "seu_ppu_causal_conv1d_decode_bf16",
            "seu_ppu_rmsnorm_decode_bf16",
            "seu_ppu_gated_rmsnorm_decode_bf16",
            "seu_ppu_qk_rmsnorm_rope_decode_bf16",
        ):
            self.assertIn(symbol, kernel)
        self.assertIn("query.stride(0)", wrapper)
        self.assertIn("query.stride(-2)", wrapper)
        self.assertIn("state must be contiguous", wrapper)
        self.assertIn("beta must be torch.bfloat16", wrapper)
        self.assertIn("expected 18 Qwen3.5 GDN modules", benchmark)
        self.assertIn("expected 49 Qwen3.5 RMSNorm modules", benchmark)
        self.assertIn("exact_output_match", benchmark)
        for variable in (
            "SEU_PPU_GDN_LIBRARY",
            "SEU_PPU_CONV_ENABLE",
            "SEU_PPU_RMSNORM_ENABLE",
            "SEU_PPU_GATED_RMSNORM_ENABLE",
            "SEU_PPU_QK_ROPE_ENABLE",
        ):
            self.assertIn(variable, integration)


if __name__ == "__main__":
    unittest.main()
