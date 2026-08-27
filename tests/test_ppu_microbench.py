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
        linear_wrapper = (ROOT / "ppu" / "custom_ops" / "ppu_linear.py").read_text(
            encoding="utf-8"
        )
        linear_extension = (
            ROOT / "ppu" / "custom_ops" / "acblas_linear_extension.cpp"
        ).read_text(encoding="utf-8")
        acblas_bridge = (
            ROOT / "ppu" / "custom_ops" / "acblas_linear_wrapper.cpp"
        ).read_text(encoding="utf-8")
        extension_builder = (
            ROOT / "ppu" / "custom_ops" / "build_acblas_linear_extension.py"
        ).read_text(encoding="utf-8")
        benchmark = (ROOT / "scripts" / "benchmark_ppu_gdn_ab.py").read_text(
            encoding="utf-8"
        )
        acblas_benchmark = (
            ROOT / "scripts" / "benchmark_ppu_acblas_multisample_ab.py"
        ).read_text(encoding="utf-8")
        packed_gdn = (
            ROOT / "ppu" / "custom_ops" / "ppu_gdn_projection_pack.py"
        ).read_text(encoding="utf-8")
        acblas_grouped_gdn = (
            ROOT / "ppu" / "custom_ops" / "ppu_acblas_gdn_projection.py"
        ).read_text(encoding="utf-8")
        packed_gdn_benchmark = (
            ROOT / "scripts" / "benchmark_ppu_packed_gdn_multisample_ab.py"
        ).read_text(encoding="utf-8")
        integration = (ROOT / "evaluation_wrapper.py").read_text(encoding="utf-8")
        residual_smoke = (
            ROOT / "ppu" / "custom_ops" / "smoke_residual_rmsnorm_integration.py"
        ).read_text(encoding="utf-8")
        swiglu_smoke = (
            ROOT / "ppu" / "custom_ops" / "smoke_swiglu_integration.py"
        ).read_text(encoding="utf-8")
        self.assertIn("stream_handle", kernel)
        self.assertIn("round_to_bf16", kernel)
        for symbol in (
            "seu_ppu_gdn_recurrent_decode_bf16",
            "seu_ppu_causal_conv1d_decode_bf16",
            "seu_ppu_rmsnorm_decode_bf16",
            "seu_ppu_residual_rmsnorm_decode_bf16",
            "seu_ppu_swiglu_decode_bf16",
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
        self.assertIn("expected 24 Qwen3.5 MLP modules", benchmark)
        self.assertIn("batch=1 single-token decode", linear_wrapper)
        self.assertIn("seu_acblas_linear_bf16", linear_extension)
        self.assertIn("seu_acblas_gdn_projections_bf16", linear_extension)
        self.assertIn("acblasGemvEx", acblas_bridge)
        self.assertIn("seu_acblas_gdn_projections_bf16", acblas_bridge)
        self.assertIn("kOutputOffsets[] = {0, 6144, 8192, 8208}", acblas_bridge)
        self.assertIn("std::lock_guard<std::mutex>", acblas_bridge)
        self.assertIn("getCurrentCUDAStream", linear_extension)
        self.assertIn("residual_rmsnorm_decode", wrapper)
        self.assertIn("pack_qwen35_decoder_residual_rmsnorm", wrapper)
        self.assertIn("next_norm=next_norm", integration)
        self.assertIn("residual_exact", residual_smoke)
        self.assertIn("normalized_exact", residual_smoke)
        self.assertIn("fused_bf16_swiglu", swiglu_smoke)
        self.assertIn("max_abs_error", swiglu_smoke)
        self.assertIn("Path(sys.executable).parent", extension_builder)
        self.assertIn("exact_output_match", benchmark)
        self.assertIn('"2048x2048": 42', acblas_benchmark)
        self.assertIn("exact_output_pairs", acblas_benchmark)
        self.assertIn("group_sizes must be a positive partition", packed_gdn)
        self.assertIn("_seu_gdn_input_weight", packed_gdn)
        self.assertIn("threading.local()", packed_gdn)
        self.assertIn("set_packed_qwen35_gdn_input_projections", packed_gdn)
        self.assertIn("threading.local()", acblas_grouped_gdn)
        self.assertIn("gdn_projections_bf16", acblas_grouped_gdn)
        self.assertIn("_seu_acblas_gdn_weight", acblas_grouped_gdn)
        self.assertIn("weights do not alias packed storage", acblas_grouped_gdn)
        self.assertIn("exact_output_pairs", packed_gdn_benchmark)
        self.assertIn("projection_group_sizes", packed_gdn_benchmark)
        for variable in (
            "SEU_PPU_GDN_LIBRARY",
            "SEU_PPU_CONV_ENABLE",
            "SEU_PPU_RMSNORM_ENABLE",
            "SEU_PPU_GATED_RMSNORM_ENABLE",
            "SEU_PPU_QK_ROPE_ENABLE",
            "SEU_PPU_PACK_MLP_ENABLE",
            "SEU_PPU_PACK_GDN_PROJECTIONS_ENABLE",
            "SEU_PPU_PACK_GDN_PROJECTIONS_GROUPS",
            "SEU_PPU_ACBLAS_GDN_BUILD_DIR",
            "SEU_PPU_ACBLAS_GDN_ALGORITHM",
            "SEU_PPU_RESIDUAL_RMSNORM_ENABLE",
        ):
            self.assertIn(variable, integration)
        self.assertIn("GDN projection backends are mutually exclusive", integration)
        self.assertIn('self._ppu_gdn_projection_backend = "acblas-grouped"', integration)


if __name__ == "__main__":
    unittest.main()
