from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ppu_first_validation.sh"


class PPUFirstValidationContractTest(unittest.TestCase):
    def test_microbenchmark_requires_explicit_opt_in(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("RUN_MICROBENCH=0", content)
        self.assertIn("--run-microbench", content)
        self.assertIn('if [[ "${RUN_MICROBENCH}" -eq 1 ]]', content)
        self.assertIn("Microbenchmark skipped", content)

    def test_compute_and_model_stages_require_explicit_opt_in(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        for default in (
            "RUN_DEVICE_SMOKE=0",
            "RUN_MODEL_LOAD=0",
            "RUN_SINGLE_SAMPLE=0",
            "VERIFY_MODEL_HASH=0",
        ):
            self.assertIn(default, content)
        for flag in (
            "--run-device-smoke",
            "--run-model-load",
            "--run-single-sample",
            "--verify-model-hash",
        ):
            self.assertIn(flag, content)

    def test_preserves_preflight_and_microbenchmark_evidence(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        for artifact in (
            "manifest.txt",
            "runtime.json",
            "runtime.stdout.log",
            "runtime-summary.md",
            "system-info.txt",
            "python-packages.json",
            "ppu-smi.txt",
            "microbench-build.log",
            "microbench-smoke.log",
            "microbench-suite.log",
        ):
            self.assertIn(artifact, content)

    def test_smoke_run_precedes_full_suite(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        smoke = content.index("--warmup 0")
        suite = content.index("run_qwen35_suite.sh")
        self.assertLess(smoke, suite)

    def test_real_sample_never_uses_auto_or_dummy_backend(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--backend transformers", content)
        self.assertIn("--num-samples 1", content)
        self.assertIn("--warmup-samples 0", content)

    def test_model_load_rejects_cpu_offload(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--require-accelerator", content)

    def test_native_failure_diagnostic_flushes_module_boundaries(self) -> None:
        content = (ROOT / "scripts" / "diagnose_qwen35_ppu.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("faulthandler.enable(all_threads=True)", content)
        self.assertIn('emit("forward_start")', content)
        self.assertIn('emit("module_enter"', content)
        self.assertIn("torch.cuda.synchronize()", content)

    def test_runner_exports_standard_sdk_for_runtime_compilation(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("-d /usr/local/PPU_SDK", content)
        self.assertIn("export PPU_SDK=/usr/local/PPU_SDK", content)
        self.assertIn('${PPU_SDK}/ppu-smi/bin', content)
        self.assertIn('echo "ppu_sdk=${PPU_SDK:-unset}"', content)

    def test_profiler_uses_real_sample_and_accelerator_activity(self) -> None:
        content = (ROOT / "scripts" / "profile_qwen35_ppu.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("load_mmbench_tsv", content)
        self.assertIn("ProfilerActivity.CUDA", content)
        self.assertIn('sort_by="self_cuda_time_total"', content)
        self.assertIn("export_chrome_trace", content)


if __name__ == "__main__":
    unittest.main()
