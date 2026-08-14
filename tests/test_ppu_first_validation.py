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
        self.assertIn('if [[ "${RUN_MICROBENCH}" -eq 0 ]]', content)

    def test_preserves_preflight_and_microbenchmark_evidence(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")

        for artifact in (
            "manifest.txt",
            "runtime.json",
            "runtime.stdout.log",
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


if __name__ == "__main__":
    unittest.main()
