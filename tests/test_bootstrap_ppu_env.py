from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "bootstrap_ppu_env.sh"
ACTIVATE = ROOT / "scripts" / "activate_ppu_env.sh"
REQUIREMENTS = ROOT / "requirements-ppu.txt"


class PPUBootstrapContractTest(unittest.TestCase):
    def test_does_not_replace_official_torch(self) -> None:
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        package_lines = [line.strip() for line in requirements if line and not line.startswith("#")]
        self.assertFalse(any(line.startswith("torch") for line in package_lines))

        content = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("venv --system-site-packages", content)
        self.assertIn('[[ "${VENV_TORCH_INFO}" == "${BASE_TORCH_INFO}" ]]', content)
        self.assertNotIn("sudo ", content)

    def test_builds_all_required_extensions_in_dependency_order(self) -> None:
        content = BOOTSTRAP.read_text(encoding="utf-8")
        gdn = content.index("build_gdn_shared.sh")
        linear = content.index("build_acblas_linear_extension.py")
        packed = content.index("build_acblas_packed_mlp_extension.py")
        self.assertLess(gdn, linear)
        self.assertLess(linear, packed)

    def test_supports_read_only_and_offline_recovery(self) -> None:
        content = BOOTSTRAP.read_text(encoding="utf-8")
        for marker in ("--check-only", "--wheelhouse", "--no-index", "--skip-smoke"):
            self.assertIn(marker, content)

    def test_smoke_checks_device_and_extensions(self) -> None:
        content = BOOTSTRAP.read_text(encoding="utf-8")
        for marker in (
            "torch.cuda.is_available()",
            "smoke_gdn_gate_prep_integration.py",
            "smoke_acblas_linear_module.py",
            "import torch  # Load the official runtime libraries",
            "set_gdn_batched_ba",
            "smoke_acblas_packed_mlp_module.py",
        ):
            self.assertIn(marker, content)

    def test_activation_uses_cache_venv_and_sdk(self) -> None:
        content = ACTIVATE.read_text(encoding="utf-8")
        self.assertIn("SEU_PPU_VENV_DIR", content)
        self.assertIn("LD_LIBRARY_PATH", content)
        self.assertIn("PYTHONPATH", content)


if __name__ == "__main__":
    unittest.main()
