from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_submission import (
    FIXED_ZIP_TIMESTAMP,
    PACKAGE_ROOT,
    collect_files,
    create_package,
)


ROOT = Path(__file__).resolve().parents[1]


class SubmissionPackageTest(unittest.TestCase):
    def test_allowlist_excludes_local_data_models_results_and_secrets(self) -> None:
        relative = {
            path.resolve().relative_to(ROOT).as_posix()
            for path in collect_files(ROOT)
        }

        self.assertIn("evaluation_wrapper.py", relative)
        self.assertIn("ppu/microbench/qwen35_bf16_gemv.hg", relative)
        self.assertIn("submission/README.md", relative)
        self.assertNotIn("configs/local.psd1", relative)
        self.assertFalse(any(path.startswith("artifacts/") for path in relative))
        self.assertFalse(any(path.startswith("data/derived/") for path in relative))
        self.assertFalse(any(path.startswith("results/raw/") for path in relative))
        self.assertFalse(
            any(
                path.startswith("models/") and path != "models/README.md"
                for path in relative
            )
        )
        self.assertFalse(any(path.endswith(".pem") for path in relative))

    def test_package_is_verified_and_uses_reproducible_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "submission.zip"
            report = create_package(ROOT, output)

            self.assertTrue(report["verified"])
            self.assertEqual(report["file_count"], report["manifest_entry_count"])
            with zipfile.ZipFile(output, "r") as archive:
                names = archive.namelist()
                self.assertIn(f"{PACKAGE_ROOT}/MANIFEST.sha256", names)
                self.assertTrue(
                    all(
                        info.date_time == FIXED_ZIP_TIMESTAMP
                        for info in archive.infolist()
                    )
                )


if __name__ == "__main__":
    unittest.main()
