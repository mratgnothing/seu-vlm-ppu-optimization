from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_ppu_runtime import _source_contains


class CheckPpuRuntimeTest(unittest.TestCase):
    def test_source_marker_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "model.py").write_text(
                "class Qwen3_5ForConditionalGeneration: pass\n",
                encoding="utf-8",
            )
            self.assertTrue(
                _source_contains(
                    root,
                    ("Qwen3_5ForConditionalGeneration",),
                )
            )
            self.assertFalse(_source_contains(root, ("GatedDeltaNet",)))


if __name__ == "__main__":
    unittest.main()
