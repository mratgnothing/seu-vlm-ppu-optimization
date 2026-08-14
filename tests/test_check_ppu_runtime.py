from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_ppu_runtime import _assess_route, _module_available, _source_contains


class CheckPpuRuntimeTest(unittest.TestCase):
    def test_nested_module_probe_is_safe_when_parent_is_missing(self) -> None:
        self.assertFalse(_module_available("definitely_missing_parent.child"))

    def test_route_assessment_lists_machine_readable_blockers(self) -> None:
        assessment = _assess_route(
            (
                (True, "hardware_missing"),
                (False, "runtime_missing"),
                (False, "model_missing"),
            )
        )

        self.assertFalse(assessment["ready"])
        self.assertEqual(
            assessment["blockers"],
            ["runtime_missing", "model_missing"],
        )

    def test_route_assessment_is_ready_when_all_checks_pass(self) -> None:
        assessment = _assess_route(((True, "unused"),))

        self.assertTrue(assessment["ready"])
        self.assertEqual(assessment["blockers"], [])

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
