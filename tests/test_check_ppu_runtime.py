from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_ppu_runtime import (
    EXPECTED_MODEL_FIELDS,
    _assess_route,
    _model_layout_probe,
    _module_available,
    _source_contains,
)


class CheckPpuRuntimeTest(unittest.TestCase):
    @staticmethod
    def _write_locked_model(root: Path) -> Path:
        config: dict[str, object] = {}
        for keys, expected in EXPECTED_MODEL_FIELDS:
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})  # type: ignore[assignment]
            current[keys[-1]] = expected
        text_config = config["text_config"]
        assert isinstance(text_config, dict)
        text_config["layer_types"] = [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(24)
        ]
        (root / "config.json").write_text(
            __import__("json").dumps(config), encoding="utf-8"
        )
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "preprocessor_config.json",
        ):
            (root / filename).write_text("{}", encoding="utf-8")
        weight = root / "model.safetensors"
        weight.write_bytes(b"ppu")
        lock = root / "model-lock.json"
        lock.write_text(
            __import__("json").dumps(
                {
                    "repo_id": "Qwen/Qwen3.5-2B",
                    "revision": "locked",
                    "files": {
                        weight.name: {
                            "size_bytes": 3,
                            "sha256": "0087db8eab997e5d95f2ff5d014e5baa9a5d109742ff240af0bbf1cb5cfc9cc7",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return lock

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

    def test_locked_qwen35_structure_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock = self._write_locked_model(root)

            report = _model_layout_probe(root, lock, verify_hash=False)

            self.assertTrue(report["valid"])
            self.assertTrue(report["config_fingerprint_matches"])
            self.assertEqual(report["weight_file_count"], 1)

    def test_qwen35_structure_mismatch_is_a_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock = self._write_locked_model(root)
            config_path = root / "config.json"
            config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
            config["text_config"]["hidden_size"] = 1024
            config_path.write_text(__import__("json").dumps(config), encoding="utf-8")

            report = _model_layout_probe(root, lock, verify_hash=False)

            self.assertFalse(report["valid"])
            self.assertIn(
                "model_config_mismatch:text_config.hidden_size",
                report["blockers"],
            )


if __name__ == "__main__":
    unittest.main()
