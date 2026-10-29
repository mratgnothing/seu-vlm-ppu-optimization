import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from host_policy_probe import extract_json_object  # noqa: E402


class HostPolicyProbeTest(unittest.TestCase):
    def test_extracts_json_from_markdown_wrapper(self):
        value = extract_json_object(
            '```json\n{"action":"handoff","risk":"high"}\n```'
        )
        self.assertEqual(value["action"], "handoff")

    def test_skips_non_json_brace_before_valid_object(self):
        value = extract_json_object('说明 {坏格式} 后面 {"action":"wait"}')
        self.assertEqual(value["action"], "wait")

    def test_rejects_output_without_object(self):
        with self.assertRaises(ValueError):
            extract_json_object("请用户自行操作")


if __name__ == "__main__":
    unittest.main()
