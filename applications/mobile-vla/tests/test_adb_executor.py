import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adb_executor import build_adb_command, hash_distance  # noqa: E402


class AdbExecutorTest(unittest.TestCase):
    def test_builds_safe_navigation_commands(self):
        self.assertEqual(
            build_adb_command({"action": "back"}),
            ["shell", "input", "keyevent", "KEYCODE_BACK"],
        )
        self.assertEqual(
            build_adb_command({"action": "tap", "coordinates": [12, 34]}),
            ["shell", "input", "tap", "12", "34"],
        )

    def test_rejects_non_ascii_adb_text(self):
        with self.assertRaises(ValueError):
            build_adb_command({"action": "type", "text": "我到家了"})

    def test_hash_distance(self):
        self.assertEqual(hash_distance("0f", "0f"), 0)
        self.assertEqual(hash_distance("00", "ff"), 8)


if __name__ == "__main__":
    unittest.main()
