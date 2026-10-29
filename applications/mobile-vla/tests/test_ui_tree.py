import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ui_tree import parse_bounds, parse_ui_xml, resolve_target  # noqa: E402


SAMPLE_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node text="" resource-id="root" class="android.widget.FrameLayout" clickable="false" enabled="true" bounds="[0,0][412,915]">
    <node text="陈小雨" content-desc="女儿 荷花头像" resource-id="contact_daughter" class="android.widget.Button" clickable="true" enabled="true" bounds="[20,170][392,245]" />
    <node text="陈小雨" content-desc="社区志愿者 蓝色头像" resource-id="contact_volunteer" class="android.widget.Button" clickable="true" enabled="true" bounds="[20,258][392,333]" />
  </node>
</hierarchy>"""


class UiTreeTest(unittest.TestCase):
    def test_parse_bounds(self):
        self.assertEqual(parse_bounds("[20,170][392,245]"), (20, 170, 392, 245))

    def test_resolves_relation_specific_contact(self):
        node = resolve_target(parse_ui_xml(SAMPLE_XML), ("陈小雨", "女儿"))
        self.assertEqual(node.resource_id, "contact_daughter")
        self.assertEqual(node.center, (206, 207))

    def test_rejects_ambiguous_name_only(self):
        with self.assertRaises(ValueError):
            resolve_target(parse_ui_xml(SAMPLE_XML), ("陈小雨",))


if __name__ == "__main__":
    unittest.main()
