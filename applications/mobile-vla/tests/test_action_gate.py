import json
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from action_gate import decide  # noqa: E402


class ActionGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(
            (ROOT / "configs" / "safety_policy.json").read_text(encoding="utf-8")
        )

    def base_action(self):
        return {
            "action": "tap",
            "operation": "open_settings",
            "coordinates": [100, 200],
            "risk": "low",
            "confidence": 0.95,
            "requires_confirmation": False,
            "reason": "设置图标清晰可见",
        }

    def test_low_risk_high_confidence_action_can_execute(self):
        self.assertEqual(decide(self.base_action(), self.policy).decision, "ACT")

    def test_low_confidence_action_must_clarify(self):
        action = self.base_action()
        action["confidence"] = 0.70
        self.assertEqual(decide(action, self.policy).decision, "CLARIFY")

    def test_message_send_requires_confirmation(self):
        action = self.base_action()
        action["operation"] = "send_message"
        action["risk"] = "medium"
        self.assertEqual(decide(action, self.policy).decision, "CONFIRM")

    def test_payment_always_hands_off(self):
        action = self.base_action()
        action["operation"] = "payment"
        self.assertEqual(decide(action, self.policy).decision, "HANDOFF")

    def test_out_of_bounds_coordinates_must_clarify(self):
        action = self.base_action()
        action["coordinates"] = [499, 240]
        decision = decide(action, self.policy, (412, 915))
        self.assertEqual(decision.decision, "CLARIFY")

    def test_required_target_tokens_detect_semantic_mismatch(self):
        action = self.base_action()
        action["target"] = "选择联系人"
        decision = decide(
            action,
            self.policy,
            (412, 915),
            ("陈小雨", "女儿"),
        )
        self.assertEqual(decision.decision, "CLARIFY")


if __name__ == "__main__":
    unittest.main()
