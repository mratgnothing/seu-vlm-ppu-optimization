import json
import unittest

from communication_skill_agent import (
    SkillCall,
    extract_skill_calls,
    validate_plan,
)


REGISTRY = {
    "skills": [
        {
            "name": "call_bound_contact",
            "parameters": {
                "properties": {
                    "contact_alias": {"enum": ["authorized_test_contact"]},
                    "max_duration_seconds": {"enum": [5]},
                },
                "required": ["contact_alias", "max_duration_seconds"],
            },
        },
        {
            "name": "send_wechat_message",
            "parameters": {
                "properties": {
                    "contact_alias": {"enum": ["authorized_test_contact"]},
                    "message_id": {"enum": ["skill_test_notice_v1"]},
                },
                "required": ["contact_alias", "message_id"],
            },
        },
    ]
}
AUTH = {
    "expected_sequence": ["call_bound_contact", "send_wechat_message"],
    "max_skill_calls": 2,
    "contact_alias": "authorized_test_contact",
}


class CommunicationSkillAgentTest(unittest.TestCase):
    def test_extracts_openai_tool_calls(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "call_bound_contact",
                                    "arguments": json.dumps(
                                        {
                                            "contact_alias": "authorized_test_contact",
                                            "max_duration_seconds": 5,
                                        }
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }
        calls = extract_skill_calls(response)
        self.assertEqual(calls[0].name, "call_bound_contact")
        self.assertEqual(calls[0].arguments["max_duration_seconds"], 5)

    def test_extracts_qwen_xml_tool_calls(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<tool_call>\n<function=call_bound_contact>\n"
                            "<parameter=contact_alias>\nauthorized_test_contact\n</parameter>\n"
                            "<parameter=max_duration_seconds>\n5\n</parameter>\n"
                            "</function>\n</tool_call>"
                        )
                    }
                }
            ]
        }
        calls = extract_skill_calls(response)
        self.assertEqual(calls[0].arguments["max_duration_seconds"], 5)

    def test_extracts_json_skill_calls_when_api_ignores_tools(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "skill_calls": [
                                    {
                                        "name": "call_bound_contact",
                                        "arguments": {
                                            "contact_alias": "authorized_test_contact",
                                            "max_duration_seconds": 5,
                                        },
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        calls = extract_skill_calls(response)
        self.assertEqual(calls[0].name, "call_bound_contact")

    def test_rejects_wrong_sequence(self):
        calls = [
            SkillCall(
                "send_wechat_message",
                {
                    "contact_alias": "authorized_test_contact",
                    "message_id": "skill_test_notice_v1",
                },
            ),
            SkillCall(
                "call_bound_contact",
                {
                    "contact_alias": "authorized_test_contact",
                    "max_duration_seconds": 5,
                },
            ),
        ]
        with self.assertRaisesRegex(ValueError, "sequence"):
            validate_plan(calls, REGISTRY, AUTH)

    def test_rejects_unbound_alias(self):
        calls = [
            SkillCall(
                "call_bound_contact",
                {"contact_alias": "someone_else", "max_duration_seconds": 5},
            ),
            SkillCall(
                "send_wechat_message",
                {
                    "contact_alias": "authorized_test_contact",
                    "message_id": "skill_test_notice_v1",
                },
            ),
        ]
        with self.assertRaises(ValueError):
            validate_plan(calls, REGISTRY, AUTH)


if __name__ == "__main__":
    unittest.main()
