"""Local small-model planner for tightly constrained communication skills.

The model sees opaque contact aliases, never raw phone numbers or account IDs.
Execution is a separate step and requires a short-lived authorization file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


@dataclass(frozen=True)
class SkillCall:
    name: str
    arguments: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def openai_tools(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": skill["name"],
                "description": skill["description"],
                "parameters": skill["parameters"],
            },
        }
        for skill in registry["skills"]
    ]


def _coerce_xml_value(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def extract_skill_calls(response: dict[str, Any]) -> list[SkillCall]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("model response has no choices")
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls")
    calls: list[SkillCall] = []
    if isinstance(tool_calls, list) and tool_calls:
        for item in tool_calls:
            function = item.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            calls.append(SkillCall(str(function.get("name") or ""), arguments))
        return calls

    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("model response content is not text")
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        parsed_content = None
    if isinstance(parsed_content, dict) and isinstance(
        parsed_content.get("skill_calls"), list
    ):
        for item in parsed_content["skill_calls"]:
            if not isinstance(item, dict) or not isinstance(item.get("arguments"), dict):
                raise ValueError("JSON skill call is malformed")
            calls.append(SkillCall(str(item.get("name") or ""), item["arguments"]))
        if calls:
            return calls
    for match in TOOL_CALL_RE.finditer(content):
        arguments = {
            parameter.group(1): _coerce_xml_value(parameter.group(2))
            for parameter in PARAM_RE.finditer(match.group(2))
        }
        calls.append(SkillCall(match.group(1), arguments))
    if not calls:
        raise ValueError("model did not produce a skill call")
    return calls


def validate_plan(
    calls: list[SkillCall],
    registry: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    skills = {skill["name"]: skill for skill in registry["skills"]}
    expected = authorization.get("expected_sequence")
    if [call.name for call in calls] != expected:
        raise ValueError("model skill sequence does not match the authorized sequence")
    if len(calls) > int(authorization.get("max_skill_calls", 0)):
        raise ValueError("model exceeded the authorized skill-call limit")

    allowed_alias = authorization.get("contact_alias")
    for call in calls:
        if call.name not in skills:
            raise ValueError(f"unknown skill: {call.name}")
        schema = skills[call.name]["parameters"]
        properties = schema["properties"]
        required = set(schema.get("required", []))
        if set(call.arguments) != required:
            raise ValueError(f"invalid arguments for {call.name}")
        for key, value in call.arguments.items():
            allowed_values = properties[key].get("enum")
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"disallowed {key} for {call.name}")
        if call.arguments.get("contact_alias") != allowed_alias:
            raise ValueError("contact alias is outside this authorization")


def request_plan(
    endpoint: str,
    api_key: str,
    model: str,
    task: str,
    registry: dict[str, Any],
) -> dict[str, Any]:
    protocol = json.dumps(
        [
            {
                "name": skill["name"],
                "parameters": skill["parameters"],
            }
            for skill in registry["skills"]
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an offline phone skill router. Copy skill names exactly; "
                    "renaming or abbreviating a skill is an error. Never invent contacts, "
                    "numbers, messages, durations, or extra actions. Output one raw JSON "
                    "object only: no Markdown, no prose, no code fence. The only valid "
                    "top-level key is skill_calls. Its value is an ordered array of objects "
                    "with exactly two keys: name and arguments. Shape: "
                    "{\"skill_calls\":[{\"name\":\"EXACT_SKILL_NAME\","
                    "\"arguments\":{}}]}. Exact schemas: "
                    + protocol
                ),
            },
            {"role": "user", "content": task},
        ],
        "tools": openai_tools(registry),
        "temperature": 0,
        "max_tokens": 384,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def redacted_plan(calls: list[SkillCall], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "planner": "on-device-mnn-loopback",
        "skill_calls": [
            {"name": call.name, "arguments": call.arguments} for call in calls
        ],
        "contains_raw_contact_data": False,
        "validated": True,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--api-key", default=os.environ.get("MNN_API_KEY"))
    parser.add_argument("--model", default="Qwen3.5-2B-MNN")
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "configs" / "communication_skills.json",
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        default=(
            "For authorized_test_contact, call for at most 5 seconds, then send the "
            "approved skill_test_notice_v1 WeChat message, then start a WeChat video "
            "call for at most 5 seconds."
        ),
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("MNN API key must be supplied through MNN_API_KEY or --api-key")

    registry = load_json(args.registry)
    authorization = load_json(args.authorization)
    response = request_plan(args.endpoint, args.api_key, args.model, args.task, registry)
    calls = extract_skill_calls(response)
    validate_plan(calls, registry, authorization)
    report = redacted_plan(calls, args.model)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
