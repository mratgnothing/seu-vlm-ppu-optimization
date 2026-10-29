"""Deterministic safety gate for GUI actions proposed by a vision-language model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_ACTIONS = {
    "tap",
    "long_press",
    "swipe",
    "type",
    "back",
    "home",
    "open_app",
    "wait",
    "clarify",
    "handoff",
    "terminate",
}
VALID_RISKS = {"low", "medium", "high"}


@dataclass(frozen=True)
class GateDecision:
    decision: str
    reason: str


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_action(action: dict[str, Any]) -> None:
    required = {"action", "risk", "confidence", "requires_confirmation", "reason"}
    missing = required - action.keys()
    if missing:
        raise ValueError(f"missing action fields: {sorted(missing)}")
    if action["action"] not in VALID_ACTIONS:
        raise ValueError(f"unsupported action: {action['action']}")
    if action["risk"] not in VALID_RISKS:
        raise ValueError(f"unsupported risk: {action['risk']}")
    confidence = action["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(action["requires_confirmation"], bool):
        raise ValueError("requires_confirmation must be boolean")
    coordinates = action.get("coordinates")
    if action["action"] in {"tap", "long_press"}:
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError("tap and long_press require [x, y] coordinates")
    if action["action"] == "swipe":
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError("swipe requires [x1, y1, x2, y2] coordinates")


def decide(
    action: dict[str, Any],
    policy: dict[str, Any],
    screen_size: tuple[int, int] | None = None,
    required_target_tokens: tuple[str, ...] | None = None,
) -> GateDecision:
    validate_action(action)
    kind = action["action"]
    if kind == "handoff":
        return GateDecision("HANDOFF", action["reason"])
    if kind == "clarify":
        return GateDecision("CLARIFY", action["reason"])
    if kind == "terminate":
        return GateDecision("TERMINATE", action["reason"])

    if screen_size and kind in {"tap", "long_press", "swipe"}:
        width, height = screen_size
        coordinates = action["coordinates"]
        points = (
            [coordinates]
            if kind in {"tap", "long_press"}
            else [coordinates[:2], coordinates[2:]]
        )
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            return GateDecision("CLARIFY", "action coordinates are outside the screen")

    if required_target_tokens and kind in {"tap", "long_press", "open_app"}:
        description = " ".join(
            str(action.get(field) or "") for field in ("target", "text", "reason")
        )
        missing = [token for token in required_target_tokens if token not in description]
        if missing:
            return GateDecision(
                "CLARIFY", f"action target is missing required tokens: {', '.join(missing)}"
            )

    operation = action.get("operation")
    if operation in set(policy["handoff_required"]):
        return GateDecision("HANDOFF", f"operation requires user control: {operation}")
    if action["risk"] == "high":
        return GateDecision("HANDOFF", "high-risk action")
    if (
        operation in set(policy["confirmation_required"])
        or action["risk"] == "medium"
        or action["requires_confirmation"]
    ):
        return GateDecision("CONFIRM", "explicit user confirmation required")
    if action["risk"] not in set(policy["auto_execute"]["allowed_risk"]):
        return GateDecision("CLARIFY", "risk level is not allowed for automatic execution")
    if action["confidence"] < policy["auto_execute"]["minimum_confidence"]:
        return GateDecision("CLARIFY", "model confidence is below the automatic threshold")
    return GateDecision("ACT", "low-risk action passed deterministic checks")
