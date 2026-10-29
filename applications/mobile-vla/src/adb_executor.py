"""Approval-gated ADB executor for a previously generated policy proposal.

The default mode is dry-run. Actual device mutation requires --execute, an ACT
decision, a low-risk action, and a sufficiently similar current screenshot.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from action_gate import decide, load_policy, validate_action
from host_policy_probe import average_hash
from ui_tree import parse_ui_xml, resolve_target


PACKAGE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+$")


def hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def build_adb_command(action: dict[str, Any]) -> list[str] | None:
    kind = action["action"]
    coordinates = action.get("coordinates")
    if kind == "tap":
        return ["shell", "input", "tap", str(coordinates[0]), str(coordinates[1])]
    if kind == "long_press":
        return [
            "shell",
            "input",
            "swipe",
            str(coordinates[0]),
            str(coordinates[1]),
            str(coordinates[0]),
            str(coordinates[1]),
            "800",
        ]
    if kind == "swipe":
        return ["shell", "input", "swipe", *(str(value) for value in coordinates), "350"]
    if kind == "back":
        return ["shell", "input", "keyevent", "KEYCODE_BACK"]
    if kind == "home":
        return ["shell", "input", "keyevent", "KEYCODE_HOME"]
    if kind == "open_app":
        package = action.get("text")
        if not isinstance(package, str) or not PACKAGE_PATTERN.fullmatch(package):
            raise ValueError("open_app requires a validated Android package name in text")
        return ["shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"]
    if kind == "type":
        value = action.get("text")
        if not isinstance(value, str) or not value.isascii():
            raise ValueError("ADB text input is limited to ASCII in the first prototype")
        return ["shell", "input", "text", value.replace(" ", "%s")]
    if kind == "wait":
        return None
    raise ValueError(f"action is not executable: {kind}")


def run_adb(adb: Path, serial: str | None, args: list[str], capture: bool = False):
    command = [str(adb)]
    if serial:
        command += ["-s", serial]
    command += args
    return subprocess.run(command, check=True, capture_output=capture)


def capture_image(adb: Path, serial: str | None) -> tuple[bytes, Image.Image]:
    result = run_adb(adb, serial, ["exec-out", "screencap", "-p"], capture=True)
    payload = result.stdout
    return payload, Image.open(io.BytesIO(payload)).convert("RGB")


def capture_ui_xml(adb: Path, serial: str | None) -> str:
    result = run_adb(
        adb,
        serial,
        ["exec-out", "uiautomator", "dump", "/dev/tty"],
        capture=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--adb",
        type=Path,
        default=root / "tools" / "platform-tools" / "adb.exe",
    )
    parser.add_argument("--serial")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-hash-distance", type=int, default=32)
    parser.add_argument("--ui-xml", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "evidence")
    args = parser.parse_args()

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    action = proposal.get("parsed_action")
    if not isinstance(action, dict):
        raise SystemExit("proposal has no parsed action")
    validate_action(action)
    policy = load_policy(root / "configs" / "safety_policy.json")
    image_size = proposal.get("image_size")
    screen_size = (
        (int(image_size[0]), int(image_size[1]))
        if isinstance(image_size, list) and len(image_size) == 2
        else None
    )
    required_tokens = tuple(proposal.get("required_target_tokens") or ())
    gate = decide(action, policy, screen_size, required_tokens)
    if proposal.get("gate_decision") != gate.decision:
        raise SystemExit("stored gate decision does not match current policy")
    if gate.decision != "ACT":
        raise SystemExit(f"execution blocked by safety gate: {gate.decision} ({gate.reason})")
    if action["risk"] != "low":
        raise SystemExit("only low-risk actions can reach the ADB executor")

    action_for_execution = dict(action)
    resolved_node = None
    if required_tokens and action["action"] in {"tap", "long_press"} and args.ui_xml:
        nodes = parse_ui_xml(args.ui_xml.read_text(encoding="utf-8"))
        resolved_node = resolve_target(nodes, required_tokens)
        action_for_execution["coordinates"] = list(resolved_node.center)
    command = build_adb_command(action_for_execution)
    report: dict[str, Any] = {
        "proposal": str(args.proposal),
        "decision": gate.decision,
        "adb_args": command,
        "executed": False,
        "required_target_tokens": list(required_tokens),
    }
    if resolved_node:
        report["resolved_node"] = {
            "text": resolved_node.text,
            "content_desc": resolved_node.content_desc,
            "resource_id": resolved_node.resource_id,
            "bounds": list(resolved_node.bounds),
            "center": list(resolved_node.center),
        }
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    before_bytes, before_image = capture_image(args.adb, args.serial)
    if required_tokens and action["action"] in {"tap", "long_press"} and not resolved_node:
        nodes = parse_ui_xml(capture_ui_xml(args.adb, args.serial))
        resolved_node = resolve_target(nodes, required_tokens)
        action_for_execution["coordinates"] = list(resolved_node.center)
        command = build_adb_command(action_for_execution)
        report["adb_args"] = command
        report["resolved_node"] = {
            "text": resolved_node.text,
            "content_desc": resolved_node.content_desc,
            "resource_id": resolved_node.resource_id,
            "bounds": list(resolved_node.bounds),
            "center": list(resolved_node.center),
        }
    current_hash = average_hash(before_image)
    distance = hash_distance(proposal["image_ahash"], current_hash)
    report["current_image_ahash"] = current_hash
    report["hash_distance"] = distance
    if distance > args.max_hash_distance:
        raise SystemExit(
            f"screen changed since proposal: hash distance {distance} > {args.max_hash_distance}"
        )
    width, height = before_image.size
    coordinates = action_for_execution.get("coordinates")
    if action["action"] in {"tap", "long_press"}:
        x, y = coordinates
        if not (0 <= x < width and 0 <= y < height):
            raise SystemExit("action coordinates are outside the current screen")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    before_path = args.output_dir / f"adb-before-{stamp}.png"
    before_path.write_bytes(before_bytes)
    if action["action"] == "wait":
        delay = min(max(float(action.get("text") or 1), 0), 5)
        time.sleep(delay)
    else:
        assert command is not None
        run_adb(args.adb, args.serial, command)
        time.sleep(0.8)
    after_bytes, after_image = capture_image(args.adb, args.serial)
    after_path = args.output_dir / f"adb-after-{stamp}.png"
    after_path.write_bytes(after_bytes)
    report.update(
        executed=True,
        before_image=str(before_path),
        after_image=str(after_path),
        after_image_ahash=average_hash(after_image),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
