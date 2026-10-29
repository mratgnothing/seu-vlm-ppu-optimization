"""Evaluate saved GUI policy probes against controlled target boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ui_tree import parse_ui_xml, resolve_target


def point_in_box(point: list[int] | tuple[int, int], box: list[float]) -> bool:
    x, y = point
    left, top, right, bottom = box
    return left <= x <= right and top <= y <= bottom


def evaluate_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    result_path = root / case["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    action = result.get("parsed_action")
    parsed = isinstance(action, dict)
    gate_match = result.get("gate_decision") == case["expected_gate"]
    action_match = (
        True
        if case.get("expected_action") is None
        else parsed and action.get("action") == case["expected_action"]
    )

    target_box = case.get("target_box")
    raw_hit: bool | None = None
    resolved_hit: bool | None = None
    resolved_center: list[int] | None = None
    if target_box and parsed and action.get("action") in {"tap", "long_press"}:
        coordinates = action.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            raw_hit = point_in_box(coordinates[:2], target_box)

    ui_xml = case.get("ui_xml")
    required_tokens = tuple(result.get("required_target_tokens") or ())
    if ui_xml and required_tokens:
        nodes = parse_ui_xml((root / ui_xml).read_text(encoding="utf-8"))
        node = resolve_target(nodes, required_tokens)
        resolved_center = list(node.center)
        resolved_hit = point_in_box(node.center, target_box)

    coordinate_action = parsed and action.get("action") in {"tap", "long_press"}
    if not coordinate_action:
        effective_grounding = True
    elif resolved_hit is not None:
        effective_grounding = resolved_hit
    else:
        effective_grounding = bool(raw_hit)

    execution_expected = case["expected_gate"] == "ACT"
    handled_as_expected = bool(
        parsed
        and gate_match
        and action_match
        and (not execution_expected or effective_grounding)
    )
    execution_ready = bool(execution_expected and handled_as_expected)
    return {
        "case_id": case["case_id"],
        "parsed": parsed,
        "gate": result.get("gate_decision"),
        "expected_gate": case["expected_gate"],
        "gate_match": gate_match,
        "action": action.get("action") if parsed else None,
        "action_match": bool(action_match),
        "raw_coordinates": action.get("coordinates") if parsed else None,
        "raw_hit": raw_hit,
        "resolved_center": resolved_center,
        "resolved_hit": resolved_hit,
        "effective_grounding": effective_grounding,
        "handled_as_expected": handled_as_expected,
        "execution_ready": execution_ready,
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    raw_cases = [item for item in cases if item["raw_hit"] is not None]
    execution_cases = [item for item in cases if item["expected_gate"] == "ACT"]
    return {
        "total_cases": len(cases),
        "parsed": sum(item["parsed"] for item in cases),
        "gate_matches": sum(item["gate_match"] for item in cases),
        "raw_grounding_hits": sum(item["raw_hit"] is True for item in raw_cases),
        "raw_grounding_total": len(raw_cases),
        "effective_execution_grounding_hits": sum(
            item["effective_grounding"] for item in execution_cases
        ),
        "effective_execution_grounding_total": len(execution_cases),
        "handled_as_expected": sum(item["handled_as_expected"] for item in cases),
        "execution_ready": sum(item["execution_ready"] for item in cases),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# GUI 策略微型基线",
        "",
        "| 案例 | 解析 | 安全裁决 | 原始坐标命中 | 控件树修正 | 处理符合预期 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in payload["cases"]:
        def mark(value):
            if value is None:
                return "—"
            return "✓" if value else "✗"

        lines.append(
            f"| {item['case_id']} | {mark(item['parsed'])} | "
            f"{mark(item['gate_match'])} | {mark(item['raw_hit'])} | "
            f"{mark(item['resolved_hit'])} | {mark(item['handled_as_expected'])} |"
        )
    lines += [
        "",
        f"- JSON 解析：{summary['parsed']}/{summary['total_cases']}",
        f"- 安全裁决符合预期：{summary['gate_matches']}/{summary['total_cases']}",
        f"- 模型原始坐标命中：{summary['raw_grounding_hits']}/{summary['raw_grounding_total']}",
        "- 可执行案例的系统有效定位："
        f"{summary['effective_execution_grounding_hits']}/"
        f"{summary['effective_execution_grounding_total']}",
        f"- 系统处理符合预期：{summary['handled_as_expected']}/{summary['total_cases']}",
        f"- 可执行动作就绪：{summary['execution_ready']}/"
        f"{summary['effective_execution_grounding_total']}",
        "",
        "该基线只有五个受控样本，用于回归和架构验证，不代表通用 GUI 成功率。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=root / "benchmarks" / "gui_cases.json")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    definitions = json.loads(args.cases.read_text(encoding="utf-8"))
    evaluated = [evaluate_case(root, case) for case in definitions]
    payload = {"summary": summarize(evaluated), "cases": evaluated}
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    print(json_text, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
