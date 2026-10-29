"""Parse and resolve Android uiautomator nodes for deterministic grounding."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass


BOUNDS_PATTERN = re.compile(r"^\[(\d+),(\d+)\]\[(\d+),(\d+)\]$")


@dataclass(frozen=True)
class UiNode:
    text: str
    content_desc: str
    resource_id: str
    class_name: str
    clickable: bool
    enabled: bool
    bounds: tuple[int, int, int, int]

    @property
    def searchable_text(self) -> str:
        return " ".join((self.text, self.content_desc, self.resource_id))

    @property
    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.bounds
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def area(self) -> int:
        left, top, right, bottom = self.bounds
        return max(0, right - left) * max(0, bottom - top)


def parse_bounds(value: str) -> tuple[int, int, int, int]:
    match = BOUNDS_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"invalid Android bounds: {value}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def parse_ui_xml(xml_text: str) -> list[UiNode]:
    start = xml_text.find("<?xml")
    if start >= 0:
        xml_text = xml_text[start:]
    root = ET.fromstring(xml_text)
    nodes: list[UiNode] = []
    for element in root.iter("node"):
        bounds = element.attrib.get("bounds")
        if not bounds:
            continue
        nodes.append(
            UiNode(
                text=element.attrib.get("text", ""),
                content_desc=element.attrib.get("content-desc", ""),
                resource_id=element.attrib.get("resource-id", ""),
                class_name=element.attrib.get("class", ""),
                clickable=element.attrib.get("clickable") == "true",
                enabled=element.attrib.get("enabled", "true") == "true",
                bounds=parse_bounds(bounds),
            )
        )
    return nodes


def resolve_target(nodes: list[UiNode], required_tokens: tuple[str, ...]) -> UiNode:
    if not required_tokens:
        raise ValueError("required target tokens are empty")
    matches = [
        node
        for node in nodes
        if node.enabled and all(token in node.searchable_text for token in required_tokens)
    ]
    if not matches:
        raise ValueError("no UI node contains all required target tokens")
    clickable = [node for node in matches if node.clickable]
    candidates = clickable or matches
    candidates.sort(key=lambda node: (node.area, node.bounds))
    if len(candidates) > 1 and candidates[0].area == candidates[1].area:
        raise ValueError("multiple equally specific UI nodes match the target")
    return candidates[0]
