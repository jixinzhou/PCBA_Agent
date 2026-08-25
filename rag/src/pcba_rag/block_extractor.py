from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable

from .ocr import OcrLine
from .page_quality import normalize_text


@dataclass(frozen=True)
class LayoutLine:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    source: str
    confidence: float


_NUMBERED_HEADING = re.compile(r"^(?P<number>\d+(?:\.\d+){0,5})(?:\s+|(?=[\u4e00-\u9fffA-Za-z]))")
_CHINESE_HEADING = re.compile(r"^第[一二三四五六七八九十百]+[章节部分]\s*")
_LIST_ITEM = re.compile(r"^(?:[-•●▪]|\(?[a-zA-Z0-9一二三四五六七八九十]+[)）、.])\s+")
_TABLE_CAPTION = re.compile(r"^(?:表|Table)\s*\d+[A-Za-z0-9.（）()\-]*(?:\s|$)", re.IGNORECASE)
_FIGURE_CAPTION = re.compile(r"^(?:图|Figure|Fig\.)\s*\d+[A-Za-z0-9.（）()\-]*(?:\s|$)", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,8}|[A-Z]-?\d{1,4})$", re.IGNORECASE)
_FORMULA = re.compile(r"(?:[A-Za-zΑ-ω]\s*=|[≤≥≈±∑√]|\d\s*[+×÷/]\s*\d)")


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _union_bbox(lines: Iterable[LayoutLine]) -> list[float]:
    values = tuple(lines)
    return [
        _clamp(min(line.bbox[0] for line in values)),
        _clamp(min(line.bbox[1] for line in values)),
        _clamp(max(line.bbox[2] for line in values)),
        _clamp(max(line.bbox[3] for line in values)),
    ]


def native_layout_lines(page: Any) -> tuple[LayoutLine, ...]:
    width = max(float(page.mediabox.width), 1.0)
    height = max(float(page.mediabox.height), 1.0)
    fragments: list[LayoutLine] = []

    def multiply(left: list[float], right: list[float]) -> list[float]:
        return [
            left[0] * right[0] + left[1] * right[2],
            left[0] * right[1] + left[1] * right[3],
            left[2] * right[0] + left[3] * right[2],
            left[2] * right[1] + left[3] * right[3],
            left[4] * right[0] + left[5] * right[2] + right[4],
            left[4] * right[1] + left[5] * right[3] + right[5],
        ]

    def visitor(text: str, cm: list[float], tm: list[float], font: Any, font_size: float) -> None:
        values = [part.strip() for part in str(text).splitlines() if part.strip()]
        if not values:
            return
        composed = multiply(tm, cm)
        scale_x = abs(float(composed[0])) or float(font_size) or 10.0
        scale_y = abs(float(composed[3])) or float(font_size) or 10.0
        effective_size = max(scale_x, scale_y, 1.0)
        x = float(composed[4])
        baseline_y = float(composed[5])
        if x < -width * 0.05 or x > width * 1.05 or baseline_y < -height * 0.05 or baseline_y > height * 1.05:
            return
        if x == 0 and baseline_y == 0 and len("".join(values)) <= 2:
            return
        for offset, value in enumerate(values):
            y = baseline_y - offset * effective_size * 1.2
            estimated_width = max(effective_size * 0.45 * len(value), effective_size)
            bbox = (
                _clamp(x / width),
                _clamp((height - y - effective_size) / height),
                _clamp((x + estimated_width) / width),
                _clamp((height - y + effective_size * 0.25) / height),
            )
            fragments.append(LayoutLine(value, bbox, effective_size, "native", 1.0))

    page.extract_text(visitor_text=visitor)
    fragments.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    deduplicated: list[LayoutLine] = []
    seen: set[tuple[str, int, int]] = set()
    for fragment in fragments:
        key = (
            re.sub(r"\s+", "", fragment.text),
            round(fragment.bbox[0] * 500),
            round(fragment.bbox[1] * 500),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(fragment)
    return tuple(_merge_same_row(deduplicated))


def ocr_layout_lines(lines: Iterable[OcrLine]) -> tuple[LayoutLine, ...]:
    result = [
        LayoutLine(
            line.text,
            line.bbox,
            max((line.bbox[3] - line.bbox[1]) * 1000.0, 1.0),
            "ocr",
            float(line.confidence) if line.confidence is not None else 0.0,
        )
        for line in lines
        if line.text.strip()
    ]
    result.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    return tuple(_merge_same_row(result))


def _merge_same_row(lines: Iterable[LayoutLine]) -> list[LayoutLine]:
    rows: list[list[LayoutLine]] = []
    for line in lines:
        center = (line.bbox[1] + line.bbox[3]) / 2
        if not rows:
            rows.append([line])
            continue
        previous_center = sum((item.bbox[1] + item.bbox[3]) / 2 for item in rows[-1]) / len(rows[-1])
        tolerance = max(line.bbox[3] - line.bbox[1], 0.008) * 0.55
        if abs(center - previous_center) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])

    merged: list[LayoutLine] = []
    for row in rows:
        row.sort(key=lambda item: item.bbox[0])
        pieces: list[str] = []
        previous_x1: float | None = None
        for item in row:
            if previous_x1 is not None and item.bbox[0] - previous_x1 > 0.04:
                pieces.append(" | ")
            elif pieces and not pieces[-1].endswith((" ", "-", "/")):
                pieces.append(" ")
            pieces.append(item.text)
            previous_x1 = item.bbox[2]
        merged.append(
            LayoutLine(
                normalize_text("".join(pieces)),
                tuple(_union_bbox(row)),
                max(item.font_size for item in row),
                row[0].source,
                sum(item.confidence for item in row) / len(row),
            )
        )
    return merged


def _heading_level(text: str) -> int | None:
    match = _NUMBERED_HEADING.match(text)
    if match:
        return min(match.group("number").count(".") + 1, 6)
    if _CHINESE_HEADING.match(text):
        return 1
    return None


def _classify(line: LayoutLine, typical_font: float, header_ratio: float, footer_ratio: float) -> str:
    text = line.text.strip()
    if _PAGE_NUMBER.fullmatch(text) and (line.bbox[1] >= 0.88 or line.bbox[3] <= 0.12):
        return "page_number"
    if line.bbox[3] <= header_ratio:
        return "header"
    if line.bbox[1] >= footer_ratio:
        return "footer"
    if _TABLE_CAPTION.match(text):
        return "table_caption"
    if _FIGURE_CAPTION.match(text):
        return "figure_caption"
    if _heading_level(text) is not None:
        return "heading"
    if (
        4 <= len(text) <= 100
        and " | " not in text
        and line.bbox[2] - line.bbox[0] >= 0.12
        and typical_font > 0
        and line.font_size >= typical_font * 1.5
        and any(char.isalpha() or "\u4e00" <= char <= "\u9fff" for char in text)
    ):
        return "heading"
    alpha_tokens = re.findall(r"[A-Z]{2,}", text)
    if 8 <= len(text) <= 100 and text.isupper() and len(alpha_tokens) >= 2:
        return "heading"
    if _LIST_ITEM.match(text):
        return "list_item"
    if " | " in text:
        return "table"
    if len(text) <= 160 and _FORMULA.search(text):
        return "formula"
    return "paragraph"


def _join_paragraph_text(previous: str, current: str) -> str:
    if previous.endswith("-") and current and current[0].islower():
        return previous[:-1] + current
    if previous and previous[-1] in "，。；：！？、,.!?;:" or current.startswith((")", "）", ",", ".")):
        return previous + current
    return previous + " " + current


def build_blocks(
    page_id: str,
    lines: Iterable[LayoutLine],
    header_ratio: float = 0.08,
    footer_ratio: float = 0.92,
    paragraph_gap_ratio: float = 0.035,
) -> list[dict[str, Any]]:
    values = [line for line in lines if line.text.strip()]
    if not values:
        return []
    content_fonts = [
        line.font_size for line in values if line.bbox[3] > header_ratio and line.bbox[1] < footer_ratio
    ]
    typical_font = median(content_fonts) if content_fonts else median(line.font_size for line in values)
    classified = [(line, _classify(line, typical_font, header_ratio, footer_ratio)) for line in values]

    groups: list[tuple[str, list[LayoutLine]]] = []
    mergeable = {"paragraph", "table"}
    for line, block_type in classified:
        if groups and groups[-1][0] == block_type and block_type in mergeable:
            previous = groups[-1][1][-1]
            gap = line.bbox[1] - previous.bbox[3]
            if gap <= paragraph_gap_ratio:
                groups[-1][1].append(line)
                continue
        groups.append((block_type, [line]))

    blocks: list[dict[str, Any]] = []
    for order, (block_type, group) in enumerate(groups):
        if block_type == "paragraph":
            text = group[0].text
            for line in group[1:]:
                text = _join_paragraph_text(text, line.text)
        else:
            text = "\n".join(line.text for line in group)
        flags: list[str] = []
        confidence = sum(line.confidence for line in group) / len(group)
        if confidence < 0.8:
            flags.append("low_block_confidence")
        if block_type == "table" and len(group) == 1:
            flags.append("table_structure_unverified")
        blocks.append(
            {
                "schema_version": "1.0.0",
                "block_id": f"{page_id}:b{order + 1:04d}",
                "block_type": block_type,
                "text": normalize_text(text),
                "reading_order": order,
                "bbox": _union_bbox(group),
                "heading_level": _heading_level(text) if block_type == "heading" else None,
                "extraction_source": group[0].source,
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "quality_flags": flags,
            }
        )
    return blocks
