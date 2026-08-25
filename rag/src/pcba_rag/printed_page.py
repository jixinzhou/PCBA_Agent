from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_LABEL = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,8}|[A-Z]-?\d{1,4})$", re.IGNORECASE)
_READER_OVERLAY = re.compile(r"(?i)(?:page\s*\(|\d+\s*/\s*\d+|100%)")
_FOOTER_LABEL = re.compile(
    r"^(?P<prefix>\d{1,4}|[ivxlcdm]{1,8}|[A-Z]-?\d{1,4})\s+.+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PrintedPage:
    label: str | None
    source: str
    confidence: float | None


def detect_printed_page(blocks: list[dict[str, Any]]) -> PrintedPage:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in blocks:
        text = block["text"].strip()
        if _READER_OVERLAY.search(text):
            continue
        bbox = block["bbox"]
        near_bottom = bbox[1] >= 0.84
        near_top = bbox[3] <= 0.12
        if not (near_bottom or near_top):
            continue
        label: str | None = text if _LABEL.fullmatch(text) else None
        if label is None and block["block_type"] == "footer":
            footer_match = _FOOTER_LABEL.match(text)
            if footer_match:
                label = footer_match.group("prefix")
        if label is None:
            continue
        score = 0.98 if block["block_type"] == "page_number" else 0.9
        if near_bottom:
            score += 0.01
        candidates.append((min(score, 0.99), {**block, "text": label}))
    if not candidates:
        return PrintedPage(None, "none", None)
    score, block = max(candidates, key=lambda item: item[0])
    source = block["extraction_source"]
    return PrintedPage(block["text"].strip(), source, round(score, 4))
