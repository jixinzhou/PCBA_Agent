from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .page_quality import normalize_text


@dataclass(frozen=True)
class PdfAudit:
    page_count: int
    encrypted: bool
    native_texts: tuple[str, ...]


def audit_pdf(path: Path) -> PdfAudit:
    reader = PdfReader(str(path))
    encrypted = bool(reader.is_encrypted)
    texts: list[str] = []
    for page in reader.pages:
        texts.append(normalize_text(page.extract_text() or ""))
    return PdfAudit(len(reader.pages), encrypted, tuple(texts))


def table_likeness(text: str) -> int:
    score = 0
    for line in text.splitlines():
        numeric_tokens = len([token for token in line.split() if any(char.isdigit() for char in token)])
        if numeric_tokens >= 2:
            score += numeric_tokens
        if "\t" in line or re_has_column_gap(line):
            score += 2
    return score


def re_has_column_gap(line: str) -> bool:
    return "  " in line


def select_sample_indices(native_texts: tuple[str, ...], sample_count: int) -> list[int]:
    page_count = len(native_texts)
    if page_count == 0:
        return []
    indices = range(page_count)
    min_text = min(indices, key=lambda index: len(native_texts[index]))
    max_text = max(indices, key=lambda index: len(native_texts[index]))
    table_page = max(indices, key=lambda index: table_likeness(native_texts[index]))
    candidates = [
        0,
        min_text,
        max_text,
        table_page,
        page_count // 2,
        page_count - 1,
        page_count // 4,
        (3 * page_count) // 4,
    ]
    selected: list[int] = []
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) == min(sample_count, page_count):
            break
    if len(selected) < min(sample_count, page_count):
        for index in indices:
            if index not in selected:
                selected.append(index)
            if len(selected) == min(sample_count, page_count):
                break
    return selected


def selection_reason(index: int, native_texts: tuple[str, ...]) -> str:
    page_count = len(native_texts)
    reasons: list[str] = []
    if index == 0:
        reasons.append("first_page")
    if index == page_count - 1:
        reasons.append("last_page")
    if index == page_count // 2:
        reasons.append("middle_page")
    if index == min(range(page_count), key=lambda item: len(native_texts[item])):
        reasons.append("lowest_native_text")
    if index == max(range(page_count), key=lambda item: len(native_texts[item])):
        reasons.append("highest_native_text")
    if index == max(range(page_count), key=lambda item: table_likeness(native_texts[item])):
        reasons.append("table_like")
    return ",".join(reasons) or "coverage_page"
