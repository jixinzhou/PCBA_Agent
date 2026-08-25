from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityResult:
    assessment: str
    text_char_count: int
    flags: tuple[str, ...]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return "\n".join(normalized).strip()


def assess_text(
    text: str,
    minimum_text_chars: int,
    maximum_replacement_ratio: float,
) -> QualityResult:
    flags: list[str] = []
    count = len(text)
    if count == 0:
        flags.append("empty_text")
    elif count < minimum_text_chars:
        flags.append("low_text")

    replacement_ratio = text.count("\ufffd") / max(count, 1)
    if replacement_ratio > maximum_replacement_ratio:
        flags.append("replacement_characters")

    if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text):
        flags.append("control_characters")

    if "empty_text" in flags:
        assessment = "fail"
    elif flags:
        assessment = "warn"
    else:
        assessment = "pass"
    return QualityResult(assessment, count, tuple(flags))
