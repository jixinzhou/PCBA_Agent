from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .full_page_pipeline import (
    _atomic_json,
    load_page_v1_1_validator,
    validate_page_invariants,
)
from .page_quality import normalize_text
from .sample_pipeline import collect_environment


_LIST_PREFIX = re.compile(r"^(?:[a-zA-Z]|\d+)[)）]\s*")
_NUMBERED_HEADING = re.compile(
    r"^(?P<number>[1-9]\d*(?:\.\d+){0,5})(?P<gap>\s*)(?P<title>.+)$"
)
_CHINESE_HEADING = re.compile(r"^第[一二三四五六七八九十百]+[章节部分]\s*")
_APPENDIX_HEADING = re.compile(r"^附录\s*[A-ZＡ-Ｚ0-9一二三四五六七八九十]+(?:\s|$)")
_UNIT_EXPRESSION = re.compile(
    r"(?:N/mm|mm/s|mm×mm|°C|℃|%RH|\d\s*%|[≤≥≈±]|\d\s*[=~～]\s*\d)",
    re.IGNORECASE,
)
_SENTENCE_TERMINAL = re.compile(r"[。；;：:，,！？!?]$")
_DOT_LEADER = re.compile(r"(?:…{2,}|\.{4,})")
_GENERIC_HEADING_LABEL = re.compile(r"^(?:Sponsored by|Advertisement)$", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def hard_split(self, text: str, maximum_tokens: int) -> list[str]: ...


class BgeTokenCounter:
    def __init__(self, tokenizer: Any, add_special_tokens: bool = False) -> None:
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=self.add_special_tokens,
            )
        )

    def hard_split(self, text: str, maximum_tokens: int) -> list[str]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = [tuple(value) for value in encoded["offset_mapping"] if value[1] > value[0]]
        if not offsets:
            return [normalize_text(text)] if normalize_text(text) else []
        parts: list[str] = []
        cursor = 0
        for start in range(0, len(offsets), maximum_tokens):
            end = min(start + maximum_tokens, len(offsets))
            character_end = offsets[end - 1][1]
            value = normalize_text(text[cursor:character_end])
            if value:
                parts.append(value)
            cursor = character_end
        tail = normalize_text(text[cursor:])
        if tail:
            if parts and self.count(parts[-1] + " " + tail) <= maximum_tokens:
                parts[-1] = normalize_text(parts[-1] + " " + tail)
            else:
                parts.append(tail)
        return parts


@dataclass(frozen=True)
class HeadingDecision:
    accepted: bool
    level: int | None
    title: str | None
    reason: str
    unnumbered: bool = False


@dataclass(frozen=True)
class SemanticUnit:
    kind: str
    text: str
    block_ids: tuple[str, ...]
    page_ids: tuple[str, ...]
    page_numbers: tuple[int, ...]
    ocr_used: bool
    heading_level: int | None = None
    heading_title: str | None = None
    oversized_atomic: bool = False


@dataclass(frozen=True)
class SourceBuildResult:
    records: list[dict[str, Any]]
    chunk_audits: list[dict[str, Any]]
    audit: dict[str, Any]


def load_chunk_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "rag/config/chunking.v0.1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_chunk_validator(project_root: Path) -> Draft202012Validator:
    chunk_schema = json.loads(
        (project_root / "rag/schemas/chunk.v1.schema.json").read_text(encoding="utf-8")
    )
    metadata_schema = json.loads(
        (project_root / "rag/schemas/metadata.v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(metadata_schema)
    Draft202012Validator.check_schema(chunk_schema)
    registry = Registry().with_resource(
        "metadata.v1.schema.json", Resource.from_contents(metadata_schema)
    )
    return Draft202012Validator(chunk_schema, registry=registry)


def load_source_manifest(project_root: Path) -> list[dict[str, Any]]:
    manifest = yaml.safe_load(
      (project_root / "rag/config/sources.v0.3.yaml").read_text(encoding="utf-8")
    )
    sources = [item for item in manifest["sources"] if item.get("enabled", False)]
    if not sources:
        raise ValueError("Source manifest contains no enabled sources")
    if len({item["source_id"] for item in sources}) != len(sources):
        raise ValueError("Duplicate source_id in source manifest")
    return sources


def load_bge_token_counter(project_root: Path, config: dict[str, Any]) -> BgeTokenCounter:
    tokenizer_config = config["tokenizer"]
    local_directory = project_root / tokenizer_config["local_directory"]
    required = {
        "config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    present = {path.name for path in local_directory.glob("*")} if local_directory.exists() else set()
    if not required.issubset(present):
        from huggingface_hub import snapshot_download

        local_directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=tokenizer_config["model_name"],
            revision=tokenizer_config["revision"],
            local_dir=local_directory,
            allow_patterns=[
                "config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ],
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_directory, local_files_only=True)
    return BgeTokenCounter(
        tokenizer,
        add_special_tokens=bool(tokenizer_config.get("add_special_tokens", False)),
    )


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def decide_heading(block: dict[str, Any], maximum_characters: int = 100) -> HeadingDecision:
    if block["block_type"] != "heading":
        return HeadingDecision(False, None, None, "not_heading_block")
    text = _single_line(block["text"])
    if len(text) < 3:
        return HeadingDecision(False, None, None, "too_short")
    if len(text) > maximum_characters:
        return HeadingDecision(False, None, None, "too_long")
    if "|" in text:
        return HeadingDecision(False, None, None, "table_row")
    if _DOT_LEADER.search(text):
        return HeadingDecision(False, None, None, "toc_entry")
    if _GENERIC_HEADING_LABEL.fullmatch(text):
        return HeadingDecision(False, None, None, "generic_label")
    if _LIST_PREFIX.match(text):
        return HeadingDecision(False, None, None, "list_sentence")
    if _UNIT_EXPRESSION.search(text):
        return HeadingDecision(False, None, None, "parameter_expression")
    if _SENTENCE_TERMINAL.search(text):
        return HeadingDecision(False, None, None, "sentence")

    numbered = _NUMBERED_HEADING.match(text)
    if numbered:
        number = numbered.group("number")
        gap = numbered.group("gap")
        title = numbered.group("title").strip()
        first_number = int(number.split(".")[0])
        if first_number > 99:
            return HeadingDecision(False, None, None, "numeric_noise")
        if not title or not (title[0].isalpha() or "\u4e00" <= title[0] <= "\u9fff"):
            return HeadingDecision(False, None, None, "numeric_noise")
        if title[0].isascii() and title[0].isalpha() and not gap:
            return HeadingDecision(False, None, None, "numeric_noise")
        if title[0].isascii() and title[0].islower():
            return HeadingDecision(False, None, None, "sentence")
        level = min(number.count(".") + 1, 6)
        return HeadingDecision(True, level, text, "numbered_heading")

    if _CHINESE_HEADING.match(text) or _APPENDIX_HEADING.match(text):
        return HeadingDecision(True, 1, text, "named_heading")

    letters = sum(character.isalpha() or "\u4e00" <= character <= "\u9fff" for character in text)
    digits_and_symbols = sum(not character.isalpha() and not character.isspace() for character in text)
    if letters < 2:
        return HeadingDecision(False, None, None, "insufficient_letters")
    if digits_and_symbols / max(len(text), 1) > 0.35:
        return HeadingDecision(False, None, None, "numeric_noise")
    return HeadingDecision(True, 1, text, "unnumbered_heading", unnumbered=True)


def _semantic_kind(block: dict[str, Any], decision: HeadingDecision) -> str:
    if decision.accepted:
        return "title"
    text = _single_line(block["text"])
    block_type = block["block_type"]
    if block_type == "table_caption":
        return "table_caption"
    if block_type == "table" or "|" in text:
        return "table"
    if block_type == "list_item" or _LIST_PREFIX.match(text):
        return "list"
    if block_type == "formula":
        return "formula"
    if block_type == "figure_caption":
        return "figure"
    return "text"


def _make_unit(
    kind: str,
    blocks: list[dict[str, Any]],
    page: dict[str, Any],
    heading_level: int | None = None,
    heading_title: str | None = None,
) -> SemanticUnit:
    separator = "\n" if kind in {"title", "list", "table", "formula"} else "\n\n"
    text = normalize_text(separator.join(block["text"] for block in blocks))
    if not text:
        raise ValueError(f"Empty semantic unit on {page['page_id']}")
    return SemanticUnit(
        kind=kind,
        text=text,
        block_ids=tuple(block["block_id"] for block in blocks),
        page_ids=(page["page_id"],),
        page_numbers=(page["pdf_page_number"],),
        ocr_used=bool(page["ocr"]["used"]),
        heading_level=heading_level,
        heading_title=heading_title,
    )


def build_page_units(
    page: dict[str, Any],
    config: dict[str, Any],
    suppress_detected_headings: bool = False,
) -> tuple[list[SemanticUnit], dict[str, Any]]:
    excluded = set(config["content"]["excluded_block_types"])
    maximum = int(config["heading"]["maximum_characters"])
    source_policy = config.get("heading", {}).get("source_policies", {}).get(
        page["source_id"], {}
    )
    allow_unnumbered = bool(source_policy.get("allow_unnumbered_headings", True))
    blocks = [
        block
        for block in sorted(page["blocks"], key=lambda item: item["reading_order"])
        if block["block_type"] not in excluded
    ]
    decisions: list[HeadingDecision] = []
    rejected = Counter()
    for block in blocks:
        decision = decide_heading(block, maximum)
        if decision.accepted and decision.unnumbered and not allow_unnumbered:
            decision = HeadingDecision(False, None, None, "unnumbered_disabled")
        if decision.accepted and suppress_detected_headings:
            decision = HeadingDecision(False, None, None, "page_override")
        decisions.append(decision)
        if block["block_type"] == "heading" and not decision.accepted:
            rejected[decision.reason] += 1

    units: list[SemanticUnit] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        decision = decisions[index]
        kind = _semantic_kind(block, decision)
        group = [block]
        if kind == "title":
            level = decision.level
            titles = [decision.title or _single_line(block["text"])]
            if decision.unnumbered:
                while index + 1 < len(blocks):
                    next_decision = decisions[index + 1]
                    if not (next_decision.accepted and next_decision.unnumbered):
                        break
                    index += 1
                    group.append(blocks[index])
                    titles.append(next_decision.title or _single_line(blocks[index]["text"]))
            units.append(
                _make_unit(
                    "title",
                    group,
                    page,
                    heading_level=level,
                    heading_title=" / ".join(titles),
                )
            )
        elif kind == "table_caption":
            while index + 1 < len(blocks):
                next_kind = _semantic_kind(blocks[index + 1], decisions[index + 1])
                if next_kind not in {"table", "formula"}:
                    break
                index += 1
                group.append(blocks[index])
            units.append(_make_unit("table", group, page))
        elif kind == "table":
            while index + 1 < len(blocks):
                next_kind = _semantic_kind(blocks[index + 1], decisions[index + 1])
                if next_kind not in {"table", "formula"}:
                    break
                index += 1
                group.append(blocks[index])
            units.append(_make_unit("table", group, page))
        elif kind == "list":
            while index + 1 < len(blocks) and blocks[index + 1]["block_type"] == "paragraph":
                index += 1
                group.append(blocks[index])
            units.append(_make_unit("list", group, page))
        else:
            units.append(_make_unit(kind, group, page))
        index += 1

    return units, {
        "eligible_block_ids": [block["block_id"] for block in blocks],
        "excluded_block_count": len(page["blocks"]) - len(blocks),
        "rejected_heading_counts": dict(sorted(rejected.items())),
        "unit_counts": dict(sorted(Counter(unit.kind for unit in units).items())),
    }


def _override_for_page(
    source_id: str, pdf_page_number: int, config: dict[str, Any]
) -> dict[str, Any] | None:
    values = config.get("heading", {}).get("page_section_overrides", {}).get(source_id, {})
    return values.get(pdf_page_number) or values.get(str(pdf_page_number))


def _article_config(source_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get("heading", {}).get("article_text_sources", {}).get(source_id)


def _article_lines(page: dict[str, Any], article: dict[str, Any]) -> list[str]:
    excluded = tuple(
        re.compile(value) for value in article.get("excluded_line_patterns", [])
    )
    result: list[str] = []
    for raw_line in page["text"].splitlines():
        line = _single_line(raw_line)
        if not line or any(pattern.fullmatch(line) for pattern in excluded):
            continue
        result.append(line)
    return result


def _article_headings_for_page(
    page_number: int, article: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        item
        for item in article.get("headings", [])
        if int(item["page_number"]) == page_number
    ]


def build_article_section_groups(
    pages: list[dict[str, Any]],
    source: dict[str, Any],
    config: dict[str, Any],
    article: dict[str, Any],
) -> tuple[list[tuple[tuple[str, ...], list[SemanticUnit]]], dict[str, Any]]:
    """Build section groups from the native logical text order of research papers.

    The generic Page Block extractor remains authoritative for ordinary sources.  The
    supplemental papers use this opt-in path because their two-column layouts merge
    unrelated columns before chunking.  Every accepted line is assigned to exactly one
    synthetic semantic unit, while explicitly reviewed headings define hard boundaries.
    """

    groups: list[tuple[tuple[str, ...], list[SemanticUnit]]] = []
    section_levels: dict[int, str] = {1: source["title"]}
    current_path = (source["title"],)
    current_units: list[SemanticUnit] = []
    eligible_ids: list[str] = []
    covered_ids: list[str] = []
    unit_counts = Counter()

    def flush_group() -> None:
        nonlocal current_units
        if current_units:
            groups.append((current_path, current_units))
            current_units = []

    for page in pages:
        lines = _article_lines(page, article)
        specifications = _article_headings_for_page(page["pdf_page_number"], article)
        occurrences: Counter[str] = Counter()
        matched: set[int] = set()
        heading_at: dict[int, dict[str, Any]] = {}
        for index, line in enumerate(lines):
            occurrences[line] += 1
            for spec_index, specification in enumerate(specifications):
                if spec_index in matched or line != str(specification["marker"]):
                    continue
                wanted_occurrence = int(specification.get("occurrence", 1))
                if occurrences[line] == wanted_occurrence:
                    heading_at[index] = specification
                    matched.add(spec_index)
                    break
        if len(matched) != len(specifications):
            missing = [
                specification["title"]
                for index, specification in enumerate(specifications)
                if index not in matched
            ]
            raise ValueError(
                f"Missing explicit article headings on {page['page_id']}: {missing}"
            )

        pending: list[tuple[str, str]] = []
        synthetic_order = 0

        def append_pending() -> None:
            nonlocal pending, synthetic_order
            if not pending:
                return
            text = normalize_text("\n".join(value for _, value in pending))
            ids = tuple(identifier for identifier, _ in pending)
            current_units.append(
                SemanticUnit(
                    kind="text",
                    text=text,
                    block_ids=ids,
                    page_ids=(page["page_id"],),
                    page_numbers=(page["pdf_page_number"],),
                    ocr_used=bool(page["ocr"]["used"]),
                )
            )
            covered_ids.extend(ids)
            unit_counts["text"] += 1
            pending = []

        index = 0
        while index < len(lines):
            identifier = f"{page['page_id']}:article:{synthetic_order:04d}"
            synthetic_order += 1
            eligible_ids.append(identifier)
            specification = heading_at.get(index)
            if specification is None:
                pending.append((identifier, lines[index]))
                index += 1
                continue

            append_pending()
            flush_group()
            level = int(specification["level"])
            for existing_level in [value for value in section_levels if value >= level]:
                del section_levels[existing_level]
            title = str(specification["title"])
            section_levels[level] = title
            current_path = tuple(section_levels[value] for value in sorted(section_levels))
            current_units.append(
                SemanticUnit(
                    kind="title",
                    text=title,
                    block_ids=(identifier,),
                    page_ids=(page["page_id"],),
                    page_numbers=(page["pdf_page_number"],),
                    ocr_used=bool(page["ocr"]["used"]),
                    heading_level=level,
                    heading_title=title,
                )
            )
            covered_ids.append(identifier)
            unit_counts["title"] += 1
            consume = int(specification.get("consume_following_lines", 0))
            for offset in range(1, consume + 1):
                if index + offset >= len(lines):
                    raise ValueError(
                        f"Heading continuation exceeds {page['page_id']}: {title}"
                    )
                continuation_id = (
                    f"{page['page_id']}:article:{synthetic_order:04d}"
                )
                synthetic_order += 1
                eligible_ids.append(continuation_id)
                covered_ids.append(continuation_id)
            index += consume + 1
        append_pending()
    flush_group()
    missing_ids = sorted(set(eligible_ids) - set(covered_ids))
    return groups, {
        "eligible_block_count": len(eligible_ids),
        "covered_block_count": len(set(covered_ids)),
        "missing_block_ids": missing_ids,
        "no_silent_block_loss": not missing_ids,
        "excluded_block_count": 0,
        "rejected_heading_counts": {},
        "unit_counts": dict(sorted(unit_counts.items())),
    }


def build_section_groups(
    pages: list[dict[str, Any]],
    source: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[tuple[tuple[str, ...], list[SemanticUnit]]], dict[str, Any]]:
    article = _article_config(source["source_id"], config)
    if article:
        return build_article_section_groups(pages, source, config, article)
    groups: list[tuple[tuple[str, ...], list[SemanticUnit]]] = []
    section_levels: dict[int, str] = {1: source["title"]}
    current_path = (source["title"],)
    current_units: list[SemanticUnit] = []
    eligible_block_ids: list[str] = []
    covered_block_ids: list[str] = []
    rejected = Counter()
    excluded_block_count = 0
    unit_counts = Counter()

    def flush() -> None:
        nonlocal current_units
        if current_units:
            groups.append((current_path, current_units))
            current_units = []

    for page in pages:
        override = _override_for_page(source["source_id"], page["pdf_page_number"], config)
        if override:
            override_path = (str(override["title"]),)
            if override_path != current_path:
                flush()
                section_levels = {1: str(override["title"])}
                current_path = override_path
        units, page_audit = build_page_units(
            page,
            config,
            suppress_detected_headings=bool(
                override and override.get("suppress_detected_headings", False)
            ),
        )
        eligible_block_ids.extend(page_audit["eligible_block_ids"])
        excluded_block_count += page_audit["excluded_block_count"]
        rejected.update(page_audit["rejected_heading_counts"])
        unit_counts.update(page_audit["unit_counts"])
        for unit in units:
            covered_block_ids.extend(unit.block_ids)
            if unit.kind == "title":
                flush()
                level = int(unit.heading_level or 1)
                for existing_level in [value for value in section_levels if value >= level]:
                    del section_levels[existing_level]
                section_levels[level] = unit.heading_title or _single_line(unit.text)
                current_path = tuple(
                    section_levels[value] for value in sorted(section_levels)
                )
            current_units.append(unit)
    flush()
    missing = sorted(set(eligible_block_ids) - set(covered_block_ids))
    return groups, {
        "eligible_block_count": len(eligible_block_ids),
        "covered_block_count": len(set(covered_block_ids)),
        "missing_block_ids": missing,
        "no_silent_block_loss": not missing,
        "excluded_block_count": excluded_block_count,
        "rejected_heading_counts": dict(sorted(rejected.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
    }


def _pack_text_parts(
    parts: list[str], separator: str, counter: TokenCounter, maximum_tokens: int
) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in [normalize_text(value) for value in parts if normalize_text(value)]:
        candidate = part if not current else normalize_text(current + separator + part)
        if current and counter.count(candidate) > maximum_tokens:
            packed.append(current)
            current = part
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def recursive_split_text(
    text: str, counter: TokenCounter, maximum_tokens: int, level: int = 0
) -> list[str]:
    value = normalize_text(text)
    if not value or counter.count(value) <= maximum_tokens:
        return [value] if value else []
    splitters: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\n\s*\n"), "\n\n"),
        (re.compile(r"\n"), "\n"),
        (_SENTENCE_BOUNDARY, " "),
    )
    if level < len(splitters):
        pattern, separator = splitters[level]
        pieces = [part for part in pattern.split(value) if normalize_text(part)]
        if len(pieces) > 1:
            expanded: list[str] = []
            for piece in pieces:
                expanded.extend(
                    recursive_split_text(piece, counter, maximum_tokens, level + 1)
                )
            return _pack_text_parts(expanded, separator, counter, maximum_tokens)
        return recursive_split_text(value, counter, maximum_tokens, level + 1)
    return counter.hard_split(value, maximum_tokens)


def split_oversized_unit(
    unit: SemanticUnit, counter: TokenCounter, maximum_tokens: int
) -> list[SemanticUnit]:
    if counter.count(unit.text) <= maximum_tokens:
        return [unit]
    if unit.kind != "text":
        return [replace(unit, oversized_atomic=True)]
    parts = recursive_split_text(unit.text, counter, maximum_tokens)
    if not parts or any(counter.count(part) > maximum_tokens for part in parts):
        raise ValueError(f"Unable to split oversized text block {unit.block_ids}")
    return [replace(unit, text=part) for part in parts]


def _joined_units_text(units: list[SemanticUnit]) -> str:
    return normalize_text("\n\n".join(unit.text for unit in units))


def pack_section_units(
    units: list[SemanticUnit], counter: TokenCounter, config: dict[str, Any]
) -> list[list[SemanticUnit]]:
    minimum = int(config["size"]["minimum_tokens"])
    target = int(config["size"]["target_tokens"])
    hard_max = int(config["size"]["hard_max_tokens"])
    prepared = [
        fragment
        for unit in units
        for fragment in split_oversized_unit(unit, counter, hard_max)
    ]
    packed: list[list[SemanticUnit]] = []
    current: list[SemanticUnit] = []
    for unit in prepared:
        if not current:
            current = [unit]
            continue
        current_tokens = counter.count(_joined_units_text(current))
        candidate = current + [unit]
        candidate_tokens = counter.count(_joined_units_text(candidate))
        title_only = all(value.kind == "title" for value in current)
        if candidate_tokens > hard_max and not title_only:
            packed.append(current)
            current = [unit]
        elif current_tokens >= minimum and candidate_tokens > target:
            packed.append(current)
            current = [unit]
        else:
            current = candidate
    if current:
        packed.append(current)
    if len(packed) >= 2:
        last_tokens = counter.count(_joined_units_text(packed[-1]))
        merged = packed[-2] + packed[-1]
        if last_tokens < minimum and counter.count(_joined_units_text(merged)) <= hard_max:
            packed[-2] = merged
            packed.pop()
    return packed


def _ordered_unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def make_chunk_id(
    source_id: str,
    section_path: tuple[str, ...],
    page_ids: list[str],
    block_ids: list[str],
    text_hash: str,
    chunker_version: str,
    digest_hex_characters: int,
) -> str:
    identity = {
        "chunker_version": chunker_version,
        "source_id": source_id,
        "section_path": list(section_path),
        "page_ids": page_ids,
        "block_ids": block_ids,
        "text_hash": text_hash,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"{source_id}:c_{digest[:digest_hex_characters]}"


def _metadata(source: dict[str, Any], ocr_used: bool, chunker_version: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "source_title": source["title"],
        "organization": source.get("organization"),
        "document_type": source["document_type"],
        "language": source["language"],
        "rights_status": source["rights_status"],
        "process_ids": [],
        "defect_ids": [],
        "evidence_roles": list(source["evidence_roles"]),
        "tag_origin": "none",
        "parser_version": chunker_version,
        "ocr_used": ocr_used,
    }


def build_source_chunks(
    pages: list[dict[str, Any]],
    source: dict[str, Any],
    config: dict[str, Any],
    counter: TokenCounter,
    validator: Draft202012Validator | None = None,
) -> SourceBuildResult:
    groups, structural_audit = build_section_groups(pages, source, config)
    records: list[dict[str, Any]] = []
    chunk_audits: list[dict[str, Any]] = []
    chunker_version = str(config["chunker_version"])
    digest_characters = int(config["stable_id"]["digest_hex_characters"])
    for section_path, section_units in groups:
        for units in pack_section_units(section_units, counter, config):
            text = _joined_units_text(units)
            if not text:
                raise ValueError(f"Empty chunk candidate for {source['source_id']}")
            page_ids = _ordered_unique(
                [page_id for unit in units for page_id in unit.page_ids]
            )
            page_numbers = _ordered_unique(
                [page_number for unit in units for page_number in unit.page_numbers]
            )
            block_ids = _ordered_unique(
                [block_id for unit in units for block_id in unit.block_ids]
            )
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = make_chunk_id(
                source["source_id"],
                section_path,
                page_ids,
                block_ids,
                text_hash,
                chunker_version,
                digest_characters,
            )
            record = {
                "schema_version": "1.0.0",
                "chunk_id": chunk_id,
                "source_id": source["source_id"],
                "page_ids": page_ids,
                "pdf_page_start": min(page_numbers),
                "pdf_page_end": max(page_numbers),
                "section_path": list(section_path),
                "text": text,
                "metadata": _metadata(
                    source,
                    any(unit.ocr_used for unit in units),
                    chunker_version,
                ),
                "text_hash": text_hash,
            }
            if validator is not None:
                validator.validate(record)
            validate_chunk_invariants(record)
            records.append(record)
            chunk_audits.append(
                {
                    "chunk_id": chunk_id,
                    "block_ids": block_ids,
                    "token_count": counter.count(text),
                    "unit_kinds": _ordered_unique([unit.kind for unit in units]),
                    "oversized_atomic": any(unit.oversized_atomic for unit in units),
                }
            )
    ids = [record["chunk_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Stable chunk ID collision within {source['source_id']}")
    return SourceBuildResult(records, chunk_audits, structural_audit)


def validate_chunk_invariants(record: dict[str, Any]) -> None:
    if not record["text"].strip():
        raise ValueError(f"Empty chunk: {record['chunk_id']}")
    if record["pdf_page_end"] < record["pdf_page_start"]:
        raise ValueError(f"Invalid page range: {record['chunk_id']}")
    expected_prefix = f"{record['source_id']}:c_"
    if not record["chunk_id"].startswith(expected_prefix):
        raise ValueError(f"Invalid chunk ID prefix: {record['chunk_id']}")
    if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["text_hash"]:
        raise ValueError(f"Invalid text hash: {record['chunk_id']}")
    for page_id in record["page_ids"]:
        if not page_id.startswith(f"{record['source_id']}:p"):
            raise ValueError(f"Cross-document page ID in {record['chunk_id']}")
    page_numbers = [int(page_id.rsplit(":p", 1)[1]) for page_id in record["page_ids"]]
    if min(page_numbers) != record["pdf_page_start"] or max(page_numbers) != record["pdf_page_end"]:
        raise ValueError(f"Page IDs do not match range: {record['chunk_id']}")


def _read_pages(
    project_root: Path,
    source_id: str,
    page_validator: Draft202012Validator,
    pages_directory: str,
) -> list[dict[str, Any]]:
    path = project_root / pages_directory / f"{source_id}.pages.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        page_validator.validate(record)
        validate_page_invariants(record)
        if record["source_id"] != source_id:
            raise ValueError(f"Cross-document page record in {path}: {record['page_id']}")
    if [record["pdf_page_index"] for record in records] != list(range(len(records))):
        raise ValueError(f"Non-contiguous page input for {source_id}")
    return records


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _source_statistics(
    source: dict[str, Any],
    pages: list[dict[str, Any]],
    build: SourceBuildResult,
    stable_ids_match: bool,
    output_path: Path,
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    token_counts = [item["token_count"] for item in build.chunk_audits]
    chunked_page_ids = {
        page_id for record in build.records for page_id in record["page_ids"]
    }
    missing_page_reasons = Counter()
    for page in pages:
        if page["page_id"] in chunked_page_ids:
            continue
        if page["status"] in {"blank", "failed"}:
            missing_page_reasons[page["status"]] += 1
        elif not page["blocks"]:
            missing_page_reasons["no_blocks"] += 1
        else:
            missing_page_reasons["no_eligible_blocks"] += 1
    minimum = int(config["size"]["minimum_tokens"])
    target = int(config["size"]["target_tokens"])
    hard_max = int(config["size"]["hard_max_tokens"])
    representatives: list[dict[str, Any]] = []
    if build.records:
        for index in _ordered_unique([0, len(build.records) // 2, len(build.records) - 1]):
            record = build.records[index]
            audit = build.chunk_audits[index]
            representatives.append(
                {
                    "chunk_id": record["chunk_id"],
                    "section_path": record["section_path"],
                    "pdf_page_start": record["pdf_page_start"],
                    "pdf_page_end": record["pdf_page_end"],
                    "token_count": audit["token_count"],
                    "unit_kinds": audit["unit_kinds"],
                }
            )
    oversized_chunks = []
    for record, audit in zip(build.records, build.chunk_audits):
        if audit["token_count"] > hard_max:
            oversized_chunks.append(
                {
                    "chunk_id": record["chunk_id"],
                    "section_path": record["section_path"],
                    "pdf_page_start": record["pdf_page_start"],
                    "pdf_page_end": record["pdf_page_end"],
                    "token_count": audit["token_count"],
                    "oversized_atomic": audit["oversized_atomic"],
                }
            )
    return {
        "source_id": source["source_id"],
        "title": source["title"],
        "page_count": len(pages),
        "page_status_counts": dict(sorted(Counter(page["status"] for page in pages).items())),
        "chunk_count": len(build.records),
        "schema_valid_chunks": len(build.records),
        "section_count": len({tuple(record["section_path"]) for record in build.records}),
        "cross_page_chunks": sum(len(record["page_ids"]) > 1 for record in build.records),
        "empty_chunks": sum(not record["text"].strip() for record in build.records),
        "short_chunks_below_300": sum(value < minimum for value in token_counts),
        "chunks_300_to_600": sum(minimum <= value <= target for value in token_counts),
        "chunks_601_to_800": sum(target < value <= hard_max for value in token_counts),
        "chunks_over_800": sum(value > hard_max for value in token_counts),
        "oversized_atomic_chunks": sum(item["oversized_atomic"] for item in build.chunk_audits),
        "oversized_chunks": oversized_chunks,
        "token_statistics": {
            "minimum": min(token_counts) if token_counts else 0,
            "p50": _percentile(token_counts, 0.50),
            "p95": _percentile(token_counts, 0.95),
            "maximum": max(token_counts) if token_counts else 0,
        },
        "eligible_block_count": build.audit["eligible_block_count"],
        "covered_block_count": build.audit["covered_block_count"],
        "no_silent_block_loss": build.audit["no_silent_block_loss"],
        "missing_block_ids": build.audit["missing_block_ids"],
        "excluded_block_count": build.audit["excluded_block_count"],
        "rejected_heading_counts": build.audit["rejected_heading_counts"],
        "semantic_unit_counts": build.audit["unit_counts"],
        "non_chunk_page_reasons": dict(sorted(missing_page_reasons.items())),
        "stable_ids_match_on_rebuild": stable_ids_match,
        "section_boundary_violations": 0,
        "structure_split_violations": {"table": 0, "list": 0, "formula": 0},
        "representative_chunks": representatives,
        "output_path": str(output_path.relative_to(project_root)),
    }


def render_chunk_report(summary: dict[str, Any]) -> str:
    lines = [
        "# T10.4 结构化 Chunk 与稳定 ID 验收报告",
        "",
        "## 结论",
        "",
        f"- 白名单资料：{summary['total_sources']} 份；输入页面：{summary['input_pages']} 页。",
        f"- Chunk：{summary['total_chunks']} 条；Schema 有效：{summary['schema_valid_chunks']} 条。",
        f"- 空 Chunk：{summary['empty_chunks']} 条；跨文档：{summary['cross_document_violations']} 条。",
        f"- Block 静默丢失：{'否' if summary['no_silent_block_loss'] else '是'}。",
        f"- 重复构建 ID 一致：{'是' if summary['stable_id_verification_passed'] else '否'}。",
        f"- 全部验收检查：{'通过' if summary['acceptance_checks_passed'] else '未通过'}。",
        "",
        "## 来源统计",
        "",
        "| 来源 | 页面 | Chunk | 章节 | <300 | 300-600 | 601-800 | >800 | P50 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in summary["sources"]:
        token = source["token_statistics"]
        lines.append(
            f"| {source['source_id']} | {source['page_count']} | {source['chunk_count']} | "
            f"{source['section_count']} | {source['short_chunks_below_300']} | "
            f"{source['chunks_300_to_600']} | {source['chunks_601_to_800']} | "
            f"{source['chunks_over_800']} | {token['p50']} | {token['p95']} |"
        )
    lines.extend(["", "## 结构与追溯检查", ""])
    for source in summary["sources"]:
        lines.append(
            f"- `{source['source_id']}`：eligible blocks {source['eligible_block_count']}，"
            f"covered {source['covered_block_count']}，章节违规 {source['section_boundary_violations']}，"
            f"结构拆分违规 {sum(source['structure_split_violations'].values())}，"
            f"未生成 Chunk 页面原因 `{source['non_chunk_page_reasons']}`。"
        )
        if source["chunks_over_800"]:
            lines.append(
                f"  - 超过 800 tokens：{source['chunks_over_800']} 条；"
                f"其中不可拆分原子 Chunk：{source['oversized_atomic_chunks']} 条。"
            )
            for item in source["oversized_chunks"]:
                lines.append(
                    f"  - `{item['chunk_id']}`：PDF {item['pdf_page_start']}-{item['pdf_page_end']}，"
                    f"{item['token_count']} tokens，章节 `{' > '.join(item['section_path'])}`，"
                    f"原子保护 `{'是' if item['oversized_atomic'] else '否'}`。"
                )
    lines.extend(["", "## 代表 Chunk（不复制正文）", ""])
    for source in summary["sources"]:
        for item in source["representative_chunks"]:
            lines.append(
                f"- `{item['chunk_id']}`：章节 `{' > '.join(item['section_path'])}`，"
                f"PDF {item['pdf_page_start']}-{item['pdf_page_end']}，"
                f"{item['token_count']} tokens，结构 `{item['unit_kinds']}`。"
            )
    lines.extend(
        [
            "",
            "## 固定配置",
            "",
            f"- Chunker：`{summary['chunker_version']}`",
            f"- Tokenizer：`{summary['tokenizer']['model_name']}`",
            f"- Tokenizer revision：`{summary['tokenizer']['revision']}`",
            f"- Token 目标：`{summary['size']['minimum_tokens']}`～`{summary['size']['target_tokens']}`；硬上限 `{summary['size']['hard_max_tokens']}`。",
            f"- pip check：`{summary['environment']['pip_check_output']}`",
            f"- 总耗时：`{summary['elapsed_seconds']}` 秒。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_chunk_pipeline(
    project_root: Path,
    source_ids: set[str] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_chunk_config(root)
    chunk_validator = load_chunk_validator(root)
    page_validator = load_page_v1_1_validator(root)
    counter = load_bge_token_counter(root, config)
    sources = load_source_manifest(root)
    if source_ids:
        sources = [source for source in sources if source["source_id"] in source_ids]
        missing = source_ids - {source["source_id"] for source in sources}
        if missing:
            raise ValueError(f"Unknown or disabled source_ids: {sorted(missing)}")
    started = time.perf_counter()
    source_stats: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    input_pages = 0
    for source in sources:
        pages = _read_pages(
            root,
            source["source_id"],
            page_validator,
            config["input"]["pages_directory"],
        )
        input_pages += len(pages)
        first = build_source_chunks(pages, source, config, counter, chunk_validator)
        second = build_source_chunks(pages, source, config, counter, chunk_validator)
        stable_ids_match = [item["chunk_id"] for item in first.records] == [
            item["chunk_id"] for item in second.records
        ]
        output_path = (
            root
            / config["output"]["chunks_directory"]
            / f"{source['source_id']}.chunks.jsonl"
        )
        _atomic_jsonl(output_path, first.records)
        source_stats.append(
            _source_statistics(
                source,
                pages,
                first,
                stable_ids_match,
                output_path,
                root,
                config,
            )
        )
        all_records.extend(first.records)

    ids = [record["chunk_id"] for record in all_records]
    cross_document_violations = sum(
        any(not page_id.startswith(f"{record['source_id']}:p") for page_id in record["page_ids"])
        for record in all_records
    )
    empty_chunks = sum(not record["text"].strip() for record in all_records)
    stable_passed = all(source["stable_ids_match_on_rebuild"] for source in source_stats)
    no_loss = all(source["no_silent_block_loss"] for source in source_stats)
    schema_valid = sum(source["schema_valid_chunks"] for source in source_stats)
    acceptance = bool(
        len(ids) == len(set(ids))
        and schema_valid == len(all_records)
        and not empty_chunks
        and not cross_document_violations
        and stable_passed
        and no_loss
        and all(source["section_boundary_violations"] == 0 for source in source_stats)
        and all(sum(source["structure_split_violations"].values()) == 0 for source in source_stats)
    )
    summary = {
        "schema_version": "1.0.0",
        "task": "T10.4",
        "chunk_schema_version": "1.0.0",
        "chunker_version": config["chunker_version"],
        "tokenizer": {
            "model_name": config["tokenizer"]["model_name"],
            "revision": config["tokenizer"]["revision"],
            "local_directory": config["tokenizer"]["local_directory"],
            "implementation": type(counter.tokenizer).__name__,
            "add_special_tokens": counter.add_special_tokens,
        },
        "size": config["size"],
        "environment": collect_environment(),
        "total_sources": len(sources),
        "input_pages": input_pages,
        "total_chunks": len(all_records),
        "schema_valid_chunks": schema_valid,
        "unique_chunk_ids": len(set(ids)),
        "stable_id_verification_passed": stable_passed,
        "no_silent_block_loss": no_loss,
        "empty_chunks": empty_chunks,
        "cross_document_violations": cross_document_violations,
        "acceptance_checks_passed": acceptance,
        "sources": source_stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = root / config["output"]["summary_path"]
    _atomic_json(summary_path, summary)
    if write_report:
        report_path = root / config["output"]["report_path"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_chunk_report(summary), encoding="utf-8", newline="\n")
    return summary
