from __future__ import annotations

import json
import re
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pypdf import PdfReader
from referencing import Registry, Resource

from .block_extractor import build_blocks, native_layout_lines, ocr_layout_lines
from .manifest import SourceSpec, load_enabled_sources, validate_source
from .ocr import LocalOcrEngine, OcrResult
from .page_quality import assess_text, normalize_text
from .printed_page import detect_printed_page
from .sample_pipeline import collect_environment


def _package_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "not-installed"


def load_full_ingestion_config(root: Path) -> dict[str, Any]:
    path = root / "rag/config/ingestion.v0.2.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_page_v1_1_validator(root: Path) -> Draft202012Validator:
    page_schema = json.loads(
        (root / "rag/schemas/page.v1.1.schema.json").read_text(encoding="utf-8")
    )
    block_schema = json.loads(
        (root / "rag/schemas/block.v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(block_schema)
    Draft202012Validator.check_schema(page_schema)
    registry = Registry().with_resource(
        "block.v1.schema.json", Resource.from_contents(block_schema)
    )
    return Draft202012Validator(page_schema, registry=registry)


def _compile_ignored_patterns(config: dict[str, Any]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value) for value in config["blank_page"]["ignored_patterns"])


def clean_reader_overlays(text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    cleaned_lines: list[str] = []
    overlay_inline = re.compile(
        r"(?i)(?:[-+]?\s*100%\s*[-+]?|\+?\s*page\s*\(\s*\d+\s*/\s*\d+\s*\))"
    )
    for line in normalize_text(text).splitlines():
        pieces = [piece.strip() for piece in line.split("|")]
        kept: list[str] = []
        for piece in pieces:
            if not piece or any(pattern.fullmatch(piece) for pattern in patterns):
                continue
            value = overlay_inline.sub("", piece).strip(" +-|")
            if value:
                kept.append(value)
        if kept:
            cleaned_lines.append(" | ".join(kept))
    return normalize_text("\n".join(cleaned_lines))


def _is_overlay_block(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return bool(text.strip()) and not clean_reader_overlays(text, patterns)


def _ocr_info(result: OcrResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "used": False,
            "average_score": None,
            "line_count": None,
            "elapsed_seconds": None,
        }
    return {
        "used": True,
        "average_score": round(result.average_score, 4) if result.average_score is not None else None,
        "line_count": result.line_count,
        "elapsed_seconds": round(result.elapsed_seconds, 4),
    }


def _failed_record(source: SourceSpec, page_index: int, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "1.1.0",
        "page_id": f"{source.source_id}:p{page_index + 1:04d}",
        "source_id": source.source_id,
        "source_sha256": source.sha256,
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "printed_page_label": None,
        "printed_page_source": "none",
        "printed_page_confidence": None,
        "status": "failed",
        "text": "",
        "blocks": [],
        "extraction_method": "failed",
        "parser": {"name": "pcba-rag-full-page", "version": "1.0.0"},
        "ocr": _ocr_info(None),
        "quality": {
            "assessment": "fail",
            "text_char_count": 0,
            "flags": ["page_processing_error"],
        },
        "error": {"type": type(error).__name__, "message": str(error) or repr(error)},
    }


def _is_confirmed_blank(
    source_id: str, pdf_page_number: int, config: dict[str, Any]
) -> bool:
    ranges = config.get("blank_page", {}).get("confirmed_pdf_page_ranges", {}).get(
        source_id, []
    )
    return any(int(start) <= pdf_page_number <= int(end) for start, end in ranges)


def _confirmed_blank_record(source: SourceSpec, page_index: int) -> dict[str, Any]:
    return {
        "schema_version": "1.1.0",
        "page_id": f"{source.source_id}:p{page_index + 1:04d}",
        "source_id": source.source_id,
        "source_sha256": source.sha256,
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "printed_page_label": None,
        "printed_page_source": "none",
        "printed_page_confidence": None,
        "status": "blank",
        "text": "",
        "blocks": [],
        "extraction_method": "none",
        "parser": {"name": "confirmed-blank-page", "version": "1.0.0"},
        "ocr": _ocr_info(None),
        "quality": {
            "assessment": "not_applicable",
            "text_char_count": 0,
            "flags": ["blank_page", "confirmed_blank_page"],
        },
        "error": None,
    }


def process_page(
    source: SourceSpec,
    page: Any,
    page_index: int,
    ocr_engine: LocalOcrEngine,
    config: dict[str, Any],
) -> dict[str, Any]:
    page_id = f"{source.source_id}:p{page_index + 1:04d}"
    if _is_confirmed_blank(source.source_id, page_index + 1, config):
        return _confirmed_blank_record(source, page_index)
    ocr_config = config["ocr"]
    quality_config = config["quality"]
    block_config = config["blocks"]
    ignored_patterns = _compile_ignored_patterns(config)

    native_text = normalize_text(page.extract_text() or "")
    native_clean = clean_reader_overlays(native_text, ignored_patterns)
    force_ocr = source.source_id in set(ocr_config["force_source_ids"])
    force_layout_ocr = source.source_id in set(ocr_config.get("force_layout_source_ids", []))
    should_ocr = (
        force_ocr
        or force_layout_ocr
        or len(native_clean) < int(ocr_config["trigger_below_chars"])
    )
    ocr_result: OcrResult | None = None

    if should_ocr:
        ocr_result = ocr_engine.recognize_page(
            source.file_path, page_index, int(ocr_config["render_dpi"])
        )
        ocr_clean = clean_reader_overlays(ocr_result.text, ignored_patterns)
        if force_layout_ocr and native_clean and len(native_clean) >= int(ocr_config["trigger_below_chars"]):
            chosen_text = native_clean
            layout_lines = ocr_layout_lines(ocr_result.lines)
            extraction_method = "mixed"
            parser_name = "pypdf+rapidocr-layout"
            parser_version = f"{_package_version('pypdf')}+{_package_version('rapidocr')}"
        elif force_ocr or ocr_clean or not native_clean:
            chosen_text = ocr_clean
            layout_lines = ocr_layout_lines(ocr_result.lines)
            extraction_method = "ocr"
            parser_name = "rapidocr"
            parser_version = _package_version("rapidocr")
        else:
            chosen_text = native_clean
            layout_lines = native_layout_lines(page)
            extraction_method = "mixed"
            parser_name = "pypdf+rapidocr"
            parser_version = f"{_package_version('pypdf')}+{_package_version('rapidocr')}"
    else:
        chosen_text = native_clean
        layout_lines = native_layout_lines(page)
        extraction_method = "native"
        parser_name = "pypdf"
        parser_version = _package_version("pypdf")

    blocks = build_blocks(
        page_id,
        layout_lines,
        float(block_config["header_ratio"]),
        float(block_config["footer_ratio"]),
        float(block_config["paragraph_gap_ratio"]),
    )
    blocks = [block for block in blocks if not _is_overlay_block(block["text"], ignored_patterns)]
    for order, block in enumerate(blocks):
        block["reading_order"] = order
        block["block_id"] = f"{page_id}:b{order + 1:04d}"

    printed = detect_printed_page(blocks)
    if printed.label is None and should_ocr and native_text:
        native_blocks = build_blocks(
            page_id,
            native_layout_lines(page),
            float(block_config["header_ratio"]),
            float(block_config["footer_ratio"]),
            float(block_config["paragraph_gap_ratio"]),
        )
        native_printed = detect_printed_page(native_blocks)
        if native_printed.label is not None:
            printed = native_printed
    if source.source_id in set(config.get("printed_page", {}).get("disabled_source_ids", [])):
        printed = detect_printed_page([])
    if not chosen_text:
        status = "partial"
        quality = {
            "assessment": "warn",
            "text_char_count": 0,
            "flags": ["empty_text", "non_text_content"],
        }
        blocks = []
        printed = detect_printed_page([])
    else:
        assessment = assess_text(
            chosen_text,
            int(quality_config["minimum_text_chars"]),
            float(quality_config["maximum_replacement_ratio"]),
        )
        flags = list(assessment.flags)
        if not blocks:
            flags.append("no_layout_blocks")
        if ocr_result is not None and ocr_result.average_score is not None and ocr_result.average_score < 0.8:
            flags.append("low_ocr_confidence")
        flags = list(dict.fromkeys(flags))
        status = "partial" if flags else "success"
        quality = {
            "assessment": "warn" if flags else "pass",
            "text_char_count": len(chosen_text),
            "flags": flags,
        }

    return {
        "schema_version": "1.1.0",
        "page_id": page_id,
        "source_id": source.source_id,
        "source_sha256": source.sha256,
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "printed_page_label": printed.label,
        "printed_page_source": printed.source,
        "printed_page_confidence": printed.confidence,
        "status": status,
        "text": chosen_text,
        "blocks": blocks,
        "extraction_method": extraction_method,
        "parser": {"name": parser_name, "version": parser_version},
        "ocr": _ocr_info(ocr_result),
        "quality": quality,
        "error": None,
    }


def validate_page_invariants(record: dict[str, Any]) -> None:
    if record["pdf_page_number"] != record["pdf_page_index"] + 1:
        raise ValueError(f"Non-contiguous page number for {record['page_id']}")
    expected_id = f"{record['source_id']}:p{record['pdf_page_number']:04d}"
    if record["page_id"] != expected_id:
        raise ValueError(f"Invalid page_id {record['page_id']}; expected {expected_id}")
    orders = [block["reading_order"] for block in record["blocks"]]
    if orders != list(range(len(orders))):
        raise ValueError(f"Invalid block reading order for {record['page_id']}")
    block_ids = [block["block_id"] for block in record["blocks"]]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError(f"Duplicate block_id for {record['page_id']}")
    for block in record["blocks"]:
        x0, y0, x1, y1 = block["bbox"]
        if x0 > x1 or y0 > y1:
            raise ValueError(f"Invalid bbox ordering for {block['block_id']}")
    if record["status"] == "blank" and (record["text"] or record["blocks"]):
        raise ValueError(f"Blank page contains content: {record['page_id']}")
    if record["status"] == "failed" and record["error"] is None:
        raise ValueError(f"Failed page lacks error details: {record['page_id']}")


def _read_existing_records(path: Path, validator: Draft202012Validator) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            validator.validate(record)
            validate_page_invariants(record)
        except Exception as error:
            raise ValueError(f"Invalid resume record {path}:{line_number}: {error}") from error
        records.append(record)
    return records


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_statistics(source: SourceSpec, records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    status_counts = Counter(record["status"] for record in records)
    method_counts = Counter(record["extraction_method"] for record in records)
    block_counts = Counter(
        block["block_type"] for record in records for block in record["blocks"]
    )
    quality_flags = Counter(
        flag for record in records for flag in record["quality"]["flags"]
    )
    return {
        "source_id": source.source_id,
        "title": source.title,
        "expected_pages": len(records),
        "processed_pages": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "extraction_method_counts": dict(sorted(method_counts.items())),
        "printed_page_labels": sum(record["printed_page_label"] is not None for record in records),
        "block_counts": dict(sorted(block_counts.items())),
        "quality_flag_counts": dict(sorted(quality_flags.items())),
        "failed_pages": [
            {"pdf_page_number": record["pdf_page_number"], "error": record["error"]}
            for record in records
            if record["status"] == "failed"
        ],
        "blank_pages": [
            record["pdf_page_number"] for record in records if record["status"] == "blank"
        ],
        "partial_pages": [
            {
                "pdf_page_number": record["pdf_page_number"],
                "quality_flags": record["quality"]["flags"],
            }
            for record in records
            if record["status"] == "partial"
        ],
        "elapsed_seconds": round(elapsed, 3),
    }


def _representative_validation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specifications = [
        ("indium_solder_defects_2021", 11, "success", "3", True, {"heading", "paragraph"}),
        ("indium_solder_defects_2021", 18, "success", "10", True, {"formula"}),
        ("indium_solder_defects_2021", 38, "success", "30", True, {"formula", "paragraph"}),
        ("solder_paste_print_inspection_guide", 4, "success", "4", True, {"heading", "paragraph"}),
        ("solder_paste_print_inspection_guide", 10, "success", "10", True, {"paragraph"}),
        ("solder_paste_print_inspection_guide", 25, "success", None, False, {"paragraph"}),
        ("ipc_7530_zh", 3, "success", None, True, {"table", "paragraph"}),
        ("ipc_7530_zh", 7, "blank", None, True, set()),
        ("ipc_7530_zh", 8, "blank", None, True, set()),
        ("gjb_3243a_2021", 2, "success", "I", True, {"heading"}),
        ("gjb_3243a_2021", 8, "success", "5", True, {"table"}),
        ("gjb_3243a_2021", 20, "success", "17", True, {"table", "heading"}),
        ("gjb_3243a_2021", 37, "success", "34", True, {"table", "formula"}),
    ]
    by_key = {(record["source_id"], record["pdf_page_number"]): record for record in records}
    results: list[dict[str, Any]] = []
    for (
        source_id,
        page_number,
        expected_status,
        expected_printed,
        check_printed,
        required_types,
    ) in specifications:
        record = by_key.get((source_id, page_number))
        actual_types = {block["block_type"] for block in record["blocks"]} if record else set()
        passed = bool(
            record
            and record["status"] == expected_status
            and (not check_printed or record["printed_page_label"] == expected_printed)
            and required_types.issubset(actual_types)
        )
        results.append(
            {
                "source_id": source_id,
                "pdf_page_number": page_number,
                "expected_status": expected_status,
                "actual_status": record["status"] if record else None,
                "expected_printed_page_label": expected_printed,
                "actual_printed_page_label": record["printed_page_label"] if record else None,
                "required_block_types": sorted(required_types),
                "actual_block_types": sorted(actual_types),
                "passed": passed,
            }
        )
    return results


def process_source(
    source: SourceSpec,
    root: Path,
    config: dict[str, Any],
    validator: Draft202012Validator,
    ocr_engine: LocalOcrEngine,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_source(source)
    reader = PdfReader(str(source.file_path))
    output_path = root / config["output"]["pages_directory"] / f"{source.source_id}.pages.jsonl"
    checkpoint_path = (
        root / config["output"]["checkpoints_directory"] / f"{source.source_id}.checkpoint.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    records = _read_existing_records(output_path, validator) if resume else []
    existing_by_index = {record["pdf_page_index"]: record for record in records}
    started = time.perf_counter()

    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for page_index, page in enumerate(reader.pages):
            if page_index in existing_by_index:
                continue
            try:
                record = process_page(source, page, page_index, ocr_engine, config)
            except Exception as error:
                record = _failed_record(source, page_index, error)
            validator.validate(record)
            validate_page_invariants(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
            existing_by_index[page_index] = record
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": "1.0.0",
                    "task": "T10.3",
                    "source_id": source.source_id,
                    "source_sha256": source.sha256,
                    "expected_pages": len(reader.pages),
                    "completed_page_numbers": sorted(index + 1 for index in existing_by_index),
                    "complete": len(existing_by_index) == len(reader.pages),
                },
            )

    records.sort(key=lambda item: item["pdf_page_index"])
    if len(records) != len(reader.pages):
        raise ValueError(
            f"Silent page loss for {source.source_id}: expected {len(reader.pages)}, got {len(records)}"
        )
    if [record["pdf_page_index"] for record in records] != list(range(len(reader.pages))):
        raise ValueError(f"Non-contiguous pages for {source.source_id}")
    elapsed = time.perf_counter() - started
    stats = _source_statistics(source, records, elapsed)
    stats["expected_pages"] = len(reader.pages)
    stats["output_path"] = str(output_path.relative_to(root))
    stats["checkpoint_path"] = str(checkpoint_path.relative_to(root))
    return records, stats


def render_full_report(summary: dict[str, Any]) -> str:
    lines = [
        "# T10.3 全量页面与结构数据验收报告",
        "",
        "## 结论",
        "",
        f"- 白名单 PDF：{summary['total_sources']} 份。",
        f"- 预期页面：{summary['expected_pages']} 页；实际记录：{summary['processed_pages']} 页。",
        f"- Schema 有效记录：{summary['schema_valid_pages']} 页。",
        f"- 静默丢页：{'否' if summary['no_silent_page_loss'] else '是'}。",
        f"- 失败页面：{summary['status_counts'].get('failed', 0)} 页。",
        "",
        "## 来源统计",
        "",
        "| 来源 | 页面 | success | partial | blank | failed | 印刷页码 | blocks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in summary["sources"]:
        status = source["status_counts"]
        block_total = sum(source["block_counts"].values())
        lines.append(
            f"| {source['source_id']} | {source['processed_pages']} | "
            f"{status.get('success', 0)} | {status.get('partial', 0)} | "
            f"{status.get('blank', 0)} | {status.get('failed', 0)} | "
            f"{source['printed_page_labels']} | {block_total} |"
        )
    lines.extend(["", "## 空白页与失败页", ""])
    for source in summary["sources"]:
        lines.append(
            f"- `{source['source_id']}`：空白页 {source['blank_pages'] or '无'}；"
            f"失败页 {source['failed_pages'] or '无'}。"
        )
        if source["partial_pages"]:
            lines.append(f"  - partial 明细：`{source['partial_pages']}`")
    lines.extend(
        [
            "",
            "## 代表页结构抽检",
            "",
            "| 来源 | PDF页 | 状态 | 印刷页码 | 必需 blocks | 结果 |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for check in summary["representative_validation"]:
        lines.append(
            f"| {check['source_id']} | {check['pdf_page_number']} | "
            f"{check['actual_status']} | {check['actual_printed_page_label'] or '-'} | "
            f"{', '.join(check['required_block_types']) or '-'} | "
            f"{'通过' if check['passed'] else '未通过'} |"
        )
    lines.extend(["", "## 处理环境", ""])
    environment = summary["environment"]
    lines.append(f"- Python：`{environment['python']}`")
    lines.append(f"- pip check：`{environment['pip_check_output']}`")
    lines.append(f"- Page Schema：`{summary['page_schema_version']}`")
    lines.append(f"- 总耗时：`{summary['elapsed_seconds']}` 秒")
    return "\n".join(lines) + "\n"


def run_full_page_pipeline(
    project_root: Path,
    source_ids: set[str] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_full_ingestion_config(root)
    validator = load_page_v1_1_validator(root)
    sources = load_enabled_sources(root)
    if source_ids:
        sources = [source for source in sources if source.source_id in source_ids]
        missing = source_ids - {source.source_id for source in sources}
        if missing:
            raise ValueError(f"Unknown or disabled source_ids: {sorted(missing)}")
    ocr_engine = LocalOcrEngine()
    started = time.perf_counter()
    all_records: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    for source in sources:
        records, stats = process_source(
            source, root, config, validator, ocr_engine, resume
        )
        all_records.extend(records)
        source_stats.append(stats)

    status_counts = Counter(record["status"] for record in all_records)
    expected_pages = sum(source["expected_pages"] for source in source_stats)
    representative_validation = _representative_validation(all_records)
    summary = {
        "schema_version": "1.0.0",
        "task": "T10.3",
        "page_schema_version": "1.1.0",
        "environment": collect_environment(),
        "total_sources": len(sources),
        "expected_pages": expected_pages,
        "processed_pages": len(all_records),
        "schema_valid_pages": len(all_records),
        "no_silent_page_loss": expected_pages == len(all_records),
        "status_counts": dict(sorted(status_counts.items())),
        "sources": source_stats,
        "representative_validation": representative_validation,
        "representative_validation_passed": all(
            item["passed"] for item in representative_validation
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = root / config["output"]["summary_path"]
    report_path = root / config["output"]["report_path"]
    _atomic_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_full_report(summary), encoding="utf-8", newline="\n")
    return summary
