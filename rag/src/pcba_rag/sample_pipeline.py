from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .manifest import SourceSpec, default_project_root, load_enabled_sources, validate_source
from .ocr import LocalOcrEngine
from .page_quality import assess_text
from .pdf_parser import audit_pdf, select_sample_indices, selection_reason


def _package_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "not-installed"


def collect_environment() -> dict[str, Any]:
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "python": sys.version.split()[0],
        "packages": {
            name: _package_version(name)
            for name in [
                "pypdf",
                "cryptography",
                "pypdfium2",
                "rapidocr",
                "onnxruntime",
                "jsonschema",
                "numpy",
                "PyYAML",
                "Pillow",
                "fastapi",
                "pydantic",
                "torch",
                "transformers",
                "huggingface-hub",
                "sentencepiece",
            ]
        },
        "pip_check_ok": pip_check.returncode == 0,
        "pip_check_output": (pip_check.stdout or pip_check.stderr).strip(),
    }


def load_ingestion_config(root: Path) -> dict[str, Any]:
    path = root / "rag/config/ingestion.v0.1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_page_validator(root: Path) -> Draft202012Validator:
    schema_path = root / "rag/schemas/page.v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def make_page_record(
    source: SourceSpec,
    page_index: int,
    text: str,
    extraction_method: str,
    parser_name: str,
    parser_version: str,
    ocr_used: bool,
    quality_config: dict[str, Any],
) -> dict[str, Any]:
    quality = assess_text(
        text,
        int(quality_config["minimum_text_chars"]),
        float(quality_config["maximum_replacement_ratio"]),
    )
    status = "failed" if quality.assessment == "fail" else "partial" if quality.assessment == "warn" else "success"
    return {
        "schema_version": "1.0.0",
        "page_id": f"{source.source_id}:p{page_index + 1:04d}",
        "source_id": source.source_id,
        "source_sha256": source.sha256,
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "status": status,
        "text": text,
        "extraction_method": extraction_method if status != "failed" else "failed",
        "parser": {"name": parser_name, "version": parser_version},
        "ocr": {"used": ocr_used},
        "quality": {
            "assessment": quality.assessment,
            "text_char_count": quality.text_char_count,
            "flags": list(quality.flags),
        },
    }


def process_source_samples(
    source: SourceSpec,
    root: Path,
    config: dict[str, Any],
    validator: Draft202012Validator,
    ocr_engine_holder: list[LocalOcrEngine],
) -> dict[str, Any]:
    validate_source(source)
    audit = audit_pdf(source.file_path)
    sample_count = int(config["sample"]["pages_per_source"])
    selected = select_sample_indices(audit.native_texts, sample_count)
    ocr_config = config["ocr"]
    quality_config = config["quality"]
    force_ocr = source.source_id in set(ocr_config["force_source_ids"])
    output_path = root / "rag/data/processed/samples" / f"{source.source_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    sample_stats: list[dict[str, Any]] = []
    for page_index in selected:
        native_text = audit.native_texts[page_index]
        should_ocr = force_ocr or len(native_text) < int(ocr_config["trigger_below_chars"])
        ocr_stats: dict[str, Any] | None = None
        if should_ocr:
            if not ocr_engine_holder:
                ocr_engine_holder.append(LocalOcrEngine())
            ocr_result = ocr_engine_holder[0].recognize_page(
                source.file_path,
                page_index,
                int(ocr_config["render_dpi"]),
            )
            text = ocr_result.text
            extraction_method = "ocr"
            parser_name = "rapidocr"
            parser_version = _package_version("rapidocr")
            ocr_used = True
            ocr_stats = {
                "elapsed_seconds": round(ocr_result.elapsed_seconds, 4),
                "average_score": round(ocr_result.average_score, 4) if ocr_result.average_score is not None else None,
                "line_count": ocr_result.line_count,
            }
        else:
            text = native_text
            extraction_method = "native"
            parser_name = "pypdf"
            parser_version = _package_version("pypdf")
            ocr_used = False

        record = make_page_record(
            source,
            page_index,
            text,
            extraction_method,
            parser_name,
            parser_version,
            ocr_used,
            quality_config,
        )
        errors = sorted(error.message for error in validator.iter_errors(record))
        if errors:
            raise ValueError(f"Page Schema validation failed for {record['page_id']}: {errors}")
        records.append(record)
        sample_stats.append(
            {
                "pdf_page_number": page_index + 1,
                "selection_reason": selection_reason(page_index, audit.native_texts),
                "native_text_chars": len(native_text),
                "final_text_chars": len(text),
                "extraction_method": record["extraction_method"],
                "status": record["status"],
                "quality_flags": record["quality"]["flags"],
                "schema_valid": True,
                "ocr": ocr_stats,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "source_id": source.source_id,
        "title": source.title,
        "path": str(source.file_path.relative_to(root)),
        "sha256_valid": True,
        "encrypted": audit.encrypted,
        "page_count": audit.page_count,
        "sample_count": len(records),
        "sample_output": str(output_path.relative_to(root)),
        "samples": sample_stats,
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    environment = summary["environment"]
    lines = [
        "# T10.2 PDF/OCR 最小链路验证报告",
        "",
        "## 结论",
        "",
        f"- 四份白名单 PDF 均完成代表页验证，共 {summary['total_samples']} 页。",
        f"- Page Schema 验证：{'通过' if summary['all_schema_valid'] else '失败'}。",
        f"- 环境 `pip check`：{'通过' if environment['pip_check_ok'] else '失败'}。",
        "- 本任务只生成页面级样本，不生成 Chunk、Embedding 或向量索引。",
        "",
        "## 环境",
        "",
        f"- Python：`{environment['python']}`",
        f"- pip check：`{environment['pip_check_output']}`",
        "",
        "| 包 | 版本 |",
        "| --- | --- |",
    ]
    for name, package_version in environment["packages"].items():
        lines.append(f"| `{name}` | `{package_version}` |")

    lines.extend(["", "## 来源与样本", ""])
    for source in summary["sources"]:
        lines.extend(
            [
                f"### {source['source_id']}",
                "",
                f"- 总页数：{source['page_count']}",
                f"- 加密：{'是' if source['encrypted'] else '否'}",
                f"- SHA-256：通过",
                f"- 样本数：{source['sample_count']}",
                "",
                "| PDF页 | 选择原因 | Native字符 | 最终字符 | 方法 | 状态 | OCR均值 |",
                "| ---: | --- | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for sample in source["samples"]:
            average = "-"
            if sample["ocr"] and sample["ocr"]["average_score"] is not None:
                average = str(sample["ocr"]["average_score"])
            lines.append(
                f"| {sample['pdf_page_number']} | {sample['selection_reason']} | "
                f"{sample['native_text_chars']} | {sample['final_text_chars']} | "
                f"{sample['extraction_method']} | {sample['status']} | {average} |"
            )

    lines.extend(
        [
            "",
            "## T10.3 建议",
            "",
            "- 原生文本质量合格的页面继续使用 pypdf。",
            "- GJB 扫描页使用 RapidOCR 本地 OCR。",
            "- 其他来源仅对低文本页面触发 OCR。",
            "- 表格页按 OCR 框坐标排序并保留为按行文本，结构化表格切分不属于 T10.2。",
            "",
        ]
    )
    return "\n".join(lines)


def run_sample_pipeline(
    project_root: Path | None = None,
    source_ids: set[str] | None = None,
) -> dict[str, Any]:
    root = (project_root or default_project_root()).resolve()
    config = load_ingestion_config(root)
    validator = load_page_validator(root)
    sources = load_enabled_sources(root)
    if source_ids:
        sources = [source for source in sources if source.source_id in source_ids]
    ocr_engine_holder: list[LocalOcrEngine] = []
    results = [
        process_source_samples(source, root, config, validator, ocr_engine_holder)
        for source in sources
    ]
    summary = {
        "schema_version": "1.0.0",
        "task": "T10.2",
        "environment": collect_environment(),
        "total_sources": len(results),
        "total_samples": sum(result["sample_count"] for result in results),
        "all_schema_valid": all(
            sample["schema_valid"] for result in results for sample in result["samples"]
        ),
        "sources": results,
    }
    report_dir = root / "rag/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "t10.2_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = report_dir / "T10.2_PDF_OCR_VALIDATION.md"
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return summary
