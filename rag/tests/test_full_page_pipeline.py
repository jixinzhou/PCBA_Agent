from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.block_extractor import LayoutLine, build_blocks
from pcba_rag.full_page_pipeline import (
    _confirmed_blank_record,
    _failed_record,
    _is_confirmed_blank,
    clean_reader_overlays,
    load_full_ingestion_config,
    load_page_v1_1_validator,
    validate_page_invariants,
)
from pcba_rag.manifest import load_enabled_sources
from pcba_rag.printed_page import detect_printed_page


class PageV11ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_page_v1_1_validator(PROJECT_ROOT)
        cls.sources = {source.source_id: source for source in load_enabled_sources(PROJECT_ROOT)}

    def test_page_and_block_examples_match_schema(self) -> None:
        page = json.loads(
            (PROJECT_ROOT / "rag/schemas/examples/page.v1.1.example.json").read_text(
                encoding="utf-8"
            )
        )
        self.validator.validate(page)
        validate_page_invariants(page)
        block_schema = json.loads(
            (PROJECT_ROOT / "rag/schemas/block.v1.schema.json").read_text(encoding="utf-8")
        )
        block = json.loads(
            (PROJECT_ROOT / "rag/schemas/examples/block.example.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(block_schema).validate(block)

    def test_confirmed_blank_has_distinct_status(self) -> None:
        source = self.sources["ipc_7530_zh"]
        record = _confirmed_blank_record(source, 6)
        self.validator.validate(record)
        validate_page_invariants(record)
        self.assertEqual("blank", record["status"])
        self.assertEqual("none", record["extraction_method"])
        self.assertIn("confirmed_blank_page", record["quality"]["flags"])

    def test_failed_page_has_explicit_error(self) -> None:
        source = self.sources["ipc_7530_zh"]
        record = _failed_record(source, 0, RuntimeError("synthetic page failure"))
        self.validator.validate(record)
        validate_page_invariants(record)
        self.assertEqual("failed", record["status"])
        self.assertEqual("RuntimeError", record["error"]["type"])

    def test_only_confirmed_pages_use_blank_status(self) -> None:
        source = self.sources["ipc_7530_zh"]
        blank = _confirmed_blank_record(source, 6)
        self.assertEqual("blank", blank["status"])
        self.assertIn("confirmed_blank_page", blank["quality"]["flags"])


class BlankAndOverlayTests(unittest.TestCase):
    def test_ipc_confirmed_blank_range(self) -> None:
        config = load_full_ingestion_config(PROJECT_ROOT)
        self.assertFalse(_is_confirmed_blank("ipc_7530_zh", 6, config))
        self.assertTrue(_is_confirmed_blank("ipc_7530_zh", 7, config))
        self.assertTrue(_is_confirmed_blank("ipc_7530_zh", 48, config))
        self.assertFalse(_is_confirmed_blank("gjb_3243a_2021", 7, config))

    def test_reader_overlay_is_removed(self) -> None:
        config = load_full_ingestion_config(PROJECT_ROOT)
        import re

        patterns = tuple(re.compile(value) for value in config["blank_page"]["ignored_patterns"])
        self.assertEqual("", clean_reader_overlays("- 100% +page(2/24)", patterns))
        self.assertEqual("正文", clean_reader_overlays("正文\npage(2/24) | 100%", patterns))


class BlockExtractionTests(unittest.TestCase):
    def test_heading_table_paragraph_and_page_number_blocks(self) -> None:
        lines = (
            LayoutLine("GJB 3243A-2021", (0.1, 0.02, 0.4, 0.04), 10, "ocr", 0.99),
            LayoutLine("6.2.6 焊接", (0.1, 0.15, 0.4, 0.18), 16, "ocr", 0.99),
            LayoutLine("再流焊接过程应满足以下要求：", (0.1, 0.22, 0.8, 0.25), 10, "ocr", 0.98),
            LayoutLine("参数 | 数值", (0.1, 0.31, 0.8, 0.34), 10, "ocr", 0.97),
            LayoutLine("峰值温度 | 245℃", (0.1, 0.35, 0.8, 0.38), 10, "ocr", 0.97),
            LayoutLine("17", (0.48, 0.95, 0.52, 0.98), 9, "ocr", 0.99),
        )
        blocks = build_blocks("gjb:p0020", lines)
        types = [block["block_type"] for block in blocks]
        self.assertIn("header", types)
        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        self.assertIn("table", types)
        self.assertIn("page_number", types)
        self.assertEqual(list(range(len(blocks))), [block["reading_order"] for block in blocks])

    def test_printed_page_uses_page_number_block(self) -> None:
        blocks = build_blocks(
            "gjb:p0020",
            (LayoutLine("17", (0.48, 0.95, 0.52, 0.98), 9, "ocr", 0.99),),
        )
        printed = detect_printed_page(blocks)
        self.assertEqual("17", printed.label)
        self.assertEqual("ocr", printed.source)

    def test_printed_page_can_be_embedded_in_footer(self) -> None:
        blocks = build_blocks(
            "guide:p0010",
            (
                LayoutLine(
                    "10 Solder Paste Print Inspection & Defect Guide",
                    (0.08, 0.96, 0.92, 0.98),
                    9,
                    "native",
                    1.0,
                ),
            ),
        )
        printed = detect_printed_page(blocks)
        self.assertEqual("10", printed.label)


class GeneratedFullDataTests(unittest.TestCase):
    EXPECTED = {
        "indium_solder_defects_2021": 52,
        "solder_paste_print_inspection_guide": 44,
        "ipc_7530_zh": 48,
        "gjb_3243a_2021": 38,
        "solder_paste_printing_critical_parameters": 13,
        "lead_free_reflow_defects": 11,
        "fine_pitch_csp_assembly_processes": 10,
    }

    def test_generated_full_data_if_present(self) -> None:
        page_dir = PROJECT_ROOT / "rag/data/processed/pages"
        paths = sorted(page_dir.glob("*.pages.jsonl")) if page_dir.exists() else []
        if not paths:
            self.skipTest("T10.3 full page outputs have not been generated yet")
        self.assertEqual(len(self.EXPECTED), len(paths))
        validator = load_page_v1_1_validator(PROJECT_ROOT)
        all_page_ids: set[str] = set()
        total = 0
        for path in paths:
            source_id = path.name.removesuffix(".pages.jsonl")
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(self.EXPECTED[source_id], len(records))
            self.assertEqual(
                list(range(len(records))), [record["pdf_page_index"] for record in records]
            )
            for record in records:
                validator.validate(record)
                validate_page_invariants(record)
                self.assertNotIn(record["page_id"], all_page_ids)
                all_page_ids.add(record["page_id"])
            total += len(records)
            checkpoint = json.loads(
                (
                    PROJECT_ROOT
                    / "rag/data/processed/checkpoints"
                    / f"{source_id}.checkpoint.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(checkpoint["complete"])
            self.assertEqual(len(records), len(checkpoint["completed_page_numbers"]))
        self.assertEqual(sum(self.EXPECTED.values()), total)
        summary = json.loads(
            (PROJECT_ROOT / "rag/reports/t10.3_summary.json").read_text(encoding="utf-8")
        )
        if summary.get("total_sources") == len(self.EXPECTED):
            self.assertTrue(summary["representative_validation_passed"])


if __name__ == "__main__":
    unittest.main()
