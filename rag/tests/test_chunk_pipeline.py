from __future__ import annotations

import json
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.chunk_pipeline import (
    build_page_units,
    build_source_chunks,
    decide_heading,
    load_chunk_config,
    load_chunk_validator,
    recursive_split_text,
    validate_chunk_invariants,
)


class FakeTokenCounter:
    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def hard_split(self, text: str, maximum_tokens: int) -> list[str]:
        words = text.split()
        return [
            " ".join(words[start : start + maximum_tokens])
            for start in range(0, len(words), maximum_tokens)
        ]


def block(
    block_id: str,
    block_type: str,
    text: str,
    order: int,
    heading_level: int | None = None,
) -> dict:
    return {
        "block_id": block_id,
        "block_type": block_type,
        "text": text,
        "reading_order": order,
        "heading_level": heading_level,
    }


def page(source_id: str, number: int, blocks: list[dict], ocr_used: bool = False) -> dict:
    return {
        "page_id": f"{source_id}:p{number:04d}",
        "source_id": source_id,
        "pdf_page_number": number,
        "text": "\n".join(value["text"] for value in blocks),
        "blocks": blocks,
        "ocr": {"used": ocr_used},
    }


SOURCE = {
    "source_id": "synthetic",
    "title": "Synthetic Guide",
    "organization": None,
    "document_type": "industry_guide",
    "language": "en",
    "rights_status": "unknown",
    "evidence_roles": ["process_guideline"],
}


class HeadingCorrectionTests(unittest.TestCase):
    def test_accepts_numbered_and_short_unnumbered_titles(self) -> None:
        numbered = decide_heading(block("b1", "heading", "6.2.2.1 印刷", 0))
        unnumbered = decide_heading(block("b2", "heading", "Solder Paste Shorts", 1))
        self.assertTrue(numbered.accepted)
        self.assertEqual(4, numbered.level)
        self.assertTrue(unnumbered.accepted)
        self.assertTrue(unnumbered.unnumbered)

    def test_rejects_table_list_parameter_and_sentence_false_titles(self) -> None:
        cases = {
            "参数 | 数值": "table_row",
            "2） 具备夹板系统及支撑系统，印刷时PCB应固定。": "list_sentence",
            "0.4N/mm，刮刀速度宜为20mm/s~50mm/s。": "parameter_expression",
            "3 minutes, 40 seconds to a peak temperature of 245°C.": "parameter_expression",
            "Sponsored by": "generic_label",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                decision = decide_heading(block("b", "heading", text, 0))
                self.assertFalse(decision.accepted)
                self.assertEqual(reason, decision.reason)


class SemanticUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_chunk_config(PROJECT_ROOT)

    def test_table_caption_and_rows_are_one_atomic_unit(self) -> None:
        value = page(
            "synthetic",
            1,
            [
                block("b1", "table_caption", "Table 1 Parameters", 0),
                block("b2", "table", "Name | Value", 1),
                block("b3", "formula", "T = 245", 2),
                block("b4", "paragraph", "After the table.", 3),
            ],
        )
        units, audit = build_page_units(value, self.config)
        self.assertEqual("table", units[0].kind)
        self.assertEqual(("b1", "b2", "b3"), units[0].block_ids)
        self.assertEqual(4, len(audit["eligible_block_ids"]))

    def test_list_item_keeps_continuation_paragraph(self) -> None:
        value = page(
            "synthetic",
            1,
            [
                block("b1", "list_item", "a) First requirement", 0),
                block("b2", "paragraph", "continued on the next line", 1),
                block("b3", "list_item", "b) Second requirement", 2),
            ],
        )
        units, _ = build_page_units(value, self.config)
        self.assertEqual(2, len(units))
        self.assertEqual(("b1", "b2"), units[0].block_ids)
        self.assertEqual(("b3",), units[1].block_ids)

    def test_page_override_suppresses_toc_headings(self) -> None:
        value = page(
            "synthetic",
            1,
            [
                block("b1", "heading", "Contents", 0),
                block("b2", "heading", "04 Introduction", 1, 1),
            ],
        )
        units, audit = build_page_units(
            value, self.config, suppress_detected_headings=True
        )
        self.assertTrue(all(unit.kind != "title" for unit in units))
        self.assertEqual(2, audit["rejected_heading_counts"]["page_override"])


class ChunkConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = FakeTokenCounter()
        self.config = deepcopy(load_chunk_config(PROJECT_ROOT))
        self.config["size"] = {
            "minimum_tokens": 3,
            "target_tokens": 6,
            "hard_max_tokens": 8,
        }
        self.validator = load_chunk_validator(PROJECT_ROOT)

    def test_recursive_split_respects_hard_limit(self) -> None:
        parts = recursive_split_text(
            " ".join(f"word{index}" for index in range(21)), self.counter, 8
        )
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(self.counter.count(part) <= 8 for part in parts))

    def test_chunks_do_not_cross_sections_and_ids_are_stable(self) -> None:
        pages = [
            page(
                "synthetic",
                1,
                [
                    block("synthetic:p0001:b0001", "heading", "1 Printing", 0, 1),
                    block(
                        "synthetic:p0001:b0002",
                        "paragraph",
                        "one two three four five",
                        1,
                    ),
                    block("synthetic:p0001:b0003", "heading", "2 Reflow", 2, 1),
                    block(
                        "synthetic:p0001:b0004",
                        "paragraph",
                        "six seven eight nine ten",
                        3,
                    ),
                ],
            )
        ]
        first = build_source_chunks(
            pages, SOURCE, self.config, self.counter, self.validator
        )
        second = build_source_chunks(
            pages, SOURCE, self.config, self.counter, self.validator
        )
        self.assertEqual(
            [record["chunk_id"] for record in first.records],
            [record["chunk_id"] for record in second.records],
        )
        self.assertEqual(2, len(first.records))
        self.assertNotEqual(
            first.records[0]["section_path"], first.records[1]["section_path"]
        )
        self.assertTrue(first.audit["no_silent_block_loss"])
        for record in first.records:
            self.validator.validate(record)
            validate_chunk_invariants(record)
            self.assertEqual([], record["metadata"]["process_ids"])
            self.assertEqual([], record["metadata"]["defect_ids"])
            self.assertEqual("none", record["metadata"]["tag_origin"])

    def test_article_text_headings_form_hard_section_boundaries(self) -> None:
        self.config["heading"]["article_text_sources"]["synthetic"] = {
            "excluded_line_patterns": [r"^PAGE \d+$"],
            "headings": [
                {
                    "page_number": 1,
                    "marker": "Introduction",
                    "title": "Introduction",
                    "level": 2,
                },
                {
                    "page_number": 1,
                    "marker": "Methods",
                    "title": "Methods",
                    "level": 2,
                },
            ],
        }
        value = page("synthetic", 1, [block("b1", "paragraph", "unused", 0)])
        value["text"] = (
            "Front matter\nIntroduction\none two three four\n"
            "Methods\nfive six seven eight\nPAGE 1"
        )
        result = build_source_chunks(
            [value], SOURCE, self.config, self.counter, self.validator
        )
        paths = {tuple(record["section_path"]) for record in result.records}
        self.assertIn(("Synthetic Guide", "Introduction"), paths)
        self.assertIn(("Synthetic Guide", "Methods"), paths)
        self.assertTrue(result.audit["no_silent_block_loss"])


class GeneratedChunkDataTests(unittest.TestCase):
    EXPECTED_SOURCES = {
        "gjb_3243a_2021",
        "indium_solder_defects_2021",
        "ipc_7530_zh",
        "solder_paste_print_inspection_guide",
        "solder_paste_printing_critical_parameters",
        "lead_free_reflow_defects",
        "fine_pitch_csp_assembly_processes",
    }

    def test_generated_chunk_data_if_present(self) -> None:
        chunk_directory = PROJECT_ROOT / "rag/data/processed/chunks"
        paths = sorted(chunk_directory.glob("*.chunks.jsonl")) if chunk_directory.exists() else []
        if not paths:
            self.skipTest("T10.4 chunk outputs have not been generated yet")
        self.assertEqual(
            self.EXPECTED_SOURCES,
            {path.name.removesuffix(".chunks.jsonl") for path in paths},
        )
        validator = load_chunk_validator(PROJECT_ROOT)
        all_ids: set[str] = set()
        total = 0
        for path in paths:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records)
            for record in records:
                validator.validate(record)
                validate_chunk_invariants(record)
                self.assertNotIn(record["chunk_id"], all_ids)
                all_ids.add(record["chunk_id"])
            total += len(records)
        summary = json.loads(
            (PROJECT_ROOT / "rag/reports/t10.4_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(total, summary["total_chunks"])
        self.assertTrue(summary["stable_id_verification_passed"])
        self.assertTrue(summary["no_silent_block_loss"])
        self.assertTrue(summary["acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
