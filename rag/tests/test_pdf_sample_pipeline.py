from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.manifest import load_enabled_sources, validate_source
from pcba_rag.ocr import group_ocr_lines
from pcba_rag.page_quality import assess_text, normalize_text
from pcba_rag.pdf_parser import select_sample_indices


class ManifestTests(unittest.TestCase):
    def test_all_enabled_sources_and_hashes(self) -> None:
        sources = load_enabled_sources(PROJECT_ROOT)
        self.assertEqual(7, len(sources))
        for source in sources:
            validate_source(source)


class QualityTests(unittest.TestCase):
    def test_normalize_and_assess_text(self) -> None:
        text = normalize_text("  回流焊  \r\n\r\n  温度曲线  ")
        self.assertEqual("回流焊\n\n温度曲线", text)
        result = assess_text(text, minimum_text_chars=4, maximum_replacement_ratio=0.01)
        self.assertEqual("pass", result.assessment)

    def test_empty_text_fails(self) -> None:
        result = assess_text("", minimum_text_chars=80, maximum_replacement_ratio=0.01)
        self.assertEqual("fail", result.assessment)
        self.assertIn("empty_text", result.flags)


class SamplingTests(unittest.TestCase):
    def test_sample_selection_is_unique_and_bounded(self) -> None:
        texts = tuple("x" * length for length in [0, 10, 100, 5, 50, 20])
        selected = select_sample_indices(texts, 5)
        self.assertEqual(5, len(selected))
        self.assertEqual(5, len(set(selected)))
        self.assertTrue(all(0 <= index < len(texts) for index in selected))


class OcrLayoutTests(unittest.TestCase):
    def test_boxes_are_grouped_into_rows_and_columns(self) -> None:
        boxes = np.asarray(
            [
                [[10, 10], [30, 10], [30, 20], [10, 20]],
                [[60, 11], [90, 11], [90, 21], [60, 21]],
                [[10, 40], [30, 40], [30, 50], [10, 50]],
            ],
            dtype=float,
        )
        text = group_ocr_lines(boxes, ["参数", "数值", "峰值温度"])
        self.assertEqual("参数 | 数值\n峰值温度", text)


class ContractTests(unittest.TestCase):
    def test_page_example_matches_schema(self) -> None:
        schema = json.loads(
            (PROJECT_ROOT / "rag/schemas/page.v1.schema.json").read_text(encoding="utf-8")
        )
        example = json.loads(
            (PROJECT_ROOT / "rag/schemas/examples/page.example.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(example)

    def test_generated_samples_match_schema_and_invariants(self) -> None:
        schema = json.loads(
            (PROJECT_ROOT / "rag/schemas/page.v1.schema.json").read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema)
        sample_dir = PROJECT_ROOT / "rag/data/processed/samples"
        paths = sorted(sample_dir.glob("*.jsonl"))
        self.assertEqual(4, len(paths))
        page_ids: set[str] = set()
        for path in paths:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(5, len(records))
            for record in records:
                validator.validate(record)
                self.assertNotIn(record["page_id"], page_ids)
                page_ids.add(record["page_id"])
                self.assertEqual(len(record["text"]), record["quality"]["text_char_count"])
                self.assertEqual(record["pdf_page_index"] + 1, record["pdf_page_number"])
                self.assertEqual(
                    record["page_id"],
                    f"{record['source_id']}:p{record['pdf_page_number']:04d}",
                )
                if record["extraction_method"] == "native":
                    self.assertFalse(record["ocr"]["used"])
                elif record["extraction_method"] == "ocr":
                    self.assertTrue(record["ocr"]["used"])


if __name__ == "__main__":
    unittest.main()
