from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.metadata_pipeline import (
    build_mapping_context,
    contains_term,
    load_metadata_validators,
    map_chunk,
    normalize_for_match,
    validate_metadata_invariants,
)


def synthetic_chunk(
    text: str,
    *,
    section: str = "Technical Content",
    source_id: str = "indium_solder_defects_2021",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "chunk_id": "synthetic:c_000000000000000000000000",
        "source_id": source_id,
        "page_ids": [f"{source_id}:p0001"],
        "pdf_page_start": 1,
        "pdf_page_end": 1,
        "section_path": [section],
        "text": text,
        "metadata": {
            "schema_version": "1.0.0",
            "source_title": "Synthetic Source",
            "organization": None,
            "document_type": "industry_guide",
            "language": "en",
            "rights_status": "unknown",
            "process_ids": [],
            "defect_ids": [],
            "evidence_roles": [],
            "tag_origin": "none",
            "parser_version": "pcba-rag-chunker/1.0.0",
            "ocr_used": False,
        },
        "text_hash": "0" * 64,
    }


class MatchBoundaryTests(unittest.TestCase):
    def test_english_terms_use_alphanumeric_boundaries(self) -> None:
        value = normalize_for_match("The short-circuit check is not a shortcut.")
        self.assertTrue(contains_term(value, "short"))
        self.assertFalse(contains_term(value, "circuit check is not a short"))
        self.assertFalse(contains_term(normalize_for_match("A shortcut."), "short"))


class MetadataMappingUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = build_mapping_context(PROJECT_ROOT)
        cls.chunk_validator, cls.trace_validator = load_metadata_validators(PROJECT_ROOT)

    def test_ambiguous_bare_short_is_not_a_defect_match(self) -> None:
        source = synthetic_chunk("Use a short heating time for this procedure.")
        enriched, trace = map_chunk(source, self.context)
        self.assertNotIn("short", enriched["metadata"]["defect_ids"])
        self.assertNotIn("short", [match["matched_term"] for match in trace["matches"]])

    def test_unambiguous_short_phrase_maps_defect(self) -> None:
        source = synthetic_chunk("Inspect the assembly for a short circuit.")
        enriched, trace = map_chunk(source, self.context)
        self.assertIn("short", enriched["metadata"]["defect_ids"])
        self.assertTrue(
            any(match["term_category"] == "synonym" for match in trace["matches"])
        )

    def test_candidate_cause_maps_through_t09_relationship(self) -> None:
        source = synthetic_chunk(
            "Low solder paste transfer can result in an incomplete joint."
        )
        enriched, trace = map_chunk(source, self.context)
        self.assertIn("printing", enriched["metadata"]["process_ids"])
        self.assertIn("insufficient_solder", enriched["metadata"]["defect_ids"])
        cause_matches = [
            match
            for match in trace["matches"]
            if match["term_category"] == "candidate_cause"
        ]
        self.assertTrue(cause_matches)
        self.assertEqual(
            "REL-INSUFFICIENT-SOLDER-PRINTING",
            cause_matches[0]["relationship_id"],
        )

    def test_chunk_can_have_multiple_processes_and_defects(self) -> None:
        source = synthetic_chunk(
            "Solder paste shorts may result from paste bridging during stencil "
            "printing. Reflow thermal imbalance can cause component shift."
        )
        enriched, _ = map_chunk(source, self.context)
        self.assertGreaterEqual(len(enriched["metadata"]["process_ids"]), 2)
        self.assertEqual(
            {"short", "shifted_component"},
            set(enriched["metadata"]["defect_ids"]),
        )

    def test_structural_section_suppresses_all_semantic_tags(self) -> None:
        source = synthetic_chunk(
            "Short circuit, paste bridging, stencil printing and reflow.",
            section="Contents",
            source_id="solder_paste_print_inspection_guide",
        )
        enriched, trace = map_chunk(source, self.context)
        self.assertTrue(trace["excluded_from_semantic_tagging"])
        self.assertEqual([], enriched["metadata"]["process_ids"])
        self.assertEqual([], enriched["metadata"]["defect_ids"])
        self.assertEqual([], enriched["metadata"]["evidence_roles"])
        self.assertEqual("none", enriched["metadata"]["tag_origin"])

    def test_mapping_is_deterministic_and_schema_valid(self) -> None:
        source = synthetic_chunk("Check solder paste insufficients after printing.")
        first = map_chunk(source, self.context)
        second = map_chunk(source, self.context)
        self.assertEqual(first, second)
        self.chunk_validator.validate(first[0])
        self.trace_validator.validate(first[1])
        validate_metadata_invariants(source, first[0], first[1])
        self.assertEqual(source["chunk_id"], first[0]["chunk_id"])
        self.assertEqual(source["text"], first[0]["text"])


class GeneratedMetadataDataTests(unittest.TestCase):
    EXPECTED_SOURCES = {
        "gjb_3243a_2021",
        "indium_solder_defects_2021",
        "ipc_7530_zh",
        "solder_paste_print_inspection_guide",
        "solder_paste_printing_critical_parameters",
        "lead_free_reflow_defects",
        "fine_pitch_csp_assembly_processes",
    }

    def test_generated_metadata_data_if_present(self) -> None:
        enriched_directory = PROJECT_ROOT / "rag/data/processed/chunks_enriched"
        trace_directory = PROJECT_ROOT / "rag/data/processed/metadata_traces"
        paths = (
            sorted(enriched_directory.glob("*.chunks.v1.1.jsonl"))
            if enriched_directory.exists()
            else []
        )
        if not paths:
            self.skipTest("T10.5 enriched Chunk outputs have not been generated yet")
        self.assertEqual(
            self.EXPECTED_SOURCES,
            {path.name.removesuffix(".chunks.v1.1.jsonl") for path in paths},
        )
        chunk_validator, trace_validator = load_metadata_validators(PROJECT_ROOT)
        defects: set[str] = set()
        total = 0
        for enriched_path in paths:
            source_id = enriched_path.name.removesuffix(".chunks.v1.1.jsonl")
            source_path = (
                PROJECT_ROOT
                / "rag/data/processed/chunks"
                / f"{source_id}.chunks.jsonl"
            )
            trace_path = trace_directory / f"{source_id}.mapping-trace.jsonl"
            source_records = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            enriched_records = [
                json.loads(line)
                for line in enriched_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            traces = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(source_records), len(enriched_records))
            self.assertEqual(len(enriched_records), len(traces))
            for source, enriched, trace in zip(
                source_records, enriched_records, traces, strict=True
            ):
                chunk_validator.validate(enriched)
                trace_validator.validate(trace)
                validate_metadata_invariants(source, enriched, trace)
                defects.update(enriched["metadata"]["defect_ids"])
            total += len(enriched_records)
        self.assertEqual(set(self.context_defects()), defects)
        summary = json.loads(
            (PROJECT_ROOT / "rag/reports/t10.5_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(total, summary["output_chunks"])
        self.assertEqual(total, summary["trace_records"])
        self.assertTrue(summary["immutable_fields_preserved"])
        self.assertEqual(0, summary["excluded_section_tag_violations"])
        self.assertTrue(summary["acceptance_checks_passed"])

    @staticmethod
    def context_defects() -> tuple[str, ...]:
        return (
            "insufficient_solder",
            "excessive_solder",
            "short",
            "shifted_component",
        )


if __name__ == "__main__":
    unittest.main()
