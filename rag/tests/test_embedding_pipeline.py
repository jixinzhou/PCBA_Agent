from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.embedding_pipeline import (
    build_embedding_records,
    build_embedding_text,
    load_embedding_config,
    load_embedding_validator,
    sha256_text,
    validate_embedding_invariants,
)


def synthetic_chunk() -> dict:
    text = "Solder paste printing guidance."
    return {
        "schema_version": "1.1.0",
        "chunk_id": "synthetic:c_000000000000000000000000",
        "source_id": "synthetic",
        "page_ids": ["synthetic:p0001"],
        "pdf_page_start": 1,
        "pdf_page_end": 1,
        "section_path": ["6 Process", "6.2 Printing"],
        "text": text,
        "metadata": {},
        "text_hash": sha256_text(text),
    }


def synthetic_trace(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "excluded_from_semantic_tagging": False,
    }


class EmbeddingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_embedding_config(PROJECT_ROOT)
        cls.validator = load_embedding_validator(PROJECT_ROOT)

    def test_embedding_text_uses_section_and_body_but_not_metadata(self) -> None:
        chunk = synthetic_chunk()
        chunk["metadata"] = {"defect_ids": ["short"]}
        value = build_embedding_text(chunk, self.config)
        self.assertEqual(
            "6 Process > 6.2 Printing\n\nSolder paste printing guidance.",
            value,
        )
        self.assertNotIn("short", value)

    def test_record_has_dense_sparse_hashes_and_structural_flag(self) -> None:
        chunk = synthetic_chunk()
        trace = synthetic_trace(chunk["chunk_id"])
        text = build_embedding_text(chunk, self.config)
        dense = np.full(
            (1, self.config["model"]["dense_dimension"]),
            1.0 / math.sqrt(self.config["model"]["dense_dimension"]),
            dtype=np.float32,
        )
        records = build_embedding_records(
            [chunk],
            {chunk["chunk_id"]: trace},
            [text],
            [12],
            dense,
            [{"42": 0.5, "7": 0.25}],
            self.config,
            "cpu",
            False,
        )
        record = records[0]
        self.validator.validate(record)
        validate_embedding_invariants(chunk, trace, record, self.config)
        self.assertEqual(1024, record["dense"]["dimension"])
        self.assertEqual([7, 42], record["sparse"]["indices"])
        self.assertEqual(sha256_text(text), record["embedding_input_hash"])
        self.assertFalse(record["semantic_tag_excluded"])

    def test_invariants_reject_changed_chunk_text(self) -> None:
        chunk = synthetic_chunk()
        trace = synthetic_trace(chunk["chunk_id"])
        text = build_embedding_text(chunk, self.config)
        dense = np.full((1, 1024), 1.0 / 32.0, dtype=np.float32)
        record = build_embedding_records(
            [chunk],
            {chunk["chunk_id"]: trace},
            [text],
            [12],
            dense,
            [{"7": 0.25}],
            self.config,
            "cpu",
            False,
        )[0]
        changed = dict(chunk)
        changed["text"] = "Changed text"
        with self.assertRaisesRegex(ValueError, "text hash"):
            validate_embedding_invariants(changed, trace, record, self.config)


class GeneratedEmbeddingDataTests(unittest.TestCase):
    EXPECTED_SOURCES = {
        "gjb_3243a_2021",
        "indium_solder_defects_2021",
        "ipc_7530_zh",
        "solder_paste_print_inspection_guide",
        "solder_paste_printing_critical_parameters",
        "lead_free_reflow_defects",
        "fine_pitch_csp_assembly_processes",
    }

    def test_generated_embedding_data_if_present(self) -> None:
        directory = PROJECT_ROOT / "rag/data/processed/embeddings"
        paths = sorted(directory.glob("*.embeddings.jsonl")) if directory.exists() else []
        if not paths:
            self.skipTest("T10.6 Embedding outputs have not been generated yet")
        self.assertEqual(
            self.EXPECTED_SOURCES,
            {path.name.removesuffix(".embeddings.jsonl") for path in paths},
        )
        validator = load_embedding_validator(PROJECT_ROOT)
        config = load_embedding_config(PROJECT_ROOT)
        all_ids: set[str] = set()
        total = 0
        for path in paths:
            source_id = path.name.removesuffix(".embeddings.jsonl")
            chunk_path = (
                PROJECT_ROOT
                / "rag/data/processed/chunks_enriched"
                / f"{source_id}.chunks.v1.1.jsonl"
            )
            trace_path = (
                PROJECT_ROOT
                / "rag/data/processed/metadata_traces"
                / f"{source_id}.mapping-trace.jsonl"
            )
            chunks = {
                record["chunk_id"]: record
                for record in (
                    json.loads(line)
                    for line in chunk_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            traces = {
                record["chunk_id"]: record
                for record in (
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(chunks), len(records))
            for record in records:
                validator.validate(record)
                validate_embedding_invariants(
                    chunks[record["chunk_id"]],
                    traces[record["chunk_id"]],
                    record,
                    config,
                )
                self.assertNotIn(record["chunk_id"], all_ids)
                all_ids.add(record["chunk_id"])
            total += len(records)
        summary = json.loads(
            (PROJECT_ROOT / "rag/reports/t10.6_summary.json").read_text(
                encoding="utf-8"
            )
        )
        reported_total = sum(summary["source_counts"].values())
        self.assertEqual(reported_total, summary["input_chunks"])
        self.assertEqual(reported_total, summary["embedding_records"])
        self.assertLessEqual(reported_total, total)
        self.assertEqual(1024, summary["dense_dimension"])
        self.assertEqual(0, summary["truncated_inputs"])
        self.assertTrue(summary["reproducibility"]["dense_allclose"])
        self.assertTrue(summary["reproducibility"]["sparse_allclose"])
        self.assertTrue(summary["acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
