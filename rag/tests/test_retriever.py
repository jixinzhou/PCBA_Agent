from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.retriever import (
    QueryTooLongError,
    QueryVectors,
    Retriever,
    build_qdrant_filter,
    load_retrieval_validator,
    load_retriever_config,
    normalize_query,
)


def empty_filters() -> dict[str, list[str]]:
    return {
        "source_ids": [],
        "process_ids": [],
        "defect_ids": [],
        "evidence_roles": [],
        "languages": [],
        "document_types": [],
    }


def request(**overrides: object) -> dict:
    value = {
        "schema_version": "1.1.0",
        "query": " solder paste  printing ",
        "top_k": 5,
        "filters": empty_filters(),
    }
    value.update(overrides)
    return value


def metadata() -> dict:
    return {
        "schema_version": "1.1.0",
        "source_title": "A Guide to Minimizing Solder Defects",
        "organization": "Indium Corporation",
        "document_type": "vendor_guide",
        "language": "en",
        "rights_status": "pending_confirmation",
        "dictionary_version": "1.1.0",
        "ontology_version": "1.1.0",
        "mapping_version": "0.1.0",
        "process_ids": ["printing"],
        "process_entities": [
            {
                "entity_id": "PROCESS-PRINTING",
                "canonical_name": "printing",
                "display_name_zh": "焊膏印刷",
            }
        ],
        "defect_ids": ["insufficient_solder"],
        "defect_entities": [
            {
                "entity_id": "DEFECT-INSUFFICIENT-SOLDER",
                "canonical_name": "insufficient_solder",
                "display_name_zh": "少锡",
            }
        ],
        "evidence_roles": ["troubleshooting_guidance"],
        "tag_origin": "rule",
        "parser_version": "pcba-rag-chunker/1.0.0",
        "mapper_version": "pcba-rag-metadata/1.0.0",
        "ocr_used": False,
    }


class FakeEncoder:
    def __init__(self, fail_long: bool = False) -> None:
        self.calls = 0
        self.fail_long = fail_long

    def encode_query(self, query: str) -> QueryVectors:
        self.calls += 1
        if self.fail_long:
            raise QueryTooLongError("Query has 513 tokens; maximum is 512")
        return QueryVectors(
            dense=[0.0] * 1023 + [1.0],
            sparse_indices=[7, 42],
            sparse_values=[0.25, 0.5],
            token_count=4,
            device="cpu",
        )


class FakeClient:
    def __init__(self, return_empty: bool = False) -> None:
        self.calls: list[dict] = []
        self.return_empty = return_empty

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.return_empty:
            return SimpleNamespace(points=[])
        payload = {
            "index_version": "0.3.0",
            "chunk_id": "indium_solder_defects_2021:c_000000000000000000000000",
            "source_id": "indium_solder_defects_2021",
            "pdf_page_start": 4,
            "pdf_page_end": 4,
            "section_path": ["Printing", "Insufficient solder"],
            "text": "Check solder paste transfer efficiency.",
            "metadata": metadata(),
        }
        return SimpleNamespace(
            points=[SimpleNamespace(id="point-1", score=0.75, payload=payload)]
        )


class RetrieverContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_retriever_config(PROJECT_ROOT)
        cls.validator = load_retrieval_validator(PROJECT_ROOT)

    def test_query_normalization_is_minimal_and_deterministic(self) -> None:
        self.assertEqual(
            "ABC 少锡",
            normalize_query("  ＡＢＣ\n\t少锡  ", self.config),
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_query(" \n\t ", self.config)

    def test_filter_uses_or_within_field_and_and_across_fields(self) -> None:
        filters = empty_filters()
        filters["process_ids"] = ["printing", "reflow"]
        filters["defect_ids"] = ["short"]
        value = build_qdrant_filter(filters, self.config)
        self.assertEqual(3, len(value.must))
        self.assertEqual("semantic_tag_excluded", value.must[0].key)
        self.assertFalse(value.must[0].match.value)
        self.assertEqual(["printing", "reflow"], value.must[1].match.any)
        self.assertEqual(["short"], value.must[2].match.any)

    def test_dense_and_sparse_responses_use_v11_metadata_and_one_encoding(self) -> None:
        encoder = FakeEncoder()
        client = FakeClient()
        retriever = Retriever(
            PROJECT_ROOT,
            client=client,
            encoder=encoder,
            check_connection=False,
        )
        dense = retriever.retrieve_dense(request())
        sparse = retriever.retrieve_sparse(request())
        self.validator.validate(dense)
        self.validator.validate(sparse)
        self.assertEqual(1, encoder.calls)
        self.assertEqual("dense", dense["retrieval_mode"])
        self.assertEqual(0.75, dense["results"][0]["dense_score"])
        self.assertIsNone(dense["results"][0]["sparse_score"])
        self.assertEqual("sparse", sparse["retrieval_mode"])
        self.assertEqual(0.75, sparse["results"][0]["sparse_score"])
        self.assertIsNone(sparse["results"][0]["dense_score"])
        self.assertEqual("1.1.0", dense["results"][0]["metadata"]["schema_version"])
        self.assertEqual("solder paste printing", dense["normalized_query"])
        self.assertEqual("dense", client.calls[0]["using"])
        self.assertEqual("sparse", client.calls[1]["using"])

    def test_valid_filters_can_return_an_empty_result_without_relaxation(self) -> None:
        client = FakeClient(return_empty=True)
        retriever = Retriever(
            PROJECT_ROOT,
            client=client,
            encoder=FakeEncoder(),
            check_connection=False,
        )
        filters = empty_filters()
        filters["document_types"] = ["standard"]
        response = retriever.retrieve_dense(request(filters=filters))
        self.assertEqual([], response["results"])
        self.assertEqual(["standard"], response["trace"]["applied_filters"]["document_types"])
        self.assertEqual(2, len(client.calls[0]["query_filter"].must))

    def test_unknown_source_and_long_query_fail_explicitly(self) -> None:
        filters = empty_filters()
        filters["source_ids"] = ["unknown_source"]
        retriever = Retriever(
            PROJECT_ROOT,
            client=FakeClient(),
            encoder=FakeEncoder(),
            check_connection=False,
        )
        with self.assertRaisesRegex(ValueError, "Unknown source_ids"):
            retriever.retrieve_dense(request(filters=filters))
        long_retriever = Retriever(
            PROJECT_ROOT,
            client=FakeClient(),
            encoder=FakeEncoder(fail_long=True),
            check_connection=False,
        )
        with self.assertRaises(QueryTooLongError):
            long_retriever.retrieve_dense(request())


class GeneratedRetrieverValidationTests(unittest.TestCase):
    def test_generated_summary_if_present(self) -> None:
        path = PROJECT_ROOT / "rag/reports/t10.8_summary.json"
        if not path.exists():
            self.skipTest("T10.8 validation has not been run yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("T10.8", summary["task"])
        self.assertEqual(3, len(summary["validation_cases"]))
        for case in summary["validation_cases"]:
            self.assertGreater(case["dense"]["result_count"], 0)
            self.assertTrue(case["dense"]["filter_compliance"])
            self.assertTrue(case["sparse"]["filter_compliance"])
        self.assertTrue(
            any(
                case["sparse"]["result_count"] > 0
                for case in summary["validation_cases"]
            )
        )
        self.assertTrue(summary["empty_results_are_valid"])
        self.assertTrue(summary["acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
