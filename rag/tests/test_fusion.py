from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))

from pcba_rag.fusion import (
    fuse_channel_responses,
    load_fusion_config,
    load_hybrid_validator,
    retrieve_hybrid,
    rrf_contribution,
)
from pcba_rag.retriever import RetrieverError


def empty_filters() -> dict[str, list[str]]:
    return {
        "source_ids": [],
        "process_ids": [],
        "defect_ids": [],
        "evidence_roles": [],
        "languages": [],
        "document_types": [],
    }


def metadata() -> dict:
    return {
        "schema_version": "1.1.0",
        "source_title": "Synthetic Guide",
        "organization": None,
        "document_type": "industry_guide",
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
        "defect_ids": [],
        "defect_entities": [],
        "evidence_roles": ["process_guideline"],
        "tag_origin": "rule",
        "parser_version": "pcba-rag-chunker/1.0.0",
        "mapper_version": "pcba-rag-metadata/1.0.0",
        "ocr_used": False,
    }


def result(mode: str, rank: int, chunk_id: str, score: float, text: str | None = None) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "text": text or f"Text for {chunk_id}",
        "citation": {
            "source_id": "synthetic",
            "source_title": "Synthetic Guide",
            "pdf_page_start": 1,
            "pdf_page_end": 1,
            "section_path": ["Printing"],
        },
        "metadata": metadata(),
        "dense_score": score if mode == "dense" else None,
        "sparse_score": score if mode == "sparse" else None,
        "fusion_score": None,
        "rerank_score": None,
    }


def response(mode: str, results: list[dict]) -> dict:
    return {
        "schema_version": "1.1.0",
        "request_id": (
            "11111111-1111-4111-8111-111111111111"
            if mode == "dense"
            else "22222222-2222-4222-8222-222222222222"
        ),
        "query": "printing defect",
        "normalized_query": "printing defect",
        "retrieval_mode": mode,
        "results": results,
        "trace": {
            "knowledge_base_version": "0.1.0",
            "index_version": "0.1.0",
            "collection_name": "pcba_industrial_knowledge_v0_1",
            "retriever_version": "pcba-rag-retriever/0.1.0",
            "embedding_model": "BAAI/bge-m3",
            "embedding_model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "query_token_count": 4,
            "applied_filters": empty_filters(),
            "system_filters": {"semantic_tag_excluded": False},
            "retrieval_time_ms": 5.0,
        },
    }


class FusionUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_fusion_config(PROJECT_ROOT)
        cls.validator = load_hybrid_validator(PROJECT_ROOT)

    def test_rrf_uses_fixed_rank_formula_and_missing_channel_zero(self) -> None:
        self.assertAlmostEqual(1.0 / 61.0, rrf_contribution(1, 60, 1.0))
        self.assertEqual(0.0, rrf_contribution(None, 60, 1.0))

    def test_cross_channel_duplicate_merges_scores_ranks_and_citations(self) -> None:
        dense = response(
            "dense",
            [result("dense", 1, "chunk-a", 0.9), result("dense", 2, "chunk-b", 0.8)],
        )
        sparse = response(
            "sparse",
            [result("sparse", 1, "chunk-b", 0.7), result("sparse", 2, "chunk-c", 0.6)],
        )
        final, stats = fuse_channel_responses(dense, sparse, self.config, 3)
        self.assertEqual("chunk-b", final[0]["chunk_id"])
        self.assertEqual(["dense", "sparse"], final[0]["retrieval_sources"])
        self.assertEqual(2, final[0]["dense_rank"])
        self.assertEqual(1, final[0]["sparse_rank"])
        self.assertEqual(1, stats["cross_channel_duplicate_count"])
        self.assertEqual(3, stats["unique_candidate_count"])
        self.assertEqual(3, len({item["chunk_id"] for item in final}))

    def test_empty_sparse_channel_falls_back_to_dense_without_padding(self) -> None:
        dense = response(
            "dense",
            [result("dense", 1, "chunk-a", 0.9), result("dense", 2, "chunk-b", 0.8)],
        )
        sparse = response("sparse", [])
        final, stats = fuse_channel_responses(dense, sparse, self.config, 5)
        self.assertEqual(2, len(final))
        self.assertEqual(0, stats["sparse_candidate_count"])
        self.assertTrue(all(item["sparse_rank"] is None for item in final))
        self.assertTrue(all(item["sparse_rrf_contribution"] == 0 for item in final))

    def test_equal_rrf_scores_use_best_rank_then_chunk_id(self) -> None:
        dense = response("dense", [result("dense", 1, "chunk-b", 0.9)])
        sparse = response("sparse", [result("sparse", 1, "chunk-a", 0.9)])
        final, _ = fuse_channel_responses(dense, sparse, self.config, 2)
        self.assertEqual(["chunk-a", "chunk-b"], [item["chunk_id"] for item in final])

    def test_payload_mismatch_for_same_chunk_is_rejected(self) -> None:
        dense = response("dense", [result("dense", 1, "chunk-a", 0.9, "Dense text")])
        sparse = response("sparse", [result("sparse", 1, "chunk-a", 0.8, "Sparse text")])
        with self.assertRaisesRegex(RetrieverError, "payload mismatch"):
            fuse_channel_responses(dense, sparse, self.config, 1)


class FakeRetriever:
    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self.requests: list[dict] = []

    def retrieve_dense(self, request: dict) -> dict:
        self.requests.append(request)
        return response(
            "dense",
            [result("dense", 1, "chunk-a", 0.9), result("dense", 2, "chunk-b", 0.8)],
        )

    def retrieve_sparse(self, request: dict) -> dict:
        self.requests.append(request)
        return response(
            "sparse",
            [result("sparse", 1, "chunk-b", 0.7), result("sparse", 2, "chunk-c", 0.6)],
        )


class HybridContractTests(unittest.TestCase):
    def test_hybrid_uses_dual_top20_and_returns_v12_trace(self) -> None:
        backend = FakeRetriever()
        request_value = {
            "schema_version": "1.2.0",
            "query": "printing defect",
            "top_k": 2,
            "filters": empty_filters(),
        }
        output = retrieve_hybrid(backend, request_value)  # type: ignore[arg-type]
        load_hybrid_validator(PROJECT_ROOT).validate(output)
        self.assertEqual([20, 20], [value["top_k"] for value in backend.requests])
        self.assertEqual("hybrid", output["retrieval_mode"])
        self.assertEqual(2, len(output["results"]))
        self.assertEqual(20, output["trace"]["candidate_top_k"])
        self.assertEqual(1, output["trace"]["cross_channel_duplicate_count"])
        self.assertEqual(3, output["trace"]["unique_candidate_count"])
        self.assertEqual(2, output["trace"]["final_result_count"])


class GeneratedFusionReportTests(unittest.TestCase):
    def test_markdown_report_if_present_and_no_json_summary(self) -> None:
        report = PROJECT_ROOT / "rag/reports/T10.9_FUSION_VALIDATION.md"
        json_summary = PROJECT_ROOT / "rag/reports/t10.9_summary.json"
        self.assertFalse(json_summary.exists())
        if not report.exists():
            self.skipTest("T10.9 validation report has not been generated yet")
        text = report.read_text(encoding="utf-8")
        self.assertIn("验证结果：通过", text)
        self.assertIn("Dense Top-20 + Sparse Top-20", text)
        self.assertIn("Fusion Score", text)
        self.assertEqual(3, text.count("## ") - 2)


if __name__ == "__main__":
    unittest.main()
