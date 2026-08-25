from __future__ import annotations

import copy
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))

from annotate_with_qwen import (
    QwenAnnotationError,
    apply_annotation,
    annotate_one_query,
    build_model_input,
    build_request_body,
    chat_completions_url,
    extract_json_content,
    load_env_file,
    request_with_retries,
    validate_annotation,
)
from build_top20_dataset import (
    EMPTY_FILTERS,
    Top20DatasetError,
    empty_retrieval_request,
    evaluation_rrf_top20,
    load_queries,
    validate_rows,
)
from evaluate_retriever import (
    RetrievalEvaluationError,
    aggregate_scores,
    build_gold_dataset,
    load_completed_rows,
    score_query,
    sha256_file,
    validate_gold_dataset,
)


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
        "process_ids": [],
        "process_entities": [],
        "defect_ids": [],
        "defect_entities": [],
        "evidence_roles": [],
        "tag_origin": "rule",
        "parser_version": "test",
        "mapper_version": "test",
        "ocr_used": False,
    }


def result(mode: str, rank: int, number: int) -> dict:
    chunk_id = f"source:c_{number:024d}"
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "text": f"Evidence {number}",
        "citation": {
            "source_id": "source",
            "source_title": "Synthetic Guide",
            "pdf_page_start": number,
            "pdf_page_end": number,
            "section_path": ["Test"],
        },
        "metadata": metadata(),
        "dense_score": float(100 - rank) if mode == "dense" else None,
        "sparse_score": float(100 - rank) if mode == "sparse" else None,
        "fusion_score": None,
        "rerank_score": None,
    }


def response(mode: str, results: list[dict]) -> dict:
    return {
        "query": "test question",
        "retrieval_mode": mode,
        "results": results,
        "trace": {},
    }


def annotation_rows() -> list[dict[str, str]]:
    return [
        {
            "query_id": "Q001",
            "query": "Test question",
            "candidate_id": f"Q001-A{index:012d}",
            "source_id": "source",
            "pdf_page_start": str(index),
            "pdf_page_end": str(index),
            "section_path": '["Test"]',
            "text": f"Evidence {index}",
            "rrf_rank": str(index),
            "dense_rank": str(index),
            "sparse_rank": "",
            "rrf_score": "0.1",
            "original_answerable": "true",
            "final_answerable": "",
            "relevance": "",
            "label_origin": "",
            "model_id": "",
            "prompt_sha256": "",
            "annotation_status": "pending",
            "annotation_error": "",
        }
        for index in range(1, 21)
    ]


def valid_annotation(model_input: dict) -> dict:
    return {
        "query_id": model_input["query_id"],
        "final_answerable": True,
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "relevance": 2 if index == 0 else 0,
            }
            for index, candidate in enumerate(model_input["candidates"])
        ],
    }


class Top20FusionTests(unittest.TestCase):
    def test_request_is_unfiltered_top20(self) -> None:
        request = empty_retrieval_request("test")
        self.assertEqual(20, request["top_k"])
        self.assertEqual(EMPTY_FILTERS, request["filters"])

    def test_evaluation_rrf_returns_stable_top20(self) -> None:
        dense = response(
            "dense", [result("dense", rank, rank) for rank in range(1, 21)]
        )
        sparse = response(
            "sparse", [result("sparse", rank, rank + 10) for rank in range(1, 21)]
        )
        first = evaluation_rrf_top20(
            dense, sparse, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
        )
        second = evaluation_rrf_top20(
            dense, sparse, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
        )
        self.assertEqual(20, len(first))
        self.assertEqual(20, len({value["chunk_id"] for value in first}))
        self.assertEqual(
            [value["chunk_id"] for value in first],
            [value["chunk_id"] for value in second],
        )
        self.assertEqual(list(range(1, 21)), [value["rrf_rank"] for value in first])

    def test_less_than_twenty_channel_results_is_rejected(self) -> None:
        dense = response(
            "dense", [result("dense", rank, rank) for rank in range(1, 20)]
        )
        sparse = response(
            "sparse", [result("sparse", rank, rank) for rank in range(1, 21)]
        )
        with self.assertRaisesRegex(Top20DatasetError, "Top-20"):
            evaluation_rrf_top20(
                dense, sparse, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
            )


class QwenAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = annotation_rows()
        self.model_input = build_model_input(self.rows)

    def test_model_input_is_blind_to_ranks_scores_and_original_label(self) -> None:
        serialized = json.dumps(self.model_input, ensure_ascii=False)
        for forbidden in (
            "rrf_rank",
            "dense_rank",
            "sparse_rank",
            "rrf_score",
            "original_answerable",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(20, len(self.model_input["candidates"]))
        self.assertEqual(
            {row["candidate_id"] for row in self.rows},
            {value["candidate_id"] for value in self.model_input["candidates"]},
        )

    def test_request_uses_qwen_and_json_object(self) -> None:
        body = build_request_body("qwen3.8-max", "prompt", self.model_input)
        self.assertEqual("qwen3.8-max", body["model"])
        self.assertEqual({"type": "json_object"}, body["response_format"])

    def test_valid_response_is_parsed_validated_and_applied(self) -> None:
        raw = {
            "choices": [
                {"message": {"content": json.dumps(valid_annotation(self.model_input))}}
            ]
        }
        parsed = validate_annotation(extract_json_content(raw), self.model_input)
        apply_annotation(self.rows, parsed, model="qwen3.8-max", prompt_hash="hash")
        self.assertTrue(all(row["final_answerable"] == "true" for row in self.rows))
        self.assertTrue(all(row["annotation_status"] == "completed" for row in self.rows))
        self.assertEqual({"0", "2"}, {row["relevance"] for row in self.rows})

    def test_invalid_candidate_set_is_rejected(self) -> None:
        value = valid_annotation(self.model_input)
        value["candidates"][0]["candidate_id"] = "unknown"
        with self.assertRaisesRegex(QwenAnnotationError, "Candidate IDs"):
            validate_annotation(value, self.model_input)

    def test_semantic_conflict_is_rejected(self) -> None:
        value = valid_annotation(self.model_input)
        value["final_answerable"] = False
        with self.assertRaisesRegex(QwenAnnotationError, "conflicts"):
            validate_annotation(value, self.model_input)

    def test_retry_accepts_third_valid_response(self) -> None:
        calls = 0

        def sender() -> dict:
            nonlocal calls
            calls += 1
            if calls < 3:
                return {"choices": [{"message": {"content": "not-json"}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(valid_annotation(self.model_input))
                        }
                    }
                ]
            }

        result_value = request_with_retries(
            sender, self.model_input, attempts=3, sleep=lambda _: None
        )
        self.assertEqual(3, calls)
        self.assertTrue(result_value["final_answerable"])

    def test_parallel_worker_returns_structured_error_without_mutating_rows(self) -> None:
        before = copy.deepcopy(self.rows)
        query_id, annotation, error = annotate_one_query(
            "Q001",
            self.rows,
            endpoint="http://127.0.0.1:1/chat/completions",
            api_key="not-a-real-key",
            model="qwen3.8-max",
            prompt="prompt",
            timeout_seconds=0.01,
            attempts=1,
        )
        self.assertEqual("Q001", query_id)
        self.assertIsNone(annotation)
        self.assertIsNotNone(error)
        self.assertEqual(before, self.rows)

    def test_endpoint_accepts_base_or_full_path(self) -> None:
        base = "https://workspace.example.com/compatible-mode/v1"
        expected = base + "/chat/completions"
        self.assertEqual(expected, chat_completions_url(base))
        self.assertEqual(expected, chat_completions_url(expected))

    def test_endpoint_rejects_template_placeholder(self) -> None:
        with self.assertRaisesRegex(QwenAnnotationError, "template placeholder"):
            chat_completions_url(
                "https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            )

    def test_local_env_file_does_not_override_existing_environment(self) -> None:
        original_model = os.environ.get("QWEN_MODEL")
        original_base = os.environ.get("QWEN_BASE_URL")
        os.environ["QWEN_MODEL"] = "existing-model"
        os.environ.pop("QWEN_BASE_URL", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ".env"
                path.write_text(
                    'QWEN_MODEL="qwen3.8-max"\nQWEN_BASE_URL=https://example.test/v1\n',
                    encoding="utf-8",
                )
                load_env_file(path)
            self.assertEqual("existing-model", os.environ["QWEN_MODEL"])
            self.assertEqual("https://example.test/v1", os.environ["QWEN_BASE_URL"])
        finally:
            if original_model is None:
                os.environ.pop("QWEN_MODEL", None)
            else:
                os.environ["QWEN_MODEL"] = original_model
            if original_base is None:
                os.environ.pop("QWEN_BASE_URL", None)
            else:
                os.environ["QWEN_BASE_URL"] = original_base


class GeneratedDatasetTests(unittest.TestCase):
    def test_source_question_file(self) -> None:
        dataset = load_queries(EVALUATION_DIR / "question/retriever_main.json")
        self.assertEqual(16, len(dataset["items"]))

    def test_frozen_annotated_csv(self) -> None:
        path = EVALUATION_DIR / "question/retriever_main_top20_v0.3.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        validate_rows(rows, require_pending=False)
        self.assertEqual(320, len(rows))
        self.assertEqual({"completed"}, {row["annotation_status"] for row in rows})
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["query_id"], []).append(row)
        for query_id, query_rows in grouped.items():
            statuses = {row["annotation_status"] for row in query_rows}
            self.assertEqual(1, len(statuses), query_id)
            status = next(iter(statuses))
            self.assertIn(status, {"pending", "completed", "failed"})
            if status == "completed":
                self.assertEqual(
                    1, len({row["final_answerable"] for row in query_rows})
                )
                self.assertTrue(
                    all(row["relevance"] in {"0", "1", "2"} for row in query_rows)
                )
            else:
                self.assertTrue(all(not row["final_answerable"] for row in query_rows))
                self.assertTrue(all(not row["relevance"] for row in query_rows))

    def test_frozen_gold(self) -> None:
        csv_path = EVALUATION_DIR / "question/retriever_main_top20_v0.3.csv"
        gold_path = EVALUATION_DIR / "question/retriever_main_gold_v0.3.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        validate_gold_dataset(gold)
        self.assertEqual(
            sha256_file(csv_path),
            gold["source"]["sha256"],
        )
        self.assertEqual(16, gold["summary"]["query_count"])
        self.assertEqual(320, gold["summary"]["candidate_count"])
        self.assertEqual(14, gold["summary"]["candidate_answerable_count"])

    def test_frozen_metric_summary(self) -> None:
        path = EVALUATION_DIR / "question/retriever_main_metrics_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(16, summary["goal_query_count"])
        self.assertEqual(14, summary["primary_answerable_query_count"])
        self.assertEqual(43, summary["rel2_gold_count"])
        self.assertAlmostEqual(
            0.76218820861678,
            summary["methods"]["rank_fusion_w_0_73"]["macro_recall_at_5"],
        )


def gold_query(candidate_answerable: bool = True) -> dict:
    labels = [
        {
            "candidate_id": f"Q001-A{index}",
            "chunk_id": f"chunk-{index}",
            "relevance": 2 if index == 1 and candidate_answerable else (1 if index == 2 else 0),
            "source_id": "source-a",
            "pdf_page_start": index,
            "pdf_page_end": index,
        }
        for index in range(1, 21)
    ]
    return {
        "query_id": "Q001",
        "query": "test",
        "candidate_answerable": candidate_answerable,
        "direct_gold_chunks": [item for item in labels if item["relevance"] == 2],
        "supporting_chunks": [item for item in labels if item["relevance"] == 1],
        "graded_relevance": labels,
        "gold_sources": ["source-a"] if candidate_answerable else [],
        "gold_pages": [
            {"source_id": "source-a", "pdf_page_start": 1, "pdf_page_end": 1}
        ]
        if candidate_answerable
        else [],
    }


def hybrid_response(order: list[int]) -> dict:
    return {
        "results": [
            {
                "rank": rank,
                "chunk_id": f"chunk-{number}",
                "citation": {
                    "source_id": "source-a",
                    "pdf_page_start": number,
                    "pdf_page_end": number,
                },
            }
            for rank, number in enumerate(order, 1)
        ],
        "trace": {"retrieval_time_ms": 1.0},
    }


class RetrievalMetricTests(unittest.TestCase):
    def test_direct_metrics_use_relevance_two_and_page_overlap(self) -> None:
        scored = score_query(
            gold_query(), hybrid_response([2, 3, 1, 4, 5, 6, 7, 8, 9, 10])
        )
        self.assertEqual(1.0, scored["recall_at_5"])
        self.assertEqual(1 / 3, scored["mrr_at_10"])
        self.assertTrue(scored["source_hit_at_5"])
        self.assertTrue(scored["page_hit_at_5"])
        self.assertGreater(scored["ndcg_at_10"], 0.0)
        self.assertLess(scored["ndcg_at_10"], 1.0)

    def test_candidate_unanswerable_has_no_primary_metrics(self) -> None:
        scored = score_query(
            gold_query(candidate_answerable=False),
            hybrid_response([2, 3, 1, 4, 5, 6, 7, 8, 9, 10]),
        )
        self.assertIsNone(scored["recall_at_5"])
        self.assertIsNone(scored["mrr_at_10"])
        self.assertIsNone(scored["ndcg_at_10"])
        self.assertTrue(scored["support_hit_at_5"])

    def test_unjudged_fresh_result_is_rejected(self) -> None:
        query = gold_query()
        gold = {"queries": [query]}
        response_value = hybrid_response([2, 3, 1, 4, 5, 6, 7, 8, 9, 99])
        with self.assertRaisesRegex(RetrievalEvaluationError, "unjudged"):
            aggregate_scores(gold, {"Q001": response_value})


if __name__ == "__main__":
    unittest.main()
