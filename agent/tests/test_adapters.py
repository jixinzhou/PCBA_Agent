from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

from pcba_agent.adapters.rag import AgentRAGAdapter
from pcba_agent.policies import assess_prediction, spi_vte_in_target
from pcba_agent.qwen_client import QwenClient


ROOT = Path(__file__).resolve().parents[2]


def retrieval(mode: str) -> dict[str, Any]:
    results = []
    for rank in range(1, 4):
        results.append({
            "chunk_id": f"C{rank}", "rank": rank, "text": f"passage {rank}",
            "citation": {"section_path": ["section"]}, "metadata": {"language": "en"},
            f"{mode}_score": float(4 - rank),
        })
    return {
        "retrieval_mode": mode, "query": "bridge", "normalized_query": "bridge",
        "results": results,
        "trace": {"applied_filters": {}, "system_filters": {}, "index_version": "test"},
    }


class FakeRetriever:
    def retrieve_dense(self, _: dict[str, Any]) -> dict[str, Any]:
        return retrieval("dense")

    def retrieve_sparse(self, _: dict[str, Any]) -> dict[str, Any]:
        return retrieval("sparse")

    def close(self) -> None:
        pass


class FailedReranker:
    def compute_score(self, _: Any) -> Any:
        raise RuntimeError("GPU unavailable")


class AdapterTests(unittest.TestCase):
    def test_reranker_failure_falls_back_to_rrf(self) -> None:
        config = {
            "candidate_pool_top_k": 3, "final_top_k": 2,
            "reranker_weight": 0.73, "rrf_k": 60,
            "reranker_config": "rag/config/retriever_main_reranker.v0.1.yaml",
        }
        adapter = AgentRAGAdapter(
            ROOT, config, retriever_factory=FakeRetriever,
            reranker_factory=FailedReranker,
        )
        result = adapter.retrieve("bridge")
        self.assertTrue(result["degraded"])
        self.assertEqual("reranker", result["stage"])
        self.assertEqual(["C1", "C2"], [row["chunk_id"] for row in result["evidence"]])

    def test_spi_vte_uses_candidate_specific_approved_thresholds(self) -> None:
        cases = [
            (94.9, "REL-INSUFFICIENT-SOLDER-PRINTING", True, "supported"),
            (95.0, "REL-INSUFFICIENT-SOLDER-PRINTING", True, "inconclusive"),
            (105.0, "REL-EXCESSIVE-SOLDER-PRINTING", True, "inconclusive"),
            (105.1, "REL-EXCESSIVE-SOLDER-PRINTING", True, "supported"),
            (106.0, "REL-INSUFFICIENT-SOLDER-PRINTING", True, "inconclusive"),
            (94.0, "REL-EXCESSIVE-SOLDER-PRINTING", True, "inconclusive"),
            (90.0, "REL-INSUFFICIENT-SOLDER-PRINTING", False, "inconclusive"),
            (90.0, "REL-UNKNOWN", True, "inconclusive"),
        ]
        for vte, relationship_id, in_domain, expected in cases:
            with self.subTest(vte=vte, relationship_id=relationship_id):
                status, _ = assess_prediction(
                    "spi_vte_prediction",
                    {"data": {
                        "vte_mean": vte,
                        "within_training_domain": in_domain,
                    }},
                    relationship_id=relationship_id,
                )
                self.assertEqual(expected, status)

    def test_spi_vte_missing_or_non_finite_value_is_inconclusive(self) -> None:
        for value in (None, float("nan"), float("inf"), "94"):
            with self.subTest(value=value):
                status, _ = assess_prediction(
                    "spi_vte_prediction",
                    {"data": {"vte_mean": value, "within_training_domain": True}},
                    relationship_id="REL-INSUFFICIENT-SOLDER-PRINTING",
                )
                self.assertEqual("inconclusive", status)

    def test_spi_revalidation_target_requires_normal_vte_and_training_domain(self) -> None:
        self.assertTrue(spi_vte_in_target({"data": {
            "vte_mean": 100.0, "within_training_domain": True,
        }}))
        self.assertFalse(spi_vte_in_target({"data": {
            "vte_mean": 94.9, "within_training_domain": True,
        }}))
        self.assertFalse(spi_vte_in_target({"data": {
            "vte_mean": 100.0, "within_training_domain": False,
        }}))

    def test_qwen_uses_strict_schema_without_thinking(self) -> None:
        captured: dict[str, Any] = {}

        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": (
                    '{"diagnosis_conclusion":"主要致因确认为回流热失衡","candidate_analysis":"分析",'
                    '"evidence_basis":"依据","recommendations":"建议","limitations":"限制"}'
                )}}]}

        def fake_post(*_: Any, **kwargs: Any) -> Response:
            captured.update(kwargs["json"])
            return Response()

        config = {
            "api_key_env": "TEST_QWEN_KEY", "base_url_env": "TEST_QWEN_URL",
            "model": "qwen3.7-flash-2026-07-15", "timeout_seconds_env": "TEST_TIMEOUT",
            "max_retries_env": "TEST_RETRIES", "generation_temperature": 0.2,
            "extraction_temperature": 0.0, "enable_thinking": False,
        }
        with patch.dict("os.environ", {
            "TEST_QWEN_KEY": "secret", "TEST_QWEN_URL": "https://example.invalid/v1",
        }), patch("pcba_agent.qwen_client.httpx.post", side_effect=fake_post):
            text = QwenClient(config).synthesize({"candidates": []})
            self.assertIn("诊断结论\n当前获得较强支持的候选路径为回流热失衡", text)
            self.assertNotIn("主要致因确认为", text)
            self.assertIn("证据依据\n依据", text)
        self.assertFalse(captured["enable_thinking"])
        self.assertTrue(captured["response_format"]["json_schema"]["strict"])


if __name__ == "__main__":
    unittest.main()
