from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

from pcba_agent.adapters.rag import AgentRAGAdapter
from pcba_agent.policies import assess_prediction
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

    def test_spi_prediction_never_invents_vte_threshold(self) -> None:
        status, _ = assess_prediction(
            "spi_vte_prediction", {"data": {"vte_mean": 123.4, "within_training_domain": True}}
        )
        self.assertEqual("inconclusive", status)

    def test_qwen_uses_strict_schema_without_thinking(self) -> None:
        captured: dict[str, Any] = {}

        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": '{"response_text":"ok"}'}}]}

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
        self.assertEqual("ok", text)
        self.assertFalse(captured["enable_thinking"])
        self.assertTrue(captured["response_format"]["json_schema"]["strict"])


if __name__ == "__main__":
    unittest.main()
