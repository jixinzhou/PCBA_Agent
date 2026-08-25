from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


def _passage(candidate: dict[str, Any]) -> str:
    citation = candidate.get("citation") or {}
    heading = " > ".join(citation.get("section_path") or [])
    return f"{heading}\n{candidate.get('text', '')}".strip()


class AgentRAGAdapter:
    """Frozen T10 route: Dense/Sparse RRF-20 -> BGE reranker -> rank fusion -> Top-5."""

    def __init__(
        self,
        project_root: Path,
        config: dict[str, Any],
        retriever_factory: Callable[[], Any] | None = None,
        reranker_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.root = project_root
        self.config = config
        self.retriever_factory = retriever_factory
        self.reranker_factory = reranker_factory
        self._reranker: Any | None = None

    def _retriever(self) -> Any:
        if self.retriever_factory:
            return self.retriever_factory()
        source = str(self.root / "rag/src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from pcba_rag.retriever import Retriever

        return Retriever(self.root)

    def _load_reranker(self) -> Any:
        if self._reranker is not None:
            return self._reranker
        if self.reranker_factory:
            self._reranker = self.reranker_factory()
            return self._reranker
        from FlagEmbedding import FlagReranker

        cfg = yaml.safe_load(
            (self.root / self.config["reranker_config"]).read_text(encoding="utf-8")
        )["model"]
        self._reranker = FlagReranker(
            cfg["model_id"],
            revision=cfg["revision"],
            use_fp16=bool(cfg["use_fp16"]),
            devices=cfg["device"],
            batch_size=int(cfg["batch_size"]),
            max_length=int(cfg["max_length"]),
            normalize=bool(cfg["normalize_score"]),
            trust_remote_code=False,
        )
        return self._reranker

    def retrieve(self, query: str) -> dict[str, Any]:
        try:
            source = str(self.root / "rag/src")
            if source not in sys.path:
                sys.path.insert(0, source)
            from pcba_rag.fusion import fuse_channel_responses, load_fusion_config

            request = {
                "schema_version": "1.1.0",
                "query": query,
                "top_k": int(self.config["candidate_pool_top_k"]),
                "filters": {
                    "source_ids": [], "process_ids": [], "defect_ids": [],
                    "evidence_roles": [], "languages": [], "document_types": [],
                },
            }
            retriever = self._retriever()
            try:
                dense = retriever.retrieve_dense(request)
                sparse = retriever.retrieve_sparse(request)
            finally:
                close = getattr(retriever, "close", None)
                if callable(close):
                    close()
            fusion_cfg = copy.deepcopy(load_fusion_config(self.root))
            fusion_cfg["ranking"]["fusion_top_k"] = int(
                self.config["candidate_pool_top_k"]
            )
            fusion_cfg["ranking"]["maximum_final_top_k"] = int(
                self.config["candidate_pool_top_k"]
            )
            rrf_rows, _ = fuse_channel_responses(
                dense, sparse, fusion_cfg, int(self.config["candidate_pool_top_k"])
            )
        except Exception as exc:
            return {"evidence": [], "degraded": True, "stage": "retriever", "error": str(exc)}

        for rank, row in enumerate(rrf_rows, 1):
            row["rrf_rank"] = rank
        try:
            reranker = self._load_reranker()
            scores = reranker.compute_score([(query, _passage(row)) for row in rrf_rows])
            if isinstance(scores, (int, float)):
                scores = [scores]
            else:
                scores = list(scores)
            ranked = sorted(
                zip(rrf_rows, (float(score) for score in scores)),
                key=lambda item: (-item[1], item[0]["rrf_rank"], item[0]["chunk_id"]),
            )
            reranker_ranks = {row["chunk_id"]: rank for rank, (row, _) in enumerate(ranked, 1)}
            reranker_scores = {row["chunk_id"]: score for row, score in ranked}
            w = float(self.config["reranker_weight"])
            k = int(self.config["rrf_k"])
            for row in rrf_rows:
                second_rank = reranker_ranks[row["chunk_id"]]
                row["reranker_rank"] = second_rank
                row["rerank_score"] = reranker_scores[row["chunk_id"]]
                row["rank_fusion_score"] = (1 - w) / (k + row["rrf_rank"]) + w / (k + second_rank)
            final = sorted(
                rrf_rows,
                key=lambda row: (-row["rank_fusion_score"], min(row["rrf_rank"], row["reranker_rank"]), row["chunk_id"]),
            )[: int(self.config["final_top_k"])]
            for rank, row in enumerate(final, 1):
                row["rank"] = rank
            return {"evidence": final, "degraded": False, "stage": None, "error": None}
        except Exception as exc:
            final = copy.deepcopy(rrf_rows[: int(self.config["final_top_k"])])
            for rank, row in enumerate(final, 1):
                row["rank"] = rank
            return {"evidence": final, "degraded": True, "stage": "reranker", "error": str(exc)}
