from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from .retriever import Retriever, RetrieverError


def load_fusion_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "rag/config/fusion.v0.1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_hybrid_validator(project_root: Path) -> Draft202012Validator:
    metadata_schema = json.loads(
        (project_root / "rag/schemas/metadata.v1.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    retrieval_schema = json.loads(
        (project_root / "rag/schemas/retrieval.v1.2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(metadata_schema)
    Draft202012Validator.check_schema(retrieval_schema)
    registry = Registry().with_resource(
        "metadata.v1.1.schema.json", Resource.from_contents(metadata_schema)
    )
    return Draft202012Validator(retrieval_schema, registry=registry)


def rrf_contribution(rank: int | None, k: int, weight: float) -> float:
    if rank is None:
        return 0.0
    if rank < 1 or k < 1 or weight <= 0:
        raise ValueError("RRF rank, k and weight must be positive")
    return float(weight / (k + rank))


def _assert_compatible_responses(
    dense: dict[str, Any], sparse: dict[str, Any]
) -> None:
    if dense["retrieval_mode"] != "dense" or sparse["retrieval_mode"] != "sparse":
        raise RetrieverError("Fusion requires one Dense and one Sparse response")
    for field in ("query", "normalized_query"):
        if dense[field] != sparse[field]:
            raise RetrieverError(f"Dense/Sparse {field} mismatch")
    for field in ("applied_filters", "system_filters", "index_version"):
        if dense["trace"][field] != sparse["trace"][field]:
            raise RetrieverError(f"Dense/Sparse trace mismatch: {field}")


def _candidate_identity(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": result["text"],
        "citation": result["citation"],
        "metadata": result["metadata"],
    }


def fuse_channel_responses(
    dense: dict[str, Any],
    sparse: dict[str, Any],
    config: dict[str, Any],
    final_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _assert_compatible_responses(dense, sparse)
    maximum_final = int(config["ranking"]["maximum_final_top_k"])
    if final_top_k < 1 or final_top_k > maximum_final:
        raise ValueError(f"final_top_k must be between 1 and {maximum_final}")
    candidates: dict[str, dict[str, Any]] = {}
    channel_ids: dict[str, set[str]] = {"dense": set(), "sparse": set()}
    for mode, response in (("dense", dense), ("sparse", sparse)):
        for result in response["results"]:
            chunk_id = result["chunk_id"]
            if chunk_id in channel_ids[mode]:
                raise RetrieverError(f"Duplicate {mode} candidate: {chunk_id}")
            channel_ids[mode].add(chunk_id)
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "citation": copy.deepcopy(result["citation"]),
                    "metadata": copy.deepcopy(result["metadata"]),
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_score": None,
                    "sparse_score": None,
                }
            elif _candidate_identity(candidates[chunk_id]) != _candidate_identity(result):
                raise RetrieverError(
                    f"Dense/Sparse payload mismatch for Chunk {chunk_id}"
                )
            candidate = candidates[chunk_id]
            candidate[f"{mode}_rank"] = int(result["rank"])
            candidate[f"{mode}_score"] = float(result[f"{mode}_score"])

    rrf = config["rrf"]
    for candidate in candidates.values():
        dense_value = rrf_contribution(
            candidate["dense_rank"], int(rrf["k"]), float(rrf["dense_weight"])
        )
        sparse_value = rrf_contribution(
            candidate["sparse_rank"], int(rrf["k"]), float(rrf["sparse_weight"])
        )
        candidate["dense_rrf_contribution"] = dense_value
        candidate["sparse_rrf_contribution"] = sparse_value
        candidate["fusion_score"] = dense_value + sparse_value
        candidate["retrieval_sources"] = [
            mode
            for mode in ("dense", "sparse")
            if candidate[f"{mode}_rank"] is not None
        ]
        candidate["rerank_score"] = None

    def sort_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
        ranks = [
            rank
            for rank in (candidate["dense_rank"], candidate["sparse_rank"])
            if rank is not None
        ]
        return (-candidate["fusion_score"], min(ranks), candidate["chunk_id"])

    ordered = sorted(candidates.values(), key=sort_key)
    fusion_top_k = int(config["ranking"]["fusion_top_k"])
    fused = ordered[:fusion_top_k]
    final = copy.deepcopy(fused[:final_top_k])
    for rank, result in enumerate(final, 1):
        result["rank"] = rank
    statistics = {
        "dense_candidate_count": len(dense["results"]),
        "sparse_candidate_count": len(sparse["results"]),
        "unique_candidate_count": len(candidates),
        "cross_channel_duplicate_count": len(
            channel_ids["dense"] & channel_ids["sparse"]
        ),
        "fusion_candidate_count": len(fused),
        "final_result_count": len(final),
    }
    return final, statistics


def retrieve_hybrid(
    retriever: Retriever,
    request: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    root = retriever.project_root
    config = load_fusion_config(root)
    validator = load_hybrid_validator(root)
    try:
        validator.validate(request)
    except ValidationError as exc:
        raise ValueError(f"Invalid Retrieval V1.2 request: {exc.message}") from exc
    maximum_final = int(config["ranking"]["maximum_final_top_k"])
    if request["top_k"] > maximum_final:
        raise ValueError("top_k exceeds Hybrid maximum_final_top_k")

    channel_request = {
        "schema_version": "1.1.0",
        "query": request["query"],
        "top_k": int(config["candidates"]["per_channel_top_k"]),
        "filters": copy.deepcopy(request["filters"]),
    }
    dense = retriever.retrieve_dense(channel_request)
    sparse = retriever.retrieve_sparse(channel_request)
    fusion_started = time.perf_counter()
    results, statistics = fuse_channel_responses(
        dense, sparse, config, int(request["top_k"])
    )
    fusion_time_ms = round((time.perf_counter() - fusion_started) * 1000, 3)
    dense_trace = dense["trace"]
    sparse_trace = sparse["trace"]
    response = {
        "schema_version": "1.2.0",
        "request_id": str(uuid.uuid4()),
        "query": request["query"],
        "normalized_query": dense["normalized_query"],
        "retrieval_mode": "hybrid",
        "results": results,
        "trace": {
            "knowledge_base_version": dense_trace["knowledge_base_version"],
            "index_version": dense_trace["index_version"],
            "collection_name": dense_trace["collection_name"],
            "retriever_version": dense_trace["retriever_version"],
            "fusion_version": config["fusion_version"],
            "embedding_model": dense_trace["embedding_model"],
            "embedding_model_revision": dense_trace["embedding_model_revision"],
            "query_token_count": dense_trace["query_token_count"],
            "applied_filters": copy.deepcopy(request["filters"]),
            "system_filters": copy.deepcopy(dense_trace["system_filters"]),
            "candidate_top_k": int(config["candidates"]["per_channel_top_k"]),
            **statistics,
            "fusion_top_k": int(config["ranking"]["fusion_top_k"]),
            "requested_final_top_k": int(request["top_k"]),
            "rrf": {
                "k": int(config["rrf"]["k"]),
                "dense_weight": float(config["rrf"]["dense_weight"]),
                "sparse_weight": float(config["rrf"]["sparse_weight"]),
            },
            "channel_request_ids": {
                "dense": dense["request_id"],
                "sparse": sparse["request_id"],
            },
            "channel_retrieval_time_ms": {
                "dense": dense_trace["retrieval_time_ms"],
                "sparse": sparse_trace["retrieval_time_ms"],
            },
            "fusion_time_ms": fusion_time_ms,
            "retrieval_time_ms": round(
                (time.perf_counter() - started) * 1000, 3
            ),
        },
    }
    try:
        validator.validate(response)
    except ValidationError as exc:
        raise RetrieverError(
            f"Generated Retrieval V1.2 response is invalid: {exc.message}"
        ) from exc
    return response


def _render_report(cases: list[dict[str, Any]], passed: bool) -> str:
    lines = [
        "# T10.9 Hybrid Fusion 最简验证",
        "",
        "## 结论",
        "",
        f"- 验证结果：{'通过' if passed else '未通过'}。",
        "- 固定流程：Dense Top-20 + Sparse Top-20 → RRF(k=60) Top-10 → 最终 Top-5。",
        "- 去重键：`chunk_id`；未启用来源多样性或Reranker。",
        "",
        "## 查询统计",
        "",
        "| 查询 | Dense | Sparse | 跨通道重复 | 唯一候选 | 融合候选 | 最终 | 总耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        trace = case["response"]["trace"]
        lines.append(
            f"| {case['case_id']} | {trace['dense_candidate_count']} | "
            f"{trace['sparse_candidate_count']} | "
            f"{trace['cross_channel_duplicate_count']} | "
            f"{trace['unique_candidate_count']} | "
            f"{trace['fusion_candidate_count']} | {trace['final_result_count']} | "
            f"{trace['retrieval_time_ms']} |"
        )
    for case in cases:
        lines.extend(
            [
                "",
                f"## {case['case_id']}",
                "",
                f"查询：{case['response']['normalized_query']}",
                "",
                "| Rank | Chunk ID | Dense Rank | Sparse Rank | Fusion Score | 引用 |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for result in case["response"]["results"]:
            citation = result["citation"]
            page_range = (
                str(citation["pdf_page_start"])
                if citation["pdf_page_start"] == citation["pdf_page_end"]
                else f"{citation['pdf_page_start']}-{citation['pdf_page_end']}"
            )
            lines.append(
                f"| {result['rank']} | `{result['chunk_id']}` | "
                f"{result['dense_rank'] or '-'} | {result['sparse_rank'] or '-'} | "
                f"{result['fusion_score']:.8f} | "
                f"{citation['source_id']} PDF {page_range} |"
            )
    return "\n".join(lines) + "\n"


def run_fusion_validation(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_fusion_config(root)
    cases: list[dict[str, Any]] = []
    with Retriever(root) as retriever:
        for case in config["validation"]["cases"]:
            request = {
                "schema_version": "1.2.0",
                "query": case["query"],
                "top_k": case["top_k"],
                "filters": copy.deepcopy(case["filters"]),
            }
            response = retriever.retrieve_hybrid(request)
            ids = [result["chunk_id"] for result in response["results"]]
            passed = bool(
                len(ids) == len(set(ids))
                and len(ids) <= int(case["top_k"])
                and all(result["fusion_score"] > 0 for result in response["results"])
                and all(result["rerank_score"] is None for result in response["results"])
            )
            cases.append(
                {"case_id": case["case_id"], "response": response, "passed": passed}
            )
    acceptance = all(case["passed"] for case in cases)
    report_path = root / config["validation"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(cases, acceptance), encoding="utf-8", newline="\n"
    )
    return {
        "task": "T10.9",
        "validation_cases": len(cases),
        "report_path": str(report_path.relative_to(root)),
        "acceptance_checks_passed": acceptance,
    }
