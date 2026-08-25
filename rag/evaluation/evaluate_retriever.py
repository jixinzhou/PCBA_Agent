from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = EVALUATION_DIR / "question/retriever_main_top20_v0.3.csv"
DEFAULT_GOLD = EVALUATION_DIR / "question/retriever_main_gold_v0.3.json"
DEFAULT_RESULTS = EVALUATION_DIR / "question/retriever_main_retrieval_results.json"
DEFAULT_REPORT = EVALUATION_DIR / "RETRIEVER_BASELINE.md"
GOLD_VERSION = "pcba-retrieval-gold/1.0.0"
RESULT_VERSION = "pcba-retrieval-evaluation/1.0.0"
EMPTY_FILTERS = {
    "source_ids": [],
    "process_ids": [],
    "defect_ids": [],
    "evidence_roles": [],
    "languages": [],
    "document_types": [],
}


class RetrievalEvaluationError(RuntimeError):
    """Raised when Gold provenance or evaluation output is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_completed_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) % 20:
        raise RetrievalEvaluationError(
            f"Annotated candidate count must be a positive multiple of 20, found {len(rows)}"
        )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    if len(grouped) != len(rows) // 20:
        raise RetrievalEvaluationError("Annotated query count does not match row count")
    candidate_ids: set[str] = set()
    for query_id, query_rows in grouped.items():
        if len(query_rows) != 20:
            raise RetrievalEvaluationError(
                f"{query_id} must contain exactly 20 candidates"
            )
        if {row["annotation_status"] for row in query_rows} != {"completed"}:
            raise RetrievalEvaluationError(f"{query_id} is not fully completed")
        if any(row["annotation_error"].strip() for row in query_rows):
            raise RetrievalEvaluationError(f"{query_id} still contains annotation errors")
        if {int(row["rrf_rank"]) for row in query_rows} != set(range(1, 21)):
            raise RetrievalEvaluationError(f"{query_id} has invalid RRF ranks")
        chunk_ids = [row["chunk_id"] for row in query_rows]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalEvaluationError(f"{query_id} contains duplicate Chunk IDs")
        for row in query_rows:
            if row["candidate_id"] in candidate_ids:
                raise RetrievalEvaluationError(
                    f"Duplicate candidate ID: {row['candidate_id']}"
                )
            candidate_ids.add(row["candidate_id"])
            if row["relevance"] not in {"0", "1", "2"}:
                raise RetrievalEvaluationError(
                    f"{row['candidate_id']} has invalid relevance"
                )
        answerable_values = {row["final_answerable"] for row in query_rows}
        if answerable_values not in ({"true"}, {"false"}):
            raise RetrievalEvaluationError(f"{query_id} has invalid final_answerable")
        has_direct = any(row["relevance"] == "2" for row in query_rows)
        if (answerable_values == {"true"}) != has_direct:
            raise RetrievalEvaluationError(
                f"{query_id} final_answerable conflicts with direct Gold"
            )
    return rows


def _one_value(rows: list[dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise RetrievalEvaluationError(
            f"Expected one {field} value, found {sorted(values)}"
        )
    return next(iter(values))


def _chunk_label(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "chunk_id": row["chunk_id"],
        "relevance": int(row["relevance"]),
        "source_id": row["source_id"],
        "pdf_page_start": int(row["pdf_page_start"]),
        "pdf_page_end": int(row["pdf_page_end"]),
    }


def build_gold_dataset(
    rows: list[dict[str, str]],
    csv_path: Path,
    *,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)

    queries: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        query_rows = sorted(grouped[query_id], key=lambda row: int(row["rrf_rank"]))
        direct = [_chunk_label(row) for row in query_rows if row["relevance"] == "2"]
        supporting = [
            _chunk_label(row) for row in query_rows if row["relevance"] == "1"
        ]
        graded = [_chunk_label(row) for row in query_rows]
        gold_sources = sorted({item["source_id"] for item in direct})
        gold_pages = [
            {
                "source_id": source_id,
                "pdf_page_start": page_start,
                "pdf_page_end": page_end,
            }
            for source_id, page_start, page_end in sorted(
                {
                    (
                        item["source_id"],
                        item["pdf_page_start"],
                        item["pdf_page_end"],
                    )
                    for item in direct
                }
            )
        ]
        queries.append(
            {
                "query_id": query_id,
                "query": _one_value(query_rows, "query"),
                "language": _one_value(query_rows, "language"),
                "query_type": _one_value(query_rows, "query_type"),
                "original_answerable": _one_value(
                    query_rows, "original_answerable"
                )
                == "true",
                "candidate_answerable": _one_value(
                    query_rows, "final_answerable"
                )
                == "true",
                "defect_ids": json.loads(_one_value(query_rows, "defect_ids")),
                "process_ids": json.loads(_one_value(query_rows, "process_ids")),
                "eval_focus": json.loads(_one_value(query_rows, "eval_focus")),
                "direct_gold_chunks": direct,
                "supporting_chunks": supporting,
                "graded_relevance": graded,
                "gold_sources": gold_sources,
                "gold_pages": gold_pages,
            }
        )

    answerable_count = sum(query["candidate_answerable"] for query in queries)
    supporting_only_count = sum(
        not query["candidate_answerable"] and bool(query["supporting_chunks"])
        for query in queries
    )
    all_zero_count = sum(
        not query["direct_gold_chunks"] and not query["supporting_chunks"]
        for query in queries
    )
    relevance_counts = Counter(
        item["relevance"]
        for query in queries
        for item in query["graded_relevance"]
    )
    return {
        "schema_version": "1.0.0",
        "gold_version": GOLD_VERSION,
        "frozen_at_utc": frozen_at_utc or utc_now(),
        "source": {
            "path": str(csv_path.resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(csv_path),
            "pool_version": _one_value(rows, "pool_version"),
            "dataset_name": _one_value(rows, "dataset_name"),
            "dataset_sha256": _one_value(rows, "dataset_sha256"),
        },
        "provenance": {
            "knowledge_base_version": _one_value(rows, "knowledge_base_version"),
            "index_version": _one_value(rows, "index_version"),
            "collection_name": _one_value(rows, "collection_name"),
            "retriever_version": _one_value(rows, "retriever_version"),
            "fusion_version": _one_value(rows, "fusion_version"),
            "embedding_model": _one_value(rows, "embedding_model"),
            "embedding_model_revision": _one_value(
                rows, "embedding_model_revision"
            ),
            "annotation_models": sorted({row["model_id"] for row in rows}),
            "annotation_prompt_sha256s": sorted({row["prompt_sha256"] for row in rows}),
        },
        "policy": {
            "direct_relevance": 2,
            "supporting_relevance": 1,
            "irrelevant_relevance": 0,
            "recall_mrr_hit_threshold": 2,
            "ndcg_gain": "2^relevance-1",
            "primary_query_scope": "candidate_answerable=true",
            "source_hit_k": 5,
            "page_hit_k": 5,
            "page_match": "same_source_and_page_range_overlap",
            "unanswerable_queries_reported_separately": True,
        },
        "summary": {
            "query_count": len(queries),
            "candidate_count": sum(len(query["graded_relevance"]) for query in queries),
            "candidate_answerable_count": answerable_count,
            "candidate_unanswerable_count": len(queries) - answerable_count,
            "supporting_only_count": supporting_only_count,
            "all_zero_count": all_zero_count,
            "relevance_counts": {
                str(value): relevance_counts.get(value, 0) for value in range(3)
            },
        },
        "queries": queries,
    }


def validate_gold_dataset(gold: dict[str, Any]) -> None:
    if gold.get("gold_version") != GOLD_VERSION:
        raise RetrievalEvaluationError("Unsupported Gold version")
    queries = gold.get("queries")
    if not isinstance(queries, list) or not queries:
        raise RetrievalEvaluationError("Gold must contain a non-empty query list")
    ids = [query.get("query_id") for query in queries]
    if len(ids) != len(set(ids)):
        raise RetrievalEvaluationError("Gold contains duplicate query IDs")
    for query in queries:
        graded = query.get("graded_relevance")
        if not isinstance(graded, list) or len(graded) != 20:
            raise RetrievalEvaluationError(
                f"{query.get('query_id')} must contain 20 graded candidates"
            )
        chunk_ids = [item.get("chunk_id") for item in graded]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalEvaluationError(
                f"{query['query_id']} has duplicate graded Chunk IDs"
            )
        if any(item.get("relevance") not in {0, 1, 2} for item in graded):
            raise RetrievalEvaluationError(
                f"{query['query_id']} has invalid graded relevance"
            )
        has_direct = any(item["relevance"] == 2 for item in graded)
        if bool(query.get("candidate_answerable")) != has_direct:
            raise RetrievalEvaluationError(
                f"{query['query_id']} answerability conflicts with direct Gold"
            )


def freeze_gold(csv_path: Path, gold_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if gold_path.exists() and not overwrite:
        raise RetrievalEvaluationError(f"Refusing to overwrite frozen Gold: {gold_path}")
    rows = load_completed_rows(csv_path)
    gold = build_gold_dataset(rows, csv_path)
    validate_gold_dataset(gold)
    atomic_write_json(gold_path, gold)
    return gold


def load_frozen_gold(gold_path: Path, csv_path: Path) -> dict[str, Any]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    validate_gold_dataset(gold)
    current_hash = sha256_file(csv_path)
    if gold["source"]["sha256"] != current_hash:
        raise RetrievalEvaluationError(
            "Annotated CSV changed after Gold freeze; explicit refreeze is required"
        )
    return gold


def _dcg(relevances: list[int]) -> float:
    return sum(
        ((2**relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, 1)
    )


def _page_overlap(result: dict[str, Any], gold_page: dict[str, Any]) -> bool:
    citation = result["citation"]
    return bool(
        citation["source_id"] == gold_page["source_id"]
        and int(citation["pdf_page_start"]) <= gold_page["pdf_page_end"]
        and gold_page["pdf_page_start"] <= int(citation["pdf_page_end"])
    )


def score_query(query: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    results = response.get("results", [])
    if len(results) != 10:
        raise RetrievalEvaluationError(
            f"{query['query_id']} expected Hybrid Top-10, found {len(results)}"
        )
    result_ids = [result["chunk_id"] for result in results]
    if len(result_ids) != len(set(result_ids)):
        raise RetrievalEvaluationError(
            f"{query['query_id']} contains duplicate retrieved Chunk IDs"
        )
    graded = {
        item["chunk_id"]: int(item["relevance"])
        for item in query["graded_relevance"]
    }
    unjudged = [chunk_id for chunk_id in result_ids if chunk_id not in graded]
    direct_ids = {item["chunk_id"] for item in query["direct_gold_chunks"]}
    top5 = results[:5]
    top5_ids = {result["chunk_id"] for result in top5}
    direct_ranks = [
        rank for rank, chunk_id in enumerate(result_ids, 1) if chunk_id in direct_ids
    ]
    recall_at_5 = (
        len(top5_ids & direct_ids) / len(direct_ids) if direct_ids else None
    )
    mrr_at_10 = (1.0 / min(direct_ranks)) if direct_ranks else (0.0 if direct_ids else None)
    observed_relevance = [graded.get(chunk_id, 0) for chunk_id in result_ids]
    ideal_relevance = sorted(graded.values(), reverse=True)[:10]
    ideal_dcg = _dcg(ideal_relevance)
    ndcg_at_10 = (
        _dcg(observed_relevance) / ideal_dcg if direct_ids and ideal_dcg else None
    )
    source_hit = (
        any(
            result["citation"]["source_id"] in set(query["gold_sources"])
            for result in top5
        )
        if direct_ids
        else None
    )
    page_hit = (
        any(
            _page_overlap(result, gold_page)
            for result in top5
            for gold_page in query["gold_pages"]
        )
        if direct_ids
        else None
    )
    supporting_ids = {item["chunk_id"] for item in query["supporting_chunks"]}
    return {
        "query_id": query["query_id"],
        "candidate_answerable": query["candidate_answerable"],
        "direct_gold_count": len(direct_ids),
        "supporting_gold_count": len(supporting_ids),
        "retrieved_chunk_ids": result_ids,
        "unjudged_top10_count": len(unjudged),
        "unjudged_top10_chunk_ids": unjudged,
        "recall_at_5": recall_at_5,
        "mrr_at_10": mrr_at_10,
        "ndcg_at_10": ndcg_at_10,
        "source_hit_at_5": source_hit,
        "page_hit_at_5": page_hit,
        "support_hit_at_5": bool(top5_ids & supporting_ids),
        "retrieval_time_ms": response["trace"]["retrieval_time_ms"],
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise RetrievalEvaluationError("Cannot calculate a metric over an empty set")
    return round(sum(values) / len(values), 6)


def aggregate_scores(
    gold: dict[str, Any], responses: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    expected_ids = {query["query_id"] for query in gold["queries"]}
    if set(responses) != expected_ids:
        missing = sorted(expected_ids - set(responses))
        extra = sorted(set(responses) - expected_ids)
        raise RetrievalEvaluationError(
            f"Response query IDs mismatch; missing={missing}, extra={extra}"
        )
    per_query = [score_query(query, responses[query["query_id"]]) for query in gold["queries"]]
    unjudged = sum(item["unjudged_top10_count"] for item in per_query)
    if unjudged:
        raise RetrievalEvaluationError(
            f"Fresh retrieval returned {unjudged} unjudged Top-10 candidates"
        )
    primary = [item for item in per_query if item["candidate_answerable"]]
    no_answer = [item for item in per_query if not item["candidate_answerable"]]
    return {
        "primary": {
            "query_count": len(primary),
            "recall_at_5": _mean([item["recall_at_5"] for item in primary]),
            "mrr_at_10": _mean([item["mrr_at_10"] for item in primary]),
            "ndcg_at_10": _mean([item["ndcg_at_10"] for item in primary]),
            "source_hit_at_5": _mean(
                [float(item["source_hit_at_5"]) for item in primary]
            ),
            "page_hit_at_5": _mean(
                [float(item["page_hit_at_5"]) for item in primary]
            ),
        },
        "candidate_unanswerable": {
            "query_count": len(no_answer),
            "supporting_only_count": sum(
                item["supporting_gold_count"] > 0 for item in no_answer
            ),
            "all_zero_count": sum(
                item["supporting_gold_count"] == 0 for item in no_answer
            ),
            "support_hit_at_5": _mean(
                [float(item["support_hit_at_5"]) for item in no_answer]
            ),
        },
        "coverage": {
            "evaluated_query_count": len(per_query),
            "retrieved_result_count": len(per_query) * 10,
            "unjudged_top10_count": unjudged,
        },
        "per_query": per_query,
    }


def _verify_trace(trace: dict[str, Any], provenance: dict[str, str]) -> None:
    fields = (
        "knowledge_base_version",
        "index_version",
        "collection_name",
        "retriever_version",
        "fusion_version",
        "embedding_model",
        "embedding_model_revision",
    )
    mismatches = {
        field: {"gold": provenance[field], "current": trace.get(field)}
        for field in fields
        if trace.get(field) != provenance[field]
    }
    if mismatches:
        raise RetrievalEvaluationError(
            f"Retriever provenance differs from frozen Gold: {mismatches}"
        )


def run_retrieval(gold: dict[str, Any], project_root: Path) -> tuple[dict[str, Any], float]:
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.retriever import Retriever

    responses: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with Retriever(project_root) as retriever:
        for query in gold["queries"]:
            request = {
                "schema_version": "1.2.0",
                "query": query["query"],
                "top_k": 10,
                "filters": {key: list(value) for key, value in EMPTY_FILTERS.items()},
            }
            response = retriever.retrieve_hybrid(request)
            _verify_trace(response["trace"], gold["provenance"])
            responses[query["query_id"]] = response
            print(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        "status": "retrieved",
                        "time_ms": response["trace"]["retrieval_time_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return responses, time.perf_counter() - started


def build_results(
    gold: dict[str, Any],
    gold_path: Path,
    scores: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    source_ids = sorted(
        {
            item["source_id"]
            for query in gold["queries"]
            for item in query["graded_relevance"]
        }
    )
    source_coverage = []
    for source_id in source_ids:
        pool_count = 0
        relevant_count = 0
        direct_count = 0
        relevant_queries: set[str] = set()
        direct_queries: set[str] = set()
        for query in gold["queries"]:
            source_labels = [
                item
                for item in query["graded_relevance"]
                if item["source_id"] == source_id
            ]
            pool_count += len(source_labels)
            relevant_count += sum(item["relevance"] >= 1 for item in source_labels)
            direct_count += sum(item["relevance"] == 2 for item in source_labels)
            if any(item["relevance"] >= 1 for item in source_labels):
                relevant_queries.add(query["query_id"])
            if any(item["relevance"] == 2 for item in source_labels):
                direct_queries.add(query["query_id"])
        source_coverage.append(
            {
                "source_id": source_id,
                "pool_candidate_count": pool_count,
                "relevant_chunk_count": relevant_count,
                "direct_gold_chunk_count": direct_count,
                "relevant_query_count": len(relevant_queries),
                "direct_gold_query_count": len(direct_queries),
            }
        )
    scope_coverage = {
        "languages": dict(
            sorted(Counter(query["language"] for query in gold["queries"]).items())
        ),
        "defect_ids": dict(
            sorted(
                Counter(
                    defect_id
                    for query in gold["queries"]
                    for defect_id in query["defect_ids"]
                ).items()
            )
        ),
        "process_ids": dict(
            sorted(
                Counter(
                    process_id
                    for query in gold["queries"]
                    for process_id in query["process_ids"]
                ).items()
            )
        ),
    }
    coverage_acceptance = {
        "four_source_pool_coverage": len(source_coverage) == 4,
        "four_source_relevant_coverage": all(
            item["relevant_chunk_count"] > 0 for item in source_coverage
        ),
        "four_source_direct_gold_coverage": all(
            item["direct_gold_chunk_count"] > 0 for item in source_coverage
        ),
        "four_defect_scope": {
            "insufficient_solder",
            "excessive_solder",
            "short",
            "shifted_component",
        }.issubset(scope_coverage["defect_ids"]),
        "three_process_scope": {"printing", "placement", "reflow"}.issubset(
            scope_coverage["process_ids"]
        ),
        "bilingual_scope": {"zh", "en"}.issubset(scope_coverage["languages"]),
        "candidate_unanswerable_scope": gold["summary"][
            "candidate_unanswerable_count"
        ]
        > 0,
    }
    return {
        "schema_version": "1.0.0",
        "evaluation_version": RESULT_VERSION,
        "evaluated_at_utc": utc_now(),
        "gold": {
            "path": str(gold_path.resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(gold_path),
            "gold_version": gold["gold_version"],
            "source_csv_sha256": gold["source"]["sha256"],
        },
        "retriever_provenance": gold["provenance"],
        "policy": gold["policy"],
        "metrics": {
            "primary": scores["primary"],
            "candidate_unanswerable": scores["candidate_unanswerable"],
            "coverage": scores["coverage"],
        },
        "gold_coverage": {
            "sources": source_coverage,
            "scope": scope_coverage,
            "acceptance_checks": coverage_acceptance,
        },
        "runtime_seconds": round(runtime_seconds, 3),
        "per_query": scores["per_query"],
        "limitations": [
            "Gold is limited to the judged evaluation RRF Top-20 candidate pool.",
            "Primary metrics exclude candidate_answerable=false queries.",
            "Qwen-generated relevance labels were structurally validated but are not exhaustive corpus annotations.",
        ],
    }


def render_report(results: dict[str, Any]) -> str:
    primary = results["metrics"]["primary"]
    no_answer = results["metrics"]["candidate_unanswerable"]
    coverage = results["metrics"]["coverage"]
    provenance = results["retriever_provenance"]
    lines = [
            "# T10.10 Retrieval Evaluation Final",
            "",
            "## 结果",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| Recall@5 | {primary['recall_at_5']:.4f} |",
            f"| MRR@10 | {primary['mrr_at_10']:.4f} |",
            f"| nDCG@10 | {primary['ndcg_at_10']:.4f} |",
            f"| Source Hit@5 | {primary['source_hit_at_5']:.4f} |",
            f"| Page Hit@5 | {primary['page_hit_at_5']:.4f} |",
            "",
            f"主指标覆盖 {primary['query_count']} 个存在直接答案的查询；"
            f"另有 {no_answer['query_count']} 个候选池不可回答查询单独统计，"
            f"其中 {no_answer['supporting_only_count']} 个仅有辅助证据、"
            f"{no_answer['all_zero_count']} 个全部无关。",
            "",
            "## 指标验收",
            "",
            f"- Recall@5 ≥ 0.85：{'通过' if primary['recall_at_5'] >= 0.85 else '未通过'}。",
            f"- MRR@10 ≥ 0.70：{'通过' if primary['mrr_at_10'] >= 0.70 else '未通过'}。",
            f"- Top-10未标注返回项为0：{'通过' if coverage['unjudged_top10_count'] == 0 else '未通过'}。",
            "",
            "## Gold来源覆盖",
            "",
            "| 来源 | 候选 | relevance≥1 | relevance=2 | 直接Gold查询 |",
            "|---|---:|---:|---:|---:|",
        ]
    for source in results["gold_coverage"]["sources"]:
        lines.append(
            f"| {source['source_id']} | {source['pool_candidate_count']} | "
            f"{source['relevant_chunk_count']} | {source['direct_gold_chunk_count']} | "
            f"{source['direct_gold_query_count']} |"
        )
    lines.extend(
        [
            "",
            "来源覆盖按最终44题及当前Top-20候选池统计；被删除问题不再参与覆盖验收。",
            "",
            "## 规则与追踪",
            "",
            "- Recall/MRR 仅以 relevance=2 为命中；nDCG 使用 0/1/2 和 `2^relevance-1` 增益。",
            "- Source/Page Hit 使用生产 Top-5；页面命中要求来源相同且页码范围相交。",
            f"- 重新检索 {coverage['evaluated_query_count']} 题、"
            f"Top-10 共 {coverage['retrieved_result_count']} 条，"
            f"未标注返回项 {coverage['unjudged_top10_count']} 条。",
            f"- Collection：`{provenance['collection_name']}`；"
            f"Index：`{provenance['index_version']}`；"
            f"Fusion：`{provenance['fusion_version']}`。",
            f"- 总耗时：{results['runtime_seconds']:.3f} 秒。",
            "",
            "## 限制",
            "",
            "Gold 仅覆盖当前 Retriever 产生并经模型判断的评测 Top-20 候选池，"
            "不是对知识库全部 Chunk 的穷尽标注；因此本结果用于当前配置的可复现基线与后续对比。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze T10.10 Gold and evaluate the production Hybrid Retriever"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--freeze-gold", action="store_true")
    parser.add_argument("--overwrite-gold", action="store_true")
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    gold_path = args.gold.resolve()
    if args.freeze_gold:
        gold = freeze_gold(csv_path, gold_path, overwrite=args.overwrite_gold)
        print(
            json.dumps(
                {
                    "gold": str(gold_path),
                    "queries": gold["summary"]["query_count"],
                    "candidate_answerable": gold["summary"][
                        "candidate_answerable_count"
                    ],
                    "candidate_unanswerable": gold["summary"][
                        "candidate_unanswerable_count"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        if not gold_path.exists():
            raise RetrievalEvaluationError(
                "Frozen Gold does not exist; run with --freeze-gold first"
            )
        gold = load_frozen_gold(gold_path, csv_path)
    if args.gold_only:
        return 0

    responses, runtime_seconds = run_retrieval(gold, args.project_root.resolve())
    scores = aggregate_scores(gold, responses)
    results = build_results(gold, gold_path, scores, runtime_seconds)
    atomic_write_json(args.results.resolve(), results)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "results": str(args.results.resolve()),
                "report": str(report_path),
                "metrics": results["metrics"]["primary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
