from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import yaml

from run_reranker_experiment import passage_text, rerank_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "rag/config/reranker.full_evaluation.v0.1.yaml"


class FullRerankerEvaluationError(RuntimeError):
    """Raised when the revised evaluation scope cannot reuse the frozen V0.3 Gold."""


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise FullRerankerEvaluationError("Config must be a mapping")
    return config


def load_scope(
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]], str]:
    input_config = config["input"]
    candidate_path = _resolve(input_config["candidate_pool"])
    gold_path = _resolve(input_config["original_gold"])
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if gold["source"]["sha256"] != _sha256(candidate_path):
        raise FullRerankerEvaluationError("Candidate pool differs from frozen Gold source")

    gold_by_id = {query["query_id"]: query for query in gold["queries"]}
    excluded = set(input_config["excluded_query_ids"])
    if not excluded <= set(gold_by_id):
        raise FullRerankerEvaluationError("Excluded query IDs are not in original Gold")
    retained_ids = set(gold_by_id) - excluded
    overrides = input_config.get("query_overrides", {})
    if not set(overrides) <= retained_ids:
        raise FullRerankerEvaluationError("Query override is not in retained scope")

    grouped: dict[str, list[dict[str, str]]] = {}
    expected = int(input_config["candidate_count_per_query"])
    for query_id in sorted(retained_ids):
        query_rows = sorted(
            (row for row in rows if row["query_id"] == query_id),
            key=lambda row: int(row["rrf_rank"]),
        )
        if len(query_rows) != expected:
            raise FullRerankerEvaluationError(
                f"{query_id} expected {expected} candidates, found {len(query_rows)}"
            )
        if {int(row["rrf_rank"]) for row in query_rows} != set(range(1, expected + 1)):
            raise FullRerankerEvaluationError(f"{query_id} has invalid RRF ranks")
        csv_labels = {row["chunk_id"]: int(row["relevance"]) for row in query_rows}
        gold_labels = {
            item["chunk_id"]: int(item["relevance"])
            for item in gold_by_id[query_id]["graded_relevance"]
        }
        if csv_labels != gold_labels:
            raise FullRerankerEvaluationError(f"{query_id} labels differ from original Gold")
        grouped[query_id] = query_rows
    return grouped, {query_id: gold_by_id[query_id] for query_id in retained_ids}, _sha256(gold_path)


def _dcg(relevances: list[int]) -> float:
    return sum(
        ((2**relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, 1)
    )


def _page_overlap(row: dict[str, Any], target: dict[str, Any]) -> bool:
    return bool(
        row["source_id"] == target["source_id"]
        and int(row["pdf_page_start"]) <= int(target["pdf_page_end"])
        and int(target["pdf_page_start"]) <= int(row["pdf_page_end"])
    )


def score_order(gold_query: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct = gold_query["direct_gold_chunks"]
    if not direct:
        return {
            "query_id": gold_query["query_id"],
            "candidate_answerable": False,
            "support_hit_at_5": any(
                row["chunk_id"]
                in {item["chunk_id"] for item in gold_query["supporting_chunks"]}
                for row in rows[:5]
            ),
        }
    direct_ids = {item["chunk_id"] for item in direct}
    graded = {
        item["chunk_id"]: int(item["relevance"])
        for item in gold_query["graded_relevance"]
    }
    top10 = rows[:10]
    top5 = rows[:5]
    top5_ids = {row["chunk_id"] for row in top5}
    direct_hit_count_at_5 = len(top5_ids & direct_ids)
    direct_ranks = [
        rank for rank, row in enumerate(top10, 1) if row["chunk_id"] in direct_ids
    ]
    observed = [graded[row["chunk_id"]] for row in top10]
    ideal = sorted(graded.values(), reverse=True)[:10]
    ideal_dcg = _dcg(ideal)
    gold_sources = {item["source_id"] for item in direct}
    return {
        "query_id": gold_query["query_id"],
        "candidate_answerable": True,
        "recall_at_5": direct_hit_count_at_5 / len(direct_ids),
        "direct_gold_count": len(direct_ids),
        "direct_hit_count_at_5": direct_hit_count_at_5,
        "mrr_at_10": (1.0 / min(direct_ranks)) if direct_ranks else 0.0,
        "ndcg_at_10": _dcg(observed) / ideal_dcg,
        "source_hit_at_5": any(row["source_id"] in gold_sources for row in top5),
        "page_hit_at_5": any(
            _page_overlap(row, target) for row in top5 for target in direct
        ),
        "first_rel2_rank": min(direct_ranks) if direct_ranks else None,
    }


def aggregate_scores(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [item for item in per_query if item["candidate_answerable"]]
    if not primary:
        raise FullRerankerEvaluationError("No answerable queries in retained scope")

    def mean(field: str) -> float:
        return sum(float(item[field]) for item in primary) / len(primary)

    macro_recall_at_5 = mean("recall_at_5")
    rel2_gold_count = sum(int(item["direct_gold_count"]) for item in primary)
    rel2_hit_at_5_count = sum(
        int(item["direct_hit_count_at_5"]) for item in primary
    )
    return {
        "query_count": len(primary),
        # Keep recall_at_5 as a compatibility alias for the established macro metric.
        "recall_at_5": macro_recall_at_5,
        "macro_recall_at_5": macro_recall_at_5,
        "micro_rel2_recall_at_5": rel2_hit_at_5_count / rel2_gold_count,
        "rel2_gold_count": rel2_gold_count,
        "rel2_hit_at_5_count": rel2_hit_at_5_count,
        "mrr_at_10": mean("mrr_at_10"),
        "ndcg_at_10": mean("ndcg_at_10"),
        "source_hit_at_5": mean("source_hit_at_5"),
        "page_hit_at_5": mean("page_hit_at_5"),
        "direct_hit_at_5_count": sum(
            item["first_rel2_rank"] is not None and item["first_rel2_rank"] <= 5
            for item in primary
        ),
        "direct_hit_at_10_count": sum(
            item["first_rel2_rank"] is not None for item in primary
        ),
        "top5_miss_query_ids": [
            item["query_id"]
            for item in primary
            if item["first_rel2_rank"] is None or item["first_rel2_rank"] > 5
        ],
        "top10_miss_query_ids": [
            item["query_id"] for item in primary if item["first_rel2_rank"] is None
        ],
    }


def build_report(
    config: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    runtime: dict[str, float],
    gold_sha256: str,
) -> str:
    input_config = config["input"]
    model = config["model"]
    deleted = "、".join(input_config["excluded_query_ids"])
    rewritten = input_config["query_overrides"]["Q019"]
    rows = [
        (
            "Macro Recall@5",
            before["macro_recall_at_5"],
            after["macro_recall_at_5"],
        ),
        (
            "Micro rel=2 Recall@5",
            before["micro_rel2_recall_at_5"],
            after["micro_rel2_recall_at_5"],
        ),
        ("MRR@10", before["mrr_at_10"], after["mrr_at_10"]),
        ("nDCG@10", before["ndcg_at_10"], after["ndcg_at_10"]),
        ("Source Hit@5", before["source_hit_at_5"], after["source_hit_at_5"]),
        ("Page Hit@5", before["page_hit_at_5"], after["page_hit_at_5"]),
    ]
    lines = [
        "# T10.10 Reranker 全量重评",
        "",
        "## 范围",
        "",
        f"- 原44题删除11题，剩余33题；删除：{deleted}。",
        f"- Q019改写为：{rewritten}",
        f"- 复用原V0.3 Gold标签，Gold SHA256：`{gold_sha256}`。",
        f"- 链路：RRF Top-20 → `{model['model_id']}` → Top-10 / Top-5。",
        "- 本次只做离线重排评测，生产Retriever未修改。",
        "",
        "## 指标",
        "",
        f"主指标覆盖{after['query_count']}个原Gold含直接答案的问题。",
        "",
        "| 指标 | RRF | Reranker | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for name, old, new in rows:
        lines.append(f"| {name} | {old:.4f} | {new:.4f} | {new-old:+.4f} |")
    lines.extend(
        [
            "",
            f"- Macro Recall@5 ≥ 0.85：{'通过' if after['macro_recall_at_5'] >= 0.85 else '未通过'}。",
            f"- MRR@10 ≥ 0.70：{'通过' if after['mrr_at_10'] >= 0.70 else '未通过'}。",
            f"- Micro rel=2 Recall@5：RRF为{before['rel2_hit_at_5_count']}/{before['rel2_gold_count']}，Reranker为{after['rel2_hit_at_5_count']}/{after['rel2_gold_count']}。",
            "- 当前T10.11的Recall@5验收门槛沿用逐题Recall平均值，即Macro Recall@5。",
            "",
            "| 直接答案覆盖 | RRF | Reranker |",
            "|---|---:|---:|",
            f"| Top-5命中问题数 | {before['direct_hit_at_5_count']}/{before['query_count']} | {after['direct_hit_at_5_count']}/{after['query_count']} |",
            f"| Top-10命中问题数 | {before['direct_hit_at_10_count']}/{before['query_count']} | {after['direct_hit_at_10_count']}/{after['query_count']} |",
            "",
            f"- Reranker Top-5仍未命中：{'、'.join(after['top5_miss_query_ids']) or '无'}。",
            f"- Reranker Top-10仍未命中：{'、'.join(after['top10_miss_query_ids']) or '无'}。",
            "",
            "## 运行",
            "",
            f"- 模型revision：`{model['revision']}`；CUDA FP16，batch={model['batch_size']}，max_length={model['max_length']}。",
            f"- 模型加载：{runtime['model_load_seconds']:.3f}秒；660对打分：{runtime['scoring_seconds']:.3f}秒。",
            f"- CUDA峰值已分配显存：{runtime['peak_cuda_memory_mb']:.2f}MB。",
            f"- 最长输入：{int(runtime['max_input_tokens'])} tokens；截断：{int(runtime['truncated_pair_count'])}对。",
            "",
            "## 限制",
            "",
            "Q019改写后只更新了Reranker查询文本；为遵循“按原来的Gold”要求，RRF Top-20候选和相关性标签仍来自原Q019。原Gold中Q019没有rel=2，因此不进入上述27题主指标。若要衡量改写问题本身的召回效果，必须重新生成其RRF候选并重新标注。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Reranker over revised V0.3 scope")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    grouped, gold_by_id, gold_sha256 = load_scope(config)
    model_config = config["model"]
    overrides = config["input"].get("query_overrides", {})

    import torch
    from FlagEmbedding import FlagReranker

    if model_config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise FullRerankerEvaluationError("Configured CUDA device is unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    reranker = FlagReranker(
        model_config["model_id"],
        revision=model_config["revision"],
        use_fp16=bool(model_config["use_fp16"]),
        devices=model_config["device"],
        batch_size=int(model_config["batch_size"]),
        max_length=int(model_config["max_length"]),
        normalize=bool(model_config["normalize_score"]),
        trust_remote_code=False,
    )
    model_load_seconds = time.perf_counter() - started

    before_scores: list[dict[str, Any]] = []
    after_scores: list[dict[str, Any]] = []
    pair_lengths: list[int] = []
    scoring_started = time.perf_counter()
    for query_id, rows in grouped.items():
        gold_query = gold_by_id[query_id]
        before_scores.append(score_order(gold_query, rows))
        query = overrides.get(query_id, gold_query["query"])
        pairs = [(query, passage_text(row)) for row in rows]
        pair_lengths.extend(
            len(reranker.tokenizer(left, right, truncation=False)["input_ids"])
            for left, right in pairs
        )
        scores = [float(value) for value in reranker.compute_score(pairs)]
        reranked = rerank_rows(rows, scores)
        after_scores.append(score_order(gold_query, reranked))
    torch.cuda.synchronize()
    runtime = {
        "model_load_seconds": model_load_seconds,
        "scoring_seconds": time.perf_counter() - scoring_started,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_input_tokens": float(max(pair_lengths)),
        "truncated_pair_count": float(
            sum(length > int(model_config["max_length"]) for length in pair_lengths)
        ),
    }
    before = aggregate_scores(before_scores)
    after = aggregate_scores(after_scores)
    report_path = _resolve(config["output"]["report"])
    atomic_write(report_path, build_report(config, before, after, runtime, gold_sha256))
    print(
        json.dumps(
            {
                "retained_queries": len(grouped),
                "answerable_queries": after["query_count"],
                "before": before,
                "after": after,
                "runtime": runtime,
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
