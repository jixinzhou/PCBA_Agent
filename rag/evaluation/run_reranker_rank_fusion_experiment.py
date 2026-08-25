from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from run_reranker_experiment import passage_text, rerank_rows
from run_reranker_full_evaluation import (
    FullRerankerEvaluationError,
    aggregate_scores,
    atomic_write,
    load_config as load_base_config,
    load_scope,
    score_order,
)


DEFAULT_CONFIG = PROJECT_ROOT / "rag/config/reranker.rank_fusion.v0.1.yaml"


class RankFusionExperimentError(RuntimeError):
    """Raised when rank-fusion inputs or settings are inconsistent."""


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RankFusionExperimentError("Experiment config must be a mapping")
    return config


def weight_grid(start: float, end: float, step: float) -> list[float]:
    start_value = Decimal(str(start))
    end_value = Decimal(str(end))
    step_value = Decimal(str(step))
    if step_value <= 0 or start_value < 0 or end_value > 1 or start_value > end_value:
        raise RankFusionExperimentError("Weights must satisfy 0 <= start <= end <= 1 and step > 0")
    values: list[float] = []
    current = start_value
    while current <= end_value:
        values.append(float(current))
        current += step_value
    if not values or Decimal(str(values[-1])) != end_value:
        raise RankFusionExperimentError("Weight grid must include the configured end value")
    return values


def fuse_rows(
    original_rows: list[dict[str, Any]],
    reranked_rows: list[dict[str, Any]],
    reranker_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    if not 0 <= reranker_weight <= 1:
        raise RankFusionExperimentError("reranker_weight must be between 0 and 1")
    if rrf_k <= 0:
        raise RankFusionExperimentError("rrf_k must be positive")
    original_by_id = {row["chunk_id"]: row for row in original_rows}
    reranker_rank = {
        row["chunk_id"]: rank for rank, row in enumerate(reranked_rows, 1)
    }
    if set(original_by_id) != set(reranker_rank):
        raise RankFusionExperimentError("RRF and Reranker candidate IDs differ")
    retriever_weight = 1.0 - reranker_weight
    fused: list[dict[str, Any]] = []
    for chunk_id, row in original_by_id.items():
        retrieval_rank = int(row["rrf_rank"])
        second_stage_rank = reranker_rank[chunk_id]
        score = (
            retriever_weight / (rrf_k + retrieval_rank)
            + reranker_weight / (rrf_k + second_stage_rank)
        )
        fused.append(
            {
                **row,
                "reranker_rank": second_stage_rank,
                "rank_fusion_score": score,
            }
        )
    return sorted(
        fused,
        key=lambda row: (
            -float(row["rank_fusion_score"]),
            min(int(row["rrf_rank"]), int(row["reranker_rank"])),
            int(row["reranker_rank"]),
            int(row["rrf_rank"]),
            row["chunk_id"],
        ),
    )


def evaluate_order(
    ordered_by_query: dict[str, list[dict[str, Any]]],
    gold_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_query = [
        score_order(gold_by_id[query_id], rows)
        for query_id, rows in ordered_by_query.items()
    ]
    summary = aggregate_scores(per_query)
    relevance_counts: Counter[int] = Counter()
    for query_id, rows in ordered_by_query.items():
        gold_query = gold_by_id[query_id]
        if not gold_query["direct_gold_chunks"]:
            continue
        graded = {
            item["chunk_id"]: int(item["relevance"])
            for item in gold_query["graded_relevance"]
        }
        relevance_counts.update(graded[row["chunk_id"]] for row in rows[:5])
    return {
        "macro_recall_at_5": summary["macro_recall_at_5"],
        "micro_rel2_recall_at_5": summary["micro_rel2_recall_at_5"],
        "rel2_hit_at_5_count": summary["rel2_hit_at_5_count"],
        "rel2_gold_count": summary["rel2_gold_count"],
        "direct_hit_at_5_count": summary["direct_hit_at_5_count"],
        "direct_hit_at_10_count": summary["direct_hit_at_10_count"],
        "mrr_at_10": summary["mrr_at_10"],
        "ndcg_at_10": summary["ndcg_at_10"],
        "source_hit_at_5": summary["source_hit_at_5"],
        "page_hit_at_5": summary["page_hit_at_5"],
        "top5_relevance_counts": {
            str(relevance): relevance_counts[relevance] for relevance in (0, 1, 2)
        },
        "top5_miss_query_ids": summary["top5_miss_query_ids"],
        "top10_miss_query_ids": summary["top10_miss_query_ids"],
    }


def select_best(grid_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[float]]:
    if not grid_results:
        raise RankFusionExperimentError("No grid results")
    best_macro = max(item["metrics"]["macro_recall_at_5"] for item in grid_results)
    macro_best = [
        item
        for item in grid_results
        if abs(item["metrics"]["macro_recall_at_5"] - best_macro) < 1e-12
    ]
    selected = max(
        macro_best,
        key=lambda item: (
            item["metrics"]["micro_rel2_recall_at_5"],
            item["metrics"]["mrr_at_10"],
            item["metrics"]["ndcg_at_10"],
            -item["reranker_weight"],
        ),
    )
    return selected, [item["reranker_weight"] for item in macro_best]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search RRF/Reranker rank fusion")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    base_config_path = _resolve(experiment["base_evaluation_config"])
    base_config = load_base_config(base_config_path)
    grouped, gold_by_id, gold_sha256 = load_scope(base_config)
    fusion_config = experiment["fusion"]
    weights = weight_grid(
        fusion_config["reranker_weight_start"],
        fusion_config["reranker_weight_end"],
        fusion_config["reranker_weight_step"],
    )

    import torch
    from FlagEmbedding import FlagReranker

    model_config = base_config["model"]
    if model_config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise FullRerankerEvaluationError("Configured CUDA device is unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
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
    model_load_seconds = time.perf_counter() - load_started

    overrides = base_config["input"].get("query_overrides", {})
    reranked_by_query: dict[str, list[dict[str, Any]]] = {}
    scoring_started = time.perf_counter()
    for query_id, rows in grouped.items():
        query = overrides.get(query_id, gold_by_id[query_id]["query"])
        pairs = [(query, passage_text(row)) for row in rows]
        scores = [float(value) for value in reranker.compute_score(pairs)]
        reranked_by_query[query_id] = rerank_rows(rows, scores)
    torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - scoring_started

    rrf_k = int(fusion_config["rrf_k"])
    grid_results: list[dict[str, Any]] = []
    for weight in weights:
        ordered = {
            query_id: fuse_rows(rows, reranked_by_query[query_id], weight, rrf_k)
            for query_id, rows in grouped.items()
        }
        grid_results.append(
            {
                "reranker_weight": weight,
                "retriever_weight": 1.0 - weight,
                "metrics": evaluate_order(ordered, gold_by_id),
            }
        )
    selected, macro_best_weights = select_best(grid_results)
    rrf_baseline = next(item for item in grid_results if item["reranker_weight"] == 0.0)
    reranker_baseline = next(
        item for item in grid_results if item["reranker_weight"] == 1.0
    )
    result = {
        "experiment_version": experiment["experiment_version"],
        "scope": {
            "retained_query_count": len(grouped),
            "answerable_query_count": sum(
                1 for query in gold_by_id.values() if query["direct_gold_chunks"]
            ),
            "candidate_count_per_query": base_config["input"]["candidate_count_per_query"],
            "gold_sha256": gold_sha256,
        },
        "formula": "(1-w)/(rrf_k+retrieval_rank) + w/(rrf_k+reranker_rank)",
        "rrf_k": rrf_k,
        "selection_policy": {
            "primary_metric": fusion_config["primary_metric"],
            "tie_break_metrics": fusion_config["tie_break_metrics"],
            "final_tie_break": fusion_config["final_tie_break"],
            "exploratory_in_sample_grid_search": True,
        },
        "baselines": {
            "rrf_weight_0": rrf_baseline,
            "reranker_weight_1": reranker_baseline,
        },
        "best": selected,
        "macro_best_weight_range": {
            "min": min(macro_best_weights),
            "max": max(macro_best_weights),
            "count": len(macro_best_weights),
            "weights": macro_best_weights,
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "scoring_seconds": scoring_seconds,
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
        "grid": grid_results,
    }
    output_path = _resolve(experiment["output"]["results"])
    atomic_write(output_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rrf": rrf_baseline,
                "reranker": reranker_baseline,
                "best": selected,
                "macro_best_weight_range": result["macro_best_weight_range"],
                "output": str(output_path.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
