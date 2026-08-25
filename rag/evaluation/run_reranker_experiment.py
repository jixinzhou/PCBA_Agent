from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "rag/config/reranker.experiment.v0.1.yaml"


class RerankerExperimentError(RuntimeError):
    """Raised when the isolated reranker experiment input is invalid."""


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RerankerExperimentError("Reranker config must be a mapping")
    return config


def load_inputs(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    input_config = config["input"]
    with _resolve(input_config["candidate_pool"]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        candidate_rows = list(csv.DictReader(stream))
    with _resolve(input_config["attribution"]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        attribution_rows = list(csv.DictReader(stream))

    target_ids = {
        row["question_id"]
        for row in attribution_rows
        if row["failure_type"] == input_config["failure_type"]
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    for query_id in sorted(target_ids):
        rows = sorted(
            (row for row in candidate_rows if row["query_id"] == query_id),
            key=lambda row: int(row["rrf_rank"]),
        )
        expected = int(input_config["candidate_count_per_query"])
        if len(rows) != expected:
            raise RerankerExperimentError(
                f"{query_id} expected {expected} candidates, found {len(rows)}"
            )
        if {int(row["rrf_rank"]) for row in rows} != set(range(1, expected + 1)):
            raise RerankerExperimentError(f"{query_id} has invalid RRF ranks")
        if not any(row["relevance"] == "2" for row in rows):
            raise RerankerExperimentError(f"{query_id} has no direct Gold")
        grouped[query_id] = rows
    if not grouped:
        raise RerankerExperimentError("No ranking_failure queries found")
    return grouped


def passage_text(row: dict[str, str]) -> str:
    section_path = json.loads(row["section_path"])
    section = " > ".join(section_path)
    return f"{section}\n\n{row['text']}" if section else row["text"]


def rerank_rows(
    rows: list[dict[str, str]], scores: list[float]
) -> list[dict[str, Any]]:
    if len(rows) != len(scores):
        raise RerankerExperimentError("Candidate and score counts differ")
    scored = [
        {**row, "reranker_score": float(score)}
        for row, score in zip(rows, scores, strict=True)
    ]
    return sorted(
        scored,
        key=lambda row: (
            -row["reranker_score"],
            int(row["rrf_rank"]),
            row["chunk_id"],
        ),
    )


def query_metrics(
    query_id: str, original: list[dict[str, str]], reranked: list[dict[str, Any]]
) -> dict[str, Any]:
    before = min(int(row["rrf_rank"]) for row in original if row["relevance"] == "2")
    after = min(
        rank
        for rank, row in enumerate(reranked, 1)
        if row["relevance"] == "2"
    )
    top5 = reranked[:5]
    return {
        "query_id": query_id,
        "before_first_rel2_rank": before,
        "after_first_rel2_rank": after,
        "before_top5_hit": before <= 5,
        "after_top5_hit": after <= 5,
        "before_mrr_at_5": (1.0 / before) if before <= 5 else 0.0,
        "after_mrr_at_5": (1.0 / after) if after <= 5 else 0.0,
        "top5": [
            {
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "relevance": int(row["relevance"]),
                "source_id": row["source_id"],
                "pdf_page_start": int(row["pdf_page_start"]),
                "pdf_page_end": int(row["pdf_page_end"]),
                "reranker_score": row["reranker_score"],
            }
            for rank, row in enumerate(top5, 1)
        ],
    }


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(metrics)
    return {
        "query_count": count,
        "before_top5_hit_count": sum(item["before_top5_hit"] for item in metrics),
        "after_top5_hit_count": sum(item["after_top5_hit"] for item in metrics),
        "before_mrr_at_5": sum(item["before_mrr_at_5"] for item in metrics) / count,
        "after_mrr_at_5": sum(item["after_mrr_at_5"] for item in metrics) / count,
    }


def _format_rank(value: int) -> str:
    return str(value) if value <= 5 else f"{value}（Top-5外）"


def build_report(
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
    summary: dict[str, float | int],
    runtime: dict[str, float],
) -> str:
    model = config["model"]
    lines = [
        "# T10.10 Reranker 隔离实验",
        "",
        "## 配置",
        "",
        f"- 候选：8个 `ranking_failure` 问题的现有RRF Top-20。",
        f"- 模型：`{model['model_id']}`，revision `{model['revision']}`。",
        f"- 推理：CUDA FP16，batch={model['batch_size']}，max_length={model['max_length']}。",
        "- 输出：按Reranker分数重排Top-20后截取Top-5；未修改生产Retriever。",
        "",
        "## 结果",
        "",
        "| 指标 | RRF Top-5 | Reranker Top-5 |",
        "|---|---:|---:|",
        f"| 含rel=2的问题数 | {summary['before_top5_hit_count']}/{summary['query_count']} | {summary['after_top5_hit_count']}/{summary['query_count']} |",
        f"| Hit@5 | {summary['before_top5_hit_count']/summary['query_count']:.4f} | {summary['after_top5_hit_count']/summary['query_count']:.4f} |",
        f"| MRR@5 | {summary['before_mrr_at_5']:.4f} | {summary['after_mrr_at_5']:.4f} |",
        "",
        "| 问题 | 重排前首个rel=2 | 重排后首个rel=2 | 进入Top-5 |",
        "|---|---:|---:|:---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['query_id']} | {item['before_first_rel2_rank']} | "
            f"{_format_rank(item['after_first_rel2_rank'])} | "
            f"{'是' if item['after_top5_hit'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 运行",
            "",
            f"- 模型加载：{runtime['model_load_seconds']:.3f} 秒。",
            f"- 160对Query/Chunk重排：{runtime['scoring_seconds']:.3f} 秒，平均每题 {runtime['scoring_seconds']/len(metrics):.3f} 秒。",
            f"- CUDA峰值已分配显存：{runtime['peak_cuda_memory_mb']:.2f} MB。",
            f"- 实际最长输入：{int(runtime['max_input_tokens'])} tokens；超过max_length的输入：{int(runtime['truncated_pair_count'])}对。",
            "",
            "本实验只判断Reranker能否把已知直接证据提升到Top-5，不代表44题总体指标，也不自动进入生产配置。",
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
    parser = argparse.ArgumentParser(description="Run isolated BGE reranker experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    grouped = load_inputs(config)
    model_config = config["model"]

    import torch
    from FlagEmbedding import FlagReranker

    if model_config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RerankerExperimentError("Configured CUDA device is unavailable")
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

    metrics: list[dict[str, Any]] = []
    pair_lengths: list[int] = []
    scoring_started = time.perf_counter()
    for query_id, rows in grouped.items():
        query = rows[0]["query"]
        pairs = [(query, passage_text(row)) for row in rows]
        pair_lengths.extend(
            len(reranker.tokenizer(left, right, truncation=False)["input_ids"])
            for left, right in pairs
        )
        raw_scores = reranker.compute_score(pairs)
        scores = [float(score) for score in raw_scores]
        reranked = rerank_rows(rows, scores)
        metrics.append(query_metrics(query_id, rows, reranked))
    torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - scoring_started
    runtime = {
        "model_load_seconds": model_load_seconds,
        "scoring_seconds": scoring_seconds,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_input_tokens": float(max(pair_lengths)),
        "truncated_pair_count": float(
            sum(length > int(model_config["max_length"]) for length in pair_lengths)
        ),
    }
    summary = aggregate(metrics)
    report_path = _resolve(config["output"]["report"])
    atomic_write(report_path, build_report(config, metrics, summary, runtime))
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "model_revision": model_config["revision"],
                "FlagEmbedding": importlib.metadata.version("FlagEmbedding"),
                "summary": summary,
                "runtime": runtime,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
