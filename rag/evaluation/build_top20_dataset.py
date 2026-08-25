from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EVALUATION_DIR / "question/retriever_main.json"
DEFAULT_OUTPUT = EVALUATION_DIR / "question/retriever_main_top20_v0.3.csv"
POOL_VERSION = "pcba-retrieval-eval-top20/1.0.0"
EMPTY_FILTERS = {
    "source_ids": [],
    "process_ids": [],
    "defect_ids": [],
    "evidence_roles": [],
    "languages": [],
    "document_types": [],
}

CSV_FIELDS = [
    "pool_version",
    "dataset_name",
    "dataset_sha256",
    "query_id",
    "query",
    "language",
    "original_answerable",
    "query_type",
    "defect_ids",
    "process_ids",
    "eval_focus",
    "candidate_id",
    "chunk_id",
    "rrf_rank",
    "dense_rank",
    "sparse_rank",
    "dense_score",
    "sparse_score",
    "dense_rrf_contribution",
    "sparse_rrf_contribution",
    "rrf_score",
    "source_id",
    "source_title",
    "pdf_page_start",
    "pdf_page_end",
    "section_path",
    "text",
    "knowledge_base_version",
    "index_version",
    "collection_name",
    "retriever_version",
    "fusion_version",
    "embedding_model",
    "embedding_model_revision",
    "query_token_count",
    "rrf_k",
    "final_answerable",
    "relevance",
    "label_origin",
    "model_id",
    "prompt_sha256",
    "annotation_status",
    "annotation_error",
]


class Top20DatasetError(RuntimeError):
    """Raised when the evaluation-only Top-20 pool is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("items"), list):
        raise Top20DatasetError("evaluation.json must contain an items array")
    items = dataset["items"]
    if dataset.get("query_count") != len(items) or not items:
        raise Top20DatasetError("query_count must match a non-empty items array")
    query_ids = [item.get("query_id") for item in items]
    duplicates = sorted(
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    )
    if duplicates:
        raise Top20DatasetError(f"Duplicate query IDs: {duplicates}")
    required = {
        "query_id",
        "query",
        "language",
        "answerable",
        "query_type",
        "defect_ids",
        "process_ids",
        "eval_focus",
    }
    for item in items:
        missing = required - set(item)
        if missing:
            raise Top20DatasetError(
                f"{item.get('query_id', '<unknown>')} missing fields: {sorted(missing)}"
            )
        if not item["query"].strip() or not isinstance(item["answerable"], bool):
            raise Top20DatasetError(f"Invalid query item: {item['query_id']}")
    return dataset


def empty_retrieval_request(query: str) -> dict[str, Any]:
    return {
        "schema_version": "1.1.0",
        "query": query,
        "top_k": 20,
        "filters": copy.deepcopy(EMPTY_FILTERS),
    }


def _identity(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": result["text"],
        "citation": result["citation"],
        "metadata": result["metadata"],
    }


def evaluation_rrf_top20(
    dense: dict[str, Any],
    sparse: dict[str, Any],
    *,
    rrf_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[dict[str, Any]]:
    if dense.get("retrieval_mode") != "dense" or sparse.get("retrieval_mode") != "sparse":
        raise Top20DatasetError("RRF requires one Dense and one Sparse response")
    if dense.get("query") != sparse.get("query"):
        raise Top20DatasetError("Dense/Sparse query mismatch")
    if len(dense.get("results", [])) != 20 or len(sparse.get("results", [])) != 20:
        raise Top20DatasetError("Evaluation pooling requires Dense Top-20 and Sparse Top-20")
    candidates: dict[str, dict[str, Any]] = {}
    seen_by_mode: dict[str, set[str]] = {"dense": set(), "sparse": set()}
    for mode, response in (("dense", dense), ("sparse", sparse)):
        for result in response["results"]:
            chunk_id = result["chunk_id"]
            if chunk_id in seen_by_mode[mode]:
                raise Top20DatasetError(f"Duplicate {mode} candidate: {chunk_id}")
            seen_by_mode[mode].add(chunk_id)
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
            elif _identity(candidates[chunk_id]) != _identity(result):
                raise Top20DatasetError(
                    f"Dense/Sparse payload mismatch for Chunk {chunk_id}"
                )
            candidate = candidates[chunk_id]
            candidate[f"{mode}_rank"] = int(result["rank"])
            candidate[f"{mode}_score"] = float(result[f"{mode}_score"])
    for candidate in candidates.values():
        dense_rank = candidate["dense_rank"]
        sparse_rank = candidate["sparse_rank"]
        dense_part = dense_weight / (rrf_k + dense_rank) if dense_rank else 0.0
        sparse_part = sparse_weight / (rrf_k + sparse_rank) if sparse_rank else 0.0
        candidate["dense_rrf_contribution"] = dense_part
        candidate["sparse_rrf_contribution"] = sparse_part
        candidate["rrf_score"] = dense_part + sparse_part

    def sort_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
        ranks = [
            rank
            for rank in (candidate["dense_rank"], candidate["sparse_rank"])
            if rank is not None
        ]
        return (-candidate["rrf_score"], min(ranks), candidate["chunk_id"])

    ordered = sorted(candidates.values(), key=sort_key)[:20]
    if len(ordered) != 20:
        raise Top20DatasetError("Evaluation RRF pool did not produce 20 candidates")
    for rank, candidate in enumerate(ordered, 1):
        candidate["rrf_rank"] = rank
    return ordered


def candidate_id(query_id: str, chunk_id: str) -> str:
    suffix = hashlib.sha256(f"{query_id}|{chunk_id}".encode("utf-8")).hexdigest()[:12]
    return f"{query_id}-A{suffix}"


def _json_array(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _number(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def rows_for_query(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    dataset_name: str,
    dataset_sha256: str,
    dense_trace: dict[str, Any],
    fusion_version: str,
    rrf_k: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        citation = candidate["citation"]
        rows.append(
            {
                "pool_version": POOL_VERSION,
                "dataset_name": dataset_name,
                "dataset_sha256": dataset_sha256,
                "query_id": item["query_id"],
                "query": item["query"],
                "language": item["language"],
                "original_answerable": str(item["answerable"]).lower(),
                "query_type": item["query_type"],
                "defect_ids": _json_array(item["defect_ids"]),
                "process_ids": _json_array(item["process_ids"]),
                "eval_focus": _json_array(item["eval_focus"]),
                "candidate_id": candidate_id(item["query_id"], candidate["chunk_id"]),
                "chunk_id": candidate["chunk_id"],
                "rrf_rank": str(candidate["rrf_rank"]),
                "dense_rank": "" if candidate["dense_rank"] is None else str(candidate["dense_rank"]),
                "sparse_rank": "" if candidate["sparse_rank"] is None else str(candidate["sparse_rank"]),
                "dense_score": _number(candidate["dense_score"]),
                "sparse_score": _number(candidate["sparse_score"]),
                "dense_rrf_contribution": _number(candidate["dense_rrf_contribution"]),
                "sparse_rrf_contribution": _number(candidate["sparse_rrf_contribution"]),
                "rrf_score": _number(candidate["rrf_score"]),
                "source_id": citation["source_id"],
                "source_title": citation["source_title"],
                "pdf_page_start": str(citation["pdf_page_start"]),
                "pdf_page_end": str(citation["pdf_page_end"]),
                "section_path": _json_array(citation["section_path"]),
                "text": candidate["text"],
                "knowledge_base_version": dense_trace["knowledge_base_version"],
                "index_version": dense_trace["index_version"],
                "collection_name": dense_trace["collection_name"],
                "retriever_version": dense_trace["retriever_version"],
                "fusion_version": fusion_version,
                "embedding_model": dense_trace["embedding_model"],
                "embedding_model_revision": dense_trace["embedding_model_revision"],
                "query_token_count": str(dense_trace["query_token_count"]),
                "rrf_k": str(rrf_k),
                "final_answerable": "",
                "relevance": "",
                "label_origin": "",
                "model_id": "",
                "prompt_sha256": "",
                "annotation_status": "pending",
                "annotation_error": "",
            }
        )
    return rows


def validate_rows(rows: list[dict[str, str]], *, require_pending: bool = False) -> None:
    if not rows or len(rows) % 20:
        raise Top20DatasetError(f"CSV row count must be a positive multiple of 20, got {len(rows)}")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    if len(grouped) != len(rows) // 20:
        raise Top20DatasetError("Query group count does not match CSV row count")
    all_candidate_ids: set[str] = set()
    for query_id, query_rows in grouped.items():
        if len(query_rows) != 20:
            raise Top20DatasetError(f"{query_id} has {len(query_rows)} candidates")
        if {int(row["rrf_rank"]) for row in query_rows} != set(range(1, 21)):
            raise Top20DatasetError(f"{query_id} has invalid RRF ranks")
        chunk_ids = [row["chunk_id"] for row in query_rows]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise Top20DatasetError(f"{query_id} contains duplicate Chunk IDs")
        for row in query_rows:
            if row["candidate_id"] in all_candidate_ids:
                raise Top20DatasetError(f"Duplicate candidate_id: {row['candidate_id']}")
            all_candidate_ids.add(row["candidate_id"])
            if int(row["pdf_page_start"]) < 1 or int(row["pdf_page_end"]) < int(row["pdf_page_start"]):
                raise Top20DatasetError(f"Invalid page range: {row['candidate_id']}")
            if require_pending and (
                row["annotation_status"] != "pending"
                or row["final_answerable"]
                or row["relevance"]
            ):
                raise Top20DatasetError("New dataset contains unexpected annotations")


def atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluation-only RRF Top-20 CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise Top20DatasetError(f"Refusing to overwrite existing CSV: {output_path}")
    dataset = load_queries(input_path)
    sys.path.insert(0, str(PROJECT_ROOT / "rag/src"))
    from pcba_rag.fusion import load_fusion_config
    from pcba_rag.retriever import Retriever

    fusion_config = load_fusion_config(PROJECT_ROOT)
    rrf_config = fusion_config["rrf"]
    rrf_k = int(rrf_config["k"])
    rows: list[dict[str, str]] = []
    with Retriever(PROJECT_ROOT) as retriever:
        for item in dataset["items"]:
            request = empty_retrieval_request(item["query"])
            dense = retriever.retrieve_dense(request)
            sparse = retriever.retrieve_sparse(request)
            candidates = evaluation_rrf_top20(
                dense,
                sparse,
                rrf_k=rrf_k,
                dense_weight=float(rrf_config["dense_weight"]),
                sparse_weight=float(rrf_config["sparse_weight"]),
            )
            rows.extend(
                rows_for_query(
                    item,
                    candidates,
                    dataset_name=dataset["dataset_name"],
                    dataset_sha256=sha256_file(input_path),
                    dense_trace=dense["trace"],
                    fusion_version=fusion_config["fusion_version"],
                    rrf_k=rrf_k,
                )
            )
    validate_rows(rows, require_pending=True)
    atomic_write_csv(output_path, rows)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "queries": len(dataset["items"]),
                "rows": len(rows),
                "candidates_per_query": 20,
                "filters": EMPTY_FILTERS,
                "production_fusion_config_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
