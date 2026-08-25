from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from annotate_with_qwen import (
    QwenAnnotationError,
    atomic_write_csv,
    build_request_body,
    chat_completions_url,
    extract_json_content,
    load_env_file,
    read_csv,
    send_chat_completion,
)


EVALUATION_DIR = Path(__file__).resolve().parent


def grouped_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    if not grouped or any(len(values) != 20 for values in grouped.values()):
        raise QwenAnnotationError("Every query must contain exactly 20 candidates")
    return grouped


def model_input(query_rows: list[dict[str, str]]) -> dict[str, Any]:
    pending = [row for row in query_rows if row["relevance"] not in {"0", "1", "2"}]
    return {
        "query_id": query_rows[0]["query_id"],
        "question": query_rows[0]["query"],
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "source_id": row["source_id"],
                "pdf_page_start": int(row["pdf_page_start"]),
                "pdf_page_end": int(row["pdf_page_end"]),
                "section_path": json.loads(row["section_path"]),
                "text": row["text"],
            }
            for row in pending
        ],
    }


def validate(value: dict[str, Any], request: dict[str, Any]) -> dict[str, int]:
    if value.get("query_id") != request["query_id"]:
        raise QwenAnnotationError("Query ID mismatch")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise QwenAnnotationError("candidates must be an array")
    expected = {item["candidate_id"] for item in request["candidates"]}
    labels: dict[str, int] = {}
    for item in candidates:
        if not isinstance(item, dict):
            raise QwenAnnotationError("Candidate label must be an object")
        candidate_id = item.get("candidate_id")
        relevance = item.get("relevance")
        if candidate_id in labels or type(relevance) is not int or relevance not in (0, 1, 2):
            raise QwenAnnotationError("Invalid or duplicate candidate label")
        labels[candidate_id] = relevance
    if set(labels) != expected:
        raise QwenAnnotationError("Candidate IDs do not match the request")
    return labels


def annotate_query(
    query_rows: list[dict[str, str]], *, endpoint: str, api_key: str, model: str,
    prompt: str, attempts: int, timeout_seconds: float
) -> tuple[str, dict[str, int] | None, str | None]:
    request = model_input(query_rows)
    query_id = request["query_id"]
    body = build_request_body(model, prompt, request)
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            response = send_chat_completion(
                endpoint=endpoint, api_key=api_key, body=body, timeout_seconds=timeout_seconds
            )
            return query_id, validate(extract_json_content(response), request), None
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    return query_id, None, "; ".join(errors)


def finalize_query(query_rows: list[dict[str, str]]) -> None:
    if not all(row["relevance"] in {"0", "1", "2"} for row in query_rows):
        return
    answerable = str(any(row["relevance"] == "2" for row in query_rows)).lower()
    for row in query_rows:
        row["final_answerable"] = answerable
        row["annotation_status"] = "completed"
        row["annotation_error"] = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally annotate only new final candidates")
    parser.add_argument(
        "--csv",
        type=Path,
        default=EVALUATION_DIR / "question/retriever_main_top20_v0.3.csv",
    )
    parser.add_argument("--prompt", type=Path, default=EVALUATION_DIR / "incremental_annotation_prompt.txt")
    parser.add_argument("--env-file", type=Path, default=EVALUATION_DIR / ".env")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file.resolve())
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    base_url = os.environ.get("QWEN_BASE_URL", "").strip()
    model = os.environ.get("QWEN_MODEL", "qwen3.8-max").strip()
    if not api_key or not base_url:
        raise QwenAnnotationError("QWEN_API_KEY and QWEN_BASE_URL are required")
    endpoint = chat_completions_url(base_url)
    prompt = args.prompt.resolve().read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    fields, rows = read_csv(args.csv.resolve())
    grouped = grouped_rows(rows)
    selected = set(args.query_id)
    unknown = selected - set(grouped)
    if unknown:
        raise QwenAnnotationError(f"Unknown query IDs: {sorted(unknown)}")
    work = [
        values
        for query_id, values in grouped.items()
        if (not selected or query_id in selected)
        and any(row["relevance"] not in {"0", "1", "2"} for row in values)
    ]
    failed = 0
    newly_labeled = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                annotate_query, values, endpoint=endpoint, api_key=api_key, model=model,
                prompt=prompt, attempts=args.attempts,
                timeout_seconds=(
                    args.timeout_seconds
                    if args.timeout_seconds is not None
                    else float(os.environ.get("QWEN_TIMEOUT_SECONDS", "180"))
                ),
            ): values
            for values in work
        }
        for future in as_completed(futures):
            query_rows = futures[future]
            query_id, labels, error = future.result()
            if labels is None:
                failed += 1
                for row in query_rows:
                    if row["relevance"] not in {"0", "1", "2"}:
                        row["annotation_status"] = "failed"
                        row["annotation_error"] = (error or "unknown error")[:1000]
                print(json.dumps({"query_id": query_id, "status": "failed", "error": error}, ensure_ascii=False), flush=True)
            else:
                for row in query_rows:
                    if row["candidate_id"] in labels:
                        row["relevance"] = str(labels[row["candidate_id"]])
                        row["label_origin"] = "llm_incremental"
                        row["model_id"] = model
                        row["prompt_sha256"] = prompt_hash
                        row["annotation_status"] = "completed"
                        row["annotation_error"] = ""
                        newly_labeled += 1
                finalize_query(query_rows)
                print(json.dumps({"query_id": query_id, "status": "completed", "new_candidates": len(labels)}, ensure_ascii=False), flush=True)
            atomic_write_csv(args.csv.resolve(), fields, rows)
    for values in grouped.values():
        finalize_query(values)
    atomic_write_csv(args.csv.resolve(), fields, rows)
    print(json.dumps({"queries": len(grouped), "queries_called": len(work), "newly_labeled": newly_labeled, "failed_queries": failed}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
