from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = EVALUATION_DIR / "question/retriever_main_top20_v0.3.csv"
DEFAULT_PROMPT = EVALUATION_DIR / "annotation_prompt.txt"
DEFAULT_ENV_FILE = EVALUATION_DIR / ".env"
SHUFFLE_SEED = "pcba-qwen-judge-v1"


class QwenAnnotationError(RuntimeError):
    """Raised when a Qwen annotation response is unusable."""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise QwenAnnotationError(
                f"Invalid .env line {line_number}: expected KEY=VALUE"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in ("'", '"'):
            value = value[1:-1]
        if not key:
            raise QwenAnnotationError(f"Invalid .env line {line_number}: empty key")
        os.environ.setdefault(key, value)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise QwenAnnotationError("CSV has no header")
        return list(reader.fieldnames), list(reader)


def atomic_write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    for query_id, query_rows in grouped.items():
        if len(query_rows) != 20:
            raise QwenAnnotationError(f"{query_id} must have exactly 20 candidates")
        if len({row["candidate_id"] for row in query_rows}) != 20:
            raise QwenAnnotationError(f"{query_id} has duplicate candidate IDs")
    if not grouped:
        raise QwenAnnotationError("CSV contains no query groups")
    return dict(grouped)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _blind_order_key(query_id: str, candidate_id: str) -> str:
    value = f"{SHUFFLE_SEED}|{query_id}|{candidate_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_model_input(query_rows: list[dict[str, str]]) -> dict[str, Any]:
    query_id = query_rows[0]["query_id"]
    ordered = sorted(
        query_rows,
        key=lambda row: _blind_order_key(query_id, row["candidate_id"]),
    )
    return {
        "query_id": query_id,
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
            for row in ordered
        ],
    }


def build_request_body(
    model: str, system_prompt: str, model_input: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(model_input, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized.startswith(("https://", "http://")):
        raise QwenAnnotationError("QWEN_BASE_URL must start with http:// or https://")
    if "YOUR_WORKSPACE_ID" in normalized.upper() or "YOUR_" in normalized.upper():
        raise QwenAnnotationError(
            "QWEN_BASE_URL still contains a template placeholder; set the real API Base URL"
        )
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def send_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:1000]
        raise QwenAnnotationError(f"Qwen HTTP {exc.code}: {details}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise QwenAnnotationError(f"Qwen API request failed: {exc}") from exc


def extract_json_content(response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QwenAnnotationError("Qwen response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise QwenAnnotationError("Qwen returned empty message content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise QwenAnnotationError(f"Qwen content is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise QwenAnnotationError("Qwen JSON output must be an object")
    return parsed


def validate_annotation(
    annotation: dict[str, Any], model_input: dict[str, Any]
) -> dict[str, Any]:
    query_id = model_input["query_id"]
    if annotation.get("query_id") != query_id:
        raise QwenAnnotationError(f"Query ID mismatch for {query_id}")
    final_answerable = annotation.get("final_answerable")
    if type(final_answerable) is not bool:
        raise QwenAnnotationError("final_answerable must be true or false")
    values = annotation.get("candidates")
    if not isinstance(values, list) or len(values) != 20:
        raise QwenAnnotationError("Annotation must contain exactly 20 candidates")
    expected_ids = {
        candidate["candidate_id"] for candidate in model_input["candidates"]
    }
    relevance_by_id: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            raise QwenAnnotationError("Candidate annotation must be an object")
        candidate_id = value.get("candidate_id")
        relevance = value.get("relevance")
        if candidate_id in relevance_by_id:
            raise QwenAnnotationError(f"Duplicate candidate annotation: {candidate_id}")
        if type(relevance) is not int or relevance not in (0, 1, 2):
            raise QwenAnnotationError(f"Invalid relevance for {candidate_id}")
        relevance_by_id[candidate_id] = relevance
    if set(relevance_by_id) != expected_ids:
        raise QwenAnnotationError("Candidate IDs do not match the request")
    if final_answerable and not any(value > 0 for value in relevance_by_id.values()):
        raise QwenAnnotationError("Answerable=true conflicts with all relevance=0")
    if not final_answerable and any(value == 2 for value in relevance_by_id.values()):
        raise QwenAnnotationError("Answerable=false conflicts with relevance=2")
    return {
        "query_id": query_id,
        "final_answerable": final_answerable,
        "relevance_by_id": relevance_by_id,
    }


def request_with_retries(
    sender: Callable[[], dict[str, Any]],
    model_input: dict[str, Any],
    *,
    attempts: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return validate_annotation(extract_json_content(sender()), model_input)
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < attempts:
                sleep(min(2 ** (attempt - 1), 8))
    raise QwenAnnotationError("; ".join(errors))


def apply_annotation(
    query_rows: list[dict[str, str]],
    annotation: dict[str, Any],
    *,
    model: str,
    prompt_hash: str,
) -> None:
    final_value = str(annotation["final_answerable"]).lower()
    relevance_by_id = annotation["relevance_by_id"]
    for row in query_rows:
        row["final_answerable"] = final_value
        row["relevance"] = str(relevance_by_id[row["candidate_id"]])
        row["label_origin"] = "llm"
        row["model_id"] = model
        row["prompt_sha256"] = prompt_hash
        row["annotation_status"] = "completed"
        row["annotation_error"] = ""


def apply_failure(
    query_rows: list[dict[str, str]],
    error: Exception,
    *,
    model: str,
    prompt_hash: str,
) -> None:
    message = str(error).replace("\r", " ").replace("\n", " ")[:1000]
    for row in query_rows:
        row["final_answerable"] = ""
        row["relevance"] = ""
        row["label_origin"] = "llm"
        row["model_id"] = model
        row["prompt_sha256"] = prompt_hash
        row["annotation_status"] = "failed"
        row["annotation_error"] = message


def annotate_one_query(
    query_id: str,
    query_rows: list[dict[str, str]],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    attempts: int,
) -> tuple[str, dict[str, Any] | None, Exception | None]:
    model_input = build_model_input(query_rows)
    body = build_request_body(model, prompt, model_input)

    def sender() -> dict[str, Any]:
        return send_chat_completion(
            endpoint=endpoint,
            api_key=api_key,
            body=body,
            timeout_seconds=timeout_seconds,
        )

    try:
        annotation = request_with_retries(
            sender, model_input, attempts=attempts
        )
        return query_id, annotation, None
    except Exception as exc:
        return query_id, None, exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate Top-20 CSV with qwen3.8-max")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    load_env_file(args.env_file.resolve())
    prompt = args.prompt.resolve().read_text(encoding="utf-8").strip()
    if not prompt:
        raise QwenAnnotationError("annotation_prompt.txt is empty")
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    base_url = os.environ.get("QWEN_BASE_URL", "").strip()
    model = os.environ.get("QWEN_MODEL", "qwen3.8-max").strip()
    timeout_seconds = float(os.environ.get("QWEN_TIMEOUT_SECONDS", "180"))
    if not api_key or not base_url:
        raise QwenAnnotationError(
            "Set QWEN_API_KEY and QWEN_BASE_URL before running annotation"
        )
    endpoint = chat_completions_url(base_url)
    fieldnames, rows = read_csv(csv_path)
    grouped = group_rows(rows)
    selected_ids = args.query_id or sorted(grouped)
    unknown = sorted(set(selected_ids) - set(grouped))
    if unknown:
        raise QwenAnnotationError(f"Unknown query IDs: {unknown}")
    if args.max_queries is not None:
        if args.max_queries < 1:
            raise QwenAnnotationError("--max-queries must be positive")
        selected_ids = selected_ids[: args.max_queries]
    if args.workers < 1:
        raise QwenAnnotationError("--workers must be positive")
    if args.attempts < 1:
        raise QwenAnnotationError("--attempts must be positive")
    completed = 0
    failed = 0
    skipped = 0
    prompt_hash = prompt_sha256(prompt)
    work_ids: list[str] = []
    for query_id in selected_ids:
        query_rows = grouped[query_id]
        if not args.force and all(
            row["annotation_status"] == "completed" for row in query_rows
        ):
            skipped += 1
        else:
            work_ids.append(query_id)
    futures: dict[Future[tuple[str, dict[str, Any] | None, Exception | None]], str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for query_id in work_ids:
            future = executor.submit(
                annotate_one_query,
                query_id,
                grouped[query_id],
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
                attempts=args.attempts,
            )
            futures[future] = query_id
        for future in as_completed(futures):
            query_id, annotation, error = future.result()
            query_rows = grouped[query_id]
            if error is None and annotation is not None:
                apply_annotation(
                    query_rows, annotation, model=model, prompt_hash=prompt_hash
                )
                completed += 1
                print(
                    json.dumps({"query_id": query_id, "status": "completed"}),
                    flush=True,
                )
            else:
                assert error is not None
                apply_failure(
                    query_rows, error, model=model, prompt_hash=prompt_hash
                )
                failed += 1
                print(
                    json.dumps(
                        {"query_id": query_id, "status": "failed", "error": str(error)},
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            atomic_write_csv(csv_path, fieldnames, rows)
    print(
        json.dumps(
            {"completed": completed, "failed": failed, "skipped": skipped},
            ensure_ascii=False,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
