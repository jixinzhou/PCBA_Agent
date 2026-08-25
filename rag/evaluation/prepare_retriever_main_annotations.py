from __future__ import annotations

import argparse
import csv
from pathlib import Path


REWRITTEN_QUERY_IDS = {"Q019", "Q024", "Q036"}
ANNOTATION_FIELDS = (
    "final_answerable",
    "relevance",
    "model_id",
    "prompt_sha256",
    "annotation_status",
    "annotation_error",
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def prepare_rows(
    new_rows: list[dict[str, str]], old_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    old_by_key = {
        (row["query_id"], row["query"], row["chunk_id"], row["text"]): row
        for row in old_rows
    }
    reused = 0
    pending = 0
    for row in new_rows:
        if row["query_id"] in REWRITTEN_QUERY_IDS:
            pending += 1
            continue
        key = (row["query_id"], row["query"], row["chunk_id"], row["text"])
        old = old_by_key.get(key)
        if old is None or old.get("annotation_status") != "completed":
            raise ValueError(f"No completed exact-match label for {row['candidate_id']}")
        for field in ANNOTATION_FIELDS:
            row[field] = old[field]
        row["label_origin"] = "reused_v0.3_exact_query_chunk"
        reused += 1
    return new_rows, {"reused": reused, "pending": pending}


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reuse only query-aware exact labels")
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    args = parser.parse_args()
    fieldnames, new_rows = read_rows(args.new)
    _, old_rows = read_rows(args.old)
    rows, counts = prepare_rows(new_rows, old_rows)
    write_rows(args.new, fieldnames, rows)
    print(f"reused={counts['reused']} pending={counts['pending']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
