from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "FROZEN_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_evaluation(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str]] = []
    for relative, expected in manifest["artifacts"].items():
        path = PROJECT_ROOT / relative
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            mismatches.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    if mismatches:
        raise RuntimeError(f"Frozen artifact mismatch: {mismatches}")

    csv_path = PROJECT_ROOT / "rag/evaluation/question/retriever_main_top20_v0.3.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320 or {row["annotation_status"] for row in rows} != {"completed"}:
        raise RuntimeError("Frozen candidate pool must contain 320 completed rows")
    relevance_counts = {
        label: sum(row["relevance"] == label for row in rows)
        for label in ("0", "1", "2")
    }
    if relevance_counts != {"0": 185, "1": 92, "2": 43}:
        raise RuntimeError(f"Unexpected relevance counts: {relevance_counts}")
    return {
        "status": "valid",
        "artifact_count": len(manifest["artifacts"]),
        "candidate_count": len(rows),
        "relevance_counts": relevance_counts,
    }


if __name__ == "__main__":
    print(json.dumps(verify_frozen_evaluation(), ensure_ascii=False))
