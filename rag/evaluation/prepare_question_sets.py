from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = EVALUATION_DIR / "archive/legacy_labeled/evaluation_final.json"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "question"

QUESTION_GROUPS = {
    "retriever_main": [
        "Q002",
        "Q003",
        "Q005",
        "Q011",
        "Q012",
        "Q015",
        "Q018",
        "Q019",
        "Q021",
        "Q022",
        "Q024",
        "Q026",
        "Q028",
        "Q031",
        "Q035",
        "Q036",
    ],
    "multi_evidence": ["Q001", "Q009", "Q029"],
    "agent_kg": ["Q038", "Q039"],
    "no_answer_abstention": ["Q041", "Q043", "Q048"],
}

QUERY_OVERRIDES = {
    "Q019": "为什么过大的焊膏沉积量可能在印刷和贴装后未出现桥连，却在回流后形成桥连？",
    "Q024": "为什么回流过程中总焊料量相对于焊点间距过大会导致桥连？",
    "Q036": "Can an initial component placement offset change during reflow due to solder self-alignment",
}

DESCRIPTIONS = {
    "retriever_main": "Formal 16-query Retriever primary evaluation goal",
    "multi_evidence": "Multi-evidence questions evaluated separately from Retriever primary metrics",
    "agent_kg": "Questions reserved for Agent and knowledge-graph evaluation",
    "no_answer_abstention": "No-answer and abstention questions evaluated separately",
}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_question_sets(source: Path, output_dir: Path) -> dict[str, Any]:
    dataset = json.loads(source.read_text(encoding="utf-8"))
    items = dataset.get("items")
    if not isinstance(items, list):
        raise ValueError("Source evaluation dataset must contain items")
    by_id = {item["query_id"]: item for item in items}
    selected_ids = [query_id for values in QUESTION_GROUPS.values() for query_id in values]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Question groups contain duplicate query IDs")
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"Question IDs missing from source dataset: {missing}")

    outputs: dict[str, str] = {}
    for group_name, query_ids in QUESTION_GROUPS.items():
        group_items: list[dict[str, Any]] = []
        for query_id in query_ids:
            item = dict(by_id[query_id])
            if query_id in QUERY_OVERRIDES:
                item["original_query"] = item["query"]
                item["query"] = QUERY_OVERRIDES[query_id]
                item["query_version"] = "rewritten_v1"
            else:
                item["query_version"] = "original"
            item["evaluation_group"] = group_name
            group_items.append(item)
        output = {
            "schema_version": "1.0.0",
            "dataset_name": f"pcba_{group_name}_v1",
            "description": DESCRIPTIONS[group_name],
            "query_count": len(group_items),
            "items": group_items,
        }
        output_path = output_dir / f"{group_name}.json"
        atomic_write_json(output_path, output)
        outputs[group_name] = str(output_path.resolve())

    manifest = {
        "schema_version": "1.0.0",
        "source_dataset": str(source.resolve()),
        "formal_retriever_goal": "retriever_main",
        "formal_retriever_goal_query_count": len(QUESTION_GROUPS["retriever_main"]),
        "query_overrides": QUERY_OVERRIDES,
        "groups": {
            name: {
                "query_count": len(query_ids),
                "query_ids": query_ids,
                "path": outputs[name],
            }
            for name, query_ids in QUESTION_GROUPS.items()
        },
        "selected_query_count": len(selected_ids),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare approved question groups")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_question_sets(args.source.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
