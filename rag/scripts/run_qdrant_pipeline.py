from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the T10.7 Qdrant V0.1 index"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Explicitly delete and recreate the derived Collection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.qdrant_pipeline import run_qdrant_pipeline

    summary = run_qdrant_pipeline(project_root, recreate=args.recreate)
    print(
        json.dumps(
            {
                "task": summary["task"],
                "collection_name": summary["collection_name"],
                "input_records": summary["input_records"],
                "exact_point_count": summary["validation"]["exact_point_count"],
                "collection_recreated": summary["collection_recreated"],
                "acceptance_checks_passed": summary[
                    "acceptance_checks_passed"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
