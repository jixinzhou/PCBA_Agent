from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T10.3 full PDF page and block extraction")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard existing T10.3 page outputs for the selected sources and process again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.full_page_pipeline import run_full_page_pipeline

    summary = run_full_page_pipeline(
        project_root,
        set(args.source_id) if args.source_id else None,
        resume=not args.restart,
    )
    print(
        json.dumps(
            {
                "task": summary["task"],
                "total_sources": summary["total_sources"],
                "expected_pages": summary["expected_pages"],
                "processed_pages": summary["processed_pages"],
                "status_counts": summary["status_counts"],
                "no_silent_page_loss": summary["no_silent_page_loss"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
