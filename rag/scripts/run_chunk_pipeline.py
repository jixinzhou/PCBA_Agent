from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T10.4 section-aware chunking with stable chunk IDs"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.chunk_pipeline import run_chunk_pipeline

    summary = run_chunk_pipeline(
        project_root,
        set(args.source_id) if args.source_id else None,
        write_report=not args.no_report,
    )
    print(
        json.dumps(
            {
                "task": summary["task"],
                "total_sources": summary["total_sources"],
                "input_pages": summary["input_pages"],
                "total_chunks": summary["total_chunks"],
                "stable_id_verification_passed": summary[
                    "stable_id_verification_passed"
                ],
                "acceptance_checks_passed": summary["acceptance_checks_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
