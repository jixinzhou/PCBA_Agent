from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T10.2 PDF/OCR representative-page validation")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--source-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.sample_pipeline import run_sample_pipeline

    summary = run_sample_pipeline(
        project_root,
        set(args.source_id) if args.source_id else None,
    )
    print(
        json.dumps(
            {
                "task": summary["task"],
                "total_sources": summary["total_sources"],
                "total_samples": summary["total_samples"],
                "all_schema_valid": summary["all_schema_valid"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
