from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.retriever import run_retriever_validation

    summary = run_retriever_validation(project_root)
    print(
        json.dumps(
            {
                "task": summary["task"],
                "validation_cases": len(summary["validation_cases"]),
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
