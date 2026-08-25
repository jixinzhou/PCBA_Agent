from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "rag/src"))
    from pcba_rag.fusion import run_fusion_validation

    result = run_fusion_validation(project_root)
    status = "passed" if result["acceptance_checks_passed"] else "failed"
    print(
        f"T10.9 fusion validation {status}: "
        f"{result['validation_cases']} cases, report={result['report_path']}"
    )
    return 0 if result["acceptance_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
