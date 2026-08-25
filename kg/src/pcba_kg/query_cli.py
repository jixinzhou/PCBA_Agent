from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import DEFAULT_ENV_FILE, DEFAULT_RUNTIME_CONFIG, PROJECT_ROOT, json_output
from .query import KGQueryError, query_causal_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query frozen PCBA causal paths from Neo4j without invoking any Tool"
    )
    parser.add_argument("--defect", required=True)
    parser.add_argument("--relationship-id")
    parser.add_argument(
        "--observations-file",
        type=Path,
        help="Optional UTF-8 JSON object containing available case data",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _observations(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observations file must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = query_causal_paths(
            args.defect,
            _observations(args.observations_file),
            args.relationship_id,
            config_path=_resolve(args.config),
            env_file=_resolve(args.env_file),
        )
    except (KGQueryError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json_output({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json_output({"success": True, "result": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
