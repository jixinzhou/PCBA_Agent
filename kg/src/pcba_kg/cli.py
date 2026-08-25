from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    DEFAULT_ENV_FILE,
    DEFAULT_RUNTIME_CONFIG,
    PROJECT_ROOT,
    build_graph_plan,
    json_output,
    load_runtime_settings,
    plan_summary,
)
from .neo4j_store import Neo4jGraphStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and validate the frozen T09 ontology")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--runs", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    env_file = args.env_file if args.env_file.is_absolute() else PROJECT_ROOT / args.env_file
    settings = load_runtime_settings(config_path=config_path, env_file=env_file)
    plan = build_graph_plan(settings)
    if args.plan_only:
        print(json_output({"success": True, "mode": "plan_only", "plan": plan_summary(plan)}))
        return 0

    validations = []
    with Neo4jGraphStore(settings) as store:
        for run_number in range(1, args.runs + 1):
            store.import_plan(plan)
            validations.append({"run": run_number, **store.validate(plan)})
    fingerprints = [item["graph_fingerprint"] for item in validations]
    idempotent = len(set(fingerprints)) == 1
    if not idempotent:
        raise RuntimeError(f"Graph fingerprint changed across imports: {fingerprints}")
    print(
        json_output(
            {
                "success": True,
                "mode": "neo4j_import_and_validation",
                "runs": args.runs,
                "idempotent": idempotent,
                "plan": plan_summary(plan),
                "validations": validations,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
