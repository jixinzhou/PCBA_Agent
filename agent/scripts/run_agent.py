from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agent/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from pcba_agent import AgentRequest, AgentRunner, ResumeInput


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume the PCBA LangGraph agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", help="AgentRequest JSON file")
    group.add_argument("--resume", help="ResumeInput JSON file")
    parser.add_argument("--thread-id", help="Required with --resume")
    args = parser.parse_args()
    with AgentRunner() as runner:
        if args.request:
            payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
            result = runner.invoke(AgentRequest.model_validate(payload))
        else:
            if not args.thread_id:
                parser.error("--thread-id is required with --resume")
            payload = json.loads(Path(args.resume).read_text(encoding="utf-8"))
            result = runner.resume(args.thread_id, ResumeInput.model_validate(payload))
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
