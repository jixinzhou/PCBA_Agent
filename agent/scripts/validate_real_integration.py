from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agent/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from pcba_agent import AgentRunner


def main() -> int:
    thread_id = f"t13-real-{uuid4()}"
    observations = {
        "input": {
            "points": [{
                "point_id": "P1", "component_x_mm": 117.729,
                "component_y_mm": 77.3908, "component_volume_mm3": 107,
            }],
            "zone_means_c": [135, 155, 165, 173, 180, 180, 190, 210, 220, 230, 255, 270, 265],
            "belt_speed_cm_min": 95,
        }
    }
    with AgentRunner() as runner:
        first = runner.invoke({
            "thread_id": thread_id,
            "user_question": "Can an initial component placement offset change during reflow due to solder self-alignment?",
            "provided_defect": "shifted_component",
            "goal": "diagnose",
            "observations": observations,
        })
        if first.status != "needs_input":
            raise RuntimeError(f"Expected needs_input, got {first.status}")
        final = runner.resume(thread_id, {
            "unavailable_inputs": first.pending_inputs
        })
        if final.status != "completed":
            raise RuntimeError(
                f"Expected completed after declaring unavailable inputs, got {final.status}: "
                f"{final.pending_inputs}"
            )
    reflow = next(
        item for item in final.candidates
        if item["relationship_id"] == "REL-SHIFTED-COMPONENT-REFLOW"
    )
    summary = {
        "first_status": first.status,
        "pending_inputs": first.pending_inputs,
        "final_status": final.status,
        "candidate_count": len(final.candidates),
        "reflow_assessment": reflow["assessment_status"],
        "tool_calls": final.tool_trace,
        "rag_degraded": any(
            item.get("stage") in {"retriever", "reranker"}
            for item in final.degradation_trace
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
