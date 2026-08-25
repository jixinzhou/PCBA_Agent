from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agent/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from pcba_agent import AgentRunner
from pcba_agent.adapters.tools import ToolAdapter


POINT = {
    "point_id": "P1",
    "component_x_mm": 117.7094,
    "component_y_mm": 28.5415,
    "component_volume_mm3": 107,
}

REFLOW_SAMPLES = [
    {
        "sample_id": "domain_min_fast",
        "zone_means_c": [135, 145, 155, 165, 173, 180, 190, 210, 220, 230, 245, 260, 255],
        "belt_speed_cm_min": 95,
    },
    {
        "sample_id": "domain_max_slow",
        "zone_means_c": [145, 155, 165, 173, 180, 190, 200, 220, 230, 240, 255, 270, 265],
        "belt_speed_cm_min": 85,
    },
    {
        "sample_id": "e2e_reference",
        "zone_means_c": [135, 155, 165, 173, 180, 180, 190, 210, 220, 230, 255, 270, 265],
        "belt_speed_cm_min": 95,
    },
]


def compact_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        citation = row.get("citation") or {}
        compact.append({
            "rank": row.get("rank"),
            "chunk_id": row.get("chunk_id"),
            "source_id": citation.get("source_id"),
            "pdf_page_start": citation.get("pdf_page_start"),
            "pdf_page_end": citation.get("pdf_page_end"),
            "rrf_rank": row.get("rrf_rank"),
            "reranker_rank": row.get("reranker_rank"),
            "rank_fusion_score": row.get("rank_fusion_score"),
        })
    return compact


def compact_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "relationship_id": row["relationship_id"],
        "cause": row["candidate_cause"]["canonical_name"],
        "verification_capability": row["verification_capability"],
        "assessment_status": row["assessment_status"],
        "recommendation_status": (row.get("optimization_result") or {}).get("recommendation_status"),
    } for row in rows]


def find_unqualified_sample(tools: ToolAdapter) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scans = []
    selected = None
    for sample in REFLOW_SAMPLES:
        response = tools.invoke("reflow_profile_prediction", {
            "request_id": f"T14-SCAN-{sample['sample_id']}",
            "input": {
                "points": [POINT],
                "zone_means_c": sample["zone_means_c"],
                "belt_speed_cm_min": sample["belt_speed_cm_min"],
            },
            "options": {"return_temperature_curve": False},
        })
        data = response["data"]
        scan = {
            "sample_id": sample["sample_id"],
            "max_pwi": data["overall"]["max_pwi"],
            "qualified": data["overall"]["qualified"],
            "within_training_domain": data["within_training_domain"],
        }
        scans.append(scan)
        if selected is None and not scan["qualified"] and scan["within_training_domain"]:
            selected = sample
    if selected is None:
        raise RuntimeError(f"No in-domain unqualified reflow sample found: {scans}")
    return selected, scans


def main() -> int:
    tools = ToolAdapter(ROOT)
    selected, scans = find_unqualified_sample(tools)
    results: dict[str, Any] = {"reflow_sample_scan": scans}

    with AgentRunner() as runner:
        short_thread = f"t14-short-{uuid4()}"
        short_first = runner.invoke({
            "thread_id": short_thread,
            "user_question": "回流焊后发现焊锡桥连，请帮我诊断可能原因。",
            "goal": "diagnose",
        })
        if short_first.status != "needs_input":
            raise RuntimeError(f"short case should request manual data: {short_first.status}")
        short_snapshot = runner.graph.get_state(runner._config(short_thread)).values
        short_final = runner.resume(short_thread, {
            "unavailable_inputs": short_first.pending_inputs,
        })
        if short_final.status != "completed":
            raise RuntimeError(f"short case did not complete: {short_final.status}")
        results["short_diagnosis"] = {
            "input": "回流焊后发现焊锡桥连，请帮我诊断可能原因。",
            "extracted_defect": short_snapshot.get("defect_name"),
            "defect_source": short_snapshot.get("defect_source"),
            "extracted_observations": short_snapshot.get("observations"),
            "first_status": short_first.status,
            "pending_inputs": short_first.pending_inputs,
            "final_status": short_final.status,
            "candidates": compact_candidates(short_final.candidates),
            "rag_evidence": compact_evidence(short_final.rag_evidence),
            "tool_trace": short_final.tool_trace,
            "degradation_trace": short_final.degradation_trace,
            "errors": short_final.errors,
            "response_text": short_final.response_text,
        }

        optimize_thread = f"t14-opt-{uuid4()}"
        optimize_first = runner.invoke({
            "thread_id": optimize_thread,
            "user_question": (
                "Can an initial component placement offset change during reflow "
                "due to solder self-alignment? Please diagnose and optimize the profile."
            ),
            "goal": "diagnose_and_optimize",
            "observations": {
                "input": {
                    "points": [POINT],
                    "zone_means_c": selected["zone_means_c"],
                    "belt_speed_cm_min": selected["belt_speed_cm_min"],
                },
                "optimization_target": {"mode": "minimize_pwi"},
                "adjustable_parameters": {
                    "zone_indexes": [8, 9, 10, 11, 12, 13],
                    "adjust_belt_speed": True,
                },
            },
        })
        if optimize_first.status != "needs_input":
            raise RuntimeError(f"optimization case should request manual data: {optimize_first.status}")
        optimize_snapshot = runner.graph.get_state(runner._config(optimize_thread)).values
        optimize_final = runner.resume(optimize_thread, {
            "unavailable_inputs": optimize_first.pending_inputs,
        })
        if optimize_final.status != "completed":
            raise RuntimeError(f"optimization case did not complete: {optimize_final.status}")
        reflow = next(
            row for row in optimize_final.candidates
            if row["relationship_id"] == "REL-SHIFTED-COMPONENT-REFLOW"
        )
        validation = reflow["validation_result"]["response"]["data"]
        optimization = reflow["optimization_result"]
        optimization_data = optimization.get("optimization_response", {}).get("data", {})
        revalidation_data = optimization.get("revalidation_response", {}).get("data", {})
        results["shifted_component_optimization"] = {
            "input_question": (
                "Can an initial component placement offset change during reflow "
                "due to solder self-alignment? Please diagnose and optimize the profile."
            ),
            "selected_sample": selected,
            "extracted_defect": optimize_snapshot.get("defect_name"),
            "defect_source": optimize_snapshot.get("defect_source"),
            "first_status": optimize_first.status,
            "pending_inputs": optimize_first.pending_inputs,
            "final_status": optimize_final.status,
            "candidates": compact_candidates(optimize_final.candidates),
            "rag_evidence": compact_evidence(optimize_final.rag_evidence),
            "initial_prediction": {
                "max_pwi": validation["overall"]["max_pwi"],
                "qualified": validation["overall"]["qualified"],
                "within_training_domain": validation["within_training_domain"],
            },
            "optimization": {
                "recommendation_status": optimization.get("recommendation_status"),
                "before": optimization_data.get("before"),
                "recommended_parameters": optimization_data.get("recommended_parameters"),
                "after": optimization_data.get("after"),
                "target_reached": optimization_data.get("target_reached"),
            },
            "revalidation": {
                "max_pwi": (revalidation_data.get("overall") or {}).get("max_pwi"),
                "qualified": (revalidation_data.get("overall") or {}).get("qualified"),
                "within_training_domain": revalidation_data.get("within_training_domain"),
            },
            "tool_trace": optimize_final.tool_trace,
            "degradation_trace": optimize_final.degradation_trace,
            "errors": optimize_final.errors,
            "response_text": optimize_final.response_text,
        }

    output = ROOT / "agent/storage/t14_sample_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "result_path": str(output.relative_to(ROOT)),
        "short_status": results["short_diagnosis"]["final_status"],
        "optimization_status": results["shifted_component_optimization"]["final_status"],
        "recommendation_status": results["shifted_component_optimization"]["optimization"]["recommendation_status"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
