from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from pcba_agent.config import load_settings
from pcba_agent.runner import AgentRunner


class FakeQwen:
    def extract(self, _: str) -> dict[str, Any]:
        return {"defect": None, "goal": None, "observations": {}}

    def synthesize(self, _: dict[str, Any]) -> str:
        return "结构化诊断已完成。"

    def clarify(self, missing: list[str], _: dict[str, Any]) -> str:
        return "请补充：" + "、".join(missing)


class FakeRAG:
    def __init__(self, *, degraded: bool = False) -> None:
        self.degraded = degraded

    def retrieve(self, _: str) -> dict[str, Any]:
        return {
            "evidence": [{"chunk_id": "C1", "rank": 1}],
            "degraded": self.degraded,
            "stage": "reranker" if self.degraded else None,
            "error": "reranker unavailable" if self.degraded else None,
        }


def candidate(
    rid: str,
    *,
    capability: str,
    validation_tool: str | None,
    missing: list[str],
    optimization_tool: str | None = None,
) -> dict[str, Any]:
    return {
        "relationship_id": rid,
        "candidate_cause": {"canonical_name": rid.lower()},
        "process": {"canonical_name": "reflow" if validation_tool == "reflow_profile_prediction" else "placement"},
        "relation_strength": "conditional",
        "verification_capability": capability,
        "assessment_status": "not_evaluated",
        "validation_action": {
            "action_type": "request_missing_data" if missing else ("invoke_tool" if validation_tool else "manual_inspection"),
            "tool_name": validation_tool,
            "missing_inputs": missing,
        },
        "optimization_action": {"tool_name": optimization_tool},
        "limitations": ["candidate limitation"],
    }


class FakeKG:
    def query(self, defect: str, observations: dict[str, Any]) -> dict[str, Any]:
        manual_path = "manual_observation.placement_offset"
        manual_missing = [] if observations.get("manual_observation", {}).get("placement_offset") else [manual_path]
        candidates = []
        if defect == "short":
            candidates.append(candidate("REL-SHORT-MANUAL", capability="unverified", validation_tool=None, missing=manual_missing))
        elif defect == "shifted_component":
            candidates.append(candidate("REL-SHIFT-MANUAL", capability="unverified", validation_tool=None, missing=manual_missing))
            inp = observations.get("input", {})
            required = ["input.points", "input.zone_means_c", "input.belt_speed_cm_min"]
            missing = [path for path in required if not inp.get(path.split(".")[-1])]
            candidates.append(candidate(
                "REL-SHIFT-REFLOW", capability="tool_supported",
                validation_tool="reflow_profile_prediction", missing=missing,
                optimization_tool="reflow_parameter_optimization",
            ))
        return {
            "defect": {"canonical_name": defect}, "candidates": candidates,
            "warnings": ["候选不是唯一根因。"],
        }


class FakeTools:
    def __init__(self, *, fail_validation: bool = False, revalidation_qualified: bool = True) -> None:
        self.fail_validation = fail_validation
        self.revalidation_qualified = revalidation_qualified
        self.calls: list[str] = []
        self.prediction_count = 0

    def classify(self, image_path: str, request_id: str) -> dict[str, Any]:
        self.calls.append("pcba_defect_classification")
        return {"data": {"predicted_class": {"class_name_en": "shifted component"}}}

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        if name == "reflow_profile_prediction":
            self.prediction_count += 1
            if self.fail_validation and self.prediction_count == 1:
                raise RuntimeError("prediction down")
            qualified = False if self.prediction_count == 1 else self.revalidation_qualified
            return {"data": {"overall": {"qualified": qualified}}}
        if name == "reflow_parameter_optimization":
            return {"data": {"recommended_parameters": {
                "zone_means_c": [150] * 13, "belt_speed_cm_min": 80,
            }, "target_reached": True}}
        raise AssertionError(name)


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "checkpoints.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runner(self, tools: FakeTools | None = None, rag: FakeRAG | None = None) -> AgentRunner:
        return AgentRunner(
            load_settings(), qwen=FakeQwen(), rag=rag or FakeRAG(), kg=FakeKG(),
            tools=tools or FakeTools(), checkpoint_path=self.db,
        )

    def test_missing_input_interrupt_and_resume(self) -> None:
        with self.runner() as runner:
            first = runner.invoke({"thread_id": "short-1", "provided_defect": "short"})
            self.assertEqual("needs_input", first.status)
            self.assertIn("manual_observation.placement_offset", first.pending_inputs)
            final = runner.resume("short-1", {"unavailable_inputs": first.pending_inputs})
        self.assertEqual("completed", final.status)
        self.assertEqual("unverified", final.candidates[0]["verification_capability"])
        self.assertEqual("not_evaluated", final.candidates[0]["assessment_status"])

    def test_image_only_classifies_then_resumes_missing_data(self) -> None:
        with self.runner() as runner:
            first = runner.invoke({"thread_id": "image-1", "image_path": "fake.png"})
        self.assertEqual("needs_input", first.status)
        self.assertIn("input.points", first.pending_inputs)

    def test_reranker_degradation_is_explicit(self) -> None:
        with self.runner(rag=FakeRAG(degraded=True)) as runner:
            first = runner.invoke({"thread_id": "fallback-1", "provided_defect": "short"})
        self.assertEqual("needs_input", first.status)
        self.assertEqual("rrf_top5", first.degradation_trace[0]["fallback"])

    def test_tool_failure_isolated_as_inconclusive(self) -> None:
        tools = FakeTools(fail_validation=True)
        observations = {"input": reflow_input()}
        with self.runner(tools=tools) as runner:
            first = runner.invoke({
                "thread_id": "tool-fail", "provided_defect": "shifted_component",
                "observations": observations,
            })
            final = runner.resume("tool-fail", {"unavailable_inputs": ["manual_observation.placement_offset"]})
        reflow = next(item for item in final.candidates if item["relationship_id"] == "REL-SHIFT-REFLOW")
        self.assertEqual("inconclusive", reflow["assessment_status"])
        self.assertTrue(final.errors)

    def test_optimization_requires_revalidation_and_accepts_only_qualified(self) -> None:
        tools = FakeTools(revalidation_qualified=True)
        with self.runner(tools=tools) as runner:
            first = runner.invoke({
                "thread_id": "opt-ok", "provided_defect": "shifted_component",
                "goal": "diagnose_and_optimize", "observations": {"input": reflow_input()},
            })
            final = runner.resume("opt-ok", {
                "unavailable_inputs": ["manual_observation.placement_offset"],
                "observations": {
                    "optimization_target": {"mode": "minimize_pwi"},
                    "adjustable_parameters": {"zone_indexes": [10, 11], "adjust_belt_speed": True},
                },
            })
        reflow = next(item for item in final.candidates if item["relationship_id"] == "REL-SHIFT-REFLOW")
        self.assertEqual("accepted", reflow["optimization_result"]["recommendation_status"])
        self.assertEqual(2, tools.calls.count("reflow_profile_prediction"))
        self.assertIn("zone_means_c=[150, 150, 150", final.response_text)
        self.assertIn("belt_speed_cm_min=80", final.response_text)
        self.assertIn("本次未写入设备", final.response_text)

    def test_failed_revalidation_rejects_recommendation(self) -> None:
        tools = FakeTools(revalidation_qualified=False)
        with self.runner(tools=tools) as runner:
            runner.invoke({
                "thread_id": "opt-reject", "provided_defect": "shifted_component",
                "goal": "diagnose_and_optimize", "observations": {"input": reflow_input()},
            })
            final = runner.resume("opt-reject", {
                "unavailable_inputs": ["manual_observation.placement_offset"],
                "observations": {
                    "optimization_target": {"mode": "minimize_pwi"},
                    "adjustable_parameters": {"zone_indexes": [10], "adjust_belt_speed": False},
                },
            })
        reflow = next(item for item in final.candidates if item["relationship_id"] == "REL-SHIFT-REFLOW")
        self.assertEqual("rejected", reflow["optimization_result"]["recommendation_status"])

    def test_unknown_defect_can_be_refused_without_tool_call(self) -> None:
        tools = FakeTools()
        with self.runner(tools=tools) as runner:
            first = runner.invoke({"thread_id": "unknown-1", "user_question": "这是什么问题？"})
            final = runner.resume("unknown-1", {"unavailable_inputs": ["provided_defect"]})
        self.assertEqual("completed", final.status)
        self.assertIsNone(final.defect)
        self.assertEqual([], tools.calls)

    def test_existing_thread_rejects_new_initial_request(self) -> None:
        with self.runner() as runner:
            runner.invoke({"thread_id": "duplicate", "provided_defect": "short"})
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.invoke({"thread_id": "duplicate", "provided_defect": "short"})


def reflow_input() -> dict[str, Any]:
    return {
        "points": [{"point_id": "P1", "component_x_mm": 1, "component_y_mm": 2, "component_volume_mm3": 3}],
        "zone_means_c": [150] * 13,
        "belt_speed_cm_min": 80,
    }


if __name__ == "__main__":
    unittest.main()
