from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pcba_agent.conversation import (
    ChatMessageInput, ConversationService, ConversationStore, _normalize_observations,
)
from pcba_agent.models import AgentResult


class FakeQwen:
    def __init__(self) -> None:
        self.intent = "diagnose_cause"

    def interpret_message(self, _: str, __: dict[str, Any]) -> dict[str, Any]:
        return {"intent": self.intent, "defect": "shifted_component",
                "observations": {}, "unavailable_inputs": []}

    def answer_follow_up(self, question: str, _: dict[str, Any]) -> str:
        return "复用上一轮诊断回答：" + question


class FakeRunner:
    def __init__(self) -> None:
        self.qwen = FakeQwen()
        self.invocations = 0
        self.resumes = 0
        self.last_request: Any = None
        self.classification = {
            "success": True,
            "data": {
                "predicted_class": {"class_name_en": "shifted_component"},
                "confidence": 0.95,
                "top_k": [{"class_name_en": "shifted_component", "confidence": 0.95}],
                "low_confidence": False,
            },
        }
        self.agent_graph = SimpleNamespace(tools=SimpleNamespace(
            classify=lambda _path, _request_id: self.classification,
        ))

    def invoke(self, request: Any) -> AgentResult:
        self.invocations += 1
        self.last_request = request
        return AgentResult(
            status="completed", request_id=request.request_id, thread_id=request.thread_id,
            defect={"canonical_name": "shifted_component", "display_name_zh": "元件偏移"},
            response_text="首次诊断报告", candidates=[{"relationship_id": "REL-1"}],
            rag_evidence=[{"chunk_id": "CHUNK-1", "text": "回流证据"}],
        )

    def resume(self, thread_id: str, _: Any) -> AgentResult:
        self.resumes += 1
        return AgentResult(status="completed", request_id="R", thread_id=thread_id,
                           response_text="补充数据后完成")


class ConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temp.name) / "conversation.sqlite3")
        self.runner = FakeRunner()
        self.service = ConversationService(self.runner, self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_follow_up_reuses_snapshot_without_reinvoking_diagnosis(self) -> None:
        conversation_id = self.service.create()["conversation_id"]
        first = self.service.send(conversation_id, ChatMessageInput(content="为什么会发生偏移？"))
        self.runner.qwen.intent = "explain_result"
        follow_up = self.service.send(conversation_id, ChatMessageInput(content="为什么支持回流路径？"))
        self.assertEqual("completed", first.status)
        self.assertTrue(follow_up.reused_context)
        self.assertEqual(1, self.runner.invocations)
        self.assertIn("复用上一轮诊断", follow_up.assistant_text)

    def test_conversation_messages_and_snapshot_survive_reload(self) -> None:
        conversation_id = self.service.create()["conversation_id"]
        self.service.send(conversation_id, ChatMessageInput(content="分析元件偏移原因"))
        restored = self.store.get(conversation_id)
        self.assertEqual("completed", restored["state"]["status"])
        self.assertEqual(2, len(restored["messages"]))
        self.assertEqual("首次诊断报告", restored["state"]["result"]["response_text"])

    def test_flat_natural_language_parameters_are_normalized_to_tool_input(self) -> None:
        normalized = _normalize_observations({
            "belt_speed_cm_min": 95,
            "zone_means_c": [150] * 13,
            "component_x_mm": 1, "component_y_mm": 2, "component_volume_mm3": 3,
        })
        self.assertEqual(95, normalized["input"]["belt_speed_cm_min"])
        self.assertEqual("P1", normalized["input"]["points"][0]["point_id"])

    def test_nested_spi_wrapper_is_flattened_and_unknown_fields_are_removed(self) -> None:
        normalized = _normalize_observations({"input": {
            "SPI": {
                "squeegee_pressure_kgf": 8,
                "squeegee_speed_m_s": 37.5,
                "separation_speed_m_s": 2,
                "separation_distance_mm": 0.6,
            },
            "unexpected": "drop-me",
        }})
        self.assertEqual({
            "squeegee_pressure_kgf": 8,
            "squeegee_speed_m_s": 37.5,
            "separation_speed_m_s": 2,
            "separation_distance_mm": 0.6,
        }, normalized["input"])

    def test_improve_parameters_is_deterministically_routed_to_optimization(self) -> None:
        conversation_id = self.service.create()["conversation_id"]
        self.service.send(conversation_id, ChatMessageInput(content="分析少锡原因"))
        self.runner.qwen.intent = "explain_result"
        result = self.service.send(conversation_id, ChatMessageInput(content="我该怎么改进参数？"))
        self.assertEqual("optimize_process", result.intent)
        self.assertFalse(result.reused_context)
        self.assertEqual("diagnose_and_optimize", self.runner.last_request.goal)

    def test_reason_question_after_identification_starts_diagnosis(self) -> None:
        conversation_id = self.service.create()["conversation_id"]
        self.runner.qwen.intent = "identify_defect"
        self.service.send(conversation_id, ChatMessageInput(
            content="识别这个缺陷", image_path="sample.png",
        ))
        self.runner.qwen.intent = "explain_result"
        result = self.service.send(conversation_id, ChatMessageInput(content="为什么会发生偏移？"))
        self.assertFalse(result.reused_context)
        self.assertEqual(1, self.runner.invocations)
        self.assertEqual("diagnosed", self.store.get(conversation_id)["state"]["case_stage"])

    def test_equivalent_image_identification_wording_bypasses_qwen_intent(self) -> None:
        self.runner.qwen.intent = "diagnose_cause"
        for wording in ("这是什么缺陷", "这张图片是什么缺陷", ""):
            with self.subTest(wording=wording):
                conversation_id = self.service.create()["conversation_id"]
                result = self.service.send(conversation_id, ChatMessageInput(
                    content=wording, image_path="sample.png",
                ))
                self.assertEqual("identify_defect", result.intent)
                self.assertEqual(0, self.runner.invocations)
                self.assertIn("元件偏移", result.assistant_text)
                self.assertIn("95.00%", result.assistant_text)

    def test_new_identification_image_clears_previous_case_observations(self) -> None:
        conversation_id = self.service.create()["conversation_id"]
        state = self.store.get(conversation_id)["state"]
        state.update({
            "image_path": "old.png", "defect": "insufficient_solder",
            "observations": {"input": {"squeegee_pressure_kgf": 8}},
            "result": {"status": "completed", "pending_inputs": []},
        })
        self.store.update_state(conversation_id, state)
        self.service.send(conversation_id, ChatMessageInput(
            content="这是什么缺陷", image_path="new.png",
        ))
        restored_state = self.store.get(conversation_id)["state"]
        self.assertNotIn("observations", restored_state)
        self.assertEqual("new.png", restored_state["image_path"])

    def test_normal_image_result_does_not_create_defect_case(self) -> None:
        self.runner.classification = {
            "success": True,
            "data": {
                "predicted_class": {"class_name_en": "normal"},
                "confidence": 0.91, "top_k": [], "low_confidence": False,
            },
        }
        conversation_id = self.service.create()["conversation_id"]
        result = self.service.send(conversation_id, ChatMessageInput(
            content="这是什么缺陷", image_path="normal.png",
        ))
        self.assertIn("正常", result.assistant_text)
        self.assertIsNone(result.result["defect"])
        self.assertEqual("identified_normal", self.store.get(conversation_id)["state"]["case_stage"])


if __name__ == "__main__":
    unittest.main()
