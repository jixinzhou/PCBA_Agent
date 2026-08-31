from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import Field

from .models import AgentRequest, AgentResult, ResumeInput, StrictModel
from .policies import deep_merge, infer_defect_from_text
from .qwen_client import QwenUnavailableError


ChatIntent = Literal[
    "identify_defect", "diagnose_cause", "optimize_process",
    "explain_result", "explain_evidence", "provide_data", "new_case",
]


class ChatMessageInput(StrictModel):
    content: str = ""
    image_path: str | None = None


class ChatTurnResult(StrictModel):
    conversation_id: str
    message_id: str
    intent: ChatIntent
    status: Literal["completed", "needs_input", "failed"]
    assistant_text: str
    result: dict[str, Any] | None = None
    reused_context: bool = False
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
            """)

    def close(self) -> None:
        self.connection.close()

    def create(self) -> dict[str, Any]:
        conversation_id = str(uuid4())
        now = _now()
        state = {"case_version": 0, "status": "empty"}
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (conversation_id, now, now, json.dumps(state, ensure_ascii=False)),
            )
        return {"conversation_id": conversation_id, "created_at": now, "state": state, "messages": []}

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                return None
            messages = self.connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        return {
            "conversation_id": conversation_id,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "state": json.loads(row["state_json"]),
            "messages": [{
                "message_id": item["message_id"], "role": item["role"],
                "content": item["content"], "created_at": item["created_at"],
                "metadata": json.loads(item["metadata_json"]),
            } for item in messages],
        }

    def update_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE conversations SET updated_at = ?, state_json = ? WHERE conversation_id = ?",
                (_now(), json.dumps(state, ensure_ascii=False), conversation_id),
            )

    def add_message(
        self, conversation_id: str, role: str, content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid4())
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, _now(),
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )
        return message_id


class ConversationService:
    def __init__(self, runner: Any, store: ConversationStore) -> None:
        self.runner = runner
        self.store = store
        self.qwen = runner.qwen

    def create(self) -> dict[str, Any]:
        return self.store.create()

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self.store.get(conversation_id)

    def _interpret(self, content: str, state: dict[str, Any], has_image: bool) -> dict[str, Any]:
        pending = (state.get("result") or {}).get("pending_inputs") or []
        if has_image and (not content.strip() or _identification_only(content)):
            return {
                "intent": "identify_defect", "defect": None,
                "observations": {}, "unavailable_inputs": [],
            }
        if pending:
            fallback_intent: ChatIntent = "provide_data"
        elif not state.get("result"):
            fallback_intent = "diagnose_cause" if not _identification_only(content) else "identify_defect"
        elif _new_case(content, has_image):
            fallback_intent = "new_case"
        elif _optimize(content):
            fallback_intent = "optimize_process"
        else:
            fallback_intent = "explain_evidence" if "证据" in content else "explain_result"
        try:
            parsed = self.qwen.interpret_message(content, {
                "has_image": has_image,
                "has_active_diagnosis": bool(state.get("result")),
                "pending_inputs": pending,
                "current_defect": state.get("defect"),
                "fallback_intent": fallback_intent,
            })
            if pending:
                parsed["intent"] = "provide_data"
            elif _optimize(content):
                # Optimization is a state-changing action. Keep common user wording
                # deterministic even when the LLM classifies it as a plain follow-up.
                parsed["intent"] = "optimize_process"
            elif state.get("case_stage") == "identified" and any(
                term in content.lower() for term in ("原因", "为什么", "怎么导致", "cause")
            ):
                parsed["intent"] = "diagnose_cause"
            parsed["observations"] = _normalize_observations(parsed.get("observations") or {})
            return parsed
        except (QwenUnavailableError, AttributeError):
            return {
                "intent": fallback_intent,
                "defect": infer_defect_from_text(content),
                "observations": {},
                "unavailable_inputs": pending if _unavailable(content) else [],
            }

    def send(
        self, conversation_id: str, message: ChatMessageInput,
        progress: Callable[[str, str], None] | None = None,
    ) -> ChatTurnResult:
        emit = progress or (lambda _stage, _status: None)
        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise ValueError("conversation not found")
        content = message.content.strip()
        if not content and not message.image_path:
            raise ValueError("message content or image is required")
        self.store.add_message(conversation_id, "user", content, {"image_path": message.image_path})
        state = conversation["state"]
        emit("理解用户意图与提取参数", "running")
        parsed = self._interpret(content, state, bool(message.image_path))
        intent: ChatIntent = parsed["intent"]
        trace = [{"stage": "intent_understanding", "status": "completed", "intent": intent}]
        emit("理解用户意图与提取参数", "completed")

        if intent in ("explain_result", "explain_evidence") and state.get("result"):
            emit("复用当前诊断快照与证据", "running")
            text = self._follow_up(content, state, conversation["messages"])
            emit("复用当前诊断快照与证据", "completed")
            message_id = self.store.add_message(conversation_id, "assistant", text, {
                "intent": intent, "reused_context": True,
                "diagnosis_thread_id": state.get("diagnosis_thread_id"),
                "evidence_ids": [row.get("chunk_id") for row in (state["result"].get("rag_evidence") or [])],
            })
            return ChatTurnResult(
                conversation_id=conversation_id, message_id=message_id, intent=intent,
                status="completed", assistant_text=text, result=state.get("result"),
                reused_context=True, execution_trace=trace + [{"stage": "context_reuse", "status": "completed"}],
            )

        if intent == "new_case":
            state = {"case_version": int(state.get("case_version", 0)) + 1, "status": "empty"}

        if intent == "provide_data" and state.get("diagnosis_thread_id"):
            observations = parsed.get("observations") or {}
            unavailable = parsed.get("unavailable_inputs") or []
            if _unavailable(content) and not unavailable:
                unavailable = (state.get("result") or {}).get("pending_inputs") or []
            emit("使用补充数据恢复诊断", "running")
            result = self.runner.resume(state["diagnosis_thread_id"], ResumeInput(
                observations=observations, unavailable_inputs=unavailable, user_message=content,
            ))
            emit("使用补充数据恢复诊断", "completed")
            state["observations"] = deep_merge(state.get("observations", {}), observations)
            trace.append({"stage": "diagnosis_resume", "status": "completed"})
            return self._save_result(conversation_id, intent, result, state, trace)

        defect = parsed.get("defect") or state.get("defect") or infer_defect_from_text(content)
        observations = _normalize_observations(deep_merge(
            state.get("observations", {}), parsed.get("observations") or {},
        ))
        if intent == "optimize_process":
            observations.setdefault("optimization_target", {"mode": "minimize_pwi"})
            observations.setdefault("adjustable_parameters", {
                "zone_indexes": [8, 9, 10, 11, 12, 13], "adjust_belt_speed": True,
            })
        image_path = message.image_path or state.get("image_path")
        if intent == "identify_defect" and image_path:
            if state.get("image_path") and state.get("image_path") != image_path:
                state = {
                    "case_version": int(state.get("case_version", 0)),
                    "status": "empty",
                }
            emit("调用AOI缺陷识别", "running")
            identified = self._identify(conversation_id, content, image_path, state, trace)
            emit("调用AOI缺陷识别", "completed")
            return identified

        thread_id = str(uuid4())
        goal = "diagnose_and_optimize" if intent == "optimize_process" else "diagnose"
        request = AgentRequest(
            thread_id=thread_id, user_question=content or None,
            image_path=image_path if not state.get("result") else None,
            provided_defect=defect, goal=goal, observations=observations,
            response_language="zh",
        )
        emit("执行RAG、KG与必要工艺Tool", "running")
        result = self.runner.invoke(request)
        emit("执行RAG、KG与必要工艺Tool", "completed")
        state.update({
            "case_version": int(state.get("case_version", 0)) + (1 if state.get("status") == "empty" else 0),
            "diagnosis_thread_id": thread_id, "defect": defect,
            "image_path": image_path, "observations": observations,
        })
        trace.append({"stage": "diagnosis_execution", "status": "completed"})
        return self._save_result(conversation_id, intent, result, state, trace)

    def _identify(
        self, conversation_id: str, content: str, image_path: str,
        state: dict[str, Any], trace: list[dict[str, Any]],
    ) -> ChatTurnResult:
        request_id = str(uuid4())
        try:
            response = self.runner.agent_graph.tools.classify(image_path, request_id)
        except Exception as exc:
            result = AgentResult(
                status="failed", request_id=request_id, thread_id=str(uuid4()),
                response_text="图片识别服务暂时不可用，请稍后重试。",
                tool_trace=[{
                    "phase": "defect_classification",
                    "tool_name": "pcba_defect_classification",
                    "success": False, "error": str(exc),
                }],
                errors=[{"stage": "aoi_classification", "error": str(exc)}],
            )
            state.update({"case_stage": "identification_failed", "image_path": image_path})
            trace.append({"stage": "defect_classification", "status": "failed"})
            return self._save_result(conversation_id, "identify_defect", result, state, trace)
        data = response.get("data") or {}
        predicted = data.get("predicted_class") or {}
        defect = infer_defect_from_text(predicted.get("class_name_en")) or predicted.get("class_name_en")
        confidence = data.get("confidence")
        if response.get("success") is not True or not defect:
            error = response.get("error") or "AOI未返回有效类别"
            result = AgentResult(
                status="failed", request_id=request_id, thread_id=str(uuid4()),
                response_text="图片识别没有得到有效结果，请确认图片后重试。",
                tool_trace=[{
                    "phase": "defect_classification",
                    "tool_name": "pcba_defect_classification",
                    "success": False, "error": str(error),
                }],
                errors=[{"stage": "aoi_classification", "error": str(error)}],
            )
            state.update({"case_stage": "identification_failed", "image_path": image_path})
            trace.append({"stage": "defect_classification", "status": "failed"})
            return self._save_result(conversation_id, "identify_defect", result, state, trace)
        text = f"图片识别结果为{_defect_zh(defect)}"
        if confidence is not None:
            text += f"，置信度为{float(confidence):.2%}"
        text += "。"
        top_k = data.get("top_k") or []
        if top_k:
            ranked = "、".join(
                f"{_defect_zh(item.get('class_name_en'))} {float(item.get('confidence', 0)):.2%}"
                for item in top_k[:3]
            )
            text += f"候选排序：{ranked}。"
        if data.get("low_confidence") is True:
            text += "当前置信度较低，建议人工复核。"
        elif defect == "normal":
            text += "当前未识别到冻结范围内的缺陷；如现场仍有异常，建议人工复核。"
        else:
            text += "如果需要分析原因或优化工艺，可以直接继续追问。"
        defect_payload = None if defect == "normal" else {
            "canonical_name": defect, "display_name_zh": _defect_zh(defect),
        }
        result = AgentResult(
            status="completed", request_id=request_id, thread_id=str(uuid4()),
            defect=defect_payload,
            response_text=text,
            tool_trace=[{
                "phase": "defect_classification",
                "tool_name": "pcba_defect_classification", "success": True,
                "predicted_defect": defect, "confidence": confidence,
            }],
        )
        state.update({"case_version": int(state.get("case_version", 0)) + 1,
                      "defect": None if defect == "normal" else defect,
                      "image_path": image_path,
                      "case_stage": "identified_normal" if defect == "normal" else "identified"})
        trace.append({"stage": "defect_classification", "status": "completed"})
        return self._save_result(conversation_id, "identify_defect", result, state, trace)

    def _follow_up(self, content: str, state: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        result = state["result"]
        try:
            return self.qwen.answer_follow_up(content, {
                "current_defect": state.get("defect"),
                "diagnosis_report": result.get("response_text"),
                "candidates": result.get("candidates") or [],
                "rag_evidence": result.get("rag_evidence") or [],
                "recent_messages": [{"role": row["role"], "content": row["content"]} for row in messages[-6:]],
            })
        except (QwenUnavailableError, AttributeError):
            return "当前追问已关联到上一轮诊断，但Qwen暂时不可用。上一轮结论如下：\n\n" + result.get("response_text", "")

    def _save_result(
        self, conversation_id: str, intent: ChatIntent, result: AgentResult,
        state: dict[str, Any], trace: list[dict[str, Any]],
    ) -> ChatTurnResult:
        payload = result.model_dump(mode="json")
        state.update({"result": payload, "status": result.status,
                      "defect": ((result.defect or {}).get("canonical_name") or state.get("defect"))})
        if intent != "identify_defect":
            state["case_stage"] = "diagnosed" if result.status == "completed" else result.status
        self.store.update_state(conversation_id, state)
        message_id = self.store.add_message(conversation_id, "assistant", result.response_text, {
            "intent": intent, "result": payload,
            "diagnosis_thread_id": state.get("diagnosis_thread_id"),
        })
        return ChatTurnResult(
            conversation_id=conversation_id, message_id=message_id, intent=intent,
            status=result.status, assistant_text=result.response_text,
            result=payload, reused_context=False, execution_trace=trace,
        )


def _identification_only(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in (
        "什么缺陷", "识别缺陷", "识别这个缺陷", "识别一下", "判断缺陷",
        "是什么缺陷", "identify defect",
    )) and not any(
        term in lowered for term in ("原因", "为什么", "优化", "调整")
    )


def _optimize(text: str) -> bool:
    return any(term in text.lower() for term in (
        "优化", "调整参数", "推荐参数", "改进参数", "参数改进",
        "改善参数", "调参", "怎么改进", "如何改进", "optimize",
    ))


def _new_case(text: str, has_image: bool) -> bool:
    return has_image and any(term in text.lower() for term in ("新", "另一", "重新", "这个"))


def _unavailable(text: str) -> bool:
    return any(term in text for term in ("无法提供", "没有数据", "暂时没有", "不知道", "不可用"))


def _defect_zh(defect: str | None) -> str:
    return {
        "insufficient_solder": "少锡", "excessive_solder": "多锡",
        "short": "短路/桥连", "shifted_component": "元件偏移",
        "normal": "正常",
    }.get(defect or "", defect or "未知缺陷")


def _normalize_observations(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    input_data: dict[str, Any] = {}
    known_input_fields = (
        "squeegee_pressure_kgf", "squeegee_speed_m_s",
        "separation_speed_m_s", "separation_distance_mm",
        "zone_means_c", "belt_speed_cm_min", "points",
    )
    point_fields = ("component_x_mm", "component_y_mm", "component_volume_mm3")
    point_data: dict[str, Any] = {}
    sources = [raw]
    for container_name in ("input", "process_params"):
        container = raw.get(container_name)
        if isinstance(container, dict):
            sources.append(container)
            for wrapper_name in ("SPI", "spi", "Reflow", "reflow"):
                wrapper = container.get(wrapper_name)
                if isinstance(wrapper, dict):
                    sources.append(wrapper)
    for source in sources:
        for field in known_input_fields:
            if source.get(field) is not None:
                input_data[field] = source[field]
        for field in point_fields:
            if source.get(field) is not None:
                point_data[field] = source[field]
    if all(field in point_data for field in point_fields) and not input_data.get("points"):
        input_data["points"] = [{
            "point_id": "P1", **point_data,
        }]
    if input_data:
        normalized["input"] = input_data
    for field in ("manual_observation", "optimization_target", "adjustable_parameters"):
        if isinstance(raw.get(field), dict):
            normalized[field] = raw[field]
    return normalized
