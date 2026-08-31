from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx


class QwenUnavailableError(RuntimeError):
    pass


class QwenClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.api_key = os.getenv(config["api_key_env"], "")
        self.base_url = os.getenv(config["base_url_env"], "").rstrip("/")
        self.model = os.getenv("QWEN_MODEL", config["model"])
        self.timeout = float(os.getenv(config["timeout_seconds_env"], "120"))
        self.max_retries = int(os.getenv(config["max_retries_env"], "2"))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        if not self.available:
            raise QwenUnavailableError("Qwen API environment is not configured")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "enable_thinking": bool(self.config.get("enable_thinking", False)),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                    trust_env=False,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(0.25 * (2**attempt))
        raise QwenUnavailableError(f"Qwen structured call failed: {last}")

    def extract(self, text: str) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "defect": {"type": ["string", "null"], "enum": [
                    "insufficient_solder", "excessive_solder", "short",
                    "shifted_component", None]},
                "goal": {"type": ["string", "null"], "enum": [
                    "diagnose", "diagnose_and_optimize", None]},
                "observations": {"type": "object"},
            },
            "required": ["defect", "goal", "observations"],
            "additionalProperties": False,
        }
        return self.structured(
            system="只抽取用户明确提供的信息，不推测工艺数值，不决定Tool调用。",
            user=text,
            schema_name="pcba_request_extraction",
            schema=schema,
            temperature=float(self.config["extraction_temperature"]),
        )

    def interpret_message(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        intents = [
            "identify_defect", "diagnose_cause", "optimize_process",
            "explain_result", "explain_evidence", "provide_data", "new_case",
        ]
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": intents},
                "defect": {"type": ["string", "null"], "enum": [
                    "insufficient_solder", "excessive_solder", "short",
                    "shifted_component", None,
                ]},
                "observations": {"type": "object"},
                "unavailable_inputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["intent", "defect", "observations", "unavailable_inputs"],
            "additionalProperties": False,
        }
        return self.structured(
            system=(
                "理解PCBA聊天消息的语义并输出结构化意图。只提取用户明确提供的缺陷、"
                "工艺参数和不可用字段，不推测任何数值，不选择Tool。若当前存在pending_inputs，"
                "用户是在补数据或说明无法提供时使用provide_data。普通追问使用explain_result，"
                "明确询问证据使用explain_evidence；优化、改进参数、调整参数或调参均使用"
                "optimize_process。工艺参数只放在observations.input中，不增加SPI、Reflow等包装层：SPI字段为"
                "squeegee_pressure_kgf、squeegee_speed_m_s、separation_speed_m_s、"
                "separation_distance_mm；回流字段为zone_means_c、belt_speed_cm_min、points，"
                "固定测点points包含point_id=P1、component_x_mm、component_y_mm、component_volume_mm3。"
            ),
            user=json.dumps({"message": text, "context": context}, ensure_ascii=False),
            schema_name="pcba_conversation_intent",
            schema=schema,
            temperature=float(self.config["extraction_temperature"]),
        )

    def answer_follow_up(self, question: str, context: dict[str, Any]) -> str:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        result = self.structured(
            system=(
                "基于上一轮PCBA诊断快照回答中文追问。必须复用给定候选状态、Tool事实和RAG证据，"
                "不得虚构新数值或声称重新执行了诊断；引用证据时注明来源或chunk_id。"
                "候选路径不能描述为已确认的唯一根因。使用纯文本回答，不使用Markdown标记。"
            ),
            user=json.dumps({"question": question, "context": context}, ensure_ascii=False),
            schema_name="pcba_conversation_follow_up",
            schema=schema,
            temperature=float(self.config["generation_temperature"]),
        )
        return str(result["answer"])

    def synthesize(self, context: dict[str, Any]) -> str:
        schema = {
            "type": "object",
            "properties": {
                "diagnosis_conclusion": {"type": "string"},
                "candidate_analysis": {"type": "string"},
                "evidence_basis": {"type": "string"},
                "recommendations": {"type": "string"},
                "limitations": {"type": "string"},
            },
            "required": [
                "diagnosis_conclusion", "candidate_analysis", "evidence_basis",
                "recommendations", "limitations",
            ],
            "additionalProperties": False,
        }
        result = self.structured(
            system=(
                "你是PCBA跨工序质量诊断工程师。请把候选致因状态、当前工艺Tool结论和"
                "RAG知识证据综合成一份简洁、专业的中文诊断报告，而不是罗列检索片段。"
                "请分别填写诊断结论、候选原因分析、证据依据、处置建议、限制说明五个字段。"
                "引用知识证据时只能使用输入中提供的证据编号，并写成[证据1]形式；知识证据"
                "只能支持一般机理，不能替代当前样本的Tool验证。不得改变候选状态、不得补造"
                "阈值、不得宣称唯一根因。对致因必须使用‘候选’或‘获得支持的候选路径’，严禁"
                "使用‘主要致因’、‘首要致因’、‘根因已确认’或‘确认为某致因’等确定性措辞。"
                "不得输出任何具体数值、温区编号、参数值或参数变化；"
                "精确工艺事实与推荐参数将由程序另行附加。全文必须使用中文。"
            ),
            user=json.dumps(context, ensure_ascii=False),
            schema_name="pcba_diagnostic_summary",
            schema=schema,
            temperature=float(self.config["generation_temperature"]),
        )
        sections = (
            ("诊断结论", result["diagnosis_conclusion"]),
            ("候选原因分析", result["candidate_analysis"]),
            ("证据依据", result["evidence_basis"]),
            ("处置建议", result["recommendations"]),
            ("限制说明", result["limitations"]),
        )
        report = "\n\n".join(f"{title}\n{str(content).strip()}" for title, content in sections)
        replacements = {
            "主要致因确认为": "当前获得较强支持的候选路径为",
            "主要致因是": "当前获得较强支持的候选路径是",
            "首要致因": "优先验证的候选路径",
            "根因已确认": "候选路径已获得支持",
            "确认为根因": "作为候选路径获得支持",
        }
        for unsafe, safe in replacements.items():
            report = report.replace(unsafe, safe)
        return report

    def clarify(self, missing_inputs: list[str], context: dict[str, Any]) -> str:
        schema = {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
            "additionalProperties": False,
        }
        result = self.structured(
            system=(
                "把程序给出的缺失字段改写为一句简洁补问。不得增删字段、不得推测数值，"
                "并提示用户无法提供时可明确说明不可用。"
            ),
            user=json.dumps(
                {"missing_inputs": missing_inputs, "context": context}, ensure_ascii=False
            ),
            schema_name="pcba_clarification",
            schema=schema,
            temperature=float(self.config["generation_temperature"]),
        )
        return str(result["prompt"])
