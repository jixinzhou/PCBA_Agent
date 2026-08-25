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

    def synthesize(self, context: dict[str, Any]) -> str:
        schema = {
            "type": "object",
            "properties": {"response_text": {"type": "string"}},
            "required": ["response_text"],
            "additionalProperties": False,
        }
        result = self.structured(
            system=(
                "根据给定结构化状态生成简洁PCBA诊断概括。不得改变候选状态、"
                "不得补造阈值、不得宣称唯一根因。不得输出任何具体数值、温区编号、"
                "参数值或参数变化；精确事实将由程序另行生成。"
            ),
            user=json.dumps(context, ensure_ascii=False),
            schema_name="pcba_diagnostic_summary",
            schema=schema,
            temperature=float(self.config["generation_temperature"]),
        )
        return str(result["response_text"])

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
