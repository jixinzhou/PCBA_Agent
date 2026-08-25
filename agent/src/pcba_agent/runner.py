from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .adapters import AgentRAGAdapter, KGAdapter, ToolAdapter
from .config import RuntimeSettings, load_settings
from .graph import PCBAAgentGraph
from .models import AgentRequest, AgentResult, ResumeInput
from .qwen_client import QwenClient


class AgentRunner:
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        *,
        qwen: Any | None = None,
        rag: Any | None = None,
        kg: Any | None = None,
        tools: Any | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        root = self.settings.project_root
        path = checkpoint_path or self.settings.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        serde = JsonPlusSerializer(pickle_fallback=False, allowed_json_modules=())
        self.checkpointer = SqliteSaver(self.connection, serde=serde)
        graph = PCBAAgentGraph(
            settings=self.settings,
            qwen=qwen or QwenClient(self.settings.raw["llm"]),
            rag=rag or AgentRAGAdapter(root, self.settings.raw["rag"]),
            kg=kg or KGAdapter(root),
            tools=tools or ToolAdapter(root),
        )
        self.graph = graph.build(self.checkpointer)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "AgentRunner":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def invoke(self, request: AgentRequest | dict[str, Any]) -> AgentResult:
        parsed = request if isinstance(request, AgentRequest) else AgentRequest.model_validate(request)
        existing = self.graph.get_state(self._config(parsed.thread_id))
        if existing.values:
            raise ValueError(
                f"thread_id={parsed.thread_id} already exists; use resume or choose a new thread_id"
            )
        output = self.graph.invoke(
            {"request": parsed.model_dump(mode="json")}, self._config(parsed.thread_id)
        )
        return self._to_result(output, parsed.thread_id, parsed.request_id)

    def resume(
        self, thread_id: str, resume: ResumeInput | dict[str, Any]
    ) -> AgentResult:
        parsed = resume if isinstance(resume, ResumeInput) else ResumeInput.model_validate(resume)
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        request = snapshot.values.get("request") or {}
        if not request:
            raise ValueError(f"No checkpoint found for thread_id={thread_id}")
        output = self.graph.invoke(Command(resume=parsed.model_dump(mode="json")), config)
        return self._to_result(output, thread_id, request["request_id"])

    @staticmethod
    def _to_result(output: dict[str, Any], thread_id: str, request_id: str) -> AgentResult:
        if output.get("result"):
            return AgentResult.model_validate(output["result"])
        interrupts = output.get("__interrupt__") or []
        if interrupts:
            payload = interrupts[0].value
            return AgentResult(
                status="needs_input", request_id=request_id, thread_id=thread_id,
                pending_inputs=payload.get("missing_inputs", []),
                pending_prompt=payload.get("prompt"), response_text=payload.get("prompt", ""),
                candidates=output.get("candidates", []), rag_evidence=output.get("rag_evidence", []),
                tool_trace=output.get("tool_trace", []),
                degradation_trace=output.get("degradation_trace", []), errors=output.get("errors", []),
            )
        raise RuntimeError("Graph returned neither a result nor an interrupt")
