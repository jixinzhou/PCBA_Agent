from __future__ import annotations

import os
import json
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import AgentRequest, AgentResult, ResumeInput
from .runner import AgentRunner
from .conversation import ChatMessageInput, ChatTurnResult, ConversationService, ConversationStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_ROOT = PROJECT_ROOT / "agent/web"
DEFAULT_UPLOAD_ROOT = PROJECT_ROOT / "agent/storage/uploads"
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_CONVERSATION_PATH = PROJECT_ROOT / "agent/storage/conversations.sqlite3"


class ConversationJobManager:
    def __init__(self, service: ConversationService, runner_lock: threading.RLock) -> None:
        self.service = service
        self.runner_lock = runner_lock
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def submit(self, conversation_id: str, message: ChatMessageInput) -> str:
        job_id = uuid4().hex
        with self.lock:
            self.jobs[job_id] = {"status": "running", "progress": [], "result": None, "error": None}

        def update(stage: str, status_value: str) -> None:
            with self.lock:
                rows = self.jobs[job_id]["progress"]
                existing = next((row for row in rows if row["stage"] == stage), None)
                if existing:
                    existing["status"] = status_value
                else:
                    rows.append({"stage": stage, "status": status_value})

        def run() -> None:
            try:
                with self.runner_lock:
                    result = self.service.send(conversation_id, message, progress=update)
                with self.lock:
                    self.jobs[job_id].update({"status": "completed", "result": result.model_dump(mode="json")})
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

        threading.Thread(target=run, name=f"pcba-chat-{job_id[:8]}", daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else json.loads(json.dumps(job, ensure_ascii=False))


def _image_suffix(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def create_app(
    *,
    runner_factory: Callable[[], Any] | None = None,
    upload_root: Path | None = None,
    conversation_path: Path | None = None,
) -> FastAPI:
    make_runner = runner_factory or AgentRunner
    uploads = (upload_root or DEFAULT_UPLOAD_ROOT).resolve()
    max_upload_bytes = int(
        os.getenv("PCBA_WEB_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        uploads.mkdir(parents=True, exist_ok=True)
        runner = make_runner()
        conversation_store = ConversationStore(conversation_path or DEFAULT_CONVERSATION_PATH)
        application.state.runner = runner
        application.state.runner_lock = threading.RLock()
        application.state.conversation_store = conversation_store
        application.state.conversation_service = (
            ConversationService(runner, conversation_store)
            if hasattr(runner, "qwen") and hasattr(runner, "agent_graph") else None
        )
        application.state.conversation_jobs = (
            ConversationJobManager(
                application.state.conversation_service, application.state.runner_lock
            ) if application.state.conversation_service is not None else None
        )
        application.state.rag_prewarm = "not_available"
        if os.getenv("PCBA_WEB_PREWARM_RAG", "1") == "1" and hasattr(runner, "agent_graph"):
            rag = getattr(runner.agent_graph, "rag", None)
            try:
                if hasattr(rag, "_load_query_encoder"):
                    rag._load_query_encoder()
                if hasattr(rag, "_load_reranker"):
                    rag._load_reranker()
                application.state.rag_prewarm = "ready"
            except Exception as exc:
                application.state.rag_prewarm = f"degraded:{type(exc).__name__}"
        try:
            yield
        finally:
            close = getattr(runner, "close", None)
            if callable(close):
                close()
            conversation_store.close()

    application = FastAPI(
        title="PCBA Quality Diagnosis Agent",
        description="Local web API for the deterministic PCBA LangGraph diagnosis workflow.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @application.get("/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "success": True,
            "data": {
                "status": "ready",
                "service": "pcba-agent-web",
                "api_version": "v1",
                "rag_prewarm": application.state.rag_prewarm,
            },
        }

    @application.post("/api/v1/agent/images", tags=["agent"])
    async def upload_image(image: UploadFile = File(...)) -> dict[str, Any]:
        content = await image.read(max_upload_bytes + 1)
        if not content:
            raise HTTPException(status_code=422, detail="上传图片不能为空。")
        if len(content) > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"图片不能超过 {max_upload_bytes // (1024 * 1024)} MB。",
            )
        suffix = _image_suffix(content)
        if suffix is None:
            raise HTTPException(status_code=422, detail="仅支持有效的 PNG 或 JPEG 图片。")
        target = uploads / f"{uuid4().hex}{suffix}"
        target.write_bytes(content)
        return {
            "success": True,
            "data": {
                "image_path": str(target),
                "original_name": image.filename,
                "content_type": "image/png" if suffix == ".png" else "image/jpeg",
                "size_bytes": len(content),
            },
        }

    @application.post("/api/v1/conversations", tags=["conversation"])
    def create_conversation() -> dict[str, Any]:
        return application.state.conversation_store.create()

    @application.get("/api/v1/conversations/{conversation_id}", tags=["conversation"])
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = application.state.conversation_store.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    @application.post(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=ChatTurnResult,
        tags=["conversation"],
    )
    def send_conversation_message(
        conversation_id: str, message: ChatMessageInput,
    ) -> ChatTurnResult:
        service = application.state.conversation_service
        if service is None:
            raise HTTPException(status_code=503, detail="Conversation service unavailable")
        try:
            with application.state.runner_lock:
                return service.send(conversation_id, message)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Conversation执行失败：{type(exc).__name__}"
            ) from exc

    @application.post(
        "/api/v1/conversations/{conversation_id}/message-jobs",
        tags=["conversation"],
    )
    def submit_conversation_job(
        conversation_id: str, message: ChatMessageInput,
    ) -> dict[str, Any]:
        if application.state.conversation_store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        manager = application.state.conversation_jobs
        if manager is None:
            raise HTTPException(status_code=503, detail="Conversation service unavailable")
        return {"job_id": manager.submit(conversation_id, message), "status": "running"}

    @application.get("/api/v1/conversation-jobs/{job_id}", tags=["conversation"])
    def get_conversation_job(job_id: str) -> dict[str, Any]:
        manager = application.state.conversation_jobs
        job = None if manager is None else manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Conversation job not found")
        return job

    @application.post(
        "/api/v1/agent/invoke",
        response_model=AgentResult,
        tags=["agent"],
    )
    def invoke_agent(request: AgentRequest) -> AgentResult:
        try:
            with application.state.runner_lock:
                return application.state.runner.invoke(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Agent执行失败：{type(exc).__name__}",
            ) from exc

    @application.post(
        "/api/v1/agent/resume/{thread_id}",
        response_model=AgentResult,
        tags=["agent"],
    )
    def resume_agent(thread_id: str, resume: ResumeInput) -> AgentResult:
        try:
            with application.state.runner_lock:
                return application.state.runner.resume(thread_id, resume)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Agent恢复失败：{type(exc).__name__}",
            ) from exc

    application.mount(
        "/static",
        StaticFiles(directory=WEB_ROOT / "static"),
        name="static",
    )
    return application


app = create_app()
