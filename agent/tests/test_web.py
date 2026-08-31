from __future__ import annotations

import base64
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from pcba_agent.models import AgentResult
from pcba_agent.web import create_app


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeRunner:
    def __init__(self) -> None:
        self.closed = False
        self.invocations: list[Any] = []
        self.resumes: list[tuple[str, Any]] = []
        self.qwen = SimpleNamespace(
            interpret_message=lambda _text, _context: {
                "intent": "diagnose_cause", "defect": "short",
                "observations": {}, "unavailable_inputs": [],
            },
            answer_follow_up=lambda question, _context: "追问：" + question,
        )
        self.agent_graph = SimpleNamespace(tools=SimpleNamespace())

    def close(self) -> None:
        self.closed = True

    def invoke(self, request: Any) -> AgentResult:
        self.invocations.append(request)
        return AgentResult(
            status="needs_input",
            request_id=request.request_id,
            thread_id=request.thread_id,
            pending_inputs=["manual_observation.paste_bridge"],
            pending_prompt="请补充焊膏桥连观察。",
            response_text="请补充焊膏桥连观察。",
            tool_trace=[{
                "phase": "defect_classification",
                "tool_name": "pcba_defect_classification",
                "success": True,
            }],
        )

    def resume(self, thread_id: str, resume: Any) -> AgentResult:
        self.resumes.append((thread_id, resume))
        return AgentResult(
            status="completed",
            request_id="REQ-WEB-1",
            thread_id=thread_id,
            defect={"canonical_name": "short", "display_name_zh": "短路"},
            response_text="诊断完成。",
        )


class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runner = FakeRunner()
        self.app = create_app(
            runner_factory=lambda: self.runner,
            upload_root=Path(self.temp.name),
            conversation_path=Path(self.temp.name) / "conversations.sqlite3",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_health_and_frontend_are_served(self) -> None:
        with TestClient(self.app) as client:
            health = client.get("/health")
            page = client.get("/")
            script = client.get("/static/app.js")
        self.assertEqual(200, health.status_code)
        self.assertEqual("ready", health.json()["data"]["status"])
        self.assertIn("PCBA Quality Copilot", page.text)
        self.assertIn('id="message-list"', page.text)
        self.assertIn('id="chat-form"', page.text)
        self.assertNotIn('id="provided-defect"', page.text)
        self.assertNotIn('id="observations-json"', page.text)
        self.assertIn("sendMessage", script.text)
        self.assertTrue(self.runner.closed)

    def test_valid_image_upload_uses_generated_safe_path(self) -> None:
        with TestClient(self.app) as client:
            response = client.post(
                "/api/v1/agent/images",
                files={"image": ("unsafe name.png", PNG_1X1, "image/png")},
            )
        self.assertEqual(200, response.status_code)
        data = response.json()["data"]
        target = Path(data["image_path"])
        self.assertEqual(Path(self.temp.name).resolve(), target.parent)
        self.assertNotEqual("unsafe name.png", target.name)
        self.assertEqual(PNG_1X1, target.read_bytes())

    def test_non_image_upload_is_rejected(self) -> None:
        with TestClient(self.app) as client:
            response = client.post(
                "/api/v1/agent/images",
                files={"image": ("note.txt", b"not an image", "text/plain")},
            )
        self.assertEqual(422, response.status_code)

    def test_invoke_and_resume_keep_existing_json_contract(self) -> None:
        request = {
            "schema_version": "1.0.0",
            "request_id": "REQ-WEB-1",
            "thread_id": "THREAD-WEB-1",
            "user_question": "这个短路应该检查什么？",
            "provided_defect": "short",
            "goal": "diagnose",
            "observations": {},
            "response_language": "zh",
        }
        with TestClient(self.app) as client:
            first = client.post("/api/v1/agent/invoke", json=request)
            final = client.post(
                "/api/v1/agent/resume/THREAD-WEB-1",
                json={
                    "observations": {},
                    "unavailable_inputs": ["manual_observation.paste_bridge"],
                    "user_message": "现场暂时无法确认。",
                },
            )
        self.assertEqual(200, first.status_code)
        self.assertEqual("needs_input", first.json()["status"])
        self.assertEqual(200, final.status_code)
        self.assertEqual("completed", final.json()["status"])
        self.assertEqual("THREAD-WEB-1", final.json()["thread_id"])
        self.assertEqual("short", self.runner.invocations[0].provided_defect)
        self.assertEqual(1, len(self.runner.resumes))

    def test_conversation_can_be_created_and_restored(self) -> None:
        with TestClient(self.app) as client:
            created = client.post("/api/v1/conversations")
            conversation_id = created.json()["conversation_id"]
            restored = client.get(f"/api/v1/conversations/{conversation_id}")
        self.assertEqual(200, created.status_code)
        self.assertEqual(200, restored.status_code)
        self.assertEqual(conversation_id, restored.json()["conversation_id"])
        self.assertEqual([], restored.json()["messages"])

    def test_conversation_job_returns_progress_and_result(self) -> None:
        with TestClient(self.app) as client:
            conversation_id = client.post("/api/v1/conversations").json()["conversation_id"]
            submitted = client.post(
                f"/api/v1/conversations/{conversation_id}/message-jobs",
                json={"content": "这个短路可能是什么原因？"},
            )
            job_id = submitted.json()["job_id"]
            for _ in range(50):
                job = client.get(f"/api/v1/conversation-jobs/{job_id}").json()
                if job["status"] != "running":
                    break
                time.sleep(0.01)
        self.assertEqual(200, submitted.status_code)
        self.assertEqual("completed", job["status"])
        self.assertTrue(job["progress"])
        self.assertEqual("needs_input", job["result"]["status"])


if __name__ == "__main__":
    unittest.main()
