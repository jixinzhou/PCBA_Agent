"""Agent Tool for AOI defect classification."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel

from .base import HTTPAgentTool, ToolTransportError
from .models import ClassificationToolInput


class PCBADefectClassificationTool(HTTPAgentTool):
    name = "pcba_defect_classification"
    description = "Classify one PCBA solder-joint image and return Top-K defect candidates."
    input_model = ClassificationToolInput

    def invoke(self, arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        values = self._validate(arguments)
        image_path = Path(values.image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        request_id = values.request_id or f"AOI-{uuid4()}"
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        try:
            with image_path.open("rb") as image_file:
                with httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                    trust_env=False,
                ) as client:
                    response = client.post(
                        "/api/v1/classify",
                        data={"request_id": request_id, "top_k": str(values.top_k)},
                        files={"image": (image_path.name, image_file, content_type)},
                    )
        except httpx.HTTPError as exc:
            raise ToolTransportError(f"Unable to call {self.name}: {exc}") from exc
        return self._parse_response(response)
