from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping


class ToolAdapter:
    def __init__(self, project_root: Path, registry: Mapping[str, Any] | None = None) -> None:
        self.project_root = project_root
        self._registry = registry

    @property
    def registry(self) -> Mapping[str, Any]:
        if self._registry is None:
            source = str(self.project_root)
            if source not in sys.path:
                sys.path.insert(0, source)
            from tool.agent_tools.registry import TOOL_REGISTRY

            self._registry = TOOL_REGISTRY
        return self._registry

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self.registry:
            raise KeyError(f"Unknown Agent Tool: {tool_name}")
        return self.registry[tool_name].invoke(arguments)

    def classify(self, image_path: str, request_id: str) -> dict[str, Any]:
        return self.invoke(
            "pcba_defect_classification",
            {"image_path": image_path, "request_id": request_id, "top_k": 3},
        )
