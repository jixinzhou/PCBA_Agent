"""Framework-neutral HTTP Tool base classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel


class ToolTransportError(RuntimeError):
    """The model service could not be reached or returned non-JSON data."""


class ToolAPIError(RuntimeError):
    """The model service returned the unified error envelope."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        error = payload.get("error") or {}
        super().__init__(f"{error.get('code', 'API_ERROR')}: {error.get('message', 'Tool call failed')}")
        self.status_code = status_code
        self.payload = payload


class ToolContractError(RuntimeError):
    """The service response does not match the expected Tool contract."""


class HTTPAgentTool:
    """Small synchronous Tool abstraction suitable for later Agent registration."""

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    def __init__(self, base_url: str, *, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def _validate(self, arguments: BaseModel | Mapping[str, Any]) -> BaseModel:
        if isinstance(arguments, self.input_model):
            return arguments
        return self.input_model.model_validate(arguments)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_s,
                trust_env=False,
            ) as client:
                response = client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise ToolTransportError(f"Unable to call {self.name}: {exc}") from exc
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ToolTransportError(
                f"{self.name} returned non-JSON HTTP {response.status_code}"
            ) from exc
        if not isinstance(payload, dict):
            raise ToolContractError(f"{self.name} response root must be an object")
        if response.is_error or payload.get("success") is not True:
            raise ToolAPIError(response.status_code, payload)
        if payload.get("tool_name") != self.name:
            raise ToolContractError(
                f"Expected tool_name={self.name!r}, got {payload.get('tool_name')!r}"
            )
        return payload

    def invoke(self, arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
