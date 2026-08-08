"""Unified response contracts for the three model services."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(ContractModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Any | None = Field(default=None, description="Optional structured details")


class ErrorResponse(ContractModel):
    success: Literal[False] = False
    request_id: str | None
    api_version: str
    tool_name: str
    tool_version: str
    model_name: str
    model_version: str
    execution_time_ms: int = Field(..., ge=0)
    data: None = None
    warnings: list[str] = Field(default_factory=list)
    error: ErrorDetail


class HealthResponse(ContractModel):
    success: Literal[True] = True
    request_id: None = None
    api_version: str
    tool_name: str
    tool_version: str
    model_name: str
    model_version: str
    execution_time_ms: int = Field(default=0, ge=0)
    data: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    error: None = None


def build_error_payload(
    *,
    request_id: str | None,
    api_version: str,
    tool_name: str,
    tool_version: str,
    model_name: str,
    model_version: str,
    execution_time_ms: int,
    code: str,
    message: str,
    details: Any | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the exact error envelope shared by all Tool APIs."""

    return ErrorResponse(
        request_id=request_id,
        api_version=api_version,
        tool_name=tool_name,
        tool_version=tool_version,
        model_name=model_name,
        model_version=model_version,
        execution_time_ms=max(0, int(execution_time_ms)),
        warnings=warnings or [],
        error=ErrorDetail(code=code, message=message, details=details),
    ).model_dump()
