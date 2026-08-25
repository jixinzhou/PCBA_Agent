from __future__ import annotations

from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


DefectName = Literal[
    "insufficient_solder", "excessive_solder", "short", "shifted_component"
]
Goal = Literal["diagnose", "diagnose_and_optimize"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AgentRequest(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    thread_id: str = Field(..., min_length=1)
    user_question: str | None = None
    image_path: str | None = None
    goal: Goal = "diagnose"
    provided_defect: DefectName | None = None
    observations: dict[str, Any] = Field(default_factory=dict)
    response_language: Literal["auto", "zh", "en"] = "auto"

    @model_validator(mode="after")
    def has_input(self) -> "AgentRequest":
        if not any((self.user_question, self.image_path, self.provided_defect)):
            raise ValueError("user_question, image_path or provided_defect is required")
        return self


class ResumeInput(StrictModel):
    observations: dict[str, Any] = Field(default_factory=dict)
    unavailable_inputs: list[str] = Field(default_factory=list)
    provided_defect: DefectName | None = None
    user_message: str | None = None


class CandidateAssessment(StrictModel):
    relationship_id: str
    knowledge_status: Literal["knowledge_supported", "insufficient"]
    verification_capability: Literal["tool_supported", "unverified"]
    assessment_status: Literal[
        "not_evaluated", "supported", "contradicted", "inconclusive"
    ]
    recommendation_status: Literal[
        "not_requested", "not_applicable", "pending_input", "accepted", "rejected", "failed"
    ] = "not_requested"
    limitations: list[str] = Field(default_factory=list)


class AgentResult(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["completed", "needs_input", "failed"]
    request_id: str
    thread_id: str
    defect: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    rag_evidence: list[dict[str, Any]] = Field(default_factory=list)
    pending_inputs: list[str] = Field(default_factory=list)
    pending_prompt: str | None = None
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    degradation_trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    response_text: str = ""
    limitations: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    request: dict[str, Any]
    defect_name: str | None
    defect_source: str | None
    observations: dict[str, Any]
    unavailable_inputs: list[str]
    rag_evidence: list[dict[str, Any]]
    kg_response: dict[str, Any]
    candidates: list[dict[str, Any]]
    validation_records: dict[str, dict[str, Any]]
    optimization_records: dict[str, dict[str, Any]]
    pending_inputs: list[str]
    pending_reason: str | None
    tool_trace: list[dict[str, Any]]
    degradation_trace: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    result: dict[str, Any]
