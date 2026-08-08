"""Agent Tools for reflow prediction and parameter optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .base import HTTPAgentTool
from .models import ReflowOptimizationToolInput, ReflowPredictionToolInput


class ReflowProfilePredictionTool(HTTPAgentTool):
    name = "reflow_profile_prediction"
    description = "Predict reflow metrics, PWI and optional temperature curves for one or more points."
    input_model = ReflowPredictionToolInput

    def invoke(self, arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        values = self._validate(arguments)
        payload = values.model_dump()
        payload["request_id"] = values.request_id or f"REFLOW-PRED-{uuid4()}"
        return self._post_json("/api/v1/reflow-profile/predict", payload)


class ReflowParameterOptimizationTool(HTTPAgentTool):
    name = "reflow_parameter_optimization"
    description = "Optimize reflow zone temperatures and belt speed for one measurement point."
    input_model = ReflowOptimizationToolInput

    def invoke(self, arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        values = self._validate(arguments)
        payload = values.model_dump()
        payload["request_id"] = values.request_id or f"REFLOW-OPT-{uuid4()}"
        return self._post_json("/api/v1/reflow-profile/optimize", payload)
