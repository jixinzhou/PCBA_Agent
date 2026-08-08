"""Agent Tools for SPI VTE prediction and parameter optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .base import HTTPAgentTool
from .models import SpiOptimizationToolInput, SpiPredictionToolInput


class SPIVTEPredictionTool(HTTPAgentTool):
    name = "spi_vte_prediction"
    description = "Predict mean solder-paste volume transfer efficiency from SPI parameters."
    input_model = SpiPredictionToolInput

    def invoke(self, arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        values = self._validate(arguments)
        payload = values.model_dump()
        payload["request_id"] = values.request_id or f"SPI-PRED-{uuid4()}"
        return self._post_json("/api/v1/tools/spi/predict", payload)


class SPIParameterOptimizationTool(HTTPAgentTool):
    name = "spi_parameter_optimization"
    description = "Recommend SPI printing parameters whose predicted VTE is near the target."
    input_model = SpiOptimizationToolInput

    def invoke(self, arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        values = self._validate(arguments)
        payload = values.model_dump()
        payload["request_id"] = values.request_id or f"SPI-OPT-{uuid4()}"
        return self._post_json("/api/v1/tools/spi/optimize", payload)
