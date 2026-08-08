"""Factory and registry for all five Agent Tools."""

from __future__ import annotations

import os

from .base import HTTPAgentTool
from .classification import PCBADefectClassificationTool
from .reflow import ReflowParameterOptimizationTool, ReflowProfilePredictionTool
from .spi import SPIParameterOptimizationTool, SPIVTEPredictionTool


DEFAULT_AOI_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REFLOW_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_SPI_BASE_URL = "http://127.0.0.1:8002"


def build_tool_registry() -> dict[str, HTTPAgentTool]:
    aoi_url = os.getenv("PCBA_AOI_BASE_URL", DEFAULT_AOI_BASE_URL)
    reflow_url = os.getenv("PCBA_REFLOW_BASE_URL", DEFAULT_REFLOW_BASE_URL)
    spi_url = os.getenv("PCBA_SPI_BASE_URL", DEFAULT_SPI_BASE_URL)
    tools: list[HTTPAgentTool] = [
        PCBADefectClassificationTool(aoi_url, timeout_s=60),
        SPIVTEPredictionTool(spi_url, timeout_s=60),
        SPIParameterOptimizationTool(spi_url, timeout_s=180),
        ReflowProfilePredictionTool(reflow_url, timeout_s=120),
        ReflowParameterOptimizationTool(reflow_url, timeout_s=180),
    ]
    return {tool.name: tool for tool in tools}


TOOL_REGISTRY = build_tool_registry()
