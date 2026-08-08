"""FastAPI entry point for SPI VTE prediction and parameter optimization."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from tool.common.schemas import ErrorResponse, HealthResponse, build_error_payload

from .optimizer import (
    OPTIMIZER_NAME,
    TARGET_VTE,
    TOLERANCE,
    optimize_parameters,
)
from .predictor import VTEPredictor


API_VERSION = "v1"
TOOL_VERSION = "0.1.0"
MODEL_NAME = "spi_gpr"
MODEL_VERSION = "0.1.0"
PREDICTION_TOOL_NAME = "spi_vte_prediction"
OPTIMIZATION_TOOL_NAME = "spi_parameter_optimization"

logger = logging.getLogger(__name__)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PrintingParameters(StrictModel):
    squeegee_pressure_kgf: float = Field(
        ..., allow_inf_nan=False, description="刮刀压力，单位 kgf"
    )
    squeegee_speed_m_s: float = Field(
        ..., allow_inf_nan=False, description="刮刀速度，单位 m/s"
    )
    separation_speed_m_s: float = Field(
        ..., allow_inf_nan=False, description="脱模速度，单位 m/s"
    )
    separation_distance_mm: float = Field(
        ..., allow_inf_nan=False, description="脱模距离，单位 mm"
    )


class PredictionRequest(StrictModel):
    request_id: str = Field(..., min_length=1, max_length=128)
    input: PrintingParameters


class OptimizationInput(StrictModel):
    current_parameters: PrintingParameters


class OptimizationRequest(StrictModel):
    request_id: str = Field(..., min_length=1, max_length=128)
    input: OptimizationInput


class PredictionData(StrictModel):
    vte_mean: float
    vte_unit: Literal["percent"] = "percent"
    within_training_domain: bool


class BeforeOptimization(StrictModel):
    parameters: PrintingParameters
    predicted_vte: float


class AfterOptimization(StrictModel):
    predicted_vte: float


class OptimizationData(StrictModel):
    target_vte: float
    tolerance: float
    before: BeforeOptimization
    recommended_parameters: PrintingParameters
    after: AfterOptimization
    objective_error: float
    target_reached: bool
    within_training_domain: bool


class PredictionResponse(StrictModel):
    success: Literal[True] = True
    request_id: str
    api_version: Literal["v1"] = API_VERSION
    tool_name: Literal["spi_vte_prediction"] = PREDICTION_TOOL_NAME
    tool_version: Literal["0.1.0"] = TOOL_VERSION
    model_name: Literal["spi_gpr"] = MODEL_NAME
    model_version: Literal["0.1.0"] = MODEL_VERSION
    execution_time_ms: int
    data: PredictionData
    warnings: list[str]
    error: None = None


class OptimizationResponse(StrictModel):
    success: Literal[True] = True
    request_id: str
    api_version: Literal["v1"] = API_VERSION
    tool_name: Literal["spi_parameter_optimization"] = OPTIMIZATION_TOOL_NAME
    tool_version: Literal["0.1.0"] = TOOL_VERSION
    model_name: Literal["spi_gpr"] = MODEL_NAME
    model_version: Literal["0.1.0"] = MODEL_VERSION
    optimizer: Literal["differential_evolution"] = OPTIMIZER_NAME
    execution_time_ms: int
    data: OptimizationData
    warnings: list[str]
    error: None = None


def elapsed_ms(request: Request) -> int:
    start_time = getattr(request.state, "start_time", time.perf_counter())
    return max(0, int(round((time.perf_counter() - start_time) * 1000)))


def extract_request_id(body: Any) -> str | None:
    if isinstance(body, dict):
        request_id = body.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
    return None


def tool_name_for_request(request: Request) -> str:
    if request.url.path.endswith("/optimize"):
        return OPTIMIZATION_TOOL_NAME
    return PREDICTION_TOOL_NAME


def error_payload(
    request: Request,
    request_id: str | None,
    code: str,
    message: str,
    details: Any | None = None,
) -> dict[str, Any]:
    return build_error_payload(
        request_id=request_id,
        api_version=API_VERSION,
        tool_name=tool_name_for_request(request),
        tool_version=TOOL_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        execution_time_ms=elapsed_ms(request),
        code=code,
        message=message,
        details=details,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = VTEPredictor()
    if predictor.model_name != MODEL_NAME or predictor.model_version != MODEL_VERSION:
        raise RuntimeError(
            "model_info.json model identity does not match the API contract"
        )
    if set(predictor.api_fields) != set(PrintingParameters.model_fields):
        raise RuntimeError(
            "model_info.json input features do not match the API parameter fields"
        )
    app.state.vte_predictor = predictor
    yield
    app.state.vte_predictor = None


app = FastAPI(
    title="焊膏印刷 VTE 预测与参数优化 Tool",
    description=(
        "共享同一个正式 GPR 模型，提供 VTE 均值预测和焊膏印刷参数优化接口。"
    ),
    version=TOOL_VERSION,
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={500: {"model": ErrorResponse}},
    summary="检查 SPI 服务和模型状态",
)
def health(request: Request) -> HealthResponse:
    predictor: VTEPredictor = request.app.state.vte_predictor
    return HealthResponse(
        api_version=API_VERSION,
        tool_name=PREDICTION_TOOL_NAME,
        tool_version=TOOL_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        data={
            "status": "ready",
            "model_loaded": True,
            "optimizer": OPTIMIZER_NAME,
            "input_features": list(predictor.api_fields),
        },
    )


@app.middleware("http")
async def add_request_timer(request: Request, call_next):
    request.state.start_time = time.perf_counter()
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            error_payload(
                request,
                extract_request_id(exc.body),
                "VALIDATION_ERROR",
                "请求 JSON 不符合接口定义，请检查必填字段、字段类型及额外字段。",
                details=exc.errors(),
            )
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(request, None, "HTTP_ERROR", str(exc.detail)),
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled SPI tool error", exc_info=exc)
    message = (
        "焊膏印刷参数优化执行失败。"
        if request.url.path.endswith("/optimize")
        else "VTE 均值预测执行失败。"
    )
    return JSONResponse(
        status_code=500,
        content=error_payload(request, None, "INTERNAL_ERROR", message),
    )


@app.post(
    "/api/v1/tools/spi/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "请求参数校验失败"},
        500: {"model": ErrorResponse, "description": "模型执行失败"},
    },
    summary="预测焊膏印刷 VTE 均值",
)
def predict_vte(payload: PredictionRequest, request: Request) -> PredictionResponse:
    values = payload.input.model_dump()
    predictor: VTEPredictor = request.app.state.vte_predictor
    within_domain, warnings = predictor.check_training_domain(values)
    prediction = predictor.predict(values)

    return PredictionResponse(
        request_id=payload.request_id,
        execution_time_ms=elapsed_ms(request),
        data=PredictionData(
            vte_mean=round(prediction, 4),
            within_training_domain=within_domain,
        ),
        warnings=warnings,
    )


@app.post(
    "/api/v1/tools/spi/optimize",
    response_model=OptimizationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "请求参数校验失败"},
        500: {"model": ErrorResponse, "description": "优化执行失败"},
    },
    summary="优化焊膏印刷工艺参数",
)
def optimize_spi_parameters(
    payload: OptimizationRequest,
    request: Request,
) -> OptimizationResponse:
    current_parameters = payload.input.current_parameters.model_dump()
    predictor: VTEPredictor = request.app.state.vte_predictor
    result = optimize_parameters(predictor, current_parameters)

    rounded_recommendation = {
        field: round(value, 4)
        for field, value in result.recommended_parameters.items()
    }
    # Recalculate after rounding so the returned prediction exactly corresponds to
    # the returned parameter values.
    rounded_after_prediction = predictor.predict(rounded_recommendation)
    rounded_error = abs(rounded_after_prediction - TARGET_VTE)
    within_domain, rounding_warnings = predictor.check_training_domain(
        rounded_recommendation
    )

    return OptimizationResponse(
        request_id=payload.request_id,
        execution_time_ms=elapsed_ms(request),
        data=OptimizationData(
            target_vte=TARGET_VTE,
            tolerance=TOLERANCE,
            before=BeforeOptimization(
                parameters=payload.input.current_parameters,
                predicted_vte=round(result.before_prediction, 4),
            ),
            recommended_parameters=PrintingParameters(**rounded_recommendation),
            after=AfterOptimization(predicted_vte=round(rounded_after_prediction, 4)),
            objective_error=round(rounded_error, 4),
            target_reached=rounded_error <= TOLERANCE,
            within_training_domain=within_domain,
        ),
        warnings=[*result.warnings, *rounding_warnings],
    )
