from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tool.common.schemas import ErrorResponse, build_error_payload

from .optimizer import (
    OptimizationInputError,
    SinglePointPSOOptimizer,
    response_number,
)
from .predictor import PredictionService, build_feature_row
from .schemas import (
    HealthResponse,
    OptimizationRequest,
    OptimizationResponse,
    PredictionRequest,
    PredictionResponse,
)
from .training_domain import TrainingDomain


API_VERSION = "v1"
TOOL_NAME = "reflow_profile_prediction"
TOOL_VERSION = "0.1.0"
MODEL_NAME = "reflow_curve_model"
MODEL_VERSION = "0.1.0"

OPTIMIZATION_TOOL_NAME = "reflow_parameter_optimization"
OPTIMIZATION_TOOL_VERSION = "0.1.0"
OPTIMIZATION_MODEL_NAME = "reflow_pso_optimizer"
OPTIMIZATION_MODEL_VERSION = "0.1.0"

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
TRAINING_DOMAIN_PATH = MODELS_DIR / "training_domain.json"

logger = logging.getLogger(__name__)


PREDICTION_METADATA = {
    "tool_name": TOOL_NAME,
    "tool_version": TOOL_VERSION,
    "model_name": MODEL_NAME,
    "model_version": MODEL_VERSION,
}
OPTIMIZATION_METADATA = {
    "tool_name": OPTIMIZATION_TOOL_NAME,
    "tool_version": OPTIMIZATION_TOOL_VERSION,
    "model_name": OPTIMIZATION_MODEL_NAME,
    "model_version": OPTIMIZATION_MODEL_VERSION,
}


def _metadata_for_path(path: str) -> dict[str, str]:
    if path == "/api/v1/reflow-profile/optimize":
        return OPTIMIZATION_METADATA
    return PREDICTION_METADATA


def _base_response(
    *,
    request_id: str | None,
    execution_time_ms: int,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected_metadata = metadata or PREDICTION_METADATA
    return {
        "request_id": request_id,
        "api_version": API_VERSION,
        **selected_metadata,
        "execution_time_ms": execution_time_ms,
    }


def _error_response(
    *,
    status_code: int,
    request_id: str | None,
    code: str,
    message: str,
    details: Any | None = None,
    execution_time_ms: int = 0,
    metadata: dict[str, str] | None = None,
) -> JSONResponse:
    selected_metadata = metadata or PREDICTION_METADATA
    content = build_error_payload(
        request_id=request_id,
        api_version=API_VERSION,
        execution_time_ms=execution_time_ms,
        code=code,
        message=message,
        details=details,
        **selected_metadata,
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(content))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService.load(MODELS_DIR)
    app.state.training_domain = TrainingDomain.load(TRAINING_DOMAIN_PATH)
    yield


app = FastAPI(
    title="回流焊预测与参数优化接口",
    description=(
        "输入测点位置/体积、13 个温区平均温度和链速，输出各测点温度曲线、"
        "工艺指标、PWI 及总体合格结论；也可针对单测点优化温区和链速。"
    ),
    version=TOOL_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = None
    if isinstance(exc.body, dict):
        raw_request_id = exc.body.get("request_id")
        if isinstance(raw_request_id, str):
            request_id = raw_request_id

    details = [
        {
            "location": [str(item) for item in error["loc"]],
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return _error_response(
        status_code=422,
        request_id=request_id,
        code="VALIDATION_ERROR",
        message="请求参数校验失败",
        details=details,
        metadata=_metadata_for_path(request.url.path),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理的接口异常", exc_info=exc)
    return _error_response(
        status_code=500,
        request_id=None,
        code="INTERNAL_ERROR",
        message="回流焊服务内部错误",
        metadata=_metadata_for_path(request.url.path),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    return _error_response(
        status_code=exc.status_code,
        request_id=None,
        code="HTTP_ERROR",
        message=str(exc.detail),
        metadata=_metadata_for_path(request.url.path),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["系统"],
)
def health(request: Request) -> dict[str, Any]:
    service: PredictionService = request.app.state.prediction_service
    training_domain: TrainingDomain = request.app.state.training_domain
    return {
        "success": True,
        "request_id": None,
        "api_version": API_VERSION,
        "tool_name": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "execution_time_ms": 0,
        "data": {
            "status": "ready",
            "model_loaded": True,
            "route_tcs": service.route_tcs,
            "curve_sample_interval_s": service.curve_sample_interval_s,
            "training_domain_source": training_domain.source,
        },
        "warnings": [],
        "error": None,
    }


@app.post(
    "/api/v1/reflow-profile/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["预测"],
)
def predict(payload: PredictionRequest, request: Request):
    start = perf_counter()
    service: PredictionService = request.app.state.prediction_service
    training_domain: TrainingDomain = request.app.state.training_domain

    try:
        input_data = payload.input_data
        options = payload.options
        within_training_domain, domain_warnings = training_domain.check(
            input_data.zone_means_c,
            input_data.belt_speed_cm_min,
        )
        feature_row = build_feature_row(
            input_data.zone_means_c,
            input_data.belt_speed_cm_min,
        )

        point_results = [
            service.predict_point(
                point_id=point.point_id,
                component_x_mm=point.component_x_mm,
                component_y_mm=point.component_y_mm,
                component_volume_mm3=point.component_volume_mm3,
                feature_row=feature_row,
                belt_speed_cm_min=input_data.belt_speed_cm_min,
                return_temperature_curve=options.return_temperature_curve,
                curve_downsample_interval_s=options.curve_downsample_interval_s,
            )
            for point in input_data.points
        ]

        worst_point = max(point_results, key=lambda item: float(item["metrics"]["pwi"]))
        overall = {
            "max_pwi": float(worst_point["metrics"]["pwi"]),
            "qualified": all(
                item["metrics"]["status"] == "qualified" for item in point_results
            ),
            "worst_point_id": worst_point["point_id"],
        }
        execution_time_ms = int(round((perf_counter() - start) * 1000.0))
        return {
            "success": True,
            **_base_response(
                request_id=payload.request_id,
                execution_time_ms=execution_time_ms,
            ),
            "data": {
                "point_results": point_results,
                "overall": overall,
                "within_training_domain": within_training_domain,
            },
            "warnings": domain_warnings,
            "error": None,
        }
    except ValueError as exc:
        execution_time_ms = int(round((perf_counter() - start) * 1000.0))
        return _error_response(
            status_code=400,
            request_id=payload.request_id,
            code="PREDICTION_INPUT_ERROR",
            message=str(exc),
            execution_time_ms=execution_time_ms,
        )
    except Exception as exc:
        execution_time_ms = int(round((perf_counter() - start) * 1000.0))
        logger.exception("模型预测失败", exc_info=exc)
        return _error_response(
            status_code=500,
            request_id=payload.request_id,
            code="PREDICTION_ERROR",
            message="模型预测失败，请检查服务日志",
            execution_time_ms=execution_time_ms,
        )


@app.post(
    "/api/v1/reflow-profile/optimize",
    response_model=OptimizationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["优化"],
)
def optimize(payload: OptimizationRequest, request: Request):
    start = perf_counter()
    service: PredictionService = request.app.state.prediction_service
    training_domain: TrainingDomain = request.app.state.training_domain

    try:
        input_data = payload.input_data
        point = input_data.points[0]
        current = input_data.current_parameters
        target = input_data.optimization_target
        adjustable = input_data.adjustable_parameters

        _, current_domain_warnings = training_domain.check(
            current.zone_means_c,
            current.belt_speed_cm_min,
        )
        warnings_out = [f"当前参数：{item}" for item in current_domain_warnings]

        target_value = None
        tolerance = None
        if target.mode == "target_peak_temperature":
            target_value = target.target_value_c
            tolerance = target.tolerance_c
        elif target.mode == "target_time_above_217":
            target_value = target.target_value_s
            tolerance = target.tolerance_s

        optimizer = SinglePointPSOOptimizer(service.delivery, training_domain)
        result = optimizer.optimize(
            component_x_mm=point.component_x_mm,
            component_y_mm=point.component_y_mm,
            component_volume_mm3=point.component_volume_mm3,
            current_zone_means_c=current.zone_means_c,
            current_belt_speed_cm_min=current.belt_speed_cm_min,
            mode=target.mode,
            target_value=target_value,
            tolerance=tolerance,
            zone_indexes=adjustable.zone_indexes,
            adjust_belt_speed=adjustable.adjust_belt_speed,
        )
        warnings_out.extend(result.warnings)

        recommended_zones = list(result.recommended_parameters[:13])
        recommended_speed = float(result.recommended_parameters[13])
        within_training_domain, unexpected_warnings = training_domain.check(
            recommended_zones,
            recommended_speed,
        )
        if not within_training_domain:
            raise RuntimeError(
                "优化器返回训练域外参数: " + "; ".join(unexpected_warnings)
            )

        before = result.before.response_dict()
        after = result.after.response_dict()
        execution_time_ms = int(round((perf_counter() - start) * 1000.0))
        return {
            "success": True,
            **_base_response(
                request_id=payload.request_id,
                execution_time_ms=execution_time_ms,
                metadata=OPTIMIZATION_METADATA,
            ),
            "data": {
                "optimization_mode": target.mode,
                "before": {
                    "zone_means_c": [response_number(value) for value in current.zone_means_c],
                    "belt_speed_cm_min": response_number(current.belt_speed_cm_min),
                    **before,
                },
                "recommended_parameters": {
                    "zone_means_c": [response_number(value) for value in recommended_zones],
                    "belt_speed_cm_min": response_number(recommended_speed),
                },
                "after": after,
                "target_reached": result.target_reached,
                "within_training_domain": within_training_domain,
            },
            "warnings": warnings_out,
            "error": None,
        }
    except (OptimizationInputError, ValueError) as exc:
        execution_time_ms = int(round((perf_counter() - start) * 1000.0))
        return _error_response(
            status_code=400,
            request_id=payload.request_id,
            code="OPTIMIZATION_INPUT_ERROR",
            message=str(exc),
            execution_time_ms=execution_time_ms,
            metadata=OPTIMIZATION_METADATA,
        )
    except Exception as exc:
        execution_time_ms = int(round((perf_counter() - start) * 1000.0))
        logger.exception("工艺参数优化失败", exc_info=exc)
        return _error_response(
            status_code=500,
            request_id=payload.request_id,
            code="OPTIMIZATION_ERROR",
            message="工艺参数优化失败，请检查服务日志",
            execution_time_ms=execution_time_ms,
            metadata=OPTIMIZATION_METADATA,
        )
