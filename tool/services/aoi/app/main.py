from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from tool.common.schemas import ErrorResponse

from .config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_FORMATS,
    API_TITLE,
    API_VERSION,
    CONFIDENCE_THRESHOLD,
    LABEL_SCHEMA_VERSION,
    MAX_UPLOAD_BYTES,
    MODEL_NAME,
    MODEL_VERSION,
    TOOL_NAME,
    TOOL_VERSION,
)
from .exception_handlers import ServiceError, install_exception_handlers
from .model_service import ModelService
from .schemas import ClassificationResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService()
    yield


app = FastAPI(
    title=API_TITLE,
    version=TOOL_VERSION,
    description="Classify PCBA solder-joint images with EfficientNet-B0.",
    lifespan=lifespan,
)
install_exception_handlers(app)


@app.middleware("http")
async def add_request_timer(request: Request, call_next):
    request.state.start_time = time.perf_counter()
    return await call_next(request)


def decode_image(
    content: bytes,
    filename: str,
    content_type: str | None,
    request_id: str,
) -> Image.Image:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ServiceError(
            415,
            "UNSUPPORTED_IMAGE_TYPE",
            "Only JPG, JPEG and PNG image files are supported.",
            request_id,
        )
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise ServiceError(
            415,
            "UNSUPPORTED_IMAGE_TYPE",
            "The uploaded file Content-Type must be image/jpeg or image/png.",
            request_id,
        )
    if not content:
        raise ServiceError(400, "EMPTY_IMAGE", "The uploaded image is empty.", request_id)
    if len(content) > MAX_UPLOAD_BYTES:
        raise ServiceError(
            413,
            "IMAGE_TOO_LARGE",
            f"The uploaded image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            request_id,
        )

    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in ALLOWED_IMAGE_FORMATS:
                raise ServiceError(
                    415,
                    "UNSUPPORTED_IMAGE_TYPE",
                    "The actual image format must be JPEG or PNG.",
                    request_id,
                )
            source.load()
            return source.convert("RGB")
    except ServiceError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ServiceError(
            400,
            "INVALID_IMAGE",
            "The uploaded file is not a valid readable image.",
            request_id,
        ) from exc


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Check service and model readiness",
)
async def health(request: Request) -> dict:
    service: ModelService = request.app.state.model_service
    return {
        "success": True,
        "api_version": API_VERSION,
        "tool_name": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "data": {
            "status": "ready",
            "model_loaded": True,
            "device": str(service.device),
            "class_count": len(service.class_names),
        },
        "error": None,
    }


@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Classify one PCBA image",
)
async def classify(
    request: Request,
    image: UploadFile = File(..., description="JPG/JPEG/PNG image file"),
    request_id: str = Form(..., min_length=1, max_length=128),
    top_k: int = Form(3, ge=1, le=5),
) -> dict:
    started = time.perf_counter()
    normalized_request_id = request_id.strip()
    if not normalized_request_id:
        raise ServiceError(
            422,
            "INVALID_REQUEST_ID",
            "request_id must not be blank.",
            request_id,
        )

    content = await image.read(MAX_UPLOAD_BYTES + 1)
    decoded = decode_image(
        content,
        image.filename or "",
        image.content_type,
        normalized_request_id,
    )
    service: ModelService = request.app.state.model_service
    prediction = await run_in_threadpool(service.predict, decoded, top_k)

    low_confidence = prediction["confidence"] < CONFIDENCE_THRESHOLD
    warnings = (
        ["Confidence is below the threshold; manual review is recommended."]
        if low_confidence
        else []
    )
    elapsed_ms = max(1, round((time.perf_counter() - started) * 1000))
    return {
        "success": True,
        "request_id": normalized_request_id,
        "api_version": API_VERSION,
        "tool_name": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "execution_time_ms": elapsed_ms,
        "data": {
            "status": "manual_review" if low_confidence else "classified",
            **prediction,
            "low_confidence": low_confidence,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "label_schema_version": LABEL_SCHEMA_VERSION,
        },
        "warnings": warnings,
        "error": None,
    }
