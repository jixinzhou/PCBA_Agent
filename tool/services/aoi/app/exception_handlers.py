import time
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tool.common.schemas import build_error_payload

from .config import (
    API_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    TOOL_NAME,
    TOOL_VERSION,
)


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        request_id: Optional[str] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details


def error_body(
    code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Any] = None,
    execution_time_ms: int = 0,
) -> dict:
    return build_error_payload(
        request_id=request_id,
        api_version=API_VERSION,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        execution_time_ms=execution_time_ms,
        code=code,
        message=message,
        details=details,
    )


def elapsed_ms(request: Request) -> int:
    started = getattr(request.state, "start_time", time.perf_counter())
    return max(0, round((time.perf_counter() - started) * 1000))


def extract_request_id(body: Any) -> str | None:
    getter = getattr(body, "get", None)
    if callable(getter):
        value = getter("request_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request,
        exc: ServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(
                error_body(
                    exc.code,
                    exc.message,
                    exc.request_id,
                    exc.details,
                    elapsed_ms(request),
                )
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                error_body(
                    "VALIDATION_ERROR",
                    "The multipart/form-data request parameters are invalid.",
                    request_id=extract_request_id(exc.body),
                    details=exc.errors(),
                    execution_time_ms=elapsed_ms(request),
                )
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        _exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_body(
                "INTERNAL_SERVER_ERROR",
                "An unexpected server error occurred.",
                execution_time_ms=elapsed_ms(request),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                "HTTP_ERROR",
                str(exc.detail),
                execution_time_ms=elapsed_ms(request),
            ),
        )
