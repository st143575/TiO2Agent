"""Maps domain errors to HTTP responses with one consistent JSON shape:

    {"error": {"code": "...", "message": "...", "details": {...}}}
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tio2agent.errors import AppError

logger = logging.getLogger(__name__)

STATUS_BY_CODE: dict[str, int] = {
    "EMPTY_FILE": 400,
    "DOCUMENT_NOT_FOUND": 404,
    "DOCUMENT_BUSY": 409,
    "FILE_TOO_LARGE": 413,
    "UNSUPPORTED_FILE_TYPE": 415,
    "PDF_ENCRYPTED": 422,
    "PDF_CORRUPTED": 422,
    "TOO_MANY_PAGES": 422,
    "INVALID_METADATA": 422,
    "VALIDATION_ERROR": 422,
}


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": jsonable_encoder(details or {})}}


def error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, message, details))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        status = STATUS_BY_CODE.get(exc.code, 500)
        if status >= 500:
            logger.error("unmapped application error %s: %s", exc.code, exc.message)
        return error_response(status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422, "VALIDATION_ERROR", "请求参数不合法", {"errors": jsonable_encoder(exc.errors())}
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return error_response(500, "INTERNAL_ERROR", "服务器内部错误")
