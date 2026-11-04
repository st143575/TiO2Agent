"""Domain errors.

Each error carries a stable machine-readable ``code``, e.g., "EMPTY_FILE", "FILE_TOO_LARGE", etc. 
The API layer maps codes to HTTP status codes (see ``api/errors.py``), so nothing below the API knows about HTTP.
"""

from typing import Any


class AppError(Exception):
    code: str = "INTERNAL_ERROR"
    message: str = "服务器内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = details or {}
        super().__init__(self.message)


# --- Upload errors -------------------------------------------------------------

class EmptyFileError(AppError):
    code = "EMPTY_FILE"
    message = "上传的文件为空"


class FileTooLargeError(AppError):
    code = "FILE_TOO_LARGE"
    message = "文件超过大小上限"


class UnsupportedFileTypeError(AppError):
    code = "UNSUPPORTED_FILE_TYPE"
    message = "仅支持 PDF 文件"


class PdfValidationError(AppError):
    """Raised with code PDF_ENCRYPTED, PDF_CORRUPTED or TOO_MANY_PAGES."""

    code = "PDF_CORRUPTED"
    message = "无法解析 PDF 文件"


class InvalidParameterError(AppError):
    code = "VALIDATION_ERROR"
    message = "请求参数不合法"


class InvalidMetadataError(AppError):
    code = "INVALID_METADATA"
    message = "metadata 必须是合法的 JSON 对象"


# --- Document errors -----------------------------------------------------------

class DocumentNotFoundError(AppError):
    code = "DOCUMENT_NOT_FOUND"
    message = "文档不存在"


class DocumentBusyError(AppError):
    code = "DOCUMENT_BUSY"
    message = "文档正在处理中，请等处理结束后再操作"


class StructureNotAvailableError(AppError):
    code = "STRUCTURE_NOT_AVAILABLE"
    message = "文档还没有结构化结果"


# --- Agent errors ----------------------------------------------------------------


class SessionNotFoundError(AppError):
    code = "SESSION_NOT_FOUND"
    message = "会话不存在"
