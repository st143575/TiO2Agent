"""Document upload and management endpoints.

Handlers are plain ``def`` (not ``async def``): FastAPI runs them in a thread pool,
which is what we want for blocking file I/O, hashing and PDF validation.
"""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile, status
from pydantic import ValidationError

from tio2agent.api.deps import get_document_service
from tio2agent.api.schemas import DocumentListOut, DocumentOut, ErrorResponse, ReindexRequest
from tio2agent.errors import InvalidMetadataError, InvalidParameterError
from tio2agent.ingestion import DocumentService
from tio2agent.domain import DocType, DocumentMetadata, DocumentStatus, IngestOptions, OcrMode

router = APIRouter(prefix="/documents", tags=["documents"])

ServiceDep = Annotated[DocumentService, Depends(get_document_service)]

_ERRORS = {
    400: {"model": ErrorResponse, "description": "空文件"},
    404: {"model": ErrorResponse, "description": "文档不存在"},
    409: {"model": ErrorResponse, "description": "文档正在处理中"},
    413: {"model": ErrorResponse, "description": "文件过大"},
    415: {"model": ErrorResponse, "description": "不是 PDF"},
    422: {"model": ErrorResponse, "description": "PDF 无法读取或参数不合法"},
}


_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    """Parse an optional boolean form field.

    FastAPI treats a *missing* ``bool`` form field as False (HTML checkbox semantics),
    which would silently override our default, so the field is received as a string.
    """
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise InvalidParameterError(
        f"{name} 必须是 true 或 false", details={"field": name, "value": raw}
    )


def _parse_metadata(raw: str | None) -> DocumentMetadata:
    if raw is None or not raw.strip():
        return DocumentMetadata()
    try:
        return DocumentMetadata.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise InvalidMetadataError(details={"reason": str(exc)}) from exc
    except ValidationError as exc:
        raise InvalidMetadataError(
            details={"errors": exc.errors(include_url=False, include_context=False)}
        ) from exc


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    responses={
        200: {"model": DocumentOut, "description": "重复上传，返回已有文档"},
        **{k: v for k, v in _ERRORS.items() if k in (400, 413, 415, 422)},
    },
    summary="上传 PDF 文档",
)
def upload_document(
    request: Request,
    response: Response,
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="PDF 文件")],
    metadata: Annotated[
        str | None, Form(description='JSON 字符串，例如 {"title": "...", "tags": ["包膜"]}')
    ] = None,
    doc_type: Annotated[DocType, Form()] = DocType.AUTO,
    ocr: Annotated[OcrMode, Form()] = OcrMode.AUTO,
    extract_facts: Annotated[str | None, Form(description="true / false，默认 true")] = None,
) -> DocumentOut:
    result = service.upload(
        file.file,
        file.filename,
        metadata=_parse_metadata(metadata),
        options=IngestOptions(
            doc_type=doc_type,
            ocr=ocr,
            extract_facts=_parse_bool("extract_facts", extract_facts, default=True),
        ),
    )
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = str(request.url_for("get_document", doc_id=result.record.id))
    return DocumentOut.from_record(result.record, duplicate=result.duplicate)


@router.get("", response_model=DocumentListOut, summary="文档列表")
def list_documents(
    service: ServiceDep,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListOut:
    records, total = service.list_documents(status=status_filter, limit=limit, offset=offset)
    return DocumentListOut(
        items=[DocumentOut.from_record(r) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{doc_id}",
    name="get_document",
    response_model=DocumentOut,
    responses={404: _ERRORS[404]},
    summary="查询文档处理状态",
)
def get_document(doc_id: str, service: ServiceDep) -> DocumentOut:
    return DocumentOut.from_record(service.get(doc_id))


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: _ERRORS[404], 409: _ERRORS[409]},
    summary="删除文档及其所有派生数据",
)
def delete_document(doc_id: str, service: ServiceDep) -> Response:
    service.delete(doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{doc_id}/reindex",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    responses={404: _ERRORS[404], 409: _ERRORS[409]},
    summary="重新处理文档",
)
def reindex_document(
    doc_id: str,
    service: ServiceDep,
    body: ReindexRequest | None = None,
) -> DocumentOut:
    options = body.merge_into(service.get(doc_id).options) if body else None
    return DocumentOut.from_record(service.reindex(doc_id, options=options))
