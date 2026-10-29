"""Request/response models for the documents API."""

from datetime import datetime

from pydantic import BaseModel, Field

from tio2agent.domain import (
    DocType,
    DocumentRecord,
    DocumentStats,
    DocumentStatus,
    IngestOptions,
    Issue,
    OcrMode,
)


class DocumentOut(BaseModel):
    id: str
    filename: str
    sha256: str
    size_bytes: int
    status: DocumentStatus
    duplicate: bool = False
    doc_type: DocType | None = None
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    stage: DocumentStatus | None = None
    options: IngestOptions
    stats: DocumentStats
    warnings: list[Issue] = Field(default_factory=list)
    error: Issue | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: DocumentRecord, *, duplicate: bool = False) -> "DocumentOut":
        return cls(
            id=record.id,
            filename=record.filename,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            status=record.status,
            duplicate=duplicate,
            doc_type=record.doc_type,
            title=record.title,
            tags=record.metadata.tags,
            stage=record.stage,
            options=record.options,
            stats=record.stats,
            warnings=record.warnings,
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class ReindexRequest(BaseModel):
    """Optional overrides; omitted fields keep the document's current options."""

    doc_type: DocType | None = None
    ocr: OcrMode | None = None
    extract_facts: bool | None = None

    def merge_into(self, current: IngestOptions) -> IngestOptions:
        return current.model_copy(update=self.model_dump(exclude_none=True))
