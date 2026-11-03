"""Document domain models shared by API, storage and ingestion."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    
    return datetime.now(timezone.utc)


class DocumentStatus(StrEnum):
    """
    Possible states of a document processing workflow.
    Currently there are 8 states, including 3 terminal states.
    """
    
    QUEUED = "queued"
    PARSING = "parsing"
    OCR = "ocr"
    STRUCTURING = "structuring"
    INDEXING = "indexing"
    EXTRACTING = "extracting"
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATUSES


TERMINAL_STATUSES: frozenset[DocumentStatus] = frozenset(
    {DocumentStatus.READY, DocumentStatus.READY_WITH_WARNINGS, DocumentStatus.FAILED}
)


class DocType(StrEnum):
    """Possible document types within the scope of the business."""
    
    AUTO = "auto"
    PATENT = "patent"
    STANDARD = "standard"
    PAPER = "paper"
    OTHER = "other"


class OcrMode(StrEnum):
    """Possible OCR modes for document processing."""
    
    AUTO = "auto"    # large embedded images (tables/figures) and pages without a text layer
    FORCE = "force"  # every embedded image that is not a tiny decoration
    OFF = "off"


class Issue(BaseModel):
    """A warning or an error occurred during document processing."""

    code: str
    message: str
    page: int | None = None
    stage: DocumentStatus | None = None


class DocumentStats(BaseModel):
    """Statistics about a processed document."""
    
    pages: int = 0
    paragraphs: int = 0
    chunks: int = 0
    duplicate_chunks: int = 0
    embodiments: int = 0
    facts: int = 0
    ocr_regions: int = 0


class IngestOptions(BaseModel):
    """Options for document ingestion."""
    
    doc_type: DocType = DocType.AUTO
    ocr: OcrMode = OcrMode.AUTO
    extract_facts: bool = True


class DocumentMetadata(BaseModel):
    """User-supplied metadata sent with the upload."""
    
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(None, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=50)


class DocumentRecord(BaseModel):
    """
    The persisted state of one uploaded document.
    This model represents the document record stored in the database.
    Currently, it consists of 15 fields.
    文档管理档案，管理一个 PDF 文件在系统中的生命周期、物理属性和任务状态。
    贯穿整个系统，直接持久化存在 SQLite 数据库的 documents 表里。
    不包含文档的实际正文。
    """

    id: str
    filename: str
    sha256: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.QUEUED
    stage: DocumentStatus | None = None
    doc_type: DocType | None = None
    title: str | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    options: IngestOptions = Field(default_factory=IngestOptions)
    stats: DocumentStats = Field(default_factory=DocumentStats)
    warnings: list[Issue] = Field(default_factory=list)
    error: Issue | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
