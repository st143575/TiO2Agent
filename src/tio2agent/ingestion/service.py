"""Document use cases: upload, query, list, delete, re-index.

This is the only entry point the API (and CLI scripts) use; it coordinates the file
store, the repository and the background runner.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

from tio2agent.config import Settings
from tio2agent.errors import DocumentBusyError, DocumentNotFoundError
from tio2agent.ingestion.runner import IngestionRunner
from tio2agent.ingestion.validation import inspect_pdf
from tio2agent.domain import (
    TERMINAL_STATUSES,
    DocType,
    DocumentMetadata,
    DocumentRecord,
    DocumentStats,
    DocumentStatus,
    IngestOptions,
)
from tio2agent.storage import DocumentRepository, DuplicateDocumentError, FileStore, Outcome
from tio2agent.storage.file_store import remove_tree

logger = logging.getLogger(__name__)

# Called with the deleted record, so other stores (vector index, BM25, fact DB) can
# remove what they hold for it. Register them via DocumentService.add_cleanup_hook.
CleanupHook = Callable[[DocumentRecord], None]


@dataclass(frozen=True)
class UploadResult:
    record: DocumentRecord
    duplicate: bool


def new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"


def safe_filename(filename: str | None) -> str:
    """Keep only the base name; browsers on Windows may send full paths."""
    if not filename:
        return "document.pdf"
    name = PureWindowsPath(PurePosixPath(filename).name).name.strip()
    return (name or "document.pdf")[:255]


class DocumentService:
    def __init__(
        self,
        *,
        repo: DocumentRepository,
        file_store: FileStore,
        runner: IngestionRunner,
        settings: Settings,
    ) -> None:
        self.repo = repo
        self.file_store = file_store
        self.runner = runner
        self.settings = settings
        self._cleanup_hooks: list[CleanupHook] = []

    def add_cleanup_hook(self, hook: CleanupHook) -> None:
        self._cleanup_hooks.append(hook)

    # --- upload ----------------------------------------------------------------

    def upload(
        self,
        stream: BinaryIO,
        filename: str | None,
        *,
        metadata: DocumentMetadata | None = None,
        options: IngestOptions | None = None,
    ) -> UploadResult:
        metadata = metadata or DocumentMetadata()
        options = options or IngestOptions()

        staged = self.file_store.stage(stream, self.settings.max_upload_bytes)
        try:
            # Check if the file is already in the database.
            # If it is already in the database, discard the temporary file
            # and return the existing record.
            existing = self.repo.get_by_sha256(staged.sha256)
            if existing is not None:
                self.file_store.discard(staged)
                return UploadResult(existing, duplicate=True)

            # Validate the PDF before committing.
            inspect_pdf(staged.path, max_pages=self.settings.max_pages)
            # Move the temporary file to the permanent location after validation.
            self.file_store.commit(staged)
        except BaseException:
            self.file_store.discard(staged)
            raise

        record = DocumentRecord(
            id=new_document_id(),
            filename=safe_filename(filename),
            sha256=staged.sha256,
            size_bytes=staged.size_bytes,
            title=metadata.title,
            doc_type=None if options.doc_type is DocType.AUTO else options.doc_type,
            metadata=metadata,
            options=options,
        )
        try:
            self.repo.insert(record)
        except DuplicateDocumentError as exc:  # concurrent upload of the same file
            existing = self.repo.get(exc.existing_id)
            if existing is None:
                raise
            return UploadResult(existing, duplicate=True)

        self.runner.submit(record.id)
        logger.info("document %s queued (%s, %d bytes)", record.id, record.filename, record.size_bytes)
        return UploadResult(record, duplicate=False)

    # --- queries ---------------------------------------------------------------

    def get(self, doc_id: str) -> DocumentRecord:
        record = self.repo.get(doc_id)
        if record is None:
            raise DocumentNotFoundError(details={"id": doc_id})
        return record

    def list_documents(
        self, *, status: DocumentStatus | None = None, limit: int = 20, offset: int = 0
    ) -> tuple[list[DocumentRecord], int]:
        return self.repo.list_page(status=status, limit=limit, offset=offset)

    # --- mutations -------------------------------------------------------------

    def delete(self, doc_id: str) -> None:
        outcome, record = self.repo.delete(doc_id, allowed_from=TERMINAL_STATUSES)
        if outcome is Outcome.MISSING:
            raise DocumentNotFoundError(details={"id": doc_id})
        if outcome is Outcome.BLOCKED:
            raise DocumentBusyError(details={"id": doc_id, "status": record.status})

        assert record is not None
        self.file_store.delete(record.sha256)
        remove_tree(self.processed_dir_for(doc_id))
        for hook in self._cleanup_hooks:
            try:
                hook(record)
            except Exception:
                logger.exception("cleanup hook failed for %s", doc_id)
        logger.info("document %s deleted", doc_id)

    def reindex(self, doc_id: str, *, options: IngestOptions | None = None) -> DocumentRecord:
        current = self.get(doc_id)
        new_options = options or current.options
        user_title = current.metadata.title

        outcome, record = self.repo.transition(
            doc_id,
            allowed_from=TERMINAL_STATUSES,
            status=DocumentStatus.QUEUED,
            stage=None,
            options=new_options,
            doc_type=None if new_options.doc_type is DocType.AUTO else new_options.doc_type,
            title=user_title,
            stats=DocumentStats(),
            warnings=[],
            error=None,
        )
        if outcome is Outcome.MISSING:
            raise DocumentNotFoundError(details={"id": doc_id})
        if outcome is Outcome.BLOCKED:
            raise DocumentBusyError(details={"id": doc_id, "status": record.status})

        assert record is not None
        self.runner.submit(doc_id)
        logger.info("document %s re-queued", doc_id)
        return record

    def recover_unfinished(self) -> int:
        """Re-submit documents interrupted by a shutdown or crash."""
        unfinished = self.repo.list_unfinished()
        for record in unfinished:
            self.runner.submit(record.id)
        return len(unfinished)

    def processed_dir_for(self, doc_id: str) -> Path:
        return self.settings.processed_dir / doc_id
