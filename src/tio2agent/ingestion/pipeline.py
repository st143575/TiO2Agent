"""Background ingestion pipeline: a sequence of stages driven by a status state machine.

Each stage declares the status it represents (parsing, structuring, ...). The pipeline
writes that status before running the stage, so GET /documents/{id} always shows where
a document is. Stages are added here as they are implemented; the pipeline itself does
not change.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tio2agent.errors import AppError
from tio2agent.domain import DocumentRecord, DocumentStats, DocumentStatus, Issue
from tio2agent.storage import DocumentRepository
from tio2agent.storage.file_store import remove_tree

logger = logging.getLogger(__name__)


@dataclass
class IngestionContext:
    record: DocumentRecord
    pdf_path: Path
    work_dir: Path
    stats: DocumentStats = field(default_factory=DocumentStats)
    warnings: list[Issue] = field(default_factory=list)
    # In-memory hand-off between stages, e.g. artifacts["pages"] from the parse stage.
    artifacts: dict[str, Any] = field(default_factory=dict)
    current_stage: DocumentStatus | None = None

    def warn(self, code: str, message: str, *, page: int | None = None) -> None:
        self.warnings.append(
            Issue(code=code, message=message, page=page, stage=self.current_stage)
        )


class Stage(Protocol):
    status: DocumentStatus

    def run(self, ctx: IngestionContext) -> None: ...


class StageError(AppError):
    """Raise from a stage to fail the document with a specific error code."""

    code = "STAGE_FAILED"
    message = "文档处理失败"


class IngestionPipeline:
    """
    Ingestion pipeline for processing documents through a series of stages.
    This class only serves as an orchestrator, delegating the actual work to individual stages.
    """
    def __init__(
        self,
        repo: DocumentRepository,
        *,
        raw_path_for: Any,
        processed_dir: Path,
        stages: Sequence[Stage],
    ) -> None:
        self.repo = repo
        self.raw_path_for = raw_path_for
        self.processed_dir = processed_dir
        self.stages = list(stages)

    def run(self, doc_id: str) -> None:
        record = self.repo.get(doc_id)
        if record is None:
            logger.info("document %s deleted before processing", doc_id)
            return
        if record.status.is_terminal:
            logger.info("document %s already %s, skipping", doc_id, record.status)
            return

        work_dir = self.processed_dir / doc_id
        remove_tree(work_dir)  # start clean: this may be a re-index or a recovered job
        work_dir.mkdir(parents=True, exist_ok=True)

        ctx = IngestionContext(
            record=record,
            pdf_path=self.raw_path_for(record.sha256),
            work_dir=work_dir,
        )

        for stage in self.stages:
            ctx.current_stage = stage.status
            # Update the document status and stage in the database.
            if self.repo.update(doc_id, status=stage.status, stage=stage.status) is None:
                logger.info("document %s deleted during processing", doc_id)
                return
            try:
                stage.run(ctx)      # The actual implementation of run() is in other modules.
            except AppError as exc:
                self._fail(ctx, exc.code, exc.message)
                return
            except Exception as exc:
                logger.exception("stage %s failed for document %s", stage.status, doc_id)
                self._fail(ctx, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}"[:300])
                return

        final = DocumentStatus.READY_WITH_WARNINGS if ctx.warnings else DocumentStatus.READY
        self.repo.update(
            doc_id,
            status=final,
            stage=None,
            stats=ctx.stats,
            warnings=ctx.warnings,
            error=None,
            doc_type=ctx.record.doc_type,
            title=ctx.record.title,
        )
        logger.info("document %s finished: %s", doc_id, final)

    def _fail(self, ctx: IngestionContext, code: str, message: str) -> None:
        self.repo.update(
            ctx.record.id,
            status=DocumentStatus.FAILED,
            stats=ctx.stats,
            warnings=ctx.warnings,
            error=Issue(code=code, message=message, stage=ctx.current_stage),
        )
        logger.warning("document %s failed at %s: %s", ctx.record.id, ctx.current_stage, code)
