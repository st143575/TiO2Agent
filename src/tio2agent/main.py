"""FastAPI application factory and server entry point.

Run locally:   uv run tio2agent
or:            uv run uvicorn tio2agent.main:create_app --factory --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tio2agent import __version__
from tio2agent.api.errors import register_exception_handlers
from tio2agent.api.middleware import UploadSizeLimitMiddleware
from tio2agent.api.routes import documents, health
from tio2agent.config import Settings, get_settings
from tio2agent.ingestion import DocumentService, IngestionPipeline, IngestionRunner
from tio2agent.ingestion.stages import default_stages
from tio2agent.storage import DocumentRepository, FileStore

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
# Multipart framing and form fields add a little on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_dirs()

    repo = DocumentRepository(settings.db_path)
    file_store = FileStore(settings.raw_dir, settings.tmp_dir, settings.upload_chunk_size)
    stale = file_store.cleanup_tmp()
    if stale:
        logger.info("removed %d stale temp uploads", stale)

    pipeline = IngestionPipeline(
        repo,
        raw_path_for=file_store.path_for,
        processed_dir=settings.processed_dir,
        stages=default_stages(),
    )
    runner = IngestionRunner(pipeline, max_workers=settings.ingest_workers)
    service = DocumentService(repo=repo, file_store=file_store, runner=runner, settings=settings)
    app.state.document_service = service

    recovered = service.recover_unfinished()
    if recovered:
        logger.info("re-queued %d unfinished documents", recovered)

    try:
        yield
    finally:
        runner.shutdown(wait=False)
        repo.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="RAG 与多步骤 Agent 智能问答后端",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        UploadSizeLimitMiddleware,
        path=f"{API_PREFIX}/documents",
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router, prefix=API_PREFIX)
    return app


def run() -> None:
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "tio2agent.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
