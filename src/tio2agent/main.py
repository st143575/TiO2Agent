"""FastAPI application factory and server entry point.

Run locally:   uv run tio2agent
or:            uv run uvicorn tio2agent.main:create_app --factory --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tio2agent import __version__
from tio2agent.agent import AgentService, SessionStore
from tio2agent.api.errors import register_exception_handlers
from tio2agent.api.middleware import UploadSizeLimitMiddleware
from tio2agent.api.routes import agent, documents, facts, health, search
from tio2agent.config import ExtractBackend, Settings, get_settings
from tio2agent.ingestion import DocumentService, IngestionPipeline, IngestionRunner
from tio2agent.ingestion.chunking import ChunkingOptions
from tio2agent.ingestion.extraction import build_extractors
from tio2agent.ingestion.ocr import OcrEngine, build_ocr_engine
from tio2agent.ingestion.stages import default_stages
from tio2agent.llm import ChatModel, build_chat_model, build_extraction_model
from tio2agent.retrieval import Embedder, Reranker, SearchService, build_embedder, build_reranker
from tio2agent.storage import ChunkRepository, DocumentRepository, FactRepository, FileStore

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

    ocr_engine: OcrEngine | None = app.state.ocr_engine_override or build_ocr_engine(settings)
    if ocr_engine is None:
        logger.info("OCR disabled (TIO2_OCR_BACKEND=none); image regions will be reported as warnings")
    else:
        logger.info("OCR enabled: %s", ocr_engine.name)

    embedder: Embedder | None = app.state.embedder_override or build_embedder(settings)
    logger.info("embeddings: %s", embedder.model if embedder else "disabled (BM25 only)")
    chunk_repo = ChunkRepository(settings.db_path)
    fact_repo = FactRepository(settings.db_path)

    extraction_model: ChatModel | None = app.state.extraction_model_override
    if extraction_model is None and settings.extract_backend is not ExtractBackend.RULES:
        extraction_model = build_extraction_model(settings)
    extractors = build_extractors(settings.extract_backend, extraction_model, max_failures=settings.extract_llm_max_failures)
    logger.info(
        "fact extraction (patents): %s%s",
        extractors["patent"].name,
        f" with {extraction_model.model}" if extraction_model and extractors["patent"].name != "patent_rules" else "",
    )

    pipeline = IngestionPipeline(
        repo,
        raw_path_for=file_store.path_for,
        processed_dir=settings.processed_dir,
        stages=default_stages(
            chunk_repo=chunk_repo,
            ocr_engine=ocr_engine,
            embedder=embedder,
            chunking=ChunkingOptions(max_chars=settings.chunk_max_chars),
            fact_repo=fact_repo,
            extractors=extractors,
        ),
    )
    runner = IngestionRunner(pipeline, max_workers=settings.ingest_workers)
    service = DocumentService(repo=repo, file_store=file_store, runner=runner, settings=settings)
    service.add_cleanup_hook(lambda record: chunk_repo.delete_document(record.id))
    service.add_cleanup_hook(lambda record: fact_repo.delete_document(record.id))
    app.state.document_service = service
    app.state.chunk_repo = chunk_repo
    app.state.fact_repo = fact_repo
    reranker: Reranker | None = app.state.reranker_override or build_reranker(settings)
    logger.info("reranking: %s", reranker.model if reranker else "disabled (fused order)")
    app.state.search_service = SearchService(
        chunk_repo,
        embedder,
        reranker,
        rrf_k=settings.search_rrf_k,
        rerank_candidates=settings.rerank_candidates,
        rerank_max_chars=settings.rerank_max_chars,
    )

    chat_model: ChatModel | None = app.state.chat_model_override or build_chat_model(settings)
    logger.info("agent LLM: %s", chat_model.model if chat_model else "disabled (/agent/chat answers 503)")
    sessions = SessionStore(settings.db_path)
    app.state.agent_service = AgentService(
        chat_model=chat_model,
        search=app.state.search_service,
        facts=fact_repo,
        documents=repo,
        chunks=chunk_repo,
        sessions=sessions,
        settings=settings,
    )

    recovered = service.recover_unfinished()
    if recovered:
        logger.info("re-queued %d unfinished documents", recovered)

    try:
        yield
    finally:
        runner.shutdown(wait=False)
        repo.close()
        chunk_repo.close()
        fact_repo.close()
        if ocr_engine is not None:
            ocr_engine.close()
        if embedder is not None:
            embedder.close()
        if reranker is not None:
            reranker.close()
        sessions.close()
        if chat_model is not None:
            chat_model.close()
        if extraction_model is not None:
            extraction_model.close()


def create_app(
    settings: Settings | None = None,
    *,
    ocr_engine: OcrEngine | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chat_model: ChatModel | None = None,
    extraction_model: ChatModel | None = None,
) -> FastAPI:
    """
    Build the app.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="RAG 与多步骤 Agent 智能问答后端",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.ocr_engine_override = ocr_engine
    app.state.embedder_override = embedder
    app.state.reranker_override = reranker
    app.state.chat_model_override = chat_model
    app.state.extraction_model_override = extraction_model

    app.add_middleware(
        UploadSizeLimitMiddleware,
        path=f"{API_PREFIX}/documents",
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(search.router, prefix=API_PREFIX)
    app.include_router(facts.router, prefix=API_PREFIX)
    app.include_router(agent.router, prefix=API_PREFIX)
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
