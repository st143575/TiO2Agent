"""Indexing stage: StructuredDocument -> chunks (+ embeddings) in the chunk store.

Output: ``{work_dir}/chunks.json`` and the ``chunks`` table. Re-indexing a document
replaces its chunks atomically. If the embedding service fails, the document is
still indexed for keyword search and a warning says so.
"""

import json
import logging

from tio2agent.domain import DocumentStatus, StructuredDocument
from tio2agent.ingestion.chunking import ChunkingOptions, chunk_document
from tio2agent.ingestion.pipeline import IngestionContext, StageError
from tio2agent.retrieval.embeddings import Embedder, EmbeddingError
from tio2agent.storage.chunk_repo import ChunkRepository

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.json"


class IndexStage:
    status = DocumentStatus.INDEXING

    def __init__(
        self,
        repo: ChunkRepository,
        embedder: Embedder | None = None,
        options: ChunkingOptions | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.options = options or ChunkingOptions()

    def run(self, ctx: IngestionContext) -> None:
        structure: StructuredDocument | None = ctx.artifacts.get("structure")
        if structure is None:
            raise StageError("缺少结构识别结果，无法建立索引", code="INDEX_NO_STRUCTURE")

        chunks = chunk_document(structure, self.options)
        vectors, model = None, None
        if self.embedder is not None and chunks:
            try:
                vectors = self.embedder.embed([c.search_text() for c in chunks])
                model = self.embedder.model
            except EmbeddingError as exc:
                ctx.warn("EMBEDDING_FAILED", f"向量化失败，本文档暂时只能关键词检索：{exc.message}")

        self.repo.replace_document(ctx.record.id, chunks, vectors=vectors, model=model)
        ctx.stats.chunks = len(chunks)
        ctx.stats.duplicate_chunks = sum(1 for c in chunks if c.restates_claims)
        ctx.artifacts["chunks"] = chunks
        (ctx.work_dir / CHUNKS_FILE).write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "document %s indexed: %d chunks (%d restate claims), embeddings=%s",
            ctx.record.id, len(chunks), ctx.stats.duplicate_chunks, model or "none",
        )
