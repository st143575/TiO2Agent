"""Ingestion stages, in execution order.

Add new stages (structuring, indexing, extracting) to ``default_stages`` as they
are implemented; each one only needs a ``status`` and a ``run(ctx)`` method.
"""

from tio2agent.ingestion.chunking import ChunkingOptions
from tio2agent.ingestion.ocr import OcrEngine
from tio2agent.ingestion.pipeline import Stage
from tio2agent.ingestion.stages.index import IndexStage
from tio2agent.ingestion.stages.ocr import OcrStage
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.stages.structure import StructureStage
from tio2agent.retrieval.embeddings import Embedder
from tio2agent.storage.chunk_repo import ChunkRepository


def default_stages(
    *,
    chunk_repo: ChunkRepository | None = None,
    ocr_engine: OcrEngine | None = None,
    embedder: Embedder | None = None,
    chunking: ChunkingOptions | None = None,
) -> list[Stage]:
    return [
        ParseStage(),
        OcrStage(ocr_engine),
        StructureStage(),   # DocumentStatus.STRUCTURING
        IndexStage(),       # DocumentStatus.INDEXING
        # ExtractStage(),     # DocumentStatus.EXTRACTING
    ]


__all__ = ["IndexStage", "OcrStage", "ParseStage", "StructureStage", "default_stages"]
