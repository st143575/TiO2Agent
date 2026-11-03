"""Ingestion stages, in execution order.

Add new stages (structuring, indexing, extracting) to ``default_stages`` as they
are implemented; each one only needs a ``status`` and a ``run(ctx)`` method.
"""

from tio2agent.ingestion.chunking import ChunkingOptions
from tio2agent.ingestion.ocr import OcrEngine
from tio2agent.ingestion.pipeline import Stage
from tio2agent.ingestion.stages.extract import ExtractStage
from tio2agent.ingestion.stages.index import IndexStage
from tio2agent.ingestion.stages.ocr import OcrStage
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.stages.structure import StructureStage
from tio2agent.retrieval.embeddings import Embedder
from tio2agent.storage.chunk_repo import ChunkRepository
from tio2agent.storage.fact_repo import FactRepository


def default_stages(
    *,
    chunk_repo: ChunkRepository | None = None,
    ocr_engine: OcrEngine | None = None,
    embedder: Embedder | None = None,
    chunking: ChunkingOptions | None = None,
    fact_repo: FactRepository | None = None,
) -> list[Stage]:
    # return [
    #     ParseStage(),           # DocumentStatus.PARSING
    #     OcrStage(ocr_engine),   # DocumentStatus.OCR
    #     StructureStage(),       # DocumentStatus.STRUCTURING
    #     IndexStage(),           # DocumentStatus.INDEXING
    #     ExtractStage(),         # DocumentStatus.EXTRACTING
    # ]
    stages: list[Stage] = [
        ParseStage(),
        OcrStage(ocr_engine),
        StructureStage(),
    ]
    
    if chunk_repo is not None:
        stages.append(IndexStage(repo=chunk_repo, embedder=embedder, options=chunking))
    
    if fact_repo is not None:
        stages.append(ExtractStage(repo=fact_repo))
    
    return stages


__all__ = ["ExtractStage", "IndexStage", "OcrStage", "ParseStage", "StructureStage", "default_stages"]
