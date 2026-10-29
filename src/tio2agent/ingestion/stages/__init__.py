"""Ingestion stages, in execution order.

Add new stages (structuring, indexing, extracting) to ``default_stages`` as they
are implemented; each one only needs a ``status`` and a ``run(ctx)`` method.
"""

from tio2agent.ingestion.ocr import OcrEngine
from tio2agent.ingestion.pipeline import Stage
from tio2agent.ingestion.stages.ocr import OcrStage
from tio2agent.ingestion.stages.parse import ParseStage


def default_stages(*, ocr_engine: OcrEngine | None = None) -> list[Stage]:
    return [
        ParseStage(),
        OcrStage(ocr_engine),
        # StructureStage(),   # DocumentStatus.STRUCTURING
        # IndexStage(),       # DocumentStatus.INDEXING
        # ExtractStage(),     # DocumentStatus.EXTRACTING
    ]


__all__ = ["OcrStage", "ParseStage", "default_stages"]
