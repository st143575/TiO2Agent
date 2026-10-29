"""Ingestion stages, in execution order.

Add new stages (structuring, indexing, extracting) to ``default_stages`` as they
are implemented; each one only needs a ``status`` and a ``run(ctx)`` method.
"""

from tio2agent.ingestion.pipeline import Stage
from tio2agent.ingestion.stages.parse import ParseStage


def default_stages() -> list[Stage]:
    return [
        ParseStage(),
        # StructureStage(),   # DocumentStatus.STRUCTURING
        # IndexStage(),       # DocumentStatus.INDEXING
        # ExtractStage(),     # DocumentStatus.EXTRACTING
    ]


__all__ = ["ParseStage", "default_stages"]
