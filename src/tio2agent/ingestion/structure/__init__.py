"""Turn page text into a StructuredDocument.

One structurer per document type; ``structure_document`` picks it and falls back to
the generic structurer when a type-specific one finds nothing it recognizes (e.g. a
document detected as a patent that is actually a scanned excerpt).
"""

from tio2agent.domain import DocType
from tio2agent.ingestion.structure.base import (
    DocumentStructurer,
    StructureInput,
    StructureIssue,
    StructureResult,
)
from tio2agent.ingestion.structure.generic import GenericStructurer
from tio2agent.ingestion.structure.normalize import join_lines, normalize_text
from tio2agent.ingestion.structure.patent import PatentStructurer

STRUCTURERS: dict[DocType, DocumentStructurer] = {
    DocType.PATENT: PatentStructurer(),
}
FALLBACK: DocumentStructurer = GenericStructurer()


def structure_document(data: StructureInput) -> StructureResult:
    try:
        doc_type = DocType(data.doc_type)
    except ValueError:
        doc_type = DocType.OTHER
    structurer = STRUCTURERS.get(doc_type, FALLBACK)
    result = structurer.structure(data)

    if structurer is not FALLBACK and not result.document.paragraphs and not result.document.claims:
        fallback = FALLBACK.structure(data)
        fallback.issues.insert(
            0,
            StructureIssue(
                "STRUCTURE_FALLBACK",
                f"按 {structurer.name} 格式未识别出任何结构，已改用通用规则",
            ),
        )
        return fallback
    return result


__all__ = [
    "DocumentStructurer",
    "GenericStructurer",
    "PatentStructurer",
    "StructureInput",
    "StructureIssue",
    "StructureResult",
    "join_lines",
    "normalize_text",
    "structure_document",
]
