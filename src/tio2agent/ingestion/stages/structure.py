"""Structuring stage: page text + OCR results -> StructuredDocument.

Output: ``{work_dir}/structure.json`` and ``ctx.artifacts["structure"]``.
Findings (gaps in paragraph numbering, empty paragraphs, a fallback to generic rules)
become document warnings; the document still completes.
"""

import logging

from tio2agent.domain import DocType, DocumentStatus, EmbodimentKind
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.structure import StructureInput, structure_document

logger = logging.getLogger(__name__)

STRUCTURE_FILE = "structure.json"


def _merge_into_ocr_warning(ctx: IngestionContext, page: int | None, paragraph_id: str) -> bool:
    """An empty paragraph on a page whose image OCR could not read is one problem,
    not two: name the paragraph in the existing OCR warning instead of adding another."""
    for warning in ctx.warnings:
        if warning.page == page and warning.code.startswith("OCR_"):
            warning.message += f"（即段落[{paragraph_id}]）"
            return True
    return False


class StructureStage:
    status = DocumentStatus.STRUCTURING

    def run(self, ctx: IngestionContext) -> None:
        data = StructureInput(
            doc_id=ctx.record.id,
            doc_type=(ctx.record.doc_type or DocType.OTHER).value,
            pages=ctx.artifacts.get("pages") or [],
            ocr_regions=ctx.artifacts.get("ocr_regions") or [],
        )
        result = structure_document(data)
        document = result.document

        for issue in result.issues:
            if issue.paragraph_id and _merge_into_ocr_warning(ctx, issue.page, issue.paragraph_id):
                continue
            ctx.warn(issue.code, issue.message, page=issue.page)

        ctx.stats.paragraphs = len(document.paragraphs)
        ctx.stats.embodiments = sum(e.kind is EmbodimentKind.EXAMPLE for e in document.embodiments)
        if document.title and not ctx.record.metadata.title:
            ctx.record.title = document.title

        ctx.artifacts["structure"] = document
        (ctx.work_dir / STRUCTURE_FILE).write_text(
            document.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "document %s structured by %s: %d paragraphs, %d claims, %d embodiments, %d tables",
            ctx.record.id, document.structurer, len(document.paragraphs),
            len(document.claims), len(document.embodiments), len(document.tables),
        )
