"""Extraction stage: StructuredDocument -> facts in the fact store.

Output: ``{work_dir}/facts.json`` and the ``facts`` table. Runs after indexing, so
a failure here never makes a document unsearchable: it becomes a warning.
Skipped (and old facts removed) when the upload set ``extract_facts=false``.
"""

import json
import logging

from tio2agent.domain import DocumentStatus, StructuredDocument
from tio2agent.ingestion.extraction import FactExtractor, extract_facts, get_extractor
from tio2agent.ingestion.pipeline import IngestionContext, StageError
from tio2agent.storage.fact_repo import FactRepository

logger = logging.getLogger(__name__)

FACTS_FILE = "facts.json"


class ExtractStage:
    status = DocumentStatus.EXTRACTING

    def __init__(self, repo: FactRepository, extractor: FactExtractor | None = None) -> None:
        self.repo = repo
        self.extractor = extractor  # None: choose by document type

    def run(self, ctx: IngestionContext) -> None:
        # If fact extraction is not enabled, delete any existing facts and return None.
        if not ctx.record.options.extract_facts:
            self.repo.delete_document(ctx.record.id)
            ctx.stats.facts = 0
            return
        structure: StructuredDocument | None = ctx.artifacts.get("structure")
        if structure is None:
            raise StageError("缺少结构识别结果，无法抽取", code="EXTRACT_NO_STRUCTURE")

        extractor = self.extractor or get_extractor(structure.doc_type)
        # If no extractor is found for the document type, delete any existing facts and return None.
        if extractor is None:
            logger.info("document %s: no extractor for doc_type=%s", ctx.record.id, structure.doc_type)
            self.repo.delete_document(ctx.record.id)
            ctx.stats.facts = 0
            return

        try:
            result = extract_facts(structure, extractor)
        except Exception as exc:  # the document is already indexed; keep it usable
            logger.exception("fact extraction failed for document %s", ctx.record.id)
            self.repo.delete_document(ctx.record.id)
            ctx.warn("EXTRACT_FAILED", f"结构化抽取失败，文档仍可检索：{type(exc).__name__}: {exc}"[:300])
            return

        for issue in result.issues:
            ctx.warn(issue.code, issue.message, page=issue.page)
        if not result.facts:
            ctx.warn("EXTRACT_NO_FACTS", "未抽取到配方或工艺参数")

        self.repo.replace_document(ctx.record.id, result.facts)
        ctx.stats.facts = len(result.facts)
        ctx.artifacts["facts"] = result.facts
        (ctx.work_dir / FACTS_FILE).write_text(
            json.dumps([f.model_dump(mode="json") for f in result.facts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "document %s: %d facts by %s, %d issues", ctx.record.id, len(result.facts), extractor.name, len(result.issues)
        )
