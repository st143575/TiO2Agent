"""Structured fact extraction: StructuredDocument -> list[Fact] (+ validation).

One extractor per document type. Only patents have one so far; other types are
skipped (no facts, no warning). For patents, TIO2_EXTRACT_BACKEND chooses:

    rules    PatentRuleExtractor                       (default; no LLM needed)
    llm      LlmPatentExtractor                        (tables still from cells)
    hybrid   HybridExtractor(rules, llm)               (merged)
"""

import logging
from collections.abc import Callable, Mapping

from tio2agent.config import ExtractBackend
from tio2agent.domain import StructuredDocument
from tio2agent.ingestion.extraction.base import (
    ExtractionIssue,
    ExtractionResult,
    FactBuilder,
    FactExtractor,
    Rejection,
    Source,
)
from tio2agent.ingestion.extraction.hybrid import HybridExtractor, merge_facts
from tio2agent.ingestion.extraction.llm_patent import LlmPatentExtractor
from tio2agent.ingestion.extraction.patent_rules import PatentRuleExtractor
from tio2agent.ingestion.extraction.validators import claimed_ranges, validate_facts
from tio2agent.llm import ChatModel

logger = logging.getLogger(__name__)

EXTRACTORS: dict[str, FactExtractor] = {"patent": PatentRuleExtractor()}


def build_extractors(
    backend: ExtractBackend,
    model: ChatModel | None,
    *,
    max_failures: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, FactExtractor]:
    """The extractor per document type for a backend. llm / hybrid without a model
    fall back to rules (logged)."""
    if backend is ExtractBackend.RULES:
        return dict(EXTRACTORS)
    if model is None:
        logger.warning("TIO2_EXTRACT_BACKEND=%s needs an LLM (TIO2_LLM_BACKEND); using rules", backend.value)
        return dict(EXTRACTORS)
    llm = LlmPatentExtractor(model, max_failures=max_failures, progress=progress)
    if backend is ExtractBackend.LLM:
        return {"patent": llm}
    return {"patent": HybridExtractor(PatentRuleExtractor(), llm)}


def get_extractor(doc_type: str, extractors: Mapping[str, FactExtractor] | None = None) -> FactExtractor | None:
    return (EXTRACTORS if extractors is None else extractors).get(doc_type)


def extract_facts(doc: StructuredDocument, extractor: FactExtractor | None = None) -> ExtractionResult:
    """Run the extractor for ``doc.doc_type`` and validate what it found."""
    extractor = extractor or get_extractor(doc.doc_type)
    if extractor is None:
        return ExtractionResult()
    result = extractor.extract(doc)
    result.issues.extend(validate_facts(result.facts))
    return result


__all__ = [
    "EXTRACTORS",
    "ExtractionIssue",
    "ExtractionResult",
    "FactBuilder",
    "FactExtractor",
    "HybridExtractor",
    "LlmPatentExtractor",
    "PatentRuleExtractor",
    "Rejection",
    "Source",
    "build_extractors",
    "claimed_ranges",
    "extract_facts",
    "get_extractor",
    "merge_facts",
    "validate_facts",
]
