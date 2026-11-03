"""Structured fact extraction: StructuredDocument -> list[Fact] (+ validation).

One extractor per document type. Only patents have one so far; other types are
skipped (no facts, no warning). An LLM-based extractor can be registered the same
way once the Agent module adds an LLM client.
"""

from tio2agent.domain import StructuredDocument
from tio2agent.ingestion.extraction.base import (
    ExtractionIssue,
    ExtractionResult,
    FactBuilder,
    FactExtractor,
    Source,
)
from tio2agent.ingestion.extraction.patent_rules import PatentRuleExtractor
from tio2agent.ingestion.extraction.validators import claimed_ranges, validate_facts

EXTRACTORS: dict[str, FactExtractor] = {"patent": PatentRuleExtractor()}


def get_extractor(doc_type: str) -> FactExtractor | None:
    return EXTRACTORS.get(doc_type)


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
    "PatentRuleExtractor",
    "Source",
    "claimed_ranges",
    "extract_facts",
    "get_extractor",
    "validate_facts",
]
