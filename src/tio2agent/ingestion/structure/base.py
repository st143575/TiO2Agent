"""Structurer interface and helpers shared by all document types."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from tio2agent.domain import StructuredDocument, TableRef
from tio2agent.ingestion.ocr import table_to_markdown


@dataclass
class StructureIssue:
    code: str
    message: str
    page: int | None = None
    paragraph_id: str | None = None


@dataclass
class StructureInput:
    doc_id: str
    doc_type: str
    pages: list[dict[str, Any]]                # ParseStage output (+ "ocr" ids from OcrStage)
    ocr_regions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StructureResult:
    document: StructuredDocument
    issues: list[StructureIssue] = field(default_factory=list)


class DocumentStructurer(Protocol):
    name: str

    def structure(self, data: StructureInput) -> StructureResult: ...


def page_text(page: dict[str, Any], regions_by_id: dict[str, dict[str, Any]]) -> str:
    """Text of a page: the text layer, or OCR text for pages that have none (scans).

    Tables are excluded here; they are attached separately as TableRefs.
    """
    if not page.get("low_text"):
        return page.get("text") or ""
    parts = []
    for region_id in page.get("ocr") or []:
        region = regions_by_id.get(region_id) or {}
        for block in region.get("blocks") or []:
            if block.get("label") != "table" and block.get("content", "").strip():
                parts.append(block["content"])
    return "\n".join(parts) if parts else page.get("text") or ""


def collect_tables(ocr_regions: list[dict[str, Any]]) -> list[TableRef]:
    """One TableRef per recognized table, not yet attached to a paragraph."""
    tables = []
    for region in ocr_regions:
        grids = [b["table"] for b in region.get("blocks") or [] if b.get("table")]
        for index, rows in enumerate(grids):
            table_id = region["id"] if len(grids) == 1 else f"{region['id']}-t{index}"
            tables.append(
                TableRef(id=table_id, page=region["page"], rows=rows, markdown=table_to_markdown(rows))
            )
    return tables
