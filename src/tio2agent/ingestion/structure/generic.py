"""Fallback structurer for documents without a known layout (standards, papers, SOPs).

Best effort, layout-agnostic:

* running headers/footers are removed by repetition across pages;
* a short line that looks like a heading ("第三章 工艺控制", "4.2 包膜", "二、原料")
  starts a new section;
* a paragraph ends where a line ends with sentence punctuation (。！？；：) or is
  clearly shorter than the page's typical line (the last line of a paragraph).

Paragraph ids are "P0001", "P0002", ... in reading order.
"""

import re
import statistics

from tio2agent.domain import Paragraph, Section, SectionKind, StructuredDocument
from tio2agent.ingestion.structure.base import (
    StructureInput,
    StructureIssue,
    StructureResult,
    collect_tables,
    page_text,
)
from tio2agent.ingestion.structure.layout import strip_repeating_edges
from tio2agent.ingestion.structure.normalize import join_lines, normalize_text, squash

_HEADING = re.compile(
    r"^(?:第[一二三四五六七八九十百零\d]+[章节部分篇]"
    r"|[一二三四五六七八九十]+[、.．]"
    r"|\d+(?:\.\d+){0,3}\s+(?=\D)"
    r"|[(（][一二三四五六七八九十]+[)）])"
)
_SENTENCE_END = re.compile(r"[。！？；：:!?;]$")


class GenericStructurer:
    name = "generic"

    def structure(self, data: StructureInput) -> StructureResult:
        regions_by_id = {r["id"]: r for r in data.ocr_regions}
        numbers = [p["page"] for p in data.pages]
        raw_pages = [[l for l in page_text(p, regions_by_id).splitlines() if l.strip()] for p in data.pages]
        pages = strip_repeating_edges(raw_pages)

        sections: list[Section] = []
        paragraphs: list[Paragraph] = []
        current = Section(kind=SectionKind.BODY, title="")
        buffer: list[tuple[int, str]] = []

        def flush() -> None:
            if not buffer:
                return
            raw = join_lines([l for _, l in buffer])
            pid = f"P{len(paragraphs) + 1:04d}"
            pages_of = sorted({p for p, _ in buffer})
            paragraphs.append(
                Paragraph(
                    id=pid,
                    section=SectionKind.BODY,
                    subsection=current.title or None,
                    text=normalize_text(raw),
                    raw=raw,
                    pages=pages_of,
                )
            )
            current.paragraph_ids.append(pid)
            current.pages = sorted(set(current.pages) | set(pages_of))
            buffer.clear()

        for number, lines in zip(numbers, pages, strict=True):
            lengths = [len(squash(l)) for l in lines if squash(l)]
            typical = statistics.median(lengths) if lengths else 0
            for line in lines:
                compact = squash(line)
                if _is_heading(line, compact):
                    flush()
                    if current.paragraph_ids:
                        sections.append(current)
                    # Keep the space after the number: "2.1 pH 控制", not "2.1pH控制".
                    current = Section(kind=SectionKind.BODY, title=" ".join(line.split()))
                    continue
                buffer.append((number, line))
                if _SENTENCE_END.search(compact) or (typical and len(compact) < 0.6 * typical):
                    flush()
        flush()
        if current.paragraph_ids:
            sections.append(current)

        issues: list[StructureIssue] = []
        # Pages without a text layer are already reported by OcrStage (LOW_TEXT_PAGE / OCR_*).
        if not paragraphs and not all(p.get("low_text") for p in data.pages):
            issues.append(StructureIssue("STRUCTURE_NO_TEXT", "文档中没有可识别的文字"))
        document = StructuredDocument(
            doc_id=data.doc_id,
            doc_type=data.doc_type,
            structurer=self.name,
            title=sections[0].title if sections and sections[0].title else None,
            sections=sections,
            paragraphs=paragraphs,
            tables=collect_tables(data.ocr_regions),
        )
        return StructureResult(document=document, issues=issues)


def _is_heading(line: str, compact: str) -> bool:
    return (
        0 < len(compact) <= 30
        and bool(_HEADING.match(line.strip()))
        and not _SENTENCE_END.search(compact)
        and not compact.endswith(("，", ","))
    )
