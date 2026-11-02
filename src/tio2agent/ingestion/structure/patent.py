"""Structurer for Chinese patent publications (CNIPA 发明/实用新型).

    page 1        扉页          -> Bibliographic           (patent_front.py)
    权利要求书     claims        -> Claim[]                 (patent_claims.py)
    说明书         description   -> Section[] / Paragraph[] (this module)
                                   Embodiment[] from 实施例N / 对比例 headings
    OCR tables                   -> TableRef[], attached to the empty paragraph
                                   that holds the image (e.g. [0043])

Pages are assigned to parts by their footer marker ("说　明　书 2/4 页"), or by
content when a page has none.
"""

import re
from collections import Counter

from tio2agent.domain import (
    Claim,
    Embodiment,
    EmbodimentKind,
    Paragraph,
    ParagraphRole,
    Section,
    SectionKind,
    StructuredDocument,
    TableRef,
)
from tio2agent.ingestion.structure.base import (
    StructureInput,
    StructureIssue,
    StructureResult,
    collect_tables,
    page_text,
)
from tio2agent.ingestion.structure.layout import CleanPage, clean_patent_page
from tio2agent.ingestion.structure.normalize import join_lines, normalize_text, squash
from tio2agent.ingestion.structure.patent_claims import parse_claims
from tio2agent.ingestion.structure.patent_front import is_front_page, parse_front_page

_PARAGRAPH = re.compile(r"^\s*[\[【]\s*(\d{4})\s*[\]】]\s*(.*)$")
_CLAIM_LINE = re.compile(r"^\s*1\s*[.．、]\s*[一-鿿]")

SECTION_HEADINGS: dict[str, SectionKind] = {
    "技术领域": SectionKind.TECHNICAL_FIELD,
    "背景技术": SectionKind.BACKGROUND,
    "发明内容": SectionKind.SUMMARY,
    "实用新型内容": SectionKind.SUMMARY,
    "附图说明": SectionKind.DRAWINGS,
    "具体实施方式": SectionKind.DETAILED_DESCRIPTION,
}

# "(一)解决的技术问题", "（三）有益效果"
_SUBHEADING = re.compile(r"^[(（]([一二三四五六七八九十]+)[)）]\s*(.{1,20})$")
# "实施例1", "实施例 2：", "对比例1", "对比案例"; not "实施例1中…"
_EMBODIMENT = re.compile(r"^(实施例|对比实施例|对比例|对比案例|比较例)\s*(\d+)?\s*(?:$|[:：、.．]\s*(.*)$)")
_STEP = re.compile(r"^(S\d+|步骤\s*\d+)\s*[、.．:：]")
_BOILERPLATE = (
    "下面将结合本发明实施例",
    "下面将结合本实用新型实施例",
    "需要说明的是",
    "尽管已经示出和描述",
    "以上所述",
    "以上实施例仅用以",
    "最后应说明的是",
    "对于本领域技术人员而言，显然本发明不限于",
)


class PatentStructurer:
    name = "patent_cn"

    def structure(self, data: StructureInput) -> StructureResult:
        regions_by_id = {r["id"]: r for r in data.ocr_regions}
        issues: list[StructureIssue] = []

        texts = {p["page"]: page_text(p, regions_by_id) for p in data.pages}
        front_pages = [n for n, t in texts.items() if n == min(texts, default=1) and is_front_page(t)]
        cleaned = [clean_patent_page(n, t) for n, t in texts.items() if n not in front_pages]
        parts = _assign_parts(cleaned)

        bibliographic = None
        if front_pages:
            front = texts[front_pages[0]]
            bibliographic = parse_front_page(front)
            if bibliographic.publication_number is None:
                bibliographic.publication_number = next(
                    (c.publication_number for c in cleaned if c.publication_number), None
                )
            if not bibliographic.publication_number:
                issues.append(StructureIssue("STRUCTURE_BIBLIO_INCOMPLETE", "扉页未识别出公布号", page=front_pages[0]))

        claim_lines = [(c.page, l) for c in cleaned if parts[c.page] == "claims" for l in c.lines]
        claims, claim_issues = parse_claims(claim_lines)
        issues += claim_issues

        desc_lines = [(c.page, l) for c in cleaned if parts[c.page] == "description" for l in c.lines]
        desc_title, sections, paragraphs = _parse_description(desc_lines)
        embodiments = _group_embodiments(paragraphs)

        tables = collect_tables(data.ocr_regions)
        _attach_tables(tables, paragraphs)
        _link_claims(claims, paragraphs)

        issues += _validate(claims, paragraphs, bool(front_pages))
        title = (bibliographic.title if bibliographic else None) or desc_title

        document = StructuredDocument(
            doc_id=data.doc_id,
            doc_type=data.doc_type,
            structurer=self.name,
            title=title,
            bibliographic=bibliographic,
            claims=claims,
            sections=sections,
            paragraphs=paragraphs,
            embodiments=embodiments,
            tables=tables,
        )
        return StructureResult(document=document, issues=issues)


# --- pages ---------------------------------------------------------------------------


def _assign_parts(pages: list[CleanPage]) -> dict[int, str]:
    parts: dict[int, str] = {}
    previous = "unknown"
    for page in pages:
        part = page.part
        if part is None:
            if any(_PARAGRAPH.match(l) for l in page.lines):
                part = "description"
            elif page.lines and _CLAIM_LINE.match(page.lines[0]):
                part = "claims"
            else:
                part = previous
        parts[page.page] = part
        previous = part
    return parts


# --- description ---------------------------------------------------------------------


def _parse_description(
    lines: list[tuple[int, str]],
) -> tuple[str | None, list[Section], list[Paragraph]]:
    title: str | None = None
    section = SectionKind.OTHER
    sections: list[Section] = []
    open_paragraphs: list[tuple[str, SectionKind, list[tuple[int, str]]]] = []

    for page, line in lines:
        if m := _PARAGRAPH.match(line):
            open_paragraphs.append((m.group(1), section, [(page, m.group(2))]))
            continue
        heading = SECTION_HEADINGS.get(squash(line))
        if heading is not None:
            section = heading
            sections.append(Section(kind=heading, title=squash(line)))
            continue
        if open_paragraphs:
            open_paragraphs[-1][2].append((page, line))
        elif title is None and not sections:
            title = normalize_text(line)  # the invention title repeated above 技术领域

    paragraphs: list[Paragraph] = []
    subsection: str | None = None
    current_section: SectionKind | None = None
    for pid, sec, body in open_paragraphs:
        if sec is not current_section:
            current_section, subsection = sec, None
        raw = join_lines([l for _, l in body])
        text = normalize_text(raw)
        role = ParagraphRole.BODY
        if m := _SUBHEADING.match(text):
            role, subsection = ParagraphRole.HEADING, m.group(2)
        elif (m := _EMBODIMENT.match(text)) and not m.group(3):
            role = ParagraphRole.EMBODIMENT_HEADING
        elif text.startswith(_BOILERPLATE):
            role = ParagraphRole.BOILERPLATE
        step = _STEP.match(text)
        paragraphs.append(
            Paragraph(
                id=pid,
                section=sec,
                subsection=subsection if role is not ParagraphRole.HEADING else None,
                role=role,
                text=text,
                raw=raw,
                pages=sorted({p for p, _ in body}),
                step=re.sub(r"\s+", "", step.group(1)) if step else None,
            )
        )

    by_kind: dict[SectionKind, Section] = {}
    for s in sections:
        by_kind.setdefault(s.kind, s)
    for p in paragraphs:
        target = by_kind.get(p.section)
        if target is None:
            target = by_kind[p.section] = Section(kind=p.section, title="")
            sections.append(target)
        target.paragraph_ids.append(p.id)
        target.pages = sorted(set(target.pages) | set(p.pages))
    return title, [s for s in sections if s.paragraph_ids], paragraphs


def _group_embodiments(paragraphs: list[Paragraph]) -> list[Embodiment]:
    embodiments: list[Embodiment] = []
    current: Embodiment | None = None
    counters: Counter[EmbodimentKind] = Counter()

    for p in paragraphs:
        if p.section is not SectionKind.DETAILED_DESCRIPTION:
            current = None
            continue
        m = _EMBODIMENT.match(p.text)
        if m:
            kind = EmbodimentKind.EXAMPLE if m.group(1) == "实施例" else EmbodimentKind.COMPARATIVE
            counters[kind] += 1
            number = int(m.group(2)) if m.group(2) else None
            prefix = "E" if kind is EmbodimentKind.EXAMPLE else "C"
            current = Embodiment(
                id=f"{prefix}{number if number is not None else counters[kind]}",
                label=f"{m.group(1)}{m.group(2) or ''}",
                kind=kind,
                number=number,
                heading_id=p.id,
            )
            embodiments.append(current)
            if p.role is ParagraphRole.EMBODIMENT_HEADING:
                p.embodiment = current.id
                current.pages = sorted(set(current.pages) | set(p.pages))
                continue
        elif p.role is ParagraphRole.BOILERPLATE:
            current = None
        if current is not None:
            p.embodiment = current.id
            current.paragraph_ids.append(p.id)
            current.pages = sorted(set(current.pages) | set(p.pages))
    return embodiments


# --- links -----------------------------------------------------------------------------


def _attach_tables(tables: list[TableRef], paragraphs: list[Paragraph]) -> None:
    """Attach each OCR table to an empty paragraph on the same page, in order."""
    used: set[str] = set()
    for table in tables:
        for p in paragraphs:
            if p.id not in used and not p.text and table.page in p.pages:
                table.paragraph_id = p.id
                p.table_ids.append(table.id)
                used.add(p.id)
                break


# A range is one value: "45-65" and "45" differ, "50℃-80℃" is a single token.
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[℃%]?[-~]\d+(?:\.\d+)?)?")


def _bigrams(text: str) -> set[str]:
    chars = re.sub(r"[^\w]", "", text)
    return {chars[i : i + 2] for i in range(len(chars) - 1)}


_CHARACTERIZING = re.compile(r"其特征(?:在于|是)\s*[:：，,]?")


def _link_claims(claims: list[Claim], paragraphs: list[Paragraph]) -> None:
    """Mark description paragraphs that restate a claim.

    The claim's characterizing part (after "其特征在于") is compared with each body
    paragraph: same wording (bigram containment >= 0.85, against the shorter text) and
    every number of the shorter text present in the longer one. Embodiment paragraphs
    are never linked: "研磨20分钟" is an instance of "研磨15-25分钟", not a restatement,
    even though 20 happens to be an endpoint of the claimed range.
    """
    candidates = [
        (p, _bigrams(p.text), set(_NUMBER.findall(p.text)))
        for p in paragraphs
        if p.role is ParagraphRole.BODY and p.embodiment is None and len(p.text) >= 8
    ]
    for claim in claims:
        parts = _CHARACTERIZING.split(claim.text, maxsplit=1)
        body = parts[1] if len(parts) == 2 and len(parts[1]) >= 8 else claim.text
        c_grams, c_nums = _bigrams(body), set(_NUMBER.findall(body))
        for p, p_grams, p_nums in candidates:
            short_grams, short_nums, long_nums = (
                (p_grams, p_nums, c_nums) if len(p_grams) <= len(c_grams) else (c_grams, c_nums, p_nums)
            )
            if not short_grams:
                continue
            overlap = len(c_grams & p_grams) / len(short_grams)
            if overlap >= 0.85 and short_nums <= long_nums:
                claim.related_paragraphs.append(p.id)


# --- validation ------------------------------------------------------------------------


def _validate(claims: list[Claim], paragraphs: list[Paragraph], has_front: bool) -> list[StructureIssue]:
    issues: list[StructureIssue] = []
    if not claims:
        issues.append(StructureIssue("STRUCTURE_NO_CLAIMS", "未识别出权利要求书"))
    if not paragraphs:
        issues.append(StructureIssue("STRUCTURE_NO_DESCRIPTION", "未识别出说明书段落"))
        return issues

    numbers = [int(p.id) for p in paragraphs]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        issues.append(
            StructureIssue("STRUCTURE_PARAGRAPH_DUPLICATE", f"段落编号重复：{[f'{n:04d}' for n in duplicates]}")
        )
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    if missing:
        shown = ", ".join(f"[{n:04d}]" for n in missing[:10]) + (" 等" if len(missing) > 10 else "")
        issues.append(StructureIssue("STRUCTURE_PARAGRAPH_GAP", f"缺少段落 {shown}，可能是图片或提取失败"))

    for p in paragraphs:
        if not p.text and not p.table_ids:
            issues.append(
                StructureIssue(
                    "STRUCTURE_EMPTY_PARAGRAPH",
                    f"段落[{p.id}]没有文字，内容可能是图片（启用 OCR 后可识别）",
                    page=p.pages[0],
                    paragraph_id=p.id,
                )
            )
    return issues
