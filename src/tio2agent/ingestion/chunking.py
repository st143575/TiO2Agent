"""Structure-aware chunking: StructuredDocument -> Chunk[].

Chunks follow the document's own units instead of a fixed window:

    扉页著录项                       -> 1 bibliographic chunk
    摘要                             -> 1 abstract chunk
    each claim                       -> 1 claim chunk
    each 实施例 / 对比例              -> 1 embodiment chunk (split only if very long)
    other description paragraphs     -> merged within the same section and subsection
                                        up to ``max_chars``; never across a boundary
    tables                           -> inlined where they sit (after "如下表所示"),
                                        or a table chunk if they have no paragraph

Headings ("(二)技术方案", "实施例1") are not chunks of their own; they appear in the
chunk's breadcrumb (``context``). Boilerplate paragraphs are not indexed.
Paragraphs longer than ``max_chars`` are split at sentence ends.
"""

import re
from dataclasses import dataclass

from tio2agent.domain import (
    Chunk,
    ChunkKind,
    Paragraph,
    ParagraphRole,
    SectionKind,
    StructuredDocument,
)

SECTION_TITLES = {
    SectionKind.TECHNICAL_FIELD: "技术领域",
    SectionKind.BACKGROUND: "背景技术",
    SectionKind.SUMMARY: "发明内容",
    SectionKind.DRAWINGS: "附图说明",
    SectionKind.DETAILED_DESCRIPTION: "具体实施方式",
}
_SENTENCE = re.compile(r"(?<=[。；！？])")
_LEAD_IN = ("：", ":")  # "具体包括以下步骤：" introduces what follows


@dataclass(frozen=True)
class ChunkingOptions:
    max_chars: int = 600            # soft limit for merged paragraphs
    embodiment_max_chars: int = 1500  # an embodiment stays whole up to this size
    include_boilerplate: bool = False


def chunk_document(doc: StructuredDocument, options: ChunkingOptions | None = None) -> list[Chunk]:
    options = options or ChunkingOptions()
    builder = _Builder(doc)
    _front(doc, builder)
    _claims(doc, builder)
    _description(doc, builder, options)
    _orphan_tables(doc, builder)
    return builder.chunks


class _Builder:
    def __init__(self, doc: StructuredDocument) -> None:
        self.doc = doc
        b = doc.bibliographic
        self.label = (b.publication_number if b and b.publication_number else None) or doc.title or doc.doc_id
        self.title = doc.title or self.label
        self.chunks: list[Chunk] = []

    def add(self, kind: ChunkKind, text: str, *, context: str, **fields) -> None:
        if not text.strip():
            return
        order = len(self.chunks) + 1
        self.chunks.append(
            Chunk(
                id=f"{self.doc.doc_id}-{order:04d}",
                doc_id=self.doc.doc_id,
                doc_label=self.label,
                order=order,
                kind=kind,
                text=text.strip(),
                context=context,
                **fields,
            )
        )

    def crumb(self, *parts: str | None) -> str:
        return " > ".join(p for p in (self.title, *parts) if p)


# --- front page and claims -------------------------------------------------------------


def _front(doc: StructuredDocument, b: _Builder) -> None:
    bib = doc.bibliographic
    if bib is None:
        return
    fields = [
        ("发明名称", bib.title),
        ("公布号", bib.publication_number),
        ("公布日", bib.publication_date),
        ("申请号", bib.application_number),
        ("申请日", bib.application_date),
        ("类型", bib.kind_text),
        ("申请人", "；".join(bib.applicants)),
        ("地址", bib.address),
        ("发明人", "、".join(bib.inventors)),
        ("专利代理机构", bib.agency),
        ("代理人", "、".join(bib.agents)),
        ("IPC分类号", "；".join(bib.ipc)),
    ]
    text = "\n".join(f"{k}：{v}" for k, v in fields if v)
    b.add(ChunkKind.BIBLIOGRAPHIC, text, context=b.crumb("著录项"), pages=[1])
    if bib.abstract:
        b.add(ChunkKind.ABSTRACT, bib.abstract, context=b.crumb("摘要"), pages=[1])


def _claims(doc: StructuredDocument, b: _Builder) -> None:
    for claim in doc.claims:
        b.add(
            ChunkKind.CLAIM,
            claim.text,
            context=b.crumb("权利要求书", f"权利要求{claim.number}"),
            claim_numbers=[claim.number],
            pages=claim.pages,
        )


# --- description -------------------------------------------------------------------------


def _description(doc: StructuredDocument, b: _Builder, options: ChunkingOptions) -> None:
    tables = {t.id: t for t in doc.tables}
    embodiments = {e.id: e for e in doc.embodiments}
    restated = {pid: claim.number for claim in doc.claims for pid in claim.related_paragraphs}

    def content(p: Paragraph) -> str:
        """
        获取段落内容，包括文本和表格。
        如果段落p下面挂载了表格，则把表格的 Markdown 文本拼接在段落文字的后面。
        """
        parts = [p.text] if p.text else []
        parts += [tables[t].markdown for t in p.table_ids if t in tables]
        return "\n".join(parts)

    group: list[Paragraph] = []
    group_key: tuple | None = None

    def flush() -> None:
        nonlocal group
        if not group:
            return
        first = group[0]
        section_title = SECTION_TITLES.get(first.section) or (first.subsection if first.section is SectionKind.BODY else None)
        if first.embodiment:
            emb = embodiments.get(first.embodiment)
            kind, context = ChunkKind.EMBODIMENT, b.crumb(section_title, emb.label if emb else first.embodiment)
        else:
            sub = first.subsection if first.section is not SectionKind.BODY else None
            kind, context = ChunkKind.PARAGRAPH, b.crumb(section_title, sub)

        limit = options.embodiment_max_chars if first.embodiment else options.max_chars
        for piece in _split_group(group, content, limit):
            claims = sorted({restated[p.id] for p in piece if p.id in restated})
            body = [p for p in piece if p.text and not p.text.endswith(_LEAD_IN)]
            fully_restated = bool(body) and all(p.id in restated for p in body)
            b.add(
                kind,
                "\n".join(c for p in piece if (c := content(p))),
                context=context,
                section=first.section.value,
                embodiment=first.embodiment,
                paragraph_ids=[p.id for p in piece],
                table_ids=[t for p in piece for t in p.table_ids],
                pages=sorted({pg for p in piece for pg in p.pages}),
                restates_claims=claims if fully_restated else [],
            )
        group = []

    for p in doc.paragraphs:
        if p.role is ParagraphRole.BOILERPLATE and not options.include_boilerplate:
            flush()
            group_key = None
            continue
        if p.role in (ParagraphRole.HEADING, ParagraphRole.EMBODIMENT_HEADING):
            flush()  # the heading text is in the breadcrumb (subsection / embodiment label)
            group_key = None
            continue
        key = (p.section, p.subsection, p.embodiment)
        if key != group_key:
            flush()
            group_key = key
        elif not p.embodiment and (
            p.text.endswith(_LEAD_IN)
            or sum(len(content(x)) for x in group) + len(content(p)) > options.max_chars
        ):
            flush()  # a lead-in starts a new group: it belongs with what follows
        group.append(p)
    flush()


def _split_group(group: list[Paragraph], content, limit: int) -> list[list[Paragraph]]:
    """
    Keep the group whole if it fits; otherwise one paragraph per piece. A single
    paragraph over the limit is split at sentence ends into pseudo-paragraphs that
    keep the original id.
    当发现一个段落组超过了限制长度，它放弃合并，尝试把组内的段落逐个切成块。
    """
    total = sum(len(content(p)) for p in group)
    if total <= limit:
        return [group]
    pieces: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    size = 0
    for p in group:
        text = content(p)
        if len(text) > limit:
            if current:
                pieces.append(current)
                current, size = [], 0
            pieces += [[p.model_copy(update={"text": s, "table_ids": []})] for s in _sentences(p.text, limit)]
            if p.table_ids:
                pieces[-1][0] = pieces[-1][0].model_copy(update={"table_ids": p.table_ids})
            continue
        if current and size + len(text) > limit:
            pieces.append(current)
            current, size = [], 0
        current.append(p)
        size += len(text)
    if current:
        pieces.append(current)
    return pieces


def _sentences(text: str, limit: int) -> list[str]:
    """
    如果连单独的一个段落都超过了限制，利用正则表达式 _SENTENCE，在句号、叹号、问号处切开。
    它把这段超长文字按句子分成了几个虚拟的（pseudo）短段落。这保证了即使在极端情况下，系统也不会把半句话硬生生截断。
    """
    out, buf = [], ""
    for sentence in (s for s in _SENTENCE.split(text) if s):
        if buf and len(buf) + len(sentence) > limit:
            out.append(buf)
            buf = ""
        buf += sentence
    if buf:
        out.append(buf)
    return out


def _orphan_tables(doc: StructuredDocument, b: _Builder) -> None:
    """
    文档中有些表格是独立的，四周没有文字说明（之前解析时没挂载到任何段落上）。
    这段代码专门把这些表格捞出来，每一个都做成一个独立的 ChunkKind.TABLE 文本块存入库中。
    """
    for table in doc.tables:
        if table.paragraph_id is None:
            b.add(
                ChunkKind.TABLE,
                table.markdown,
                context=b.crumb(f"第{table.page}页表格"),
                table_ids=[table.id],
                pages=[table.page],
            )
