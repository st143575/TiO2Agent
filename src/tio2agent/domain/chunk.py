"""Retrieval units built from a StructuredDocument.

A chunk is what search returns and what the LLM reads. It keeps enough provenance to
cite it ("说明书 [0026]-[0029]", "权利要求 8") and to collapse duplicates: description
paragraphs that only restate claims carry ``restates_claims``.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkKind(StrEnum):
    """
    Chunk types for retrieval and LLM context.
    This class adopts semantic chunking (按语义结构切块).
    """
    BIBLIOGRAPHIC = "bibliographic"  # 整个扉页（申请人、发明人、申请日、专利代理机构、IPC分类号等）作为一个独立 Chunk
    ABSTRACT = "abstract"            # 摘要作为一个独立的 Chunk
    CLAIM = "claim"                  # 每一条权利要求严格对应一个独立的 Chunk
    PARAGRAPH = "paragraph"          # 说明书里普通的业务段落
    EMBODIMENT = "embodiment"        # 每个实施例/对比例被打包成一个 Chunk（非常关键！防止实验数据或配方在分块时被强行截断，导致大模型读取到残缺的实验步骤）
    TABLE = "table"                  # 无法归属到段落的表格作为一个独立的 Chunk


class Chunk(BaseModel):
    id: str                          # chunk ID, e.g. "{doc_id}-0007"
    doc_id: str                      # document ID
    doc_label: str                   # publication number or title, for display
    order: int                       # reading order within the document (为了后续能够让 LLM 按照原文顺序阅读检索到的多个 Chunk)
    kind: ChunkKind                  # chunk type
    text: str                        # content shown to the LLM
    context: str = ""                # breadcrumb(面包屑导航): "一种高耐候钛白粉及制备方法 > 具体实施方式 > 实施例1"；这解决了 RAG 里的经典痛点：当一段文字太短时，向量模型会丢失它的全局上下文。
    section: str | None = None       # SectionKind value
    embodiment: str | None = None    # Embodiment id, e.g. "E1"
    paragraph_ids: list[str] = Field(default_factory=list)
    claim_numbers: list[int] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    restates_claims: list[int] = Field(default_factory=list)    # 如果一段说明书只是把权利要求重复了一遍，这个字段会记录对应的权利要求号。在检索时，系统就可以借此过滤掉重复信息。

    @property
    def citation(self) -> str:
        """Human-readable source, e.g. "说明书[0026]-[0029]" or "权利要求8"."""
        if self.kind is ChunkKind.CLAIM:
            return "权利要求" + "、".join(str(n) for n in self.claim_numbers)
        if self.kind is ChunkKind.BIBLIOGRAPHIC:
            return "扉页著录项"
        if self.kind is ChunkKind.ABSTRACT:
            return "摘要"
        if self.paragraph_ids:
            prefix = "说明书" if self.paragraph_ids[0].isdigit() else ""
            return prefix + _id_ranges(self.paragraph_ids)
        if self.table_ids:
            return f"第{self.pages[0]}页表格" if self.pages else "表格"
        return f"第{self.pages[0]}页" if self.pages else ""

    def search_text(self) -> str:
        """
        Text that is indexed: the breadcrumb gives short chunks their topic.
        这是实际喂给向量数据库（Embedding 模型）的字符串。它巧妙地把“面包屑导航（context）”拼在了正文（text）前面。
        例如，原文只有“研磨20分钟”几个字，拼上上下文后变成了“XXX专利 > 具体实施方式 > 实施例3 \n 研磨20分钟”。
        这极大地提升了向量检索的命中率和准确度。
        """
        return f"{self.context}\n{self.text}" if self.context else self.text


def _id_ranges(ids: list[str]) -> str:
    """["0026","0027","0028","0030"] -> "[0026]-[0028]、[0030]"."""
    if not all(i.isdigit() for i in ids):
        return "、".join(f"[{i}]" for i in ids)
    numbers = sorted(int(i) for i in ids)
    width = len(ids[0])
    parts, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"[{start:0{width}d}]" if start == prev else f"[{start:0{width}d}]-[{prev:0{width}d}]")
        if n is not None:
            start = prev = n
    return "、".join(parts)
