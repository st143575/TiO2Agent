"""Structured representation of a parsed document.

Produced by the structuring stage and saved as ``structure.json``; read by later
stages (chunking, fact extraction) and by the agent's tools. Every piece of text
keeps where it came from (paragraph id, page), so answers can cite "说明书 [0028]".
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class SectionKind(StrEnum):
    TECHNICAL_FIELD = "technical_field"          # 技术领域
    BACKGROUND = "background"                    # 背景技术
    SUMMARY = "summary"                          # 发明内容 / 实用新型内容
    DRAWINGS = "drawings"                        # 附图说明
    DETAILED_DESCRIPTION = "detailed_description"  # 具体实施方式
    BODY = "body"                                # generic documents: text under a heading
    OTHER = "other"


class ParagraphRole(StrEnum):
    BODY = "body"
    HEADING = "heading"                          # e.g. [0004] (一)解决的技术问题
    EMBODIMENT_HEADING = "embodiment_heading"    # e.g. [0025] 实施例1
    BOILERPLATE = "boilerplate"                  # closing legal text, low retrieval value


class EmbodimentKind(StrEnum):
    EXAMPLE = "example"          # 实施例
    COMPARATIVE = "comparative"  # 对比例 / 对比案例


class ClaimKind(StrEnum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"


class ClaimCategory(StrEnum):
    PRODUCT = "product"
    METHOD = "method"
    USE = "use"


class Bibliographic(BaseModel):
    """Front-page data of a patent, keyed by INID code where one exists."""

    publication_number: str | None = None   # 申请公布号 (10) / (45) e.g. CN116694108A
    publication_date: str | None = None     # 申请公布日 (43) / (45) ISO date
    application_number: str | None = None   # 申请号 (21)
    application_date: str | None = None     # 申请日 (22) ISO date
    kind_text: str | None = None            # 专利类型 (12) e.g. 发明专利申请
    applicants: list[str] = Field(default_factory=list)  # 申请人 (71) / (73)
    address: str | None = None              # 地址
    inventors: list[str] = Field(default_factory=list)   # 发明人 (72)
    agency: str | None = None               # 专利代理机构 (74)
    agents: list[str] = Field(default_factory=list)     # 专利代理师
    ipc: list[str] = Field(default_factory=list)          # 国际专利分类号(International Patent Classification, IPC) (51) e.g. Int. Cl.
    title: str | None = None                # 发明名称 (54)
    abstract: str | None = None             # 摘要 (57)
    page_counts: dict[str, int] = Field(default_factory=dict)  # {"权利要求书": 1, "说明书": 4}


class Paragraph(BaseModel):
    id: str                                 # "0028" for patents, "P0001" for generic documents
    section: SectionKind
    subsection: str | None = None           # e.g. "技术方案" under 发明内容
    role: ParagraphRole = ParagraphRole.BODY
    text: str                               # normalized text
    raw: str                                # text as extracted, for audit
    pages: list[int]
    embodiment: str | None = None           # Embodiment.id
    step: str | None = None                 # e.g. "S1"
    table_ids: list[str] = Field(default_factory=list)


class Section(BaseModel):
    kind: SectionKind
    title: str
    paragraph_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)


class Embodiment(BaseModel):
    id: str                                 # "E1", "E2", ... ; "C1" for comparatives
    label: str                              # as written: "实施例1", "对比案例"
    kind: EmbodimentKind
    number: int | None = None
    heading_id: str | None = None           # paragraph holding the heading
    paragraph_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)


class Claim(BaseModel):
    number: int
    text: str
    raw: str
    pages: list[int]
    kind: ClaimKind
    category: ClaimCategory
    refers_to: list[int] = Field(default_factory=list)
    # Description paragraphs that restate this claim (near-duplicate text).
    related_paragraphs: list[str] = Field(default_factory=list)


class TableRef(BaseModel):
    id: str                                 # OCR region id, e.g. "p6-img0"
    page: int
    paragraph_id: str | None = None         # paragraph the table belongs to, if known
    source: str = "ocr"
    rows: list[list[str]]
    markdown: str


class StructuredDocument(BaseModel):
    """
    The structured representation of a document.
    This model represents the structured document stored in the database.
    Currently, it consists of 15 fields.
    结构化文档对象，承载对 PDF 文本进行深度 NLP/正则解析后得到的业务语义结构。
    只在后台解析流水线（Ingestion Pipeline）的内存中短暂或作为中间产物（Artifact）存在。
    包含高度业务化的结构化信息，如权利要求、实施例、表格等。
    """
    schema_version: int = 1
    doc_id: str
    doc_type: str
    structurer: str
    title: str | None = None
    bibliographic: Bibliographic | None = None
    claims: list[Claim] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    embodiments: list[Embodiment] = Field(default_factory=list)
    tables: list[TableRef] = Field(default_factory=list)

    def paragraph(self, paragraph_id: str) -> Paragraph | None:
        return next((p for p in self.paragraphs if p.id == paragraph_id), None)
