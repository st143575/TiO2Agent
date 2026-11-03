"""PatentStructurer on the sample patent CN116694108A.

Every expected value below was checked by hand against the PDF text.
"""

import tempfile
from pathlib import Path

import pytest

from conftest import SAMPLES_DIR
from ocr_fakes import P6_TABLE_HTML
from tio2agent.domain import (
    ClaimCategory,
    ClaimKind,
    DocumentRecord,
    EmbodimentKind,
    ParagraphRole,
    SectionKind,
    StructuredDocument,
)
from tio2agent.ingestion.ocr import parse_table
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.structure import StructureInput, StructureResult, structure_document

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")

P6_REGION = {
    "id": "p6-img0",
    "page": 6,
    "blocks": [{"label": "table", "content": P6_TABLE_HTML, "table": parse_table(P6_TABLE_HTML)}],
}


def _pages() -> list[dict]:
    record = DocumentRecord(id="doc_t", filename=SAMPLE.name, sha256="x", size_bytes=1)
    ctx = IngestionContext(record=record, pdf_path=SAMPLE, work_dir=Path(tempfile.mkdtemp()))
    ParseStage().run(ctx)
    return ctx.artifacts["pages"]


@pytest.fixture(scope="module")
def pages() -> list[dict]:
    return _pages()


@pytest.fixture(scope="module")
def result(pages) -> StructureResult:
    return structure_document(StructureInput("doc_t", "patent", pages, []))


@pytest.fixture(scope="module")
def doc(result) -> StructuredDocument:
    return result.document


def text(doc: StructuredDocument, pid: str) -> str:
    paragraph = doc.paragraph(pid)
    assert paragraph is not None, pid
    return paragraph.text


def test_bibliographic(doc: StructuredDocument) -> None:
    b = doc.bibliographic
    assert doc.structurer == "patent_cn"
    assert b.publication_number == "CN116694108A"
    assert b.publication_date == "2023-09-05"
    assert b.application_number == "202310432662.5"
    assert b.application_date == "2023-04-21"
    assert b.kind_text == "发明专利申请"
    assert b.applicants == ["淮安市鸿洋钛业有限公司"]
    assert b.address == "223000 江苏省淮安市淮安区经济开发区"
    assert b.inventors == ["陆台芬", "杨文赞", "陈长华", "陆明", "刘超", "胡飞", "韩小艳"]
    assert b.agency == "盐城汇聪知识产权代理事务所(普通合伙) 32581"
    assert b.agents == ["王金燃"]
    assert b.ipc == ["C09C 1/36", "C09C 3/04", "C09C 3/06", "C09C 3/08"]
    assert b.title == doc.title == "一种高耐候钛白粉及制备方法"
    assert b.abstract.startswith("本发明公开了一种高耐候钛白粉及制备方法，所述高耐候钛白粉的原料按重量份比包括：")
    assert "分散剂0.5-0.9份" in b.abstract
    assert b.abstract.endswith("提高其耐候性、抗粉化性和保光性。")
    assert b.page_counts == {"权利要求书": 1, "说明书": 4}


def test_claims(doc: StructuredDocument) -> None:
    assert [c.number for c in doc.claims] == list(range(1, 9))
    assert all(c.pages == [2] for c in doc.claims)

    c1, c8 = doc.claims[0], doc.claims[7]
    assert (c1.kind, c1.category, c1.refers_to) == (ClaimKind.INDEPENDENT, ClaimCategory.PRODUCT, [])
    assert "分散剂0.5-0.9份" in c1.text and "0 . 5" in c1.raw
    for c in doc.claims[1:7]:
        assert (c.kind, c.category, c.refers_to) == (ClaimKind.DEPENDENT, ClaimCategory.PRODUCT, [1])
    # A method claim referring to product claims is an independent claim of another category.
    assert (c8.kind, c8.category, c8.refers_to) == (ClaimKind.INDEPENDENT, ClaimCategory.METHOD, [1, 2, 3, 4, 5, 6, 7])
    assert c8.text.count("\n") == 4  # preamble + steps S1..S4 on their own lines
    assert "将pH值调节至5-7" in c8.text
    assert doc.claims[6].text.endswith("所述抗氧化剂为2，6-二叔丁基甲酚、丁基羟基茴香醚或叔丁基对苯二酚。")


def test_claims_linked_to_restating_paragraphs(doc: StructuredDocument) -> None:
    links = {c.number: c.related_paragraphs for c in doc.claims}
    assert links == {
        1: ["0007"], 2: ["0008"], 3: ["0009"], 4: ["0010"],
        5: ["0011"], 6: ["0012"], 7: ["0013"],
        8: ["0015", "0016", "0017", "0018"],
    }


def test_sections(doc: StructuredDocument) -> None:
    spans = [(s.kind, s.title, s.paragraph_ids[0], s.paragraph_ids[-1], s.pages) for s in doc.sections]
    assert spans == [
        (SectionKind.TECHNICAL_FIELD, "技术领域", "0001", "0001", [3]),
        (SectionKind.BACKGROUND, "背景技术", "0002", "0003", [3]),
        (SectionKind.SUMMARY, "发明内容", "0004", "0022", [3, 4]),
        (SectionKind.DETAILED_DESCRIPTION, "具体实施方式", "0023", "0045", [4, 5, 6]),
    ]


def test_paragraphs(doc: StructuredDocument) -> None:
    assert [p.id for p in doc.paragraphs] == [f"{n:04d}" for n in range(1, 46)]

    # Normalization of the extraction artefacts
    assert "分散剂0.5-0.9份" in text(doc, "0007")
    assert "温度设为50℃-80℃" in text(doc, "0015") and "转速设为150-250转/分钟" in text(doc, "0015")
    assert "随机选取30名工作人员在相同的时间以及条件下" in text(doc, "0041")
    assert text(doc, "0026").startswith("S1、以重量份计将55份二氧化钛、45份去离子水、0.7份分散剂、0.4份消泡剂和0.5份抗氧化剂")

    # [0014] continues on page 3 only; [0028] is wrapped over several lines
    assert doc.paragraph("0014").pages == [3]
    assert text(doc, "0028").endswith("将pH值调节至5；")

    # Subsections of 发明内容
    assert doc.paragraph("0004").role is ParagraphRole.HEADING
    assert doc.paragraph("0005").subsection == "解决的技术问题"
    assert doc.paragraph("0007").subsection == "技术方案"
    assert doc.paragraph("0021").subsection == "有益效果"
    assert doc.paragraph("0021").role is ParagraphRole.BODY  # "(1)、该高耐候…" is content

    assert [doc.paragraph(f"00{n}").step for n in (15, 16, 17, 18)] == ["S1", "S2", "S3", "S4"]
    assert [p.id for p in doc.paragraphs if p.role is ParagraphRole.BOILERPLATE] == ["0023", "0044", "0045"]
    assert text(doc, "0043") == ""


def test_embodiments(doc: StructuredDocument) -> None:
    got = [(e.id, e.label, e.kind, e.number, e.heading_id, e.paragraph_ids, e.pages) for e in doc.embodiments]
    assert got == [
        ("E1", "实施例1", EmbodimentKind.EXAMPLE, 1, "0025", ["0026", "0027", "0028", "0029"], [4]),
        ("E2", "实施例2", EmbodimentKind.EXAMPLE, 2, "0030", ["0031", "0032", "0033", "0034"], [5]),
        ("E3", "实施例3", EmbodimentKind.EXAMPLE, 3, "0035", ["0036", "0037", "0038", "0039"], [5]),
        ("C1", "对比案例", EmbodimentKind.COMPARATIVE, None, "0040", ["0041", "0042", "0043"], [5, 6]),
    ]
    assert doc.paragraph("0042").embodiment == "C1"
    assert doc.paragraph("0044").embodiment is None
    # "本实施例1中制备的…" mentions an embodiment but is not a heading
    assert doc.paragraph("0042").role is ParagraphRole.BODY


def test_without_ocr_the_image_paragraph_is_reported(result: StructureResult) -> None:
    assert [(i.code, i.page, i.paragraph_id) for i in result.issues] == [
        ("STRUCTURE_EMPTY_PARAGRAPH", 6, "0043")
    ]
    assert result.document.tables == []


def test_ocr_table_is_attached_to_paragraph_0043(pages) -> None:
    result = structure_document(StructureInput("doc_t", "patent", pages, [P6_REGION]))
    doc = result.document
    assert result.issues == []
    (table,) = doc.tables
    assert (table.id, table.page, table.paragraph_id) == ("p6-img0", 6, "0043")
    assert table.rows[1] == ["老化试验色差值", "1.21", "1.35", "1.46"]
    assert table.markdown.startswith("| 统计对象 统计项目 | 实施例1 |")
    assert doc.paragraph("0043").table_ids == ["p6-img0"]
    assert doc.paragraph("0043").embodiment == "C1"  # the results table belongs to 对比案例


def test_round_trip_json(doc: StructuredDocument) -> None:
    assert StructuredDocument.model_validate_json(doc.model_dump_json()) == doc
