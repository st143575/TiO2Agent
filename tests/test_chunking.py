"""Structure-aware chunking on the sample patent and on constructed documents."""

import tempfile
from pathlib import Path

import pytest

from conftest import SAMPLES_DIR
from ocr_fakes import P6_TABLE_HTML
from tio2agent.domain import ChunkKind, DocumentRecord, Paragraph, SectionKind, StructuredDocument
from tio2agent.ingestion.chunking import ChunkingOptions, chunk_document
from tio2agent.ingestion.ocr import parse_table
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.structure import StructureInput, structure_document

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
P6_REGION = {"id": "p6-img0", "page": 6, "blocks": [{"label": "table", "content": P6_TABLE_HTML, "table": parse_table(P6_TABLE_HTML)}]}


def sample_structure(with_ocr: bool) -> StructuredDocument:
    record = DocumentRecord(id="doc_t", filename=SAMPLE.name, sha256="x", size_bytes=1)
    ctx = IngestionContext(record=record, pdf_path=SAMPLE, work_dir=Path(tempfile.mkdtemp()))
    ParseStage().run(ctx)
    regions = [P6_REGION] if with_ocr else []
    return structure_document(StructureInput("doc_t", "patent", ctx.artifacts["pages"], regions)).document


@pytest.fixture(scope="module")
def sample_chunks():
    return chunk_document(sample_structure(with_ocr=True))


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")
class TestSamplePatent:
    @pytest.fixture
    def chunks(self, sample_chunks):
        return sample_chunks

    def test_layout(self, chunks) -> None:
        got = [(c.kind, c.citation) for c in chunks]
        assert got == [
            (ChunkKind.BIBLIOGRAPHIC, "扉页著录项"),
            (ChunkKind.ABSTRACT, "摘要"),
            *[(ChunkKind.CLAIM, f"权利要求{n}") for n in range(1, 9)],
            (ChunkKind.PARAGRAPH, "说明书[0001]"),
            (ChunkKind.PARAGRAPH, "说明书[0002]-[0003]"),
            (ChunkKind.PARAGRAPH, "说明书[0005]"),
            (ChunkKind.PARAGRAPH, "说明书[0007]-[0013]"),
            (ChunkKind.PARAGRAPH, "说明书[0014]-[0018]"),  # lead-in [0014] stays with its steps
            (ChunkKind.PARAGRAPH, "说明书[0020]-[0022]"),
            (ChunkKind.PARAGRAPH, "说明书[0024]"),
            (ChunkKind.EMBODIMENT, "说明书[0026]-[0029]"),
            (ChunkKind.EMBODIMENT, "说明书[0031]-[0034]"),
            (ChunkKind.EMBODIMENT, "说明书[0036]-[0039]"),
            (ChunkKind.EMBODIMENT, "说明书[0041]-[0043]"),
        ]
        assert [c.id for c in chunks][:2] == ["doc_t-0001", "doc_t-0002"]
        assert all(c.doc_label == "CN116694108A" for c in chunks)

    def test_headings_and_boilerplate_are_not_chunks(self, chunks) -> None:
        ids = {pid for c in chunks for pid in c.paragraph_ids}
        assert {"0004", "0006", "0019", "0025", "0030", "0035", "0040"}.isdisjoint(ids)  # headings
        assert {"0023", "0044", "0045"}.isdisjoint(ids)                                   # boilerplate

    def test_breadcrumbs(self, chunks) -> None:
        by_cite = {c.citation: c for c in chunks}
        assert by_cite["说明书[0007]-[0013]"].context == "一种高耐候钛白粉及制备方法 > 发明内容 > 技术方案"
        assert by_cite["说明书[0026]-[0029]"].context == "一种高耐候钛白粉及制备方法 > 具体实施方式 > 实施例1"
        assert by_cite["说明书[0026]-[0029]"].embodiment == "E1"
        assert by_cite["权利要求8"].context.endswith("权利要求书 > 权利要求8")

    def test_restated_claims(self, chunks) -> None:
        by_cite = {c.citation: c for c in chunks}
        assert by_cite["说明书[0007]-[0013]"].restates_claims == [1, 2, 3, 4, 5, 6, 7]
        assert by_cite["说明书[0014]-[0018]"].restates_claims == [8]
        assert by_cite["说明书[0026]-[0029]"].restates_claims == []  # embodiments never restate

    def test_table_inlined_after_its_caption(self, chunks) -> None:
        c = next(c for c in chunks if c.citation == "说明书[0041]-[0043]")
        assert c.table_ids == ["p6-img0"]
        caption = c.text.index("如下表所示")
        table = c.text.index("| 老化试验色差值 | 1.21 | 1.35 | 1.46 |")
        assert caption < table

    def test_bibliographic_text(self, chunks) -> None:
        bib = chunks[0].text
        assert "申请人：淮安市鸿洋钛业有限公司" in bib
        assert "发明人：陆台芬、杨文赞、陈长华、陆明、刘超、胡飞、韩小艳" in bib
        assert "IPC分类号：C09C 1/36；C09C 3/04；C09C 3/06；C09C 3/08" in bib

    def test_without_ocr_no_table(self) -> None:
        chunks = chunk_document(sample_structure(with_ocr=False))
        c = next(c for c in chunks if c.citation == "说明书[0041]-[0043]")
        assert "|" not in c.text and c.table_ids == []


def _doc(paragraphs: list[Paragraph], **extra) -> StructuredDocument:
    return StructuredDocument(doc_id="d", doc_type="other", structurer="generic", title="规程", paragraphs=paragraphs, **extra)


def _p(pid: str, text: str, subsection: str | None = "1 范围") -> Paragraph:
    return Paragraph(id=pid, section=SectionKind.BODY, subsection=subsection, text=text, raw=text, pages=[1])


def test_long_paragraph_is_split_at_sentence_ends() -> None:
    sentence = "包膜温度应控制在六十摄氏度至七十摄氏度之间。"  # 22 chars
    doc = _doc([_p("P0001", sentence * 10)])
    chunks = chunk_document(doc, ChunkingOptions(max_chars=100))
    assert len(chunks) == 3
    assert all(c.paragraph_ids == ["P0001"] for c in chunks)
    assert all(len(c.text) <= 100 and c.text.endswith("。") for c in chunks)
    assert "".join(c.text for c in chunks) == sentence * 10


def test_merging_respects_size_and_subsection() -> None:
    doc = _doc([
        _p("P0001", "甲" * 40), _p("P0002", "乙" * 40), _p("P0003", "丙" * 40),
        _p("P0004", "丁" * 10, subsection="2 术语"),
    ])
    chunks = chunk_document(doc, ChunkingOptions(max_chars=100))
    assert [c.paragraph_ids for c in chunks] == [["P0001", "P0002"], ["P0003"], ["P0004"]]
    assert [c.citation for c in chunks] == ["[P0001]、[P0002]", "[P0003]", "[P0004]"]
    assert chunks[0].context == "规程 > 1 范围" and chunks[2].context == "规程 > 2 术语"


def test_orphan_table_becomes_its_own_chunk() -> None:
    from tio2agent.domain import TableRef

    table = TableRef(id="p2-img0", page=2, rows=[["a", "b"]], markdown="| a | b |\n| --- | --- |")
    chunks = chunk_document(_doc([_p("P0001", "正文。")], tables=[table]))
    assert chunks[-1].kind is ChunkKind.TABLE
    assert chunks[-1].citation == "第2页表格"
