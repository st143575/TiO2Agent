"""Heading and step styles that differ between patents.

Unit tests for each rule, then the three further sample patents end to end. Each of
them broke a rule written for CN116694108A:

    CN115403940A  "实施例一" (Chinese numerals), "对比实验", boilerplate [0040]
    CN114573021A  "步骤1：" steps, results table + summary after the last example
    CN115466527A  (already worked; guards against regressions)
"""

import tempfile
from pathlib import Path

import pytest

from conftest import SAMPLES_DIR
from tio2agent.domain import DocumentRecord, EmbodimentKind, Paragraph, ParagraphRole, SectionKind
from tio2agent.ingestion.extraction import extract_facts
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.structure import StructureInput, structure_document
from tio2agent.ingestion.structure.normalize import chinese_number, parse_number, step_id
from tio2agent.ingestion.structure.patent import _BOILERPLATE, _EMBODIMENT, _group_embodiments

NYLON = SAMPLES_DIR / "一种水分散性好的锦纶化纤用钛白粉及其 制备方法.pdf"      # CN115403940A
ACRYLIC = SAMPLES_DIR / "一种过滤性好的腈纶化纤用钛白粉及其生 产方法.pdf"      # CN115466527A
MIXED = SAMPLES_DIR / "一种高纯度混晶型钛白粉及其生产方法.pdf"               # CN114573021A


# --- rules ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "number"),
    [
        ("实施例1", "实施例", 1),
        ("实施例 2：", "实施例", 2),
        ("实施例1：", "实施例", 1),
        ("实施例一", "实施例", 1),
        ("实施例十二", "实施例", 12),
        ("具体实施例3", "实施例", 3),
        ("对比例1", "对比例", 1),
        ("对比案例", "对比案例", None),
        ("对比实验", "对比实验", None),
        ("对比试验", "对比试验", None),
        ("比较例二", "比较例", 2),
    ],
)
def test_embodiment_headings(text: str, kind: str, number: int | None) -> None:
    m = _EMBODIMENT.match(text)
    assert m and m.group(1) == kind and not m.group(3)
    assert (parse_number(m.group(2)) if m.group(2) else None) == number


@pytest.mark.parametrize("text", ["实施例1中二氧化钛用量最少", "实施例一中", "对比实验结果表明…", "本实施例提供"])
def test_references_are_not_headings(text: str) -> None:
    m = _EMBODIMENT.match(text)
    assert m is None or m.group(3)


@pytest.mark.parametrize(
    ("text", "step"),
    [
        ("S1、将55份二氧化钛…", "S1"),
        ("S 2：研磨", "S2"),
        ("步骤1：预处理，将钛铁矿…", "S1"),
        ("步骤 5、煅烧", "S5"),
        ("步骤三：除杂质", "S3"),
        ("第一步，将钛铁矿…", "S1"),
        ("第2步：酸解", "S2"),
        ("步骤1中预处理后的钛铁矿…", None),   # a reference, not a step marker
        ("所述步骤2中酸解时候…", None),
        ("(1)、该高耐候钛白粉…", None),       # numbered benefits, not steps
    ],
)
def test_step_markers(text: str, step: str | None) -> None:
    assert step_id(text) == step


def test_chinese_numbers() -> None:
    assert [chinese_number(s) for s in ("一", "十", "十二", "二十", "二十一", "百", "")] == [1, 10, 12, 20, 21, None, None]
    assert parse_number("7") == 7 and parse_number("七") == 7


def test_boilerplate_openings() -> None:
    for text in (
        "同时本说明书中未作详细描述的内容均属于本领域技术人员公知的现有技术。",
        "以上显示和描述了本发明的基本原理和主要特征和本发明的优点。",
        "在本发明的描述中，需要理解的是，术语“上”、“下”……",
        "下面将对本发明实施例中的技术方案进行清楚、完整地描述。",
    ):
        assert text.startswith(_BOILERPLATE)


def _p(pid: str, text: str, role: ParagraphRole = ParagraphRole.BODY) -> Paragraph:
    return Paragraph(id=pid, section=SectionKind.DETAILED_DESCRIPTION, role=role, text=text, raw=text, pages=[5])


def test_example_ends_before_results_for_several_examples() -> None:
    paragraphs = [
        _p("0010", "实施例1", ParagraphRole.EMBODIMENT_HEADING),
        _p("0011", "步骤1：预处理"),
        _p("0012", "所述步骤2中酸解时候先进行空压"),
        _p("0013", ""),                                         # results table (image)
        _p("0014", "上表为对本发明实施例1 -实施例3制备得到的钛白粉的检测结果"),
        _p("0015", "对比例1", ParagraphRole.EMBODIMENT_HEADING),
        _p("0016", "取实施例1、实施例2和实施例3的产品各10份"),  # comparatives may discuss all examples
    ]
    e1, c1 = _group_embodiments(paragraphs)
    assert (e1.id, e1.paragraph_ids) == ("E1", ["0011", "0012"])
    assert paragraphs[3].embodiment is None and paragraphs[4].embodiment is None
    assert (c1.id, c1.kind, c1.paragraph_ids) == ("C1", EmbodimentKind.COMPARATIVE, ["0016"])


# --- the three further sample patents ----------------------------------------------------


def _structure(pdf: Path):
    ctx = IngestionContext(
        record=DocumentRecord(id="d", filename=pdf.name, sha256="x", size_bytes=1),
        pdf_path=pdf,
        work_dir=Path(tempfile.mkdtemp()),
    )
    ParseStage().run(ctx)
    return structure_document(StructureInput("d", "patent", ctx.artifacts["pages"], [])).document


def _embodiments(doc) -> list[tuple]:
    return [(e.id, e.label, e.heading_id, e.paragraph_ids[0], e.paragraph_ids[-1]) for e in doc.embodiments]


@pytest.mark.skipif(not NYLON.exists(), reason="sample PDF not present")
def test_nylon_patent_chinese_numerals() -> None:
    doc = _structure(NYLON)
    assert doc.bibliographic.publication_number == "CN115403940A"
    assert _embodiments(doc) == [
        ("E1", "实施例一", "0025", "0026", "0029"),
        ("E2", "实施例二", "0030", "0031", "0034"),
        ("E3", "实施例三", "0035", "0036", "0039"),
        ("C1", "对比实验", "0041", "0042", "0043"),
    ]
    assert doc.paragraph("0040").role is ParagraphRole.BOILERPLATE
    assert [p.step for p in doc.paragraphs if p.embodiment == "E2" and p.step] == ["S1", "S2", "S3", "S4"]
    # The examples now reach extraction. (Many component names come out wrong here,
    # e.g. "去离子水中", "有机硅进行混合": a limit of the rule extractor, not of the
    # structure, so only a correctly named component is checked.)
    facts = extract_facts(doc).facts
    assert {f.subject for f in facts if f.subject.startswith("E")} == {"E1", "E2", "E3"}
    acid = [(f.subject, f.step, f.value) for f in facts if f.name == "枸橼酸" and f.subject.startswith("E")]
    assert acid == [("E1", "S3", 4), ("E2", "S3", 5), ("E3", "S3", 6)]


@pytest.mark.skipif(not MIXED.exists(), reason="sample PDF not present")
def test_mixed_crystal_patent_numbered_steps() -> None:
    doc = _structure(MIXED)
    assert _embodiments(doc) == [
        ("E1", "实施例1", "0022", "0023", "0029"),
        ("E2", "实施例2", "0030", "0031", "0037"),
        ("E3", "实施例3", "0038", "0039", "0046"),
    ]
    # the results table [0047] and its summary [0048] are not part of 实施例3
    assert doc.paragraph("0047").embodiment is None and doc.paragraph("0048").embodiment is None
    assert doc.paragraph("0019").role is ParagraphRole.BOILERPLATE   # "下面将对本发明实施例…"
    assert doc.paragraph("0020").role is ParagraphRole.BOILERPLATE   # "在本发明的描述中…"
    assert [p.step for p in doc.paragraphs if p.embodiment == "E1" and p.step] == ["S1", "S2", "S3", "S4", "S5"]
    facts = extract_facts(doc).facts
    assert [(f.subject, f.step, f.name, f.value) for f in facts if f.name == "pH" and f.subject.startswith("E")] == [
        ("E1", "S4", "pH", 2), ("E2", "S4", "pH", 3), ("E3", "S4", "pH", 3),
    ]


@pytest.mark.skipif(not ACRYLIC.exists(), reason="sample PDF not present")
def test_acrylic_patent_unchanged() -> None:
    doc = _structure(ACRYLIC)
    assert _embodiments(doc) == [
        ("E1", "实施例1", "0026", "0027", "0030"),
        ("E2", "实施例2", "0031", "0032", "0035"),
        ("E3", "实施例3", "0036", "0037", "0040"),
        ("C1", "对比案例", "0041", "0042", "0043"),
    ]
    result = extract_facts(doc)
    assert len(result.facts) == 105
    # the document itself gives no amount of 沉淀剂 in examples 2 and 3 ("加入沉淀剂")
    assert [i.message for i in result.issues] == [
        "实施例2：缺少权利要求1列出的组分：沉淀剂",
        "实施例3：缺少权利要求1列出的组分：沉淀剂",
    ]
