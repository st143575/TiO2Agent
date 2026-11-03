"""Structuring rules on constructed text: cases the sample patent does not contain."""

from tio2agent.domain import ClaimCategory, ClaimKind, SectionKind
from tio2agent.ingestion.structure import StructureInput, structure_document
from tio2agent.ingestion.structure.patent_claims import parse_claims
from tio2agent.ingestion.structure.patent_front import parse_front_page


def lines(text: str, page: int = 2) -> list[tuple[int, str]]:
    return [(page, l) for l in text.strip().splitlines()]


# --- claims -----------------------------------------------------------------------


def test_claim_reference_forms() -> None:
    claims, issues = parse_claims(lines("""
1 .一种包膜钛白粉，其特征在于：包括二氧化钛。
2 .根据权利要求1所述的钛白粉，其特征在于：包膜剂为氧化铝。
3 .如权利要求1或2所述的钛白粉，其特征在于：分散剂为
0 .5份六偏磷酸钠。
4 .根据权利要求1至3中任一项所述的钛白粉，其特征在于：粒径为
5 .0 μm。
5 .根据权利要求2、4所述的钛白粉在涂料中的应用。
"""))
    assert [c.number for c in claims] == [1, 2, 3, 4, 5]
    assert claims[2].text.endswith("分散剂为0.5份六偏磷酸钠。")  # "0 .5份" is not claim 0
    assert claims[3].text.endswith("粒径为5.0 μm。")              # "5 .0 μm" is not claim 5
    assert [c.refers_to for c in claims] == [[], [1], [1, 2], [1, 2, 3], [2, 4]]
    assert claims[4].category is ClaimCategory.USE
    assert claims[4].kind is ClaimKind.INDEPENDENT  # use claim referring to product claims
    assert issues == []


def test_claim_referring_forward_is_reported() -> None:
    claims, issues = parse_claims(lines("""
1 .根据权利要求2所述的方法，其特征在于：搅拌。
2 .一种方法，其特征在于：加热。
"""))
    assert [i.code for i in issues] == ["STRUCTURE_CLAIM_REF_INVALID"]


def test_out_of_order_number_is_continuation() -> None:
    claims, _ = parse_claims(lines("""
1 .一种钛白粉，其特征在于：
3 .批次混合。
2 .根据权利要求1所述的钛白粉，其特征在于：干燥。
"""))
    assert [c.number for c in claims] == [1, 2]
    assert claims[0].text.endswith("批次混合。")  # the stray "3 ." line stays in claim 1


# --- front page --------------------------------------------------------------------


def test_granted_patent_front_page() -> None:
    bib = parse_front_page("""
(19)国家知识产权局
(12)发明专利
(10)授权公告号 CN 112345678 B
(45)授权公告日 2024.01.09
(21)申请号 202011223344.X
(22)申请日 2020.11.03
(73)专利权人 某某钛业股份有限公司；某某大学
地址 400000 重庆市
(72)发明人 张三　李四
(51)Int.Cl.
C01G 23/047(2006.01)
(54)发明名称
一种氯化法钛白粉的
制备方法
(57)摘要
本发明公开了一种方法。
权利要求书2页  说明书6页  附图1页
""")
    assert bib.publication_number == "CN112345678B"
    assert bib.publication_date == "2024-01-09"
    assert bib.application_number == "202011223344.X"
    assert bib.applicants == ["某某钛业股份有限公司", "某某大学"]
    assert bib.address == "400000 重庆市"
    assert bib.inventors == ["张三", "李四"]
    assert bib.ipc == ["C01G 23/047"]
    assert bib.title == "一种氯化法钛白粉的制备方法"
    assert bib.abstract == "本发明公开了一种方法。"
    assert bib.page_counts == {"权利要求书": 2, "说明书": 6, "附图": 1}


# --- description gaps, fallback, generic documents -------------------------------------


def _patent_pages(description: str) -> list[dict]:
    return [
        {"page": 1, "text": "(19)国家知识产权局\n(54)发明名称\n一种钛白粉\n(57)摘要\n一种钛白粉。", "char_count": 30, "low_text": False},
        {"page": 2, "text": "1 .一种钛白粉，其特征在于：二氧化钛。\n权　利　要　求　书 1/1 页\n2", "char_count": 30, "low_text": False},
        {"page": 3, "text": description + "\n说　明　书 1/1 页\n3", "char_count": 99, "low_text": False},
    ]


def test_paragraph_gap_and_duplicate_are_reported() -> None:
    pages = _patent_pages("技术领域\n[0001] 钛白粉。\n[0003] 背景。\n[0003] 重复。")
    result = structure_document(StructureInput("d", "patent", pages))
    codes = {i.code: i.message for i in result.issues}
    assert "[0002]" in codes["STRUCTURE_PARAGRAPH_GAP"]
    assert "0003" in codes["STRUCTURE_PARAGRAPH_DUPLICATE"]


def test_fallback_to_generic_when_nothing_patent_like() -> None:
    pages = [{"page": 1, "text": "第一章 总则\n本规程适用于包膜工序。\n第二章 参数\n温度为60℃。", "char_count": 30, "low_text": False}]
    result = structure_document(StructureInput("d", "patent", pages))
    assert result.document.structurer == "generic"
    assert result.issues[0].code == "STRUCTURE_FALLBACK"


def test_generic_document() -> None:
    header = "ACME 包膜工序作业指导书"
    pages = [
        {"page": 1, "text": f"{header}\n1 范围\n本文件规定了钛白粉包膜工序的操作要求，适用于\n连续化生产线。\n2 工艺参数\n包膜温度控制在 60 ℃‑70 ℃。\n第 1 页", "low_text": False},
        {"page": 2, "text": f"{header}\n2.1 pH 控制\n加入铝盐后将PH值调节至 6 . 5。\n第 2 页", "low_text": False},
        {"page": 3, "text": f"{header}\n3 检验\n白度按国家标准测定。\n第 3 页", "low_text": False},
    ]
    doc = structure_document(StructureInput("d", "other", pages)).document
    assert doc.structurer == "generic"
    assert [s.title for s in doc.sections] == ["1 范围", "2 工艺参数", "2.1 pH 控制", "3 检验"]
    assert [p.text for p in doc.paragraphs] == [
        "本文件规定了钛白粉包膜工序的操作要求，适用于连续化生产线。",
        "包膜温度控制在60℃-70℃。",
        "加入铝盐后将pH值调节至6.5。",
        "白度按国家标准测定。",
    ]
    assert [p.id for p in doc.paragraphs] == ["P0001", "P0002", "P0003", "P0004"]
    assert doc.paragraphs[2].pages == [2] and doc.paragraphs[2].subsection == "2.1 pH 控制"
    assert all(p.section is SectionKind.BODY for p in doc.paragraphs)


def test_scanned_page_uses_ocr_text() -> None:
    pages = [{"page": 1, "text": "", "char_count": 0, "low_text": True, "ocr": ["p1-img0"]}]
    regions = [{"id": "p1-img0", "page": 1, "blocks": [
        {"label": "paragraph_title", "content": "一、原料"},
        {"label": "text", "content": "二氧化钛含量不低于93%。"},
    ]}]
    doc = structure_document(StructureInput("d", "other", pages, regions)).document
    assert [s.title for s in doc.sections] == ["一、原料"]
    assert doc.paragraphs[0].text == "二氧化钛含量不低于93%。"
