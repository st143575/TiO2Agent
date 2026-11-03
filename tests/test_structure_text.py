import re

import pytest

from tio2agent.ingestion.structure.layout import clean_patent_page, strip_repeating_edges
from tio2agent.ingestion.structure.normalize import join_lines, normalize_text, squash

# Each case is copied from the text pypdf extracts from the sample patent.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("分散剂0 . 5‑0 .9份、包膜剂25‑35份", "分散剂0.5-0.9份、包膜剂25-35份"),
        ("将反应釜的温度设为50 ℃‑80℃，", "将反应釜的温度设为50℃-80℃，"),
        ("去离子水 40 ‑50份、 分散剂", "去离子水40-50份、分散剂"),
        ("加入二氧化钛 、去离子水、分散剂", "加入二氧化钛、去离子水、分散剂"),
        ("某高耐候钛白粉的 加工厂随机选取30 名工作人员", "某高耐候钛白粉的加工厂随机选取30名工作人员"),
        ("将PH值调节至5 ‑7，", "将pH值调节至5-7，"),
        ("所述抗氧化剂为2， 6‑二叔丁基甲酚", "所述抗氧化剂为2，6-二叔丁基甲酚"),
        ("搅拌时间为10 ‑20分钟", "搅拌时间为10-20分钟"),
        ("C09C 1/36(2006.01)", "C09C 1/36(2006.01)"),
        ("2023.04.21", "2023.04.21"),
        ("研磨15～25分钟", "研磨15~25分钟"),
        ("温度为 1–3 h", "温度为1-3 h"),
        ("PHB 材料", "PHB材料"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_join_lines() -> None:
    assert join_lines(["二氧化钛的化学性质极为稳定，是一种", "偏酸性的两性氧化物"]) == (
        "二氧化钛的化学性质极为稳定，是一种偏酸性的两性氧化物"
    )
    assert join_lines(["the rutile", "phase"]) == "the rutile phase"
    steps = re.compile(r"^S\d+、")
    assert join_lines(["具体包括以下步骤：", "S1、混合", "均匀；", "S2、研磨"], hard_break=steps) == (
        "具体包括以下步骤：\nS1、混合均匀；\nS2、研磨"
    )


def test_squash() -> None:
    assert squash("说　明　书 2/4 页") == "说明书2/4页"


def test_clean_patent_page_strips_trailing_furniture() -> None:
    text = "[0044] 需要说明的是\n所固有的要素。\n说　明　书 4/4 页\n6\nCN 116694108 A\n6"
    page = clean_patent_page(6, text)
    assert page.lines == ["[0044] 需要说明的是", "所固有的要素。"]
    assert page.part == "description"
    assert page.part_index == (4, 4)
    assert page.publication_number == "CN116694108A"


def test_clean_patent_page_claims_marker() -> None:
    page = clean_patent_page(2, "1 .一种钛白粉。\n权　利　要　求　书 1/1 页\n2\nCN 116694108 A\n2")
    assert page.part == "claims"
    assert page.lines == ["1 .一种钛白粉。"]


def test_content_lines_ending_in_numbers_survive() -> None:
    # Only the trailing furniture block is removed, not numbers inside the body.
    page = clean_patent_page(4, "[0028] 搅拌\n6分钟时\n说　明　书 2/4 页\n4")
    assert page.lines == ["[0028] 搅拌", "6分钟时"]


def test_strip_repeating_edges() -> None:
    bodies = [
        ["1 范围", "本规程规定了金红石型钛白粉包膜工序的操作要求。", "适用于连续化生产线。"],
        ["2 术语", "包膜：在二氧化钛颗粒表面沉积无机氧化物的过程。", "固含量以质量分数计。"],
        ["3 工艺参数", "包膜温度控制在60℃至70℃之间。", "pH值按工艺卡执行。"],
        ["4 检验", "白度按国家标准方法测定。", "每批次留样不少于500克。"],
    ]
    pages = [["ACME 技术规程", *body, f"第 {i} 页 共 4 页"] for i, body in enumerate(bodies, 1)]
    cleaned = strip_repeating_edges(pages)
    assert cleaned == bodies  # header and "第 n 页 共 4 页" removed, body kept
    assert strip_repeating_edges(pages[:2]) == pages[:2]  # too few pages to judge
