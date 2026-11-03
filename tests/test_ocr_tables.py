import pytest

from ocr_fakes import P6_TABLE_HTML
from tio2agent.ingestion.ocr import normalize_cell, parse_table, table_to_markdown
from tio2agent.ingestion.ocr.tables import parse_html_table, parse_otsl


def test_html_table_from_pipeline() -> None:
    assert parse_table(P6_TABLE_HTML) == [
        ["统计对象 统计项目", "实施例1", "实施例2", "实施例3"],
        ["老化试验色差值", "1.21", "1.35", "1.46"],
        ["耐候性能", "最佳", "一般", "较差"],
    ]


def test_html_spans_are_expanded() -> None:
    html = (
        "<table>"
        "<tr><th rowspan=\"2\">项目</th><th colspan=\"2\">实施例</th></tr>"
        "<tr><th>1</th><th>2</th></tr>"
        "<tr><td>pH</td><td>7.5</td><td>8.0</td></tr>"
        "</table>"
    )
    assert parse_html_table(html) == [
        ["项目", "实施例", "实施例"],
        ["项目", "1", "2"],
        ["pH", "7.5", "8.0"],
    ]


def test_html_entities_line_breaks_and_nested_tables() -> None:
    html = (
        "<table><tr><td>A &amp; B<br/>C</td>"
        "<td><table><tr><td>nested</td></tr></table>outer</td></tr></table>"
    )
    assert parse_html_table(html) == [["A & B C", "outer"]]


def test_otsl_with_merged_cells() -> None:
    otsl = (
        "<fcel>项目<fcel>实施例<lcel><nl>"
        "<ucel><fcel>1<fcel>2<nl>"
        "<fcel>色差<fcel>1.21<ecel><nl>"
    )
    assert parse_otsl(otsl) == [
        ["项目", "实施例", "实施例"],
        ["项目", "1", "2"],
        ["色差", "1.21", ""],
    ]


def test_parse_table_dispatch() -> None:
    assert parse_table("<fcel>a<fcel>b<nl>") == [["a", "b"]]
    assert parse_table("just text") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1. 21", "1.21"), ("１．２１", "1.21"), ("统计对象 统计项目", "统计对象统计项目"), (" 最佳\n", "最佳")],
)
def test_normalize_cell(raw: str, expected: str) -> None:
    assert normalize_cell(raw) == expected


def test_table_to_markdown() -> None:
    assert table_to_markdown([["a", "b|c"], ["1", "2"]]) == "| a | b\\|c |\n| --- | --- |\n| 1 | 2 |"
