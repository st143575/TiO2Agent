"""Numbers, ranges, units and names as written in Chinese patents."""

import re

import pytest

from tio2agent.ingestion.extraction.quantities import (
    QUANTITY,
    canonical_name,
    canonical_unit,
    normalize_chemical_name,
    parse_number,
    parse_quantity,
)


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("45-65", 45, 65),
        ("50℃-80℃", 50, 80),
        ("0.5-0.9", 0.5, 0.9),
        ("5~7", 5, 7),
        ("5至7", 5, 7),
        ("55", 55, 55),
        ("65-45", 45, 65),  # reversed range is normalized
    ],
)
def test_parse_quantity(text: str, low: float, high: float) -> None:
    m = re.fullmatch(rf"{QUANTITY}℃?", text)  # the trailing unit is matched by the caller
    assert m, text
    q = parse_quantity(m.group("low"), m.group("high"))
    assert (q.low, q.high) == (low, high)
    assert q.is_range is (low != high)
    assert q.value == (None if low != high else low)


def test_units_are_canonical() -> None:
    assert canonical_unit("重量份") == "份"
    assert canonical_unit("转/分钟") == "rpm"
    assert canonical_unit("分钟") == "min"
    assert canonical_unit("小时") == "h"
    assert canonical_unit("Pa") == "Pa"  # unknown units pass through
    assert canonical_unit(None) is None


def test_aliases_keep_the_written_name() -> None:
    assert canonical_name("去离子") == ("去离子水", "去离子")
    assert canonical_name("去离子水") == ("去离子水", None)


def test_parse_number_and_chemical_names() -> None:
    assert parse_number("1.21") == 1.21
    assert parse_number("1. 21") == 1.21
    assert parse_number("最佳") is None
    assert normalize_chemical_name("2，6-二叔丁基甲酚") == "2,6-二叔丁基甲酚"
