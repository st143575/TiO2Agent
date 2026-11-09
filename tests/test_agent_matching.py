"""Lenient name / subject matching used by query_facts and get_formulation."""

import pytest

from tio2agent.agent.matching import chinese_number, name_candidates, normalize_subject


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("二氧化钛用量", ["二氧化钛用量", "二氧化钛"]),
        ("二氧化钛的用量是多少", ["二氧化钛的用量是多少", "二氧化钛"]),
        ("pH值", ["pH值", "pH"]),
        ("PH", ["PH", "pH"]),
        ("反应釜温度", ["反应釜温度", "温度"]),
        ("实施例2的S1反应釜温度", ["实施例2的S1反应釜温度", "温度"]),
        ("TiO2", ["TiO2", "二氧化钛"]),
        ("钛白粉含量", ["钛白粉含量", "钛白粉", "二氧化钛"]),
        ("研磨", ["研磨", "研磨时间"]),
        ("锆盐加入时间", ["锆盐加入时间", "加入时间"]),
        ("色差值", ["色差值", "色差"]),
        ("温度", ["温度"]),
        (None, [None]),
        ("  ", [None]),
    ],
)
def test_name_candidates(written, expected) -> None:
    assert name_candidates(written) == expected


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("实施例一", "实施例1"),
        ("实施例 3", "实施例3"),
        ("实施例第2", "实施例2"),
        ("实施例十二", "实施例12"),
        ("权利要求八", "权利要求8"),
        ("对比例一", "对比例1"),
        ("claim 8", "claim:8"),
        ("E1", "E1"),
        (None, None),
    ],
)
def test_normalize_subject(written, expected) -> None:
    assert normalize_subject(written) == expected


def test_chinese_number() -> None:
    assert [chinese_number(s) for s in ("三", "十", "十二", "二十", "二十一", "百", "一二", "")] == [
        3, 10, 12, 20, 21, None, None, None,
    ]
