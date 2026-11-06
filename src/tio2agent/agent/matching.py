"""Lenient matching of what a model writes to what the fact store holds.

Models phrase lookups naturally ("二氧化钛用量", "pH值", "反应釜温度", "实施例一"),
while facts use short canonical names ("二氧化钛", "pH", "温度") and subjects with
Arabic numerals ("实施例1"). Exact substring matching made such lookups return
nothing, and the model concluded the document did not say. These helpers produce
fallback candidates, tried in order until one matches.
"""

import re

from tio2agent.ingestion.extraction.quantities import NAME_ALIASES
from tio2agent.ingestion.structure.normalize import chinese_number

# Endings that describe the quantity rather than name it.
_SUFFIXES = (
    "的用量", "用量", "的含量", "含量", "加入量", "添加量", "的数值", "数值", "的大小",
    "的份数", "份数", "的值", "值", "是多少", "多少",
)
# Whole phrases that mean a canonical fact name.
_SYNONYMS = {
    "反应釜温度": "温度", "反应温度": "温度", "釜温": "温度",
    "搅拌速度": "转速", "搅拌转速": "转速", "转数": "转速",
    "ph": "pH", "酸碱度": "pH",
    "钛白粉": "二氧化钛", "钛白": "二氧化钛", "tio2": "二氧化钛",
    "水": "去离子水", "纯水": "去离子水",
    "色差值": "色差", "色差": "色差",
    "耐候性": "耐候性能",
    "研磨": "研磨时间", "砂磨时间": "研磨时间",
}
# If one of these occurs inside a longer phrase, it is the name ("S1搅拌温度" -> 温度).
_KEYWORDS = ("加入时间", "搅拌时间", "研磨时间", "温度", "转速", "pH", "色差")

_CN_NUMBER = re.compile(r"(实施例|对比例|比较例|权利要求)\s*第?\s*([零一二两三四五六七八九十]+)")


def normalize_subject(subject: str | None) -> str | None:
    """"实施例一" -> "实施例1", "实施例 1" -> "实施例1", "claim 8" -> "claim:8"."""
    if not subject:
        return None
    s = re.sub(r"\s+", "", subject.strip())

    def arabic(m: re.Match) -> str:
        n = chinese_number(m.group(2))
        return f"{m.group(1)}{n}" if n is not None else m.group(0)

    s = _CN_NUMBER.sub(arabic, s)
    s = re.sub(r"(实施例|对比例|权利要求)第(\d+)", r"\1\2", s)
    if s.lower().startswith("claim") and not s.lower().startswith("claim:"):
        s = "claim:" + s[5:]
    return s


def name_candidates(name: str | None) -> list[str | None]:
    """The name as written, then shorter canonical forms, without duplicates."""
    if not name or not name.strip():
        return [None]
    written = re.sub(r"\s+", "", name.strip())
    out: list[str] = [written]

    def add(candidate: str | None) -> None:
        if candidate and candidate not in out:
            out.append(candidate)

    stripped = written
    changed = True
    while changed:  # "二氧化钛的用量是多少" -> "二氧化钛"
        changed = False
        for suffix in _SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) > len(suffix):
                stripped = stripped[: -len(suffix)]
                changed = True
    for candidate in (written, stripped):
        add(candidate)
        add(_SYNONYMS.get(candidate.lower()) or _SYNONYMS.get(candidate))
        add(NAME_ALIASES.get(candidate))
    lower = written.lower()
    for keyword in _KEYWORDS:
        if keyword.lower() in lower:
            add(keyword)
    return out
