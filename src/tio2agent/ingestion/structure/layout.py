"""Remove running headers and footers from extracted page text.

Two mechanisms:

* Patent-specific markers. pypdf puts them at the *end* of each page's text in CNIPA
  publications ("说　明　书 2/4 页", the page number, "CN 116694108 A"). They are
  removed and, where useful, returned: the marker tells which part of the patent a
  page belongs to, and the publication number is used by the front-page parser.
* Repetition. For any document, a line that appears at the top or bottom edge of
  most pages (digits ignored) is treated as a running header/footer.
"""

import re
from collections import Counter
from dataclasses import dataclass

from tio2agent.ingestion.structure.normalize import squash

# "说明书1/4页", "权利要求书1/1页", "说明书附图1/2页", "说明书摘要1/1页"
_PART_MARKER = re.compile(r"^(说明书附图|说明书摘要|摘要附图|权利要求书|说明书)(\d+)/(\d+)页$")  # 专利部分标记，如 说明书1/4页
_PUB_NUMBER = re.compile(r"^CN\d{9}[A-Z]\d?$")  # 专利申请公布号，如 CN 116694108 A
_BARE_NUMBER = re.compile(r"^\d{1,4}$")     # 纯数字，如页码

PART_KINDS = {
    "权利要求书": "claims",
    "说明书": "description",
    "说明书附图": "drawings",
    "说明书摘要": "abstract",
    "摘要附图": "abstract_drawing",
}


@dataclass
class CleanPage:
    page: int
    lines: list[str]
    part: str | None = None            # "claims", "description", ... from the footer marker
    part_index: tuple[int, int] | None = None  # (2, 4) for "说明书 2/4 页"
    publication_number: str | None = None   # CN 116694108 A


def clean_patent_page(page: int, text: str) -> CleanPage:
    """Strip CNIPA running headers/footers from both ends of the page text."""
    lines = [line.rstrip() for line in text.splitlines()]
    result = CleanPage(page=page, lines=[])

    def is_furniture(line: str) -> bool:
        """判断一行文本是否为‘排版废料’（包括页脚、页码、申请公布号等）。"""
        compact = squash(line)
        if not compact:
            return True
        if m := _PART_MARKER.match(compact):
            result.part = PART_KINDS[m.group(1)]
            result.part_index = (int(m.group(2)), int(m.group(3)))
            return True
        if _PUB_NUMBER.match(compact):
            result.publication_number = compact
            return True
        return bool(_BARE_NUMBER.match(compact))

    # 从文本末尾开始，逐行检查是否为排版废料。
    start, end = 0, len(lines)
    while end > start and is_furniture(lines[end - 1]):
        end -= 1
    # Leading furniture is rarer; don't treat a leading bare number as a page number
    # unless a marker was also found there.
    while start < end and squash(lines[start]) and (
        _PART_MARKER.match(squash(lines[start])) or _PUB_NUMBER.match(squash(lines[start]))
    ):
        is_furniture(lines[start])
        start += 1
    # if line.strip() 条件的作用：过滤掉只包含空白字符的行
    # 如果一行文本只包含空白字符，strip() 处理后会变成空字符串 ""。
    # 在 Python 中，空字符串在 if 条件下会被计算为 False，而任何非空字符串会被计算为 True。
    # 这个 if 条件确保了只有那些包含实际可见字符的行，才会最终被加入到 result.lines 列表中。
    result.lines = [line for line in lines[start:end] if line.strip()]
    return result


def strip_repeating_edges(
    pages: list[list[str]], *, edge: int = 3, min_ratio: float = 0.6, min_pages: int = 3
) -> list[list[str]]:
    """
    Drop lines that recur at the top/bottom ``edge`` lines of most pages.
    切除页眉页脚
    """
    if len(pages) < min_pages:
        return pages

    def key(line: str) -> str:
        compact = squash(line)
        # Running headers/footers are short; only there do digits (page numbers) vary.
        return re.sub(r"\d+", "#", compact) if len(compact) <= 25 else compact

    counts: Counter[str] = Counter()
    for lines in pages:
        edges = {key(l) for l in lines[:edge] + lines[-edge:] if squash(l)}
        counts.update(edges)
    threshold = max(2, int(len(pages) * min_ratio + 0.999))
    repeating = {k for k, n in counts.items() if n >= threshold}
    if not repeating:
        return pages

    cleaned = []
    for lines in pages:
        n = len(lines)
        cleaned.append(
            [
                l for i, l in enumerate(lines)
                if not ((i < edge or i >= n - edge) and key(l) in repeating)
            ]
        )
    return cleaned
