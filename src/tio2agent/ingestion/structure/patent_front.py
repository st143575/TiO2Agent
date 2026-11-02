"""Front page (扉页) of a CNIPA patent publication: bibliographic data by INID code.

    (21)申请号 202310432662.5          -> application_number
    (22)申请日 2023.04.21              -> application_date "2023-04-21"
    (71)申请人 淮安市鸿洋钛业有限公司  -> applicants, followed by 地址 ...
    (72)发明人 陆台芬　杨文赞 ...       -> inventors (separated by full-width spaces)
    (51)Int.Cl. C09C 1/36(2006.01) ... -> ipc
    (54)发明名称 / (57)摘要             -> title / abstract

In the sample, the (10) publication number and (43) publication date lines are
empty in the extracted text; their values sit at the end of the page
("CN 116694108 A", "2023.09.05"). They are recovered from there.
"""

import re

from tio2agent.domain import Bibliographic
from tio2agent.ingestion.structure.normalize import join_lines, normalize_text, squash

_INID = re.compile(r"^[(（](\d{2})[)）]\s*(.*)$")  # 专利文献著录项目识别代码(INID)模式
_DATE = re.compile(r"(\d{4})\s*[.\-/年]\s*(\d{1,2})\s*[.\-/月]\s*(\d{1,2})")  # 日期模式
_PUB_NUMBER = re.compile(r"CN\s?(\d{9})\s?([A-Z]\d?)")  # 专利申请公布号模式
_IPC = re.compile(r"[A-H]\d{2}[A-Z]\s*\d{1,4}/\d{1,6}")  # 国际专利分类号(IPC)模式
_COUNTS = re.compile(r"(权利要求书|说明书附图|说明书|附图)\s*(\d+)\s*页")  # 页数模式
_LABELS = re.compile(r"^(申请公布号|申请公布日|授权公告号|授权公告日|申请号|申请日|申请人|专利权人|发明人|专利代理机构|发明名称|摘要|Int\.?\s*Cl\.?)")  # INID代码后的标签的模式
_NAME_SPLIT = re.compile(r"[\s　;；,，、]+")  # 人名分割模式


def is_front_page(text: str) -> bool:
    """
    判断是否为专利扉页
    """
    compact = squash(text)
    return "(19)国家知识产权局" in compact or "(54)发明名称" in compact or "(57)摘要" in compact


def _iso(text: str) -> str | None:
    """
    将日期转换为ISO格式，即 YYYY-MM-DD
    """
    m = _DATE.search(text)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def _is_tail_line(line: str) -> bool:
    """
    判断是否为尾部行，即在摘要之后的行，包括页数、公告号、公告日等
    """
    compact = squash(line)
    return bool(
        _COUNTS.match(compact)
        or re.fullmatch(r"CN\d{9}[A-Z]\d?", compact)
        or re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}", compact)
    )


def parse_front_page(text: str) -> Bibliographic:
    blocks: dict[str, list[str]] = {}
    tail: list[str] = []
    current: str | None = None

    for line in text.splitlines():
        if not line.strip():
            continue
        if _is_tail_line(line):
            tail.append(line)
            continue
        if m := _INID.match(line.strip()):
            current = m.group(1)
            blocks.setdefault(current, [])
            remainder = _LABELS.sub("", m.group(2).strip()).strip()
            if remainder:
                blocks[current].append(remainder)
        elif current is not None:
            blocks[current].append(line)

    bib = Bibliographic()
    get = lambda code: blocks.get(code, [])  # noqa: E731

    bib.kind_text = normalize_text("".join(get("12"))) or None
    if m := _PUB_NUMBER.search(" ".join(get("10") + get("45"))):
        bib.publication_number = f"CN{m.group(1)}{m.group(2)}"
    bib.publication_date = _iso(" ".join(get("43") + get("45")))
    bib.application_number = squash("".join(get("21"))) or None
    bib.application_date = _iso(" ".join(get("22")))

    holder = get("71") or get("73")
    owner_lines, address_lines = [], []
    for line in holder:
        target = address_lines if address_lines or squash(line).startswith("地址") else owner_lines
        target.append(line)
    if owner_lines:
        names = normalize_text(join_lines(owner_lines))
        bib.applicants = [n for n in re.split(r"[;；]", names) if n]
    if address_lines:
        address = re.sub(r"\s+", " ", join_lines(address_lines)).strip()
        bib.address = re.sub(r"^地址\s*", "", address) or None  # keep "223000 江苏省…"

    bib.inventors = [n for n in _NAME_SPLIT.split(" ".join(get("72"))) if n]

    agency_lines, agents = [], []
    for line in get("74"):
        compact = squash(line)
        if compact.startswith(("专利代理师", "代理人")):
            agents += [n for n in _NAME_SPLIT.split(re.sub(r"^\s*(专利代理师|代理人)", "", line)) if n]
        else:
            agency_lines.append(line)
    bib.agency = normalize_text(join_lines(agency_lines)) or None
    bib.agents = agents

    bib.ipc = [re.sub(r"\s+", " ", code) for code in _IPC.findall("\n".join(get("51")))]
    bib.title = normalize_text(join_lines(get("54"))) or None
    bib.abstract = normalize_text(join_lines(get("57"))) or None

    # Values that the layout pushed to the end of the page.
    tail_text = "\n".join(tail)
    if bib.publication_number is None and (m := _PUB_NUMBER.search(tail_text)):
        bib.publication_number = f"CN{m.group(1)}{m.group(2)}"
    if bib.publication_date is None:
        dates = [_iso(l) for l in tail if _DATE.search(l)]
        dates = [d for d in dates if d and d != bib.application_date]
        bib.publication_date = dates[-1] if dates else None
    bib.page_counts = {part: int(n) for part, n in _COUNTS.findall(squash(tail_text))}
    return bib
