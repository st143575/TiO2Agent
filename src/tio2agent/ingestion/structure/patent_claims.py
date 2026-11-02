"""Claims (权利要求书): numbering, dependencies, category.

pypdf renders claim numbers as "1 .一种…"; a new claim starts only when the line
begins with the *next* expected number followed by a Chinese character, so wrapped
lines such as "0 .5份…" or "6分钟时…" are never mistaken for claims.

    "根据权利要求1所述的…"          -> refers_to [1]
    "根据权利要求1-7任意一项所述的…" -> refers_to [1, ..., 7]
    "如权利要求2或3所述的…"         -> refers_to [2, 3]
"""

import re

from tio2agent.domain import Claim, ClaimCategory, ClaimKind
from tio2agent.ingestion.structure.base import StructureIssue
from tio2agent.ingestion.structure.normalize import join_lines, normalize_text

_CLAIM_START = re.compile(r"^\s*(\d{1,3})\s*[.．、]\s*(?=[一-鿿])")  # 权利要求(claim)的模式
_STEP = re.compile(r"^\s*(?:S|步骤)\s*\d+\s*[、.．:：)）]")  # 步骤(step)的模式
_REFS = re.compile(r"权利要求\s*(\d+(?:\s*(?:-|~|至|到|、|,|，|或者|或|和|及|以及)\s*\d+)*)")  # 权利要求引用的模式
_RANGE = re.compile(r"(\d+)\s*(?:-|~|至|到)\s*(\d+)")  # 范围的模式


def parse_claims(lines: list[tuple[int, str]]) -> tuple[list[Claim], list[StructureIssue]]:
    """``lines`` are (page, line) pairs from the claims pages, furniture removed."""
    groups: list[tuple[int, list[tuple[int, str]]]] = []
    for page, line in lines:
        m = _CLAIM_START.match(line)
        if m and int(m.group(1)) == len(groups) + 1:  # 强制要求序号必须连续递增 (期望的下一个序号是 len(groups) + 1)
            groups.append((int(m.group(1)), [(page, line[m.end():])]))
        elif groups:
            groups[-1][1].append((page, line))
        # text before claim 1 (rare) is ignored

    claims: list[Claim] = []
    issues: list[StructureIssue] = []
    for number, body in groups:
        raw = join_lines([l for _, l in body], hard_break=_STEP)
        text = normalize_text(raw)
        refs = _references(_preamble(text))
        claims.append(
            Claim(
                number=number,
                text=text,
                raw=raw,
                pages=sorted({p for p, _ in body}),
                kind=ClaimKind.DEPENDENT if refs else ClaimKind.INDEPENDENT,
                category=_category(_preamble(text)),
                refers_to=refs,
            )
        )
        # A claim that refers to claims of another category (a method "according to
        # claim 1" product) is an independent claim of its own category.
        referenced = [c for c in claims[:-1] if c.number in refs]
        if referenced and all(c.category is not claims[-1].category for c in referenced):
            claims[-1].kind = ClaimKind.INDEPENDENT
        # 从属权利要求只能引用排在它前面的权利要求。如果r >= number (如：第 5 条引用了第 6 条），
        # 则视为无效引用 (bad reference)，生成一个 STRUCTURE_CLAIM_REF_INVALID 问题。
        bad = [r for r in refs if r >= number]
        if bad:
            issues.append(
                StructureIssue(
                    "STRUCTURE_CLAIM_REF_INVALID",
                    f"权利要求{number}引用了权利要求{bad}，引用的编号应小于自身",
                    page=claims[-1].pages[0],
                )
            )
    return claims, issues


def _preamble(text: str) -> str:
    """The part before "其特征在于" holds the references and the claimed subject."""
    head = re.split(r"其特征在于|其特征是", text, maxsplit=1)[0]
    return head if len(head) < len(text) else text[:120]


def _references(preamble: str) -> list[int]:
    refs: set[int] = set()
    for m in _REFS.finditer(preamble):
        group = m.group(1)
        for a, b in _RANGE.findall(group):
            lo, hi = sorted((int(a), int(b)))
            refs.update(range(lo, min(hi, lo + 200) + 1))
        refs.update(int(n) for n in re.findall(r"\d+", _RANGE.sub("", group)))
    return sorted(refs)


def _category(preamble: str) -> ClaimCategory:
    if re.search(r"用途|应用", preamble):
        return ClaimCategory.USE
    if re.search(r"方法|工艺", preamble):
        return ClaimCategory.METHOD
    return ClaimCategory.PRODUCT
