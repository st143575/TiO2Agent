"""Cross-checks between facts, attached to each fact as ``checks``.

A patent states every number twice: as a claimed range and as an example value. The
two must agree, so comparing them catches both extraction bugs and errors in the
document itself:

    within_claimed_range   each single value lies inside the claimed range for the
                           same quantity (component: same name; parameter: same
                           name, step and qualifier)
    complete_formulation   each example uses every component the claims list
    name_alias             the document wrote the name differently ("去离子")

Failures become document warnings; passes are only recorded on the fact.
"""

from tio2agent.domain import Fact, FactCheck, FactKind
from tio2agent.ingestion.extraction.base import ExtractionIssue

_EPS = 1e-9


def _key(f: Fact) -> tuple:
    if f.kind is FactKind.COMPONENT:
        return (f.kind, f.name)
    return (f.kind, f.name, f.step, f.qualifier)


def _is_claim(f: Fact) -> bool:
    return f.claim_number is not None


def claimed_ranges(facts: list[Fact]) -> dict[tuple, Fact]:
    """
    The first (lowest-numbered) claim range for each quantity.
    范围核对：实施例的值必须落在权利要求的范围内。参数按名称 + 步骤 + 限定语匹配。
    """
    ranges: dict[tuple, Fact] = {}
    for f in sorted(facts, key=lambda f: f.claim_number or 0):
        if _is_claim(f) and f.is_range and f.kind in (FactKind.COMPONENT, FactKind.PARAMETER):
            ranges.setdefault(_key(f), f)
    return ranges


def validate_facts(facts: list[Fact]) -> list[ExtractionIssue]:
    issues: list[ExtractionIssue] = []
    ranges = claimed_ranges(facts)

    for f in facts:
        if f.name_raw:
            f.checks.append(
                FactCheck(rule="name_alias", ok=True, message=f"原文写作“{f.name_raw}”，按“{f.name}”归并")
            )
        if f.value is None or f.kind not in (FactKind.COMPONENT, FactKind.PARAMETER):
            continue
        claimed = ranges.get(_key(f))
        if claimed is None or claimed.subject == f.subject:
            continue
        ok = claimed.low - _EPS <= f.value <= claimed.high + _EPS  # The fact's value lies within the range claimed in the patent.
        label = _label(f)
        message = (
            f"{label} {f.display_value} 在{claimed.citation}的范围 {claimed.display_value} 内"
            if ok
            else f"{label} {f.display_value} 超出{claimed.citation}的范围 {claimed.display_value}"
        )
        f.checks.append(FactCheck(rule="within_claimed_range", ok=ok, message=message, reference=claimed.citation))
        if not ok:
            issues.append(
                ExtractionIssue("EXTRACT_RANGE_VIOLATION", f"{f.subject_label}：{message}", page=_page(f))
            )

    issues.extend(_check_completeness(facts, ranges))
    return issues


def _check_completeness(facts: list[Fact], ranges: dict[tuple, Fact]) -> list[ExtractionIssue]:
    """
    完整性核对：每个实施例要包含权利要求列出的全部组分。
    """
    claimed = [r for k, r in ranges.items() if k[0] is FactKind.COMPONENT]
    if not claimed:
        return []
    by_subject: dict[str, list[Fact]] = {}
    for f in facts:
        if f.kind is FactKind.COMPONENT and not _is_claim(f):
            by_subject.setdefault(f.subject, []).append(f)
    issues = []
    for components in by_subject.values():
        names = {c.name for c in components}
        missing = [r.name for r in claimed if r.name not in names]
        ok = not missing
        message = (
            f"包含{claimed[0].citation}列出的全部 {len(claimed)} 种组分"
            if ok
            else f"缺少{claimed[0].citation}列出的组分：{'、'.join(missing)}"
        )
        check = FactCheck(rule="complete_formulation", ok=ok, message=message, reference=claimed[0].citation)
        for c in components:
            c.checks.append(check)
        if not ok:
            issues.append(
                ExtractionIssue(
                    "EXTRACT_MISSING_COMPONENT", f"{components[0].subject_label}：{message}", page=_page(components[0])
                )
            )
    return issues


def _label(f: Fact) -> str:
    parts = [f.step, f.qualifier if f.kind is FactKind.PARAMETER and f.name != "加入时间" else None, f.name]
    if f.name == "加入时间" and f.qualifier:
        parts[-1] = f"{f.qualifier}加入时间"
    return " ".join(p for p in parts if p)


def _page(f: Fact) -> int | None:
    return f.pages[0] if f.pages else None
