"""Structured facts extracted from documents: formulations, process parameters, results.

Facts are flat rows so that they are easy to filter ("all 温度 facts of E1") and to
compute with ("sum the parts of E1's formulation"). Every fact keeps the exact text
it came from (``evidence``) and where that text is (paragraph ids / claim / table),
so a number in an answer can always be traced back to the document.

Quantities are either a single value (``value``; ``low`` = ``high`` = value) or a
range (``value`` is None, ``low`` / ``high`` set), as patents claim ranges ("45-65份")
and embodiments use single values ("55份").
"""

from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class FactKind(StrEnum):
    COMPONENT = "component"   # 配方组分：二氧化钛 55 份
    PARAMETER = "parameter"   # 工艺参数：S1 温度 65 ℃
    RESULT = "result"         # 测试结果：老化试验色差值 1.21
    OPTION = "option"         # 可选材料：分散剂 = 聚乙二醇 / 三聚磷酸钠 / …


class FactCheck(BaseModel):
    """Outcome of one validation rule applied to a fact."""

    rule: str                 # e.g. "within_claimed_range"
    ok: bool
    message: str
    reference: str | None = None  # what it was checked against, e.g. "权利要求1"


class Fact(BaseModel):
    id: str                   # "{doc_id}-f0001"
    doc_id: str
    kind: FactKind
    subject: str              # "E1" (embodiment), "claim:1", ...
    subject_label: str        # "实施例1", "权利要求1"
    name: str                 # canonical name: "去离子水", "温度", "pH"
    name_raw: str | None = None   # as written, when different ("去离子")
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str | None = None   # canonical: 份 ℃ rpm min h % …
    text_value: str | None = None  # non-numeric results ("最佳") or the option list text
    options: list[str] = Field(default_factory=list)
    step: str | None = None   # process step, e.g. "S1"
    qualifier: str | None = None  # disambiguates repeated parameters: "加入锆盐后"
    evidence: str             # exact text the fact was read from
    paragraph_ids: list[str] = Field(default_factory=list)
    claim_number: int | None = None
    table_id: str | None = None
    pages: list[int] = Field(default_factory=list)
    extractor: str            # "patent_rules", later e.g. "llm"
    checks: list[FactCheck] = Field(default_factory=list)

    @property
    def is_range(self) -> bool:
        return self.value is None and self.low is not None and self.high is not None

    # Computed fields appear in API responses; they are ignored when a stored fact is
    # validated back from JSON.
    @computed_field
    @property
    def display_value(self) -> str:
        if self.text_value is not None and self.value is None and self.low is None:
            return self.text_value
        number = _fmt(self.value) if self.value is not None else f"{_fmt(self.low)}-{_fmt(self.high)}"
        return f"{number}{self.unit or ''}" if self.unit not in (None, "pH") else number

    @computed_field
    @property
    def citation(self) -> str:
        if self.claim_number is not None:
            return f"权利要求{self.claim_number}"
        if self.table_id is not None:
            anchor = f"[{self.paragraph_ids[0]}]" if self.paragraph_ids else ""
            return f"说明书{anchor}表格"
        if self.paragraph_ids:
            return "说明书" + "、".join(f"[{p}]" for p in self.paragraph_ids)
        return ""


class Formulation(BaseModel):
    """The components of one subject (an embodiment or a composition claim)."""

    doc_id: str
    subject: str
    subject_label: str
    unit: str | None
    components: list[Fact]
    total: float | None = None  # sum of single values; None if any component is a range

    @computed_field
    @property
    def is_range(self) -> bool:
        return any(c.is_range for c in self.components)


def group_formulations(facts: list[Fact]) -> list[Formulation]:
    """Group component facts by (document, subject), keeping first-seen order."""
    groups: dict[tuple[str, str], list[Fact]] = {}
    for fact in facts:
        if fact.kind is FactKind.COMPONENT:
            groups.setdefault((fact.doc_id, fact.subject), []).append(fact)
    out = []
    for (doc_id, subject), components in groups.items():
        units = {c.unit for c in components}
        singles = [c.value for c in components if c.value is not None]
        total = round(sum(singles), 6) if len(singles) == len(components) and len(units) == 1 else None
        out.append(
            Formulation(
                doc_id=doc_id,
                subject=subject,
                subject_label=components[0].subject_label,
                unit=units.pop() if len(units) == 1 else None,
                components=components,
                total=total,
            )
        )
    return out


def _fmt(x: float | None) -> str:
    if x is None:
        return "?"
    return str(int(x)) if float(x).is_integer() else f"{x:g}"
