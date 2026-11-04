"""The ``final_answer`` tool and the checks that turn its arguments into an AgentAnswer.

The model ends a run by calling ``final_answer`` with a fixed schema, so the output
format does not depend on how the model phrases things. Its arguments are then
checked against the run's SourceLedger:

    - cited ids must exist in the ledger (unknown ones are removed, with a warning);
    - each value must equal its source: a fact or calculation numerically (allowing
      for the rounding the model wrote), a text chunk by the number appearing in it;
    - calculations are expanded to the facts they were computed from, so citations
      always reach a document location.
"""

import math
import re
from typing import Literal

from pydantic import BaseModel, Field

from tio2agent.agent.ledger import CITATION_GROUP, SOURCE_ID, SourceLedger, SourceRecord
from tio2agent.agent.tools import json_schema
from tio2agent.domain import AgentAnswer, AnswerStatus, AnswerValue, SourceKind
from tio2agent.ingestion.extraction.quantities import canonical_unit
from tio2agent.llm import ToolSpec

FINAL_ANSWER = "final_answer"


class FinalValue(BaseModel):
    label: str = Field(description="数值的含义，如 实施例1 二氧化钛用量")
    value: float | None = Field(None, description="单个数值")
    low: float | None = Field(None, description="范围下限")
    high: float | None = Field(None, description="范围上限")
    unit: str | None = None
    text: str | None = Field(None, description="非数值结果，如 最佳")
    source: str = Field(description="来源 id：F…、C… 或 T…")


class FinalAnswerArgs(BaseModel):
    status: Literal["answered", "partial", "insufficient_evidence"] = Field(
        "answered", description="answered 已回答；partial 只回答了一部分；insufficient_evidence 文档中没有依据"
    )
    answer: str = Field(min_length=1, description="中文回答，数值和结论后用 [F3] [C1] [T2] 标注来源")
    values: list[FinalValue] = Field(default_factory=list, description="回答中的关键数值，每个都要有来源")
    sources: list[str] = Field(default_factory=list, description="引用的全部来源 id")


FINAL_ANSWER_SPEC = ToolSpec(
    FINAL_ANSWER,
    "提交最终回答并结束。只能引用本轮工具返回过的来源 id。",
    json_schema(FinalAnswerArgs),
)


def build_answer(final: FinalAnswerArgs, ledger: SourceLedger) -> tuple[AgentAnswer, list[str]]:
    """Check the model's final answer against the ledger. Returns (answer, warnings)."""
    warnings: list[str] = []
    unknown: list[str] = []

    def known(sid: str) -> bool:
        if sid in ledger:
            return True
        if sid not in unknown:
            unknown.append(sid)
        return False

    # inline citations: keep known ids, drop unknown ones (and empty brackets)
    inline: list[str] = []

    def rewrite(m: re.Match) -> str:
        ids = [sid for sid in SOURCE_ID.findall(m.group(1)) if known(sid)]
        inline.extend(i for i in ids if i not in inline)
        return f"[{', '.join(ids)}]" if ids else ""

    text = CITATION_GROUP.sub(rewrite, final.answer).strip()

    values = []
    for v in final.values:
        source = v.source.strip().split(".")[0]
        record = ledger.get(source) if known(source) else None
        verified, note = _verify(v, record)
        if verified is False:
            warnings.append(f"数值“{v.label}”与来源不一致：{note}")
        values.append(AnswerValue(
            label=v.label, value=v.value, low=v.low, high=v.high, unit=v.unit, text=v.text,
            source_id=record.id if record else None, verified=verified, note=note,
        ))

    listed = [s.strip() for s in final.sources if known(s.strip())]
    cited = inline + [v.source_id for v in values if v.source_id] + listed
    ordered = ledger.expand(list(dict.fromkeys(cited)))
    citations = [ledger.get(sid).to_citation() for sid in ordered]

    if unknown:
        warnings.append(f"回答引用了本轮没有出现过的来源 {', '.join(unknown)}，已删除")
    status = AnswerStatus(final.status)
    if status is AnswerStatus.ANSWERED and not citations:
        warnings.append("回答没有引用任何来源")
    return AgentAnswer(status=status, answer=text, values=values, citations=citations), warnings


def _verify(v: FinalValue, record: SourceRecord | None) -> tuple[bool | None, str | None]:
    numbers = [n for n in (v.value, v.low, v.high) if n is not None]
    if record is None:
        return (False, f"来源 {v.source} 不存在") if numbers or v.text else (None, None)
    if not numbers:
        return None, None
    if record.kind is SourceKind.CHUNK:
        compact = re.sub(r"\s+", "", record.quote)
        missing = [n for n in numbers if not re.search(rf"(?<![\d.]){re.escape(_fmt(n))}(?![\d]|\.\d)", compact)]
        if missing:
            return False, f"{record.id} 的原文中没有 {', '.join(_fmt(n) for n in missing)}"
        return True, None

    note = None
    if v.unit and record.unit and canonical_unit(v.unit) != canonical_unit(record.unit):
        note = f"单位与来源不同（来源：{record.unit}）"
    if v.value is not None:
        if record.value is None:
            return False, f"{record.id} 是范围 {_fmt(record.low)}-{_fmt(record.high)}，不是单值"
        ok = _same(v.value, record.value)
        return ok, note if ok else f"{record.id} 的值是 {_fmt(record.value)}"
    ok = (v.low is None or _same(v.low, record.low)) and (v.high is None or _same(v.high, record.high))
    return ok, note if ok else f"{record.id} 的范围是 {_fmt(record.low)}-{_fmt(record.high)}"


def _same(reported: float, actual: float | None) -> bool:
    """Equal, or ``actual`` rounded to the decimals the model wrote (38.92 ≈ 38.9222)."""
    if actual is None:
        return False
    if math.isclose(reported, actual, rel_tol=1e-9, abs_tol=1e-9):
        return True
    decimals = len(repr(float(reported)).split(".")[1].rstrip("0")) if "." in repr(float(reported)) else 0
    return abs(reported - actual) <= 0.5 * 10 ** (-decimals) + 1e-9


def _fmt(x: float | None) -> str:
    if x is None:
        return "?"
    return str(int(x)) if float(x).is_integer() else f"{x:g}"
