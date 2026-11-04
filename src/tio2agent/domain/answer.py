"""The Agent's structured answer: text, key values and the sources behind them.

Every source the Agent saw during one run gets a short id: ``T1`` for a retrieved
text chunk (not S: patents name process steps S1, S2, …), ``F1`` for a fact from the
fact store, ``C1`` for a calculation. The answer text cites them inline ("55 份[F3]"),
each key value names its source, and ``citations`` resolves the ids to document
locations ("说明书[0026]").
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class SourceKind(StrEnum):
    CHUNK = "chunk"              # T1, T2, … : retrieved text
    FACT = "fact"                # F1, F2, … : fact store rows
    CALCULATION = "calculation"  # C1, C2, … : calculate() results


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"                              # part of the question answered
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # the documents do not say
    INCOMPLETE = "incomplete"                        # step limit reached before an answer


class Citation(BaseModel):
    id: str                       # "T1" / "F3" / "C1"
    kind: SourceKind
    doc_id: str | None = None
    doc_label: str | None = None  # publication number, e.g. CN116694108A
    citation: str                 # "说明书[0026]", "权利要求1", "计算"
    quote: str                    # chunk text excerpt, fact evidence, or the formula
    ref_id: str | None = None     # chunk id / fact id in the stores
    pages: list[int] = Field(default_factory=list)


class AnswerValue(BaseModel):
    label: str                     # "实施例1 二氧化钛用量"
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str | None = None
    text: str | None = None        # non-numeric values ("最佳")
    source_id: str | None = None
    verified: bool | None = None   # True: equals its source; False: does not; None: nothing to check
    note: str | None = None

    @computed_field
    @property
    def display(self) -> str:
        if self.value is not None:
            number = _fmt(self.value)
        elif self.low is not None and self.high is not None:
            number = f"{_fmt(self.low)}-{_fmt(self.high)}"
        else:
            return self.text or ""
        return f"{number}{self.unit or ''}"


class AgentAnswer(BaseModel):
    status: AnswerStatus
    answer: str
    values: list[AnswerValue] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class AgentStep(BaseModel):
    """One tool call in the trace."""

    index: int
    tool: str
    arguments: dict[str, Any] | str      # str if the model sent invalid JSON
    ok: bool
    summary: str
    sources: list[str] = Field(default_factory=list)   # ids registered by this call
    thought: str | None = None           # text the model wrote alongside the call
    duration_ms: int = 0


class AgentUsage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"
