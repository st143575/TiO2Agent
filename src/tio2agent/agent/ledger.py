"""Every source the Agent sees in one run, under a short id the model can cite.

Tools register what they return here (T = text chunk, F = fact, C = calculation; not
S, which patents use for process steps S1, S2, …). The same chunk or fact keeps its
id when it is returned again. After the run the ledger is the ground truth for
checking the final answer: a cited id must exist here, and a reported value must
equal the value of its source.
"""

import re
from dataclasses import dataclass, field

from tio2agent.domain import Chunk, Citation, Fact, SourceKind

_PREFIX = {SourceKind.CHUNK: "T", SourceKind.FACT: "F", SourceKind.CALCULATION: "C"}
# Inline citations: "[F3]", "[F3, C1]", "[T2、F5]".
CITATION_GROUP = re.compile(r"\[((?:[TFC]\d+)(?:\s*[,，、;；]\s*[TFC]\d+)*)\]")
SOURCE_ID = re.compile(r"[TFC]\d+")
_REF = re.compile(r"^([TFC]\d+)(?:\.(low|high|value))?$")


@dataclass
class SourceRecord:
    id: str
    kind: SourceKind
    citation: str
    quote: str
    doc_id: str | None = None
    doc_label: str | None = None
    ref_id: str | None = None
    pages: list[int] = field(default_factory=list)
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)  # calculations: variable -> source ref

    def to_citation(self) -> Citation:
        return Citation(
            id=self.id, kind=self.kind, doc_id=self.doc_id, doc_label=self.doc_label,
            citation=self.citation, quote=self.quote, ref_id=self.ref_id, pages=self.pages,
        )


class SourceLedger:
    """
    记录本轮问答见过的所有来源及其 T/F/C 编号，后面核对回答时以它为准。
    """
    def __init__(self) -> None:
        self._records: dict[str, SourceRecord] = {}
        self._by_ref: dict[tuple[SourceKind, str], str] = {}
        self._counters = dict.fromkeys(SourceKind, 0)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, source_id: str) -> SourceRecord | None:
        return self._records.get(source_id)

    def records(self) -> list[SourceRecord]:
        return list(self._records.values())

    def _add(self, record_kind: SourceKind, ref: str | None, make) -> str:
        if ref is not None and (record_kind, ref) in self._by_ref:
            return self._by_ref[(record_kind, ref)]
        self._counters[record_kind] += 1
        source_id = f"{_PREFIX[record_kind]}{self._counters[record_kind]}"
        self._records[source_id] = make(source_id)
        if ref is not None:
            self._by_ref[(record_kind, ref)] = source_id
        return source_id

    def add_chunk(self, chunk: Chunk) -> str:
        return self._add(SourceKind.CHUNK, chunk.id, lambda sid: SourceRecord(
            id=sid, kind=SourceKind.CHUNK, citation=chunk.citation, quote=chunk.text,
            doc_id=chunk.doc_id, doc_label=chunk.doc_label, ref_id=chunk.id, pages=chunk.pages,
        ))

    def add_fact(self, fact: Fact, doc_label: str | None) -> str:
        return self._add(SourceKind.FACT, fact.id, lambda sid: SourceRecord(
            id=sid, kind=SourceKind.FACT, citation=fact.citation, quote=fact.evidence,
            doc_id=fact.doc_id, doc_label=doc_label, ref_id=fact.id, pages=fact.pages,
            value=fact.value, low=fact.low, high=fact.high, unit=fact.unit,
        ))

    def add_calculation(self, expression: str, inputs: dict[str, str], value: float, unit: str | None) -> str:
        shown = expression
        for name, ref in inputs.items():
            shown = re.sub(rf"\b{re.escape(name)}\b", ref, shown)
        return self._add(SourceKind.CALCULATION, None, lambda sid: SourceRecord(
            id=sid, kind=SourceKind.CALCULATION, citation="计算", quote=f"{shown} = {value:g}",
            value=value, low=value, high=value, unit=unit, inputs=dict(inputs),
        ))

    def number(self, ref: str) -> float:
        """Value behind "F3", "F3.low", "F3.high" or "C1". Raises ValueError."""
        m = _REF.match(ref.strip())
        if not m:
            raise ValueError(f"“{ref}”不是来源 id（应形如 F3、F3.low、C1）")
        record = self.get(m.group(1))
        if record is None:
            raise ValueError(f"来源 {m.group(1)} 不存在：只能引用本轮工具返回过的 id")
        attr = m.group(2) or "value"
        if attr == "value" and record.value is None:
            if record.low is not None:
                raise ValueError(f"{record.id} 是范围 {record.low:g}-{record.high:g}，请用 {record.id}.low 或 {record.id}.high")
            raise ValueError(f"{record.id} 没有数值")
        value = getattr(record, attr)
        if value is None:
            raise ValueError(f"{record.id} 没有 {attr}")
        return value

    def expand(self, source_ids: list[str]) -> list[str]:
        """Ids plus, for calculations, the sources they were computed from (recursively)."""
        out: list[str] = []

        def visit(sid: str) -> None:
            if sid in out or sid not in self._records:
                return
            out.append(sid)
            for ref in self._records[sid].inputs.values():
                visit(ref.split(".")[0])

        for sid in source_ids:
            visit(sid)
        return out
