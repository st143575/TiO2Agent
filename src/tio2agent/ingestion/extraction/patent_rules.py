"""Rule-based fact extraction for Chinese process patents.

Reads only authoritative places, never restatements:

    claims                 product claims -> components (ranges or values), options
                           method claims  -> process parameters per step (S1, S2, ...)
    examples (实施例N)      components and process parameters per step
    tables (from OCR)      results per subject: "老化试验色差值 × 实施例1 = 1.21"

Description paragraphs that restate claims and comparative/test sections are skipped,
so the same number is not stored twice and "取各组别…各10份" is not read as a recipe.

Components: an amount "N份" names the material on its right ("55份二氧化钛", the
style of examples) or, if nothing follows, on its left ("二氧化钛45-65份", the style
of claims).

Parameters: a fixed set of patterns (temperature, stirring speed and time, milling
time, addition time, pH, and calcination / drying / ageing temperature and time). A
pH value is qualified by the last material added before it ("加入锆盐后"), because
the same step adjusts pH twice.
"""

import re

from tio2agent.domain import ClaimCategory, EmbodimentKind, FactKind, StructuredDocument
from tio2agent.ingestion.extraction.base import ExtractionResult, FactBuilder, Source
from tio2agent.ingestion.extraction.quantities import (
    NUMBER,
    QUANTITY,
    canonical_name,
    canonical_unit,
    normalize_chemical_name,
    parse_number,
    parse_quantity,
)

_CJK = r"一-鿿"

# --- components ---------------------------------------------------------------------
_AMOUNT = re.compile(rf"{QUANTITY}\s*(?P<unit>重量份|质量份|份)(?![计比数])")
_NAME_RIGHT = re.compile(rf"[{_CJK}][{_CJK}\d,\-]*?(?=加入|和|及|与|[，、。；：,;:\s]|$)")
_NAME_LEFT = re.compile(rf"[{_CJK}][{_CJK}\d,\-]*$")
_LEAD_WORDS = re.compile(r"^(?:具体包括|包括|包含|含有|以及|和|及|与|将|加入)")
_NOT_NAMES = {"各", "每", "共", "约", "重量", "质量"}

# --- parameters (most specific first; a span is used by one pattern only) --------------
_OPS = "煅烧|焙烧|干燥|烘干|陈化|熟化|保温"
_PARAMS: list[tuple[str, re.Pattern[str]]] = [
    (
        "加入时间",
        re.compile(
            rf"搅拌(?:至)?\s*{QUANTITY}\s*(?P<unit>分钟|min)时[，,]?\s*(?:缓慢|分批次?|逐滴)?加入"
            rf"(?:{NUMBER}\s*份)?(?P<what>[{_CJK}]{{1,8}}?)(?=[，,。；;]|$)"
        ),
    ),
    ("{op}温度", re.compile(rf"(?P<op>{_OPS})(?:温度)?(?:为|在|控制在|设为)?\s*{QUANTITY}\s*(?P<unit>℃)")),
    ("{op}时间", re.compile(rf"(?P<op>{_OPS})(?:时间)?(?:为)?\s*{QUANTITY}\s*(?P<unit>小时|h|分钟|min)")),
    ("温度", re.compile(rf"温度(?:设为|设定为|控制在|保持在|为|升至|至)?\s*{QUANTITY}\s*(?P<unit>℃)")),
    ("转速", re.compile(rf"转速(?:设为|设定为|为|控制在)?\s*{QUANTITY}\s*(?P<unit>转/分钟|转/分|r/min|rpm)")),
    ("搅拌时间", re.compile(rf"搅拌时间(?:为|控制在)?\s*{QUANTITY}\s*(?P<unit>分钟|小时|min|h)")),
    ("研磨时间", re.compile(rf"研磨(?:时间为)?\s*{QUANTITY}\s*(?P<unit>分钟|小时|min|h)")),
    ("pH", re.compile(rf"pH值?(?:调节至|调至|调节为|控制在|为)\s*{QUANTITY}")),
]
_STEP_LINE = re.compile(r"^(S\d+)\s*[、.．:：]", re.MULTILINE)

# --- options -------------------------------------------------------------------------
_OPTION = re.compile(rf"所述(?P<name>[{_CJK}]{{2,8}}?)(?:为|选自|是)(?P<values>.+)$")
_OPTION_TAIL = re.compile(r"(?:中的(?:一种|任意一种|至少一种)(?:或(?:多种|几种|两种以上))?(?:的组合|的混合物)?|等)$")
_OPTION_SPLIT = re.compile(r"、|或者|以及|或|和|及")
_EMBODIMENT_HEADER = re.compile(r"(实施例|对比例|比较例)\s*(\d+)")


class PatentRuleExtractor:
    name = "patent_rules"

    def extract(self, doc: StructuredDocument) -> ExtractionResult:
        builder = FactBuilder(doc.doc_id, self.name)

        for claim in doc.claims:
            source = Source(
                text=claim.text,
                subject=f"claim:{claim.number}",
                subject_label=f"权利要求{claim.number}",
                claim_number=claim.number,
                pages=claim.pages,
            )
            if claim.category is ClaimCategory.METHOD:
                for step_source in _split_steps(source):
                    _parameters(builder, step_source)
            else:
                _components(builder, source)
            _options(builder, source)

        paragraphs = {p.id: p for p in doc.paragraphs}
        for emb in doc.embodiments:
            if emb.kind is not EmbodimentKind.EXAMPLE:
                continue
            for pid in emb.paragraph_ids:
                p = paragraphs.get(pid)
                if p is None or not p.text:
                    continue
                source = Source(
                    text=p.text, subject=emb.id, subject_label=emb.label,
                    paragraph_ids=[p.id], pages=p.pages, step=p.step,
                )
                _components(builder, source)
                _parameters(builder, source)

        labels = {e.label: e.id for e in doc.embodiments}
        for table in doc.tables:
            _table_results(builder, table, labels)

        return ExtractionResult(facts=builder.facts, issues=builder.rejected)


# --- helpers ---------------------------------------------------------------------------


def _split_steps(source: Source) -> list[Source]:
    """A method claim lists its steps as lines "S1、…"; one Source per step."""
    starts = list(_STEP_LINE.finditer(source.text))
    if not starts:
        return [source]
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(source.text)
        out.append(Source(**{**source.__dict__, "text": source.text[m.start():end].strip(), "step": m.group(1)}))
    return out


def _components(b: FactBuilder, source: Source) -> None:
    text = source.text
    seen: set[str] = set()
    for m in _AMOUNT.finditer(text):
        right = _NAME_RIGHT.match(text, m.end())
        if right and right.end() > m.end():
            raw, evidence = right.group(), text[m.start():right.end()]
        else:
            left = _NAME_LEFT.search(text, 0, m.start())
            if not left:
                continue
            raw = _LEAD_WORDS.sub("", left.group())
            evidence = text[m.end() - len(m.group()) - len(raw):m.end()]
        if not (2 <= len(raw) <= 12) or raw in _NOT_NAMES:
            continue
        name, name_raw = canonical_name(raw)
        if name in seen:  # the same material twice in one paragraph: keep the first
            continue
        seen.add(name)
        q = parse_quantity(m.group("low"), m.group("high"))
        b.add(
            source, FactKind.COMPONENT, name, evidence, name_raw=name_raw,
            value=q.value, low=q.low, high=q.high, unit=canonical_unit(m.group("unit")),
        )


def _parameters(b: FactBuilder, source: Source) -> None:
    text = source.text
    taken: list[tuple[int, int]] = []
    found = []
    for template, pattern in _PARAMS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append(m.span())
            found.append((m.start(), template, m))
    last_added: str | None = None
    for _, template, m in sorted(found, key=lambda x: x[0]):
        groups = m.groupdict()
        name = template.format(op=groups.get("op") or "")
        qualifier = None
        if name == "加入时间":
            last_added = groups["what"]
            qualifier = last_added
        elif name == "pH" and last_added:
            qualifier = f"加入{last_added}后"
        q = parse_quantity(groups["low"], groups["high"])
        b.add(
            source, FactKind.PARAMETER, name, m.group(), qualifier=qualifier,
            value=q.value, low=q.low, high=q.high,
            unit=canonical_unit(groups.get("unit")) if name != "pH" else None,
        )


def _options(b: FactBuilder, source: Source) -> None:
    for clause in re.split(r"[。；;]|[，,](?=所述)", source.text):
        m = _OPTION.search(clause)
        if not m:
            continue
        values = _OPTION_TAIL.sub("", m.group("values").strip())
        options = [normalize_chemical_name(v) for v in _OPTION_SPLIT.split(values) if v.strip()]
        if len(options) < 2 or any(len(o) > 20 for o in options):
            continue
        name, name_raw = canonical_name(m.group("name"))
        b.add(
            source, FactKind.OPTION, name, m.group(), name_raw=name_raw,
            options=options, text_value="、".join(options),
        )


def _table_results(b: FactBuilder, table, labels: dict[str, str]) -> None:
    if len(table.rows) < 2:
        return
    header = table.rows[0]
    flat = "\n".join(" | ".join(r) for r in table.rows)
    for col in range(1, len(header)):
        m = _EMBODIMENT_HEADER.search(header[col])
        label = f"{m.group(1)}{m.group(2)}" if m else header[col].strip()
        subject = labels.get(label, label)
        for row in table.rows[1:]:
            if col >= len(row) or not row[col].strip() or not row[0].strip():
                continue
            cell = row[col].strip()
            name = re.sub(r"\s+", "", row[0])
            number = parse_number(cell)
            source = Source(
                text=flat, subject=subject, subject_label=label,
                paragraph_ids=[table.paragraph_id] if table.paragraph_id else [],
                table_id=table.id, pages=[table.page],
            )
            b.add(
                source, FactKind.RESULT, name, cell,
                value=number, low=number, high=number,
                text_value=None if number is not None else cell,
            )
