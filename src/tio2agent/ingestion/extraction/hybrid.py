"""Rules and LLM together: keep what either finds, merge what both find.

The two extractors fail differently. The rules read numbers reliably but cut names
at the wrong place when the wording differs from what they were written for
("去离子水中", "再使用" in CN115403940A); the LLM names things well but may miss or
misread. Both only produce facts whose evidence is in the text.

    same fact      same subject and kind, same location (claim / paragraph), same
                   numbers (options: any), overlapping evidence in the text
                   -> one fact: the LLM's name, step and qualifier; extractor "hybrid"
    one side only  kept as it is (tables come from the rules)
    same key       (subject, step, kind, name, qualifier) twice with different
                   numbers -> keep the rule fact, warn EXTRACT_CONFLICT

Facts are then put in document order and renumbered.
"""

from tio2agent.domain import Fact, FactKind, StructuredDocument
from tio2agent.ingestion.extraction import evidence as ev
from tio2agent.ingestion.extraction.base import ExtractionIssue, ExtractionResult, FactExtractor

_EPS = 1e-9
_RANK = {"hybrid": 0}  # then rules, then llm


class HybridExtractor:
    name = "hybrid"

    def __init__(self, rules: FactExtractor, llm: FactExtractor) -> None:
        self.rules = rules
        self.llm = llm

    def extract(self, doc: StructuredDocument) -> ExtractionResult:
        by_rules = self.rules.extract(doc)
        by_llm = self.llm.extract(doc)
        facts, issues = merge_facts(doc, by_rules.facts, by_llm.facts, rules_name=self.rules.name)
        return ExtractionResult(
            facts=facts,
            issues=by_rules.issues + by_llm.issues + issues,
            rejected=by_llm.rejected,
            stats={**by_llm.stats, "rule_facts": len(by_rules.facts), "llm_facts": len(by_llm.facts)},
        )


def merge_facts(
    doc: StructuredDocument, rule_facts: list[Fact], llm_facts: list[Fact], *, rules_name: str = "patent_rules"
) -> tuple[list[Fact], list[ExtractionIssue]]:
    texts = _TextLookup(doc)
    consumed: set[int] = set()
    merged: list[Fact] = []
    for fact in llm_facts:
        if fact.table_id is not None:  # tables are read from cells by the rules
            continue
        twin = next(
            (r for r in rule_facts if id(r) not in consumed and _same_fact(fact, r, texts)),
            None,
        )
        if twin is not None:
            consumed.add(id(twin))
            merged.append(fact.model_copy(update={"extractor": "hybrid"}))
        else:
            merged.append(fact)
    merged += [r for r in rule_facts if id(r) not in consumed]

    kept: dict[tuple, Fact] = {}
    issues: list[ExtractionIssue] = []
    rank = {**_RANK, rules_name: 1}
    for fact in sorted(merged, key=lambda f: rank.get(f.extractor, 2)):
        key = (fact.subject, fact.step, fact.kind, fact.name, fact.qualifier)
        other = kept.get(key)
        if other is None:
            kept[key] = fact
        elif not _same_value(other, fact):
            issues.append(ExtractionIssue(
                "EXTRACT_CONFLICT",
                f"{fact.subject_label} {' '.join(p for p in (fact.step, fact.qualifier, fact.name) if p)}："
                f"保留{other.display_value}（{other.citation}「{other.evidence}」），"
                f"舍弃{fact.display_value}（{fact.citation}「{fact.evidence}」）",
                page=fact.pages[0] if fact.pages else None,
            ))
    ordered = sorted(kept.values(), key=lambda f: _position(f, doc, texts))
    return [f.model_copy(update={"id": f"{doc.doc_id}-f{i:04d}"}) for i, f in enumerate(ordered, 1)], issues


def _same_fact(a: Fact, b: Fact, texts: "_TextLookup") -> bool:
    if (a.kind, a.subject, a.claim_number, a.paragraph_ids, a.table_id) != (
        b.kind, b.subject, b.claim_number, b.paragraph_ids, b.table_id
    ):
        return False
    if a.kind is not FactKind.OPTION and not _same_value(a, b):
        return False
    text = texts.of(a)
    if text is None:
        return False
    return any(s1 < e2 and s2 < e1 for s1, e1 in ev.spans(text, a.evidence) for s2, e2 in ev.spans(text, b.evidence))


def _same_value(a: Fact, b: Fact) -> bool:
    if a.kind is FactKind.OPTION or b.kind is FactKind.OPTION:
        return set(a.options) == set(b.options)
    return all(
        (x is None and y is None) or (x is not None and y is not None and abs(x - y) < _EPS)
        for x, y in ((a.low, b.low), (a.high, b.high))
    )


class _TextLookup:
    def __init__(self, doc: StructuredDocument) -> None:
        self.claims = {c.number: c.text for c in doc.claims}
        self.paragraphs = {p.id: p.text for p in doc.paragraphs}
        self.order = {p.id: i for i, p in enumerate(doc.paragraphs)}

    def of(self, fact: Fact) -> str | None:
        if fact.claim_number is not None:
            return self.claims.get(fact.claim_number)
        if fact.table_id is None and fact.paragraph_ids:
            return self.paragraphs.get(fact.paragraph_ids[0])
        return None


def _position(fact: Fact, doc: StructuredDocument, texts: _TextLookup) -> tuple:
    text = texts.of(fact) or ""
    offset = text.find(fact.evidence) if fact.evidence else -1
    if fact.claim_number is not None:
        return (0, fact.claim_number, offset)
    if fact.table_id is not None:
        tables = [t.id for t in doc.tables]
        return (2, tables.index(fact.table_id) if fact.table_id in tables else len(tables), 0)
    return (1, texts.order.get(fact.paragraph_ids[0], 0) if fact.paragraph_ids else 0, offset)
