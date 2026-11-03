"""Validators and the evidence guard, on small synthetic fact sets."""

from tio2agent.domain import Fact, FactKind
from tio2agent.ingestion.extraction import FactBuilder, Source, validate_facts


def fact(subject: str, name: str, *, low: float, high: float | None = None, kind=FactKind.COMPONENT, **kw) -> Fact:
    high = low if high is None else high
    claim = int(subject.split(":")[1]) if subject.startswith("claim:") else None
    return Fact(
        id=f"{subject}-{name}", doc_id="d", kind=kind, subject=subject,
        subject_label=f"权利要求{claim}" if claim else f"实施例{subject[1:]}",
        name=name, low=low, high=high, value=None if high != low else low,
        unit=kw.pop("unit", "份"), evidence="x", claim_number=claim, extractor="test",
        paragraph_ids=[] if claim else ["0026"], pages=[5], **kw,
    )


def test_value_outside_claimed_range_is_reported() -> None:
    facts = [
        fact("claim:1", "二氧化钛", low=45, high=65),
        fact("E1", "二氧化钛", low=70),
    ]
    issues = validate_facts(facts)
    assert [i.code for i in issues] == ["EXTRACT_RANGE_VIOLATION"]
    assert "超出权利要求1的范围 45-65份" in issues[0].message
    assert issues[0].page == 5
    (check,) = [c for c in facts[1].checks if c.rule == "within_claimed_range"]
    assert (check.rule, check.ok, check.reference) == ("within_claimed_range", False, "权利要求1")


def test_range_endpoints_are_inside() -> None:
    facts = [fact("claim:1", "分散剂", low=0.5, high=0.9), fact("E1", "分散剂", low=0.9)]
    assert validate_facts(facts) == []
    assert facts[1].checks[0].ok


def test_parameters_match_on_step_and_qualifier() -> None:
    p = {"kind": FactKind.PARAMETER, "unit": None}
    facts = [
        fact("claim:8", "pH", low=5, high=7, step="S3", qualifier="加入锆盐后", **p),
        fact("claim:8", "pH", low=4, high=6, step="S3", qualifier="加入铝盐后", **p),
        fact("E1", "pH", low=4, step="S3", qualifier="加入铝盐后", **p),   # ok for 铝盐, would fail for 锆盐
        fact("E1", "pH", low=4, step="S3", qualifier="加入锆盐后", **p),   # violation
        fact("E1", "pH", low=4, step="S1", qualifier=None, **p),           # no claimed range: unchecked
    ]
    issues = validate_facts(facts)
    assert len(issues) == 1 and "加入锆盐后 pH 4 超出权利要求8的范围 5-7" in issues[0].message
    assert [len(f.checks) for f in facts[2:]] == [1, 1, 0]


def test_missing_component_is_reported_once_per_example() -> None:
    facts = [
        fact("claim:1", "二氧化钛", low=45, high=65),
        fact("claim:1", "锆盐", low=6, high=8),
        fact("E1", "二氧化钛", low=55),
        fact("E2", "二氧化钛", low=50),
        fact("E2", "锆盐", low=7),
    ]
    issues = validate_facts(facts)
    assert [(i.code, i.message) for i in issues] == [
        ("EXTRACT_MISSING_COMPONENT", "实施例1：缺少权利要求1列出的组分：锆盐")
    ]


def test_builder_rejects_evidence_not_in_source() -> None:
    builder = FactBuilder("d", "test")
    source = Source(text="温度设为65℃", subject="E1", subject_label="实施例1", pages=[5])
    assert builder.add(source, FactKind.PARAMETER, "温度", "温度设为65℃", value=65, low=65, high=65) is not None
    assert builder.add(source, FactKind.PARAMETER, "温度", "温度设为66℃", value=66, low=66, high=66) is None
    assert [f.id for f in builder.facts] == ["d-f0001"]
    assert builder.rejected[0].code == "EXTRACT_EVIDENCE_MISMATCH"
