"""SourceLedger and the checks applied to final_answer."""

import pytest

from tio2agent.agent import FinalAnswerArgs, FinalValue, SourceLedger, build_answer
from tio2agent.domain import AnswerStatus, Chunk, ChunkKind, Fact, FactKind


def fact(fid: str, name: str, *, value=None, low=None, high=None, unit="份", para="0026") -> Fact:
    if value is not None:
        low = high = value
    return Fact(
        id=fid, doc_id="d1", kind=FactKind.COMPONENT, subject="E1", subject_label="实施例1", name=name,
        value=value, low=low, high=high, unit=unit, evidence=f"{name}{value or low}", paragraph_ids=[para],
        pages=[4], extractor="test",
    )


CHUNK = Chunk(
    id="d1-0007", doc_id="d1", doc_label="CN116694108A", order=7, kind=ChunkKind.EMBODIMENT,
    text="将反应釜温度设为 65℃，转速设为200转/分钟，搅拌时间为15分钟", paragraph_ids=["0026"], pages=[4],
)


@pytest.fixture
def ledger() -> SourceLedger:
    led = SourceLedger()
    assert led.add_fact(fact("d1-f1", "二氧化钛", value=55), "CN116694108A") == "F1"
    assert led.add_fact(fact("d1-f2", "锆盐", low=6, high=8, para="0008"), "CN116694108A") == "F2"
    assert led.add_chunk(CHUNK) == "T1"
    assert led.add_calculation("a / b * 100", {"a": "F1", "b": "F3"}, 38.922155, "%") == "C1"
    return led


def test_ledger_ids_and_numbers(ledger: SourceLedger) -> None:
    assert ledger.add_fact(fact("d1-f1", "二氧化钛", value=55), None) == "F1"  # same fact, same id
    assert ledger.add_chunk(CHUNK) == "T1"
    assert ledger.number("F1") == 55 and ledger.number("F2.low") == 6 and ledger.number("F2.high") == 8
    assert ledger.number("C1") == pytest.approx(38.922155)
    with pytest.raises(ValueError, match="是范围 6-8"):
        ledger.number("F2")
    with pytest.raises(ValueError, match="不存在"):
        ledger.number("F9")
    with pytest.raises(ValueError, match="不是来源 id"):
        ledger.number("55")
    with pytest.raises(ValueError, match="没有数值"):
        ledger.number("T1")
    assert ledger.get("C1").quote == "F1 / F3 * 100 = 38.9222"


def test_verified_values_and_citation_order(ledger: SourceLedger) -> None:
    answer, warnings = build_answer(FinalAnswerArgs(
        answer="实施例1中二氧化钛 55 份[F1]，占配方的 38.92%[C1]；反应温度 65℃[T1]。",
        values=[
            FinalValue(label="二氧化钛", value=55, unit="份", source="F1"),
            FinalValue(label="二氧化钛占比", value=38.92, unit="%", source="C1"),   # rounded: ok
            FinalValue(label="温度", value=65, unit="℃", source="T1"),              # number is in the chunk
            FinalValue(label="锆盐范围", low=6, high=8, unit="份", source="F2.low"),
        ],
    ), ledger)
    assert warnings == []
    assert [v.verified for v in answer.values] == [True, True, True, True]
    assert answer.values[1].display == "38.92%"
    # inline order first, then values; C1 brings in its inputs (F3 does not exist and is skipped)
    assert [c.id for c in answer.citations] == ["F1", "C1", "T1", "F2"]
    assert answer.citations[0].citation == "说明书[0026]" and answer.citations[0].doc_label == "CN116694108A"
    assert answer.citations[2].ref_id == "d1-0007"


def test_mismatched_values_are_flagged(ledger: SourceLedger) -> None:
    answer, warnings = build_answer(FinalAnswerArgs(
        answer="二氧化钛 56 份[F1]",
        values=[
            FinalValue(label="二氧化钛", value=56, source="F1"),
            FinalValue(label="占比", value=38.8, unit="%", source="C1"),
            FinalValue(label="锆盐", value=7, source="F2"),
            FinalValue(label="转速", value=300, source="T1"),
            FinalValue(label="温度", value=5, source="T1"),            # "5" only occurs inside 65 / 15
            FinalValue(label="分散剂", value=0.7, unit="g", source="F1"),
        ],
    ), ledger)
    assert [v.verified for v in answer.values] == [False] * 6
    assert answer.values[0].note == "F1 的值是 55"
    assert answer.values[2].note == "F2 是范围 6-8，不是单值"
    assert answer.values[3].note == "T1 的原文中没有 300"
    assert len(warnings) == 6


def test_unit_difference_is_noted_not_failed(ledger: SourceLedger) -> None:
    answer, _ = build_answer(FinalAnswerArgs(
        answer="x", values=[FinalValue(label="二氧化钛", value=55, unit="g", source="F1")]
    ), ledger)
    assert answer.values[0].verified is True and answer.values[0].note == "单位与来源不同（来源：份）"


def test_unknown_ids_are_removed(ledger: SourceLedger) -> None:
    answer, warnings = build_answer(FinalAnswerArgs(
        answer="二氧化钛 55 份[F1, F7]，见原文[T9]。工艺步骤 S1 不受影响。",
        values=[FinalValue(label="x", value=1, source="F8")],
        sources=["F1", "C5"],
    ), ledger)
    assert answer.answer == "二氧化钛 55 份[F1]，见原文。工艺步骤 S1 不受影响。"
    assert [c.id for c in answer.citations] == ["F1"]
    assert answer.values[0].verified is False and answer.values[0].source_id is None
    assert warnings[-1] == "回答引用了本轮没有出现过的来源 F7, T9, F8, C5，已删除"


def test_answered_without_sources_is_warned(ledger: SourceLedger) -> None:
    answer, warnings = build_answer(FinalAnswerArgs(answer="不知道"), ledger)
    assert answer.status is AnswerStatus.ANSWERED and warnings == ["回答没有引用任何来源"]
    _, warnings = build_answer(FinalAnswerArgs(answer="文档中没有提到粒径", status="insufficient_evidence"), ledger)
    assert warnings == []
