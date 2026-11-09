"""LLM fact extraction: verification of every proposed fact, merging with the rules,
and the wiring through settings and the API.

The model is GoldChatModel (tests/extraction_fakes.py): it submits the hand-labelled
gold facts for whatever claim or example it is asked about, optionally edited to
inject mistakes. So these tests show that correct facts are all accepted and that
each kind of mistake is caught (or, for one kind, that it is not).
"""

import tempfile
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, UPLOAD_URL, upload, wait_until_done
from extraction_fakes import GoldChatModel
from llm_fakes import call, reply, text
from tio2agent.config import ExtractBackend, Settings
from tio2agent.domain import DocumentRecord
from tio2agent.ingestion.extraction import (
    HybridExtractor,
    LlmPatentExtractor,
    PatentRuleExtractor,
    build_extractors,
)
from tio2agent.ingestion.extraction import evidence as ev
from tio2agent.ingestion.extraction.base import Source
from tio2agent.ingestion.extraction.llm_patent import Unit, parse_response, verify
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.structure import StructureInput, structure_document
from tio2agent.llm import ChatMessage, LlmError
from tio2agent.main import create_app

NYLON = SAMPLES_DIR / "一种水分散性好的锦纶化纤用钛白粉及其 制备方法.pdf"      # CN115403940A
ACRYLIC = SAMPLES_DIR / "一种过滤性好的腈纶化纤用钛白粉及其生 产方法.pdf"      # CN115466527A
needs_pdfs = pytest.mark.skipif(not (NYLON.exists() and ACRYLIC.exists()), reason="sample PDFs not present")


def _structure(pdf: Path):
    ctx = IngestionContext(
        record=DocumentRecord(id="d", filename=pdf.name, sha256="x", size_bytes=1),
        pdf_path=pdf,
        work_dir=Path(tempfile.mkdtemp()),
    )
    ParseStage().run(ctx)
    return structure_document(StructureInput("d", "patent", ctx.artifacts["pages"], [])).document


@pytest.fixture(scope="module")
def nylon():
    return _structure(NYLON)


@pytest.fixture(scope="module")
def acrylic():
    return _structure(ACRYLIC)


class Wrapped:
    """Passes requests to ``inner`` and lets a test rewrite or replace the response."""

    def __init__(self, inner, transform) -> None:
        self.inner, self.transform, self.model, self.calls = inner, transform, "wrapped", 0

    def chat(self, messages, *, tools=None, tool_choice=None):
        self.calls += 1
        return self.transform(self.calls, messages, lambda: self.inner.chat(messages, tools=tools, tool_choice=tool_choice))

    def close(self) -> None:
        pass


# --- evidence ------------------------------------------------------------------------------


def test_locate_returns_the_original_text() -> None:
    assert ev.locate("二氧化钛95-97％、金红石型", "二氧化钛95-97%") == "二氧化钛95-97％"   # CN114573021A claim 1
    assert ev.locate("升温至50-70℃后滴入", "升温至 50-70°C") == "升温至50-70℃"
    assert ev.locate("调整钛白粉料浆pH值为8-10", "PH值为8-10") == "pH值为8-10"
    assert ev.locate("加入3份有机硅", "加入5份有机硅") is None
    assert ev.locate("任何文字", "  ") is None


def test_numbers_and_step_references() -> None:
    assert ev.numbers("以800-1000r/min的转速") == {800.0, 1000.0}
    assert ev.numbers("熟化0.5小时") == {0.5}
    assert ev.step_references("所述步骤S2中包膜后") == {"S2"}           # CN115403940A claim 8
    assert ev.step_references("所述步骤4中经过漂白后的偏钛酸") == {"S4"}   # CN114573021A claim 8
    assert ev.step_references("S1、锐钛型…\n步骤5：煅烧") == {"S1", "S5"}


# --- reading the model's answer -------------------------------------------------------------


def test_parse_tool_call_and_text_forms() -> None:
    facts = [{"kind": "component", "name": "枸橼酸", "low": 4, "high": 4, "evidence": "4份枸橼酸"}]
    assert parse_response(reply(call("record_facts", facts=facts)).message)[0] == facts
    as_json = '{"facts": [{"kind": "component", "name": "枸橼酸", "low": 4, "high": 4, "evidence": "4份枸橼酸"}]}'
    for content in (
        f"<think>先找用量……</think>\n```json\n{as_json}\n```",
        f'<tool_call>{{"name": "record_facts", "arguments": {as_json}}}</tool_call>',
        f"结果如下：{as_json}",
        '[{"kind": "component", "name": "枸橼酸", "low": 4, "high": 4, "evidence": "4份枸橼酸"}]',
    ):
        assert parse_response(ChatMessage.assistant(content))[0] == facts, content
    assert parse_response(ChatMessage.assistant("原文中没有数值。")) == (None, "没有调用工具，也没有 JSON")
    assert parse_response(ChatMessage.assistant('{"facts": 3}'))[0] is None


# --- the extractor ---------------------------------------------------------------------------


@needs_pdfs
def test_every_gold_fact_passes_verification(nylon) -> None:
    model = GoldChatModel()
    result = LlmPatentExtractor(model).extract(nylon)
    assert (result.rejected, result.issues) == ([], [])
    assert len(result.facts) == 68 and result.stats["llm_calls"] == 11   # 8 claims + 3 examples
    by_key = {(f.subject, f.step, f.name, f.qualifier): f for f in result.facts}
    assert by_key[("claim:8", "S2", "料浆浓度", None)].display_value == "600-800g/L"   # "所述步骤S2中"
    assert by_key[("claim:7", "S3", "熟化时间", None)].evidence == "熟化0.5-2小时"     # step line of claim 7
    assert by_key[("E1", "S3", "pH调节剂", None)].paragraph_ids == ["0028"]
    assert ("E3", "S2", "有机硅", None) not in by_key        # the document gives no amount
    assert {f.extractor for f in result.facts} == {"llm"}
    # each example is sent with the component names read from the claims
    assert "权利要求中的组分名称：锐钛型钛白粉粗料、去离子水、pH调节剂、有机分散剂、枸橼酸、有机硅" in model.requests[8]


@needs_pdfs
def test_wrong_facts_are_rejected_with_reasons(nylon) -> None:
    bad = [
        {"kind": "component", "name": "锐钛型钛白粉粗料", "low": 65, "high": 65, "unit": "份", "evidence": "锐钛型钛白粉粗料65份"},
        {"kind": "component", "name": "去离子水", "low": 35, "high": 50, "unit": "份", "evidence": "去离子水30-50份"},
        {"kind": "component", "name": "枸橼酸", "low": 6, "high": 8, "unit": "份", "evidence": "有机分散剂6-8份"},
        {"kind": "option", "name": "有机分散剂", "options": ["聚羧酸钠", "聚乙二醇"], "evidence": "有机分散剂6-8份"},
        {"kind": "result", "name": "水分散性", "low": 92, "high": 92, "evidence": "有机硅3-7份"},
        {"kind": "component", "name": "有机硅", "low": 3, "high": 3, "unit": "份", "evidence": "有机硅3-7份"},
    ]

    model = GoldChatModel(edit=lambda subject, facts: facts + bad if subject == "claim:1" else facts)
    result = LlmPatentExtractor(model).extract(nylon)
    assert [r.reason for r in result.rejected] == [
        "证据不在原文中", "数值不在证据中", "名称不在证据中", "可选材料不在证据中", "类型无效",
        "与同一位置的另一条事实重复但数值不同",
    ]
    assert [(i.code, i.message.split("（")[0]) for i in result.issues] == [
        ("EXTRACT_LLM_REJECTED", "权利要求1：大模型给出的 6 条事实未通过核对，已丢弃"),
    ]
    assert len(result.facts) == 68       # the correct ones are all still there


def test_tolerated_differences_in_a_proposed_fact() -> None:
    text = "S3、…升温至50℃后滴入4份枸橼酸，保温搅拌，搅拌时间为30min"          # CN115403940A [0028]
    unit = Unit("E1", "实施例一", "", [Source(text, "E1", "实施例一", paragraph_ids=["0028"], step="S3")], False)
    source, fields = verify(
        {"kind": "parameter", "name": "温度", "low": "50", "unit": "°C", "evidence": "[0028] 升温至 50°C"}, unit
    )
    assert (fields["evidence"], fields["low"], fields["high"], fields["value"], fields["unit"], fields["step"]) == (
        "升温至50℃", 50.0, 50.0, 50.0, "℃", "S3"
    )
    _, fields = verify({"kind": "component", "name": "枸橼酸", "high": 4, "unit": "份", "evidence": "4份枸橼酸"}, unit)
    assert (fields["low"], fields["high"]) == (4.0, 4.0)                    # only one bound given
    assert verify({"kind": "parameter", "name": "PH值", "low": 8, "evidence": "4份枸橼酸"}, unit) == "数值不在证据中"
    assert verify({"kind": "parameter", "name": "温度", "low": True, "evidence": "升温至50℃"}, unit) == "缺少数值"


@needs_pdfs
def test_steps_come_from_the_text_not_the_model(nylon) -> None:
    def edit(subject, facts):
        if subject == "claim:8":
            return [{**f, "step": "S3"} for f in facts]       # claim 8 only names S2
        if subject == "E1":
            return [{**f, "step": "S1"} for f in facts]       # paragraphs carry S1-S4
        return facts

    facts = LlmPatentExtractor(GoldChatModel(edit=edit)).extract(nylon).facts
    assert [f.step for f in facts if f.subject == "claim:8"] == [None]
    e1 = [(f.paragraph_ids[0], f.step) for f in facts if f.subject == "E1"]
    assert {pid: step for pid, step in e1} == {"0026": "S1", "0027": "S2", "0028": "S3", "0029": "S4"}


@needs_pdfs
def test_llm_errors_and_bad_output_do_not_stop_extraction(nylon) -> None:
    def some_fail(n, messages, ask):                       # claims 1 and 3 time out
        if any(f"来源：claim:{k}（" in messages[1].content for k in (1, 3)):
            raise LlmError("无法连接大模型服务：timed out")
        return ask()

    lines: list[str] = []
    result = LlmPatentExtractor(Wrapped(GoldChatModel(), some_fail), progress=lines.append).extract(nylon)
    assert {f.subject for f in result.facts} >= {"claim:2", "claim:4", "E1", "E2", "E3"}
    assert Counter(i.code for i in result.issues) == {"EXTRACT_LLM_FAILED": 2}
    assert (result.stats["failed_units"], result.stats["skipped_units"]) == (2, 0)
    assert lines[0].startswith("权利要求1：失败") and lines[1].startswith("权利要求2：6 条，丢弃 0 条")

    def down(n, messages, ask):                            # every request fails: stop after 2
        raise LlmError("无法连接大模型服务：timed out")

    model = Wrapped(GoldChatModel(), down)
    result = LlmPatentExtractor(model, max_failures=2).extract(nylon)
    assert (model.calls, result.facts) == (2, [])
    assert [i.code for i in result.issues] == ["EXTRACT_LLM_FAILED", "EXTRACT_LLM_FAILED", "EXTRACT_LLM_STOPPED"]
    assert result.stats["skipped_units"] == 9               # 8 claims + 3 examples - 2 tried

    def first_answer_is_prose(n, messages, ask):
        return text("这条权利要求没有数值。") if n == 1 else ask()

    model = Wrapped(GoldChatModel(), first_answer_is_prose)
    result = LlmPatentExtractor(model).extract(nylon)
    assert len(result.facts) == 68 and model.calls == 12     # claim 1 asked twice
    assert result.issues == []


@needs_pdfs
def test_json_written_as_text_is_accepted(nylon) -> None:
    def as_text(n, messages, ask):
        arguments = ask().message.tool_calls[0].arguments
        return text(f"<think>逐句检查……</think>\n```json\n{arguments}\n```")

    result = LlmPatentExtractor(Wrapped(GoldChatModel(), as_text)).extract(nylon)
    assert len(result.facts) == 68 and result.rejected == []


# --- hybrid ----------------------------------------------------------------------------------


@needs_pdfs
def test_hybrid_takes_names_from_the_llm(nylon) -> None:
    rules = PatentRuleExtractor().extract(nylon).facts
    assert {"去离子水中", "有机硅进行混合", "再使用"} <= {f.name for f in rules}     # rule names cut wrongly

    result = HybridExtractor(PatentRuleExtractor(), LlmPatentExtractor(GoldChatModel())).extract(nylon)
    names = {f.name for f in result.facts}
    assert not names & {"去离子水中", "有机硅进行混合", "再使用"}
    assert len(result.facts) == 68
    assert Counter(f.extractor for f in result.facts) == {"hybrid": len(rules), "llm": 68 - len(rules)}
    assert [f.id for f in result.facts][:2] == ["d-f0001", "d-f0002"]
    assert result.stats["rule_facts"] == len(rules)


@needs_pdfs
def test_hybrid_keeps_rule_facts_when_the_llm_finds_nothing(acrylic) -> None:
    rules = PatentRuleExtractor().extract(acrylic).facts
    result = HybridExtractor(PatentRuleExtractor(), LlmPatentExtractor(GoldChatModel(edit=lambda s, f: []))).extract(acrylic)
    # the same facts, now in document order (the rules list components before parameters)
    assert sorted((f.subject, f.step, f.name, f.qualifier, f.low, f.high, f.extractor) for f in result.facts) == sorted(
        (f.subject, f.step, f.name, f.qualifier, f.low, f.high, "patent_rules") for f in rules
    )
    s3 = [f.name for f in result.facts if (f.subject, f.step) == ("E1", "S3")]
    assert s3 == ["加入时间", "锆盐", "pH", "加入时间", "铝盐", "pH"]   # no 温度: rules miss "升温至50℃"


@needs_pdfs
def test_a_misnamed_parameter_is_only_caught_by_the_rules(acrylic) -> None:
    """The LLM calls the 40 in "40份去离子水" the S1 temperature. Number and quote are
    right, so verification accepts it (a known limit); the hybrid keeps the rules'
    55 ℃ and reports the conflict."""
    wrong = {"kind": "parameter", "name": "温度", "step": "S1", "low": 40, "high": 40, "unit": "℃",
             "evidence": "40份去离子水", "paragraph": "0027"}

    def edit(subject, facts):
        if subject != "E1":
            return facts
        return [wrong if (f["name"], f.get("step")) == ("温度", "S1") else f for f in facts]

    alone = LlmPatentExtractor(GoldChatModel(edit=edit)).extract(acrylic).facts
    assert [f.value for f in alone if (f.subject, f.step, f.name) == ("E1", "S1", "温度")] == [40]

    result = HybridExtractor(PatentRuleExtractor(), LlmPatentExtractor(GoldChatModel(edit=edit))).extract(acrylic)
    assert [(f.value, f.evidence) for f in result.facts if (f.subject, f.step, f.name) == ("E1", "S1", "温度")] == [
        (55, "温度设为55℃")
    ]
    conflicts = [i.message for i in result.issues if i.code == "EXTRACT_CONFLICT"]
    assert conflicts == ["实施例1 S1 温度：保留55℃（说明书[0027]「温度设为55℃」），舍弃40℃（说明书[0027]「40份去离子水」）"]


# --- wiring ----------------------------------------------------------------------------------


def test_build_extractors_by_backend() -> None:
    model = GoldChatModel()
    assert build_extractors(ExtractBackend.RULES, model)["patent"].name == "patent_rules"
    assert build_extractors(ExtractBackend.LLM, None)["patent"].name == "patent_rules"   # no LLM: fall back
    assert isinstance(build_extractors(ExtractBackend.LLM, model)["patent"], LlmPatentExtractor)
    assert isinstance(build_extractors(ExtractBackend.HYBRID, model)["patent"], HybridExtractor)


@needs_pdfs
def test_upload_with_hybrid_extraction(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", max_upload_mb=50, max_pages=200, ingest_workers=1,
        extract_backend=ExtractBackend.HYBRID,
    )
    with TestClient(create_app(settings, extraction_model=GoldChatModel())) as client:
        response = upload(client, NYLON.read_bytes(), filename="nylon.pdf")
        done = wait_until_done(client, response.json()["id"])
        # the document itself gives no amount of 有机硅 in 实施例三 (CN115403940A [0037])
        assert {w["code"] for w in done["warnings"]} == {"OCR_SKIPPED", "EXTRACT_MISSING_COMPONENT"}
        assert done["stats"]["facts"] == 68
        items = client.get(f"{UPLOAD_URL}/{done['id']}/facts", params={"subject": "E1", "step": "S2"}).json()["items"]
        assert [(f["name"], f["display_value"], f["extractor"]) for f in items] == [
            ("去离子水", "30份", "hybrid"), ("有机硅", "3份", "hybrid"),
            ("转速", "800rpm", "llm"), ("反应时间", "3h", "llm"),
        ]


def test_extraction_model_turns_thinking_off() -> None:
    import json

    import httpx

    from tio2agent.config import LlmBackend
    from tio2agent.llm import OpenAICompatibleChatModel, build_extraction_model

    settings = Settings(_env_file=None, llm_backend=LlmBackend.OPENAI_COMPATIBLE)
    model = build_extraction_model(settings)
    assert (model.extra_body, model.max_retries, model.max_tokens) == ({"reasoning_effort": "none"}, 0, 4096)
    model.close()
    model = build_extraction_model(settings.model_copy(update={"extract_llm_reasoning_effort": ""}))
    assert model.extra_body == {}
    model.close()

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "{}"},
                                                      "finish_reason": "stop"}]})

    client = OpenAICompatibleChatModel("http://x/v1", "qwen3:8b", extra_body={"reasoning_effort": "none"},
                                       transport=httpx.MockTransport(handler))
    client.chat([ChatMessage.user("hi")])
    assert bodies[0]["reasoning_effort"] == "none"
