"""evals/run_agent_eval.py: scoring rules, and the full run with a scripted model."""

import importlib.util
from pathlib import Path

import pytest

from llm_fakes import ScriptedChatModel, call, fact_id, reply

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/samples/一种高耐候钛白粉及制备方法.pdf"


def _load():
    spec = importlib.util.spec_from_file_location("run_agent_eval", ROOT / "evals/run_agent_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load()


def test_citation_tokens() -> None:
    assert ev.citation_tokens("说明书[0026]-[0028]、[0031]") == {"0026", "0027", "0028", "0031"}
    assert ev.citation_tokens("权利要求1、2") == {"权利要求1", "权利要求2"}
    assert ev.citation_tokens("扉页著录项") == {"扉页著录项"}


def response(status="answered", text="", values=(), citations=(), warnings=()):
    return {
        "answer": {"status": status, "answer": text, "values": list(values),
                   "citations": [{"citation": c} for c in citations]},
        "warnings": list(warnings),
    }


def test_score() -> None:
    item = {"expected": {"values": [{"value": 38.92, "tolerance": 0.05}], "citations": ["0036"], "keywords": ["二氧化钛"]}}
    good = response(text="二氧化钛占 38.9%", values=[{"value": 38.9, "verified": True}], citations=["说明书[0036]", "计算"])
    assert ev.score(item, good)["passed"]

    unverified = response(text="二氧化钛占 38.92%", values=[{"value": 38.92, "verified": False}], citations=["说明书[0036]"])
    s = ev.score(item, unverified)
    assert not s["passed"] and s["checks"]["values"] is False and s["values_in_text_only"]

    wrong_place = response(text="二氧化钛 38.92", values=[{"value": 38.92, "verified": True}], citations=["说明书[0026]"])
    assert ev.score(item, wrong_place)["checks"] == {"status": True, "values": True, "citations": False, "keywords": True}

    negative = {"expected": {"status": "insufficient_evidence"}}
    assert ev.score(negative, response(status="insufficient_evidence"))["passed"]
    assert not ev.score(negative, response(status="answered", text="粒径 0.2 μm"))["passed"]

    rng = {"expected": {"values": [{"low": 6, "high": 8}]}}
    assert ev.score(rng, response(values=[{"low": 6, "high": 8, "verified": True}]))["passed"]


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")
def test_full_run_with_scripted_model() -> None:
    model = ScriptedChatModel(
        # a01: look the fact up, answer with it
        reply(call("query_facts", subject="实施例1", name="二氧化钛")),
        lambda msgs: reply(call(
            "final_answer", answer=f"实施例1中二氧化钛用量为 55 份[{fact_id(msgs, '二氧化钛', '实施例1')}]。",
            values=[{"label": "二氧化钛", "value": 55, "unit": "份", "source": fact_id(msgs, "二氧化钛", "实施例1")}],
        )),
        # a14: nothing about particle size
        reply(call("search_documents", query="二氧化钛 粒径")),
        reply(call("final_answer", answer="文档中没有给出二氧化钛原料的粒径。", status="insufficient_evidence")),
    )
    report = ev.evaluate(only={"a01", "a14", "a15"}, env_file=None, chat_model=model)
    rows = {r["id"]: r for r in report["items"]}
    assert rows["a01"]["passed"] and rows["a01"]["tools"] == ["query_facts"]
    assert rows["a14"]["passed"]
    assert rows["a15"]["skipped"] == "需要 OCR"
    assert report["summary"] | {"avg_seconds": 0} == {
        "questions": 3, "skipped": 1, "passed": 2, "pass_rate": 1.0, "by_type": {"fact": "1/1", "negative": "1/1"},
        "errors": 0, "values_in_text_only": 0, "with_grounding_warnings": 0,
        "avg_llm_calls": 2.0, "avg_tokens": 240, "avg_seconds": 0,
    }
    assert report["model"] == "scripted"
