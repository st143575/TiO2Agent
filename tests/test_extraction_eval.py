"""evals/run_extraction_eval.py: scoring, and full runs on the three gold documents."""

import importlib.util
from pathlib import Path

import pytest

from conftest import SAMPLES_DIR
from extraction_fakes import GoldChatModel

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("run_extraction_eval", ROOT / "evals/run_extraction_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load()


def _fact(subject, step, name, low, high=None, *, kind="parameter", qualifier=None, where=None, unit=None):
    return {"kind": kind, "subject": subject, "step": step, "name": name, "qualifier": qualifier, "low": low,
            "high": low if high is None else high, "unit": unit, "evidence": f"{name}{low}",
            "where": where or {"paragraph": "0027"}}


def test_score_counts_matches_near_misses_and_scope() -> None:
    gold = {
        "scope": {"claims": [1], "paragraphs": ["0027"]},
        "facts": [
            _fact("E1", "S2", "去离子水", 30, kind="component", unit="份"),
            _fact("E1", "S2", "转速", 800, unit="rpm"),
            _fact("E1", "S2", "反应时间", 3, unit="h"),
            _fact("claim:1", None, "有机硅", 3, 7, kind="component", where={"claim": 1}),
            {"kind": "result", "subject": "E1", "name": "水分散性", "where": {"figure": "图2"}},  # not scored
        ],
    }
    predicted = [
        _fact("E1", "S2", "去离子水中", 30, kind="component", unit="份"),      # right number, wrong name
        _fact("E1", "S2", "转速", 800, unit="rpm"),                           # correct
        _fact("claim:1", None, "有机硅", 3, 7, kind="component", where={"claim": 1}),   # correct
        _fact("E3", "S2", "搅拌时间", 30, where={"paragraph": "0046"}),        # outside the gold scope
    ]
    s = ev.score(gold, predicted)
    assert (s["gold"], s["predicted"], s["correct"]) == (4, 4, 2)
    assert (s["precision"], s["recall"], s["f1"]) == (0.5, 0.5, 0.5)
    assert [(n["gold"]["name"], n["predicted"]["name"], n["fields"]) for n in s["near_misses"]] == [
        ("去离子水", "去离子水中", ["name"])
    ]
    assert [g["name"] for g in s["missed"]] == ["反应时间"]
    assert [(x["name"], x["out_of_scope"]) for x in s["extra"]] == [("搅拌时间", True)]


def test_names_are_compared_after_the_alias_table() -> None:
    gold = {"scope": {"claims": [2], "paragraphs": []},
            "facts": [_fact("claim:2", None, "去离子", 45, kind="component", where={"claim": 2})]}
    predicted = [_fact("claim:2", None, "去离子水", 45, kind="component", where={"claim": 2})]
    assert ev.score(gold, predicted)["correct"] == 1


@pytest.mark.skipif(not all((SAMPLES_DIR / g["file"]).exists() for g in ev.load_gold()), reason="sample PDFs not present")
def test_full_run_on_the_gold_documents() -> None:
    """With a model that answers from the gold standard, LLM and hybrid extraction are
    exact; the rules' numbers are the current baseline (README, docs/extraction.md)."""
    report = ev.evaluate(ev.load_gold(), ["rules", "llm", "hybrid"], GoldChatModel())
    rules = {r["doc"]: (r["gold"], r["predicted"], r["correct"]) for r in report["modes"]["rules"]["documents"]}
    assert rules == {
        "CN114573021A": (46, 4, 3),
        "CN115403940A": (68, 46, 27),
        "CN115466527A": (117, 105, 105),
    }
    for mode in ("llm", "hybrid"):
        t = report["modes"][mode]["total"]
        assert (t["gold"], t["predicted"], t["correct"]) == (231, 231, 231), mode


@pytest.mark.skipif(not all((SAMPLES_DIR / g["file"]).exists() for g in ev.load_gold()), reason="sample PDFs not present")
def test_an_unreachable_model_does_not_hang_the_run() -> None:
    from tio2agent.llm import LlmError

    class Down:
        model, calls = "down", 0

        def chat(self, messages, *, tools=None, tool_choice=None):
            Down.calls += 1
            raise LlmError("无法连接大模型服务：timed out")

    lines: list[str] = []
    report = ev.evaluate(ev.load_gold(), ["llm", "hybrid"], Down(), progress=lines.append)
    assert Down.calls == 6                      # 2 per document, shared by llm and hybrid
    hybrid = report["modes"]["hybrid"]["total"]
    assert (hybrid["predicted"], hybrid["correct"]) == (155, 135)   # = the rules alone
    assert sum(line.endswith("跳过（大模型已连续失败 2 次）") for line in lines) == 3 * 11 - 6
