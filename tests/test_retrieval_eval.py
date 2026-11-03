"""Regression guard on retrieval quality (BM25, sample patent, evals/datasets/retrieval_gold.jsonl).

Thresholds sit a little below the measured values (hit@1 0.875, hit@5 0.958, MRR 0.917
on 24 questions) so that a change which makes retrieval worse fails here. They are not
targets to tune towards: 24 questions on one document are far too few for that.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/samples/一种高耐候钛白粉及制备方法.pdf"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", ROOT / "evals/run_retrieval_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bm25_report() -> dict:
    return _load_eval_module().evaluate(modes=("bm25",), k=10, env_file=None)["modes"]["bm25"]


def test_overall_bm25_quality(bm25_report) -> None:
    m = bm25_report["metrics"]
    assert m["mode_used"] == ["bm25"]
    assert m["hit@5"] >= 0.9
    assert m["hit@1"] >= 0.8
    assert m["mrr@10"] >= 0.85


def test_exact_fact_questions_rank_first(bm25_report) -> None:
    by_type = bm25_report["by_type"]
    # Questions that reuse the document's own terms must be answered by the first hit.
    for kind in ("bibliographic", "fact", "comparison", "background"):
        assert by_type[kind]["hit@1"] == 1.0, kind


def test_relevance_labels() -> None:
    module = _load_eval_module()
    hit = {"chunk": {"kind": "paragraph", "claim_numbers": [], "paragraph_ids": ["0014", "0015"]}, "also_in": ["权利要求8"]}
    assert module.is_relevant(hit, ["claim:8"])          # claim absorbed by its restatement
    assert module.is_relevant(hit, ["p:0015"])
    assert not module.is_relevant(hit, ["p:0026", "abstract"])
