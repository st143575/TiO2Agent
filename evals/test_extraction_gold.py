"""Consistency checks for the hand-labelled extraction gold (evals/datasets/extraction_gold).

These tests do not judge any extractor. They check the labels against the PDF text:

    evidence     every text-layer evidence string occurs in the cited claim / paragraph
    numbers      every labelled number occurs in its evidence; every option too
    coverage     every number in an in-scope claim or paragraph is labelled
                 (step and claim references such as "S2", "步骤1", "权利要求1-4" excepted)
    steps        a fact's step matches the step line its evidence comes from
    uniqueness   (subject, step, name, qualifier) occurs once per document

Results read from images (requires_ocr) cannot be checked against the text layer;
they were copied by hand from the figure and are reviewed by eye.
"""

import json
import re
import tempfile
from pathlib import Path

import pytest

from conftest import SAMPLES_DIR
from tio2agent.domain import DocumentRecord
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.structure import StructureInput, structure_document
from tio2agent.ingestion.structure.normalize import step_id

GOLD_DIR = Path(__file__).parents[1] / "evals" / "datasets" / "extraction_gold"
GOLD_FILES = sorted(GOLD_DIR.glob("CN*.json"))

KINDS = {"component", "parameter", "option", "result"}
UNITS = {None, "份", "%", "℃", "rpm", "min", "h", "g/L", "μm"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_REFERENCES = re.compile(r"S\d+|步骤\d+|权利要求\d+(?:-\d+)?|实施例\d+")


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(_REFERENCES.sub("", text))}


def _values(fact: dict) -> set[float]:
    return {float(v) for v in (fact.get("low"), fact.get("high")) if v is not None}


def _load(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


@pytest.fixture(scope="module", params=GOLD_FILES, ids=lambda p: p.stem)
def case(request):
    gold = _load(request.param)
    pdf = SAMPLES_DIR / gold["file"]
    if not pdf.exists():
        pytest.skip(f"sample PDF not present: {gold['file']}")
    ctx = IngestionContext(
        record=DocumentRecord(id="d", filename=pdf.name, sha256="x", size_bytes=1),
        pdf_path=pdf,
        work_dir=Path(tempfile.mkdtemp()),
    )
    ParseStage().run(ctx)
    doc = structure_document(StructureInput("d", "patent", ctx.artifacts["pages"], [])).document
    return gold, doc


def _source(doc, where: dict) -> str | None:
    if "claim" in where:
        return doc.claims[where["claim"] - 1].text
    if "figure" in where:
        return None
    return doc.paragraph(where["paragraph"]).text


# --- schema (no PDF needed) -----------------------------------------------------------


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_schema(path: Path) -> None:
    gold = _load(path)
    assert gold["doc"] == path.stem and gold["status"] in {"draft", "reviewed"}
    assert set(gold["scope"]) >= {"claims", "paragraphs"}
    ids = [f["id"] for f in gold["facts"]]
    assert len(ids) == len(set(ids))
    for f in gold["facts"]:
        assert f["kind"] in KINDS, f
        assert f["name"] and f["evidence"] and f["where"], f
        if f["kind"] == "option":
            assert len(f["options"]) >= 2, f
            continue
        assert f.get("unit") in UNITS, f
        low, high = f.get("low"), f.get("high")
        if f["kind"] == "result":
            assert f.get("requires_ocr") is True and "figure" in f["where"], f
            assert low is not None or high is not None or f.get("text"), f
        else:
            assert low is not None and high is not None and low <= high, f
        if f["subject"].startswith("claim:"):
            assert f["where"] == {"claim": int(f["subject"].split(":")[1])}, f


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_unique_keys(path: Path) -> None:
    keys = [(f["subject"], f["step"], f["name"], f["qualifier"]) for f in _load(path)["facts"]]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes


# --- against the PDF text -------------------------------------------------------------


def test_evidence_in_source(case) -> None:
    gold, doc = case
    for f in gold["facts"]:
        text = _source(doc, f["where"])
        if text is None:
            continue
        assert _squash(f["evidence"]) in _squash(text), (f["id"], f["evidence"])


def test_values_in_evidence(case) -> None:
    gold, _ = case
    for f in gold["facts"]:
        if f["kind"] == "option":
            assert all(o in f["evidence"] for o in f["options"]), f["id"]
        else:
            assert _values(f) <= _numbers(f["evidence"]), (f["id"], f["evidence"])


def test_every_number_in_scope_is_labelled(case) -> None:
    gold, doc = case
    labelled: dict[str, set[float]] = {}
    for f in gold["facts"]:
        w = f["where"]
        key = f"claim:{w['claim']}" if "claim" in w else w.get("paragraph", "")
        labelled.setdefault(key, set()).update(_values(f))
    sources = [(f"claim:{n}", doc.claims[n - 1].text) for n in gold["scope"]["claims"]]
    sources += [(pid, doc.paragraph(pid).text) for pid in gold["scope"]["paragraphs"]]
    missing = {key: sorted(_numbers(text) - labelled.get(key, set())) for key, text in sources}
    assert {k: v for k, v in missing.items() if v} == {}


def test_steps_match_step_lines(case) -> None:
    gold, doc = case
    for f in gold["facts"]:
        text = _source(doc, f["where"])
        if text is None:
            continue
        if "paragraph" in f["where"]:
            assert step_id(text) == f["step"], f["id"]
            continue
        lines = [line for line in text.splitlines() if step_id(line)]
        if not lines:  # "所述步骤S2中…": the step comes from the reference
            continue
        line = next(line for line in lines if _squash(f["evidence"]) in _squash(line))
        assert step_id(line) == f["step"], f["id"]


def test_scope_paragraphs_are_examples(case) -> None:
    """The scope lists example paragraphs only (no restatements, no comparative text)."""
    gold, doc = case
    for pid in gold["scope"]["paragraphs"]:
        emb = doc.paragraph(pid).embodiment
        assert emb is not None and emb.startswith("E"), pid
