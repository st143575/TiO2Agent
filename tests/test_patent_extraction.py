"""PatentRuleExtractor on the sample patent, compared with hand-checked facts.

tests/fixtures/CN116694108A.facts.expected.json was checked line by line against the
PDF: claims 1-8, examples [0026]-[0039] and the page-6 results table.
"""

import json
from pathlib import Path

import pytest

from conftest import SAMPLES_DIR
from ocr_fakes import P6_TABLE_HTML
from tio2agent.domain import DocumentRecord, FactKind, group_formulations
from tio2agent.ingestion.extraction import extract_facts
from tio2agent.ingestion.ocr import parse_table
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import ParseStage
from tio2agent.ingestion.structure import StructureInput, structure_document

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
EXPECTED = Path(__file__).parent / "fixtures" / "CN116694108A.facts.expected.json"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


def structure(tmp_path: Path, *, with_table: bool = True):
    ctx = IngestionContext(
        record=DocumentRecord(id="d", filename="x.pdf", sha256="x", size_bytes=1),
        pdf_path=SAMPLE,
        work_dir=tmp_path,
    )
    ParseStage().run(ctx)
    regions = []
    if with_table:
        block = {"label": "table", "content": P6_TABLE_HTML, "table": parse_table(P6_TABLE_HTML)}
        regions = [{"id": "p6-img0", "page": 6, "blocks": [block]}]
    return structure_document(StructureInput("d", "patent", ctx.artifacts["pages"], regions)).document


@pytest.fixture(scope="module")
def facts(tmp_path_factory):
    result = extract_facts(structure(tmp_path_factory.mktemp("sample")))
    assert result.issues == []
    return result.facts


def test_matches_hand_checked_facts(facts) -> None:
    got = [[f.kind.value, f.subject, f.step, f.name, f.qualifier, f.display_value, f.citation] for f in facts]
    assert got == json.loads(EXPECTED.read_text("utf-8"))


def test_every_evidence_is_in_the_source(facts, tmp_path) -> None:
    doc = structure(tmp_path)
    for f in facts:
        if f.claim_number is not None:
            assert f.evidence in doc.claims[f.claim_number - 1].text
        elif f.table_id is None:
            assert f.evidence in doc.paragraph(f.paragraph_ids[0]).text


def test_restatements_and_comparatives_are_not_read(facts) -> None:
    paragraphs = {p for f in facts for p in f.paragraph_ids}
    assert paragraphs <= {"0026", "0027", "0028", "0031", "0032", "0033", "0036", "0037", "0038", "0043"}
    assert not any(f.subject == "C1" for f in facts)


def test_alias_and_options(facts) -> None:
    (alias,) = [f for f in facts if f.name_raw]
    assert (alias.subject, alias.name, alias.name_raw, alias.evidence) == ("claim:2", "去离子水", "去离子", "去离子45份")
    options = {f.name: f.options for f in facts if f.kind is FactKind.OPTION}
    assert options["抗氧化剂"] == ["2,6-二叔丁基甲酚", "丁基羟基茴香醚", "叔丁基对苯二酚"]
    assert options["包膜剂"] == ["氧化硅", "氧化铝", "氧化锆"]


def test_ph_is_qualified_by_the_preceding_addition(facts) -> None:
    ph = [(f.qualifier, f.value) for f in facts if f.subject == "E2" and f.name == "pH"]
    assert ph == [("加入锆盐后", 5), ("加入铝盐后", 4)]


def test_all_examples_are_within_claimed_ranges(facts) -> None:
    checks = [c for f in facts for c in f.checks if c.rule == "within_claimed_range"]
    assert len(checks) == 78 and all(c.ok for c in checks)


def test_formulation_totals(facts) -> None:
    totals = {fm.subject: fm.total for fm in group_formulations(facts)}
    assert totals == {"claim:1": None, "claim:2": 144.6, "claim:3": 122.2, "claim:4": 167.0,
                      "E1": 144.6, "E2": 122.2, "E3": 167.0}


def test_without_ocr_there_are_no_results(tmp_path) -> None:
    result = extract_facts(structure(tmp_path, with_table=False))
    assert len(result.facts) == 100
    assert not any(f.kind is FactKind.RESULT for f in result.facts)
