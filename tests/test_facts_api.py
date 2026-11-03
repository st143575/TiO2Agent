"""Facts end to end: upload the sample (fake OCR for the results table), query the API."""

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, UPLOAD_URL, upload, wait_until_done
from ocr_fakes import FakeLayoutParsingServer
from tio2agent.config import Settings
from tio2agent.ingestion.ocr import PaddleOcrVlClient
from tio2agent.main import create_app
from tio2agent.storage import FactRepository

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    settings = Settings(
        _env_file=None, data_dir=tmp_path_factory.mktemp("data"), max_upload_mb=50, max_pages=200, ingest_workers=1
    )
    engine = PaddleOcrVlClient("http://ocr:8080", transport=httpx.MockTransport(FakeLayoutParsingServer()))
    with TestClient(create_app(settings, ocr_engine=engine)) as client:
        response = upload(client, SAMPLE.read_bytes(), filename="doc.pdf")
        assert response.status_code == 202, response.text
        done = wait_until_done(client, response.json()["id"])
        yield client, done, settings


def test_document_reports_fact_count(ingested) -> None:
    _, done, settings = ingested
    assert done["status"] == "ready", done["warnings"]
    assert done["stats"]["facts"] == 106
    assert (settings.processed_dir / done["id"] / "facts.json").exists()


def test_filter_document_facts(ingested) -> None:
    client, done, _ = ingested
    url = f"{UPLOAD_URL}/{done['id']}/facts"
    body = client.get(url, params={"subject": "实施例1", "name": "温度"}).json()
    assert [(f["step"], f["display_value"], f["citation"]) for f in body["items"]] == [
        ("S1", "65℃", "说明书[0026]"),
        ("S3", "35℃", "说明书[0028]"),
    ]
    results = client.get(url, params={"kind": "result", "name": "色差"}).json()["items"]
    assert [(f["subject"], f["value"], f["citation"]) for f in results] == [
        ("E1", 1.21, "说明书[0043]表格"), ("E2", 1.35, "说明书[0043]表格"), ("E3", 1.46, "说明书[0043]表格"),
    ]
    # the written alias is searchable too
    (alias,) = client.get(url, params={"name": "去离子", "subject": "claim:2"}).json()["items"]
    assert alias["name"] == "去离子水" and alias["name_raw"] == "去离子"


def test_formulations(ingested) -> None:
    client, done, _ = ingested
    body = client.get(f"{UPLOAD_URL}/{done['id']}/formulations", params={"subject": "E3"}).json()
    (e3,) = body["items"]
    assert e3["total"] == 167.0 and e3["unit"] == "份" and e3["is_range"] is False
    assert [c["name"] for c in e3["components"]] == ["二氧化钛", "去离子水", "分散剂", "消泡剂", "抗氧化剂", "包膜剂", "锆盐", "铝盐"]


def test_cross_document_query_and_validation(ingested) -> None:
    client, _, _ = ingested
    body = client.get("/api/v1/facts", params={"name": "pH", "step": "S3", "kind": "parameter"}).json()
    assert body["total"] == 8  # claim 8 (2 ranges) + 3 examples × 2
    example = next(f for f in body["items"] if f["subject"] == "E1")
    assert example["checks"][0]["ok"] and example["checks"][0]["reference"] == "权利要求8"
    assert client.get("/api/v1/facts", params={"doc_id": "nope"}).json()["total"] == 0
    assert client.get(f"{UPLOAD_URL}/nope/facts").status_code == 404


def test_opt_out_and_delete(ingested) -> None:
    client, done, settings = ingested
    response = upload(client, SAMPLE.read_bytes() + b"\n%copy", filename="copy.pdf", extract_facts="false")
    assert response.status_code == 202, response.text
    copy = wait_until_done(client, response.json()["id"])
    assert copy["stats"]["facts"] == 0
    assert client.get(f"{UPLOAD_URL}/{copy['id']}/facts").json()["total"] == 0

    repo = FactRepository(settings.db_path)
    try:
        assert repo.count(done["id"]) == 106
        assert client.delete(f"{UPLOAD_URL}/{copy['id']}").status_code == 204
        assert repo.count() == 106
    finally:
        repo.close()
