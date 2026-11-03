"""GET /api/v1/documents/{id}/structure and the structuring stage in the pipeline."""

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, UPLOAD_URL, upload, wait_until_done
from ocr_fakes import FakeLayoutParsingServer
from tio2agent.config import Settings
from tio2agent.ingestion.ocr import PaddleOcrVlClient
from tio2agent.main import create_app

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


@pytest.fixture
def roomy(settings: Settings) -> Settings:
    return settings.model_copy(update={"max_upload_mb": 50, "max_pages": 200})


def test_structure_endpoint_without_ocr(roomy: Settings) -> None:
    with TestClient(create_app(roomy)) as client:
        doc_id = upload(client, SAMPLE.read_bytes()).json()["id"]
        done = wait_until_done(client, doc_id)
        response = client.get(f"{UPLOAD_URL}/{doc_id}/structure")

    assert done["stats"]["paragraphs"] == 45
    assert done["stats"]["embodiments"] == 3
    # One problem, one warning: the OCR warning names the empty paragraph.
    (warning,) = done["warnings"]
    assert warning["code"] == "OCR_SKIPPED"
    assert warning["message"].endswith("（即段落[0043]）")

    assert response.status_code == 200
    body = response.json()
    assert body["structurer"] == "patent_cn"
    assert body["bibliographic"]["publication_number"] == "CN116694108A"
    assert len(body["claims"]) == 8
    assert "raw" not in body["paragraphs"][0] and "raw" not in body["claims"][0]


def test_structure_with_ocr_table_and_raw(roomy: Settings) -> None:
    engine = PaddleOcrVlClient("http://ocr:8080", transport=httpx.MockTransport(FakeLayoutParsingServer()))
    with TestClient(create_app(roomy, ocr_engine=engine)) as client:
        doc_id = upload(client, SAMPLE.read_bytes()).json()["id"]
        done = wait_until_done(client, doc_id)
        body = client.get(f"{UPLOAD_URL}/{doc_id}/structure", params={"include_raw": True}).json()

    assert done["status"] == "ready" and done["warnings"] == []
    assert body["tables"][0]["paragraph_id"] == "0043"
    p7 = next(p for p in body["paragraphs"] if p["id"] == "0007")
    assert "0 . 5" in p7["raw"] and "0.5-0.9" in p7["text"]


def test_structure_errors(roomy: Settings, make_pdf) -> None:
    with TestClient(create_app(roomy)) as client:
        assert client.get(f"{UPLOAD_URL}/doc_missing/structure").status_code == 404

        doc_id = upload(client, make_pdf()).json()["id"]
        wait_until_done(client, doc_id)
        (roomy.processed_dir / doc_id / "structure.json").unlink()  # e.g. imported before this feature
        response = client.get(f"{UPLOAD_URL}/{doc_id}/structure")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "STRUCTURE_NOT_AVAILABLE"
    assert "reindex" in response.json()["error"]["message"]
