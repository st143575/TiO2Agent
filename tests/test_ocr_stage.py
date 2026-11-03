"""OcrStage end to end: upload through the API, with a fake PaddleOCR-VL service."""

import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from conftest import SAMPLES_DIR, upload, wait_until_done
from ocr_fakes import FakeLayoutParsingServer, layout_parsing_body
from tio2agent.config import Settings
from tio2agent.ingestion.ocr import PaddleOcrVlClient
from tio2agent.main import create_app

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


@pytest.fixture
def roomy(settings: Settings) -> Settings:
    return settings.model_copy(update={"max_upload_mb": 50, "max_pages": 200})


def run_upload(settings: Settings, server, content: bytes, **form: str) -> dict:
    engine = PaddleOcrVlClient("http://ocr:8080", transport=httpx.MockTransport(server))
    with TestClient(create_app(settings, ocr_engine=engine)) as client:
        response = upload(client, content, filename="doc.pdf", **form)
        assert response.status_code == 202, response.text
        return wait_until_done(client, response.json()["id"])


def test_sample_page6_table_is_recognized(roomy: Settings) -> None:
    server = FakeLayoutParsingServer()
    done = run_upload(roomy, server, SAMPLE.read_bytes())

    assert done["status"] == "ready", done["warnings"]
    assert done["stats"]["ocr_regions"] == 1
    # Only the 754x389 results table is sent; header strips and the cover logo are not.
    assert server.image_sizes() == [(754, 389)]

    work = roomy.processed_dir / done["id"]
    (region,) = json.loads((work / "ocr.json").read_text("utf-8"))
    assert region["id"] == "p6-img0"
    assert region["page"] == 6
    assert region["reason"] == "large_image"
    assert region["blocks"][0]["table"][1] == ["老化试验色差值", "1.21", "1.35", "1.46"]

    pages = json.loads((work / "pages.json").read_text("utf-8"))
    assert pages[5]["ocr"] == ["p6-img0"]
    assert all(p["ocr"] == [] for p in pages[:5])


def test_ocr_off_sends_nothing(roomy: Settings) -> None:
    server = FakeLayoutParsingServer()
    done = run_upload(roomy, server, SAMPLE.read_bytes(), ocr="off")
    assert server.requests == []
    assert done["stats"]["ocr_regions"] == 0
    # The structurer still notices that [0043] has no text.
    assert [(w["code"], w["page"]) for w in done["warnings"]] == [("STRUCTURE_EMPTY_PARAGRAPH", 6)]


def test_force_also_sends_medium_images(roomy: Settings) -> None:
    server = FakeLayoutParsingServer()
    run_upload(roomy, server, SAMPLE.read_bytes(), ocr="force")
    # Cover images of 113x89 and 74x74 now qualify; 116x17 / 134x45 strips are still skipped.
    assert sorted(server.image_sizes()) == [(74, 74), (113, 89), (754, 389)]


def test_service_error_becomes_warning(roomy: Settings) -> None:
    server = FakeLayoutParsingServer(
        lambda _: httpx.Response(500, json={"errorCode": 500, "errorMsg": "model not loaded"})
    )
    done = run_upload(roomy, server, SAMPLE.read_bytes())
    assert done["status"] == "ready_with_warnings"
    (warning,) = done["warnings"]
    assert (warning["code"], warning["page"], warning["stage"]) == ("OCR_FAILED", 6, "ocr")
    assert "model not loaded" in warning["message"]


def test_unavailable_service_skips_remaining_images(roomy: Settings) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    done = run_upload(roomy, refuse, SAMPLE.read_bytes(), ocr="force")
    codes = [(w["code"], w["page"]) for w in done["warnings"]]
    # One failed call, then no further requests: the rest are reported per page.
    assert codes == [("OCR_UNAVAILABLE", 1), ("OCR_SKIPPED", 1), ("OCR_SKIPPED", 6)]


def test_empty_result_is_flagged(roomy: Settings) -> None:
    server = FakeLayoutParsingServer(lambda _: httpx.Response(200, json=layout_parsing_body([])))
    done = run_upload(roomy, server, SAMPLE.read_bytes())
    assert [w["code"] for w in done["warnings"]] == ["OCR_EMPTY"]


def _scanned_pdf() -> bytes:
    """One page that is only an image, as a scanner would produce."""
    page = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(page).text((100, 100), "scanned", fill="black")
    buffer = io.BytesIO()
    page.save(buffer, format="PDF", resolution=150)
    return buffer.getvalue()


def test_scanned_page_is_covered_by_ocr(settings: Settings) -> None:
    server = FakeLayoutParsingServer(
        lambda _: httpx.Response(
            200, json=layout_parsing_body([{"label": "text", "content": "scanned"}], "scanned")
        )
    )
    done = run_upload(settings, server, _scanned_pdf())
    assert done["status"] == "ready", done["warnings"]  # no LOW_TEXT_PAGE once OCR filled it
    assert server.image_sizes() == [(1240, 1754)]


def test_scanned_page_without_ocr_is_reported(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:  # default: TIO2_OCR_BACKEND=none
        done = wait_until_done(client, upload(client, _scanned_pdf()).json()["id"])
    # One warning per page: OCR_SKIPPED already explains why the page has no text.
    assert [(w["code"], w["page"]) for w in done["warnings"]] == [("OCR_SKIPPED", 1)]
