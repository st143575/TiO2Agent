import json

import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, UPLOAD_URL, upload, wait_until_done
from tio2agent.main import create_app
from tio2agent.domain import DocumentStatus


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- happy path ------------------------------------------------------------------


def test_upload_returns_202_and_finishes(client: TestClient, make_pdf, settings) -> None:
    response = upload(client, make_pdf(2), filename="blank.pdf")

    assert response.status_code == 202
    body = response.json()
    assert body["id"].startswith("doc_")
    assert body["filename"] == "blank.pdf"
    assert body["duplicate"] is False
    assert body["status"] in {"queued", "parsing", "ready_with_warnings"}
    assert response.headers["location"].endswith(f"{UPLOAD_URL}/{body['id']}")
    assert (settings.raw_dir / f"{body['sha256']}.pdf").exists()

    done = wait_until_done(client, body["id"])
    # Blank pages have no text layer, so they are flagged for OCR.
    assert done["status"] == "ready_with_warnings"
    assert done["stats"]["pages"] == 2
    assert {w["code"] for w in done["warnings"]} == {"LOW_TEXT_PAGE"}
    assert [w["page"] for w in done["warnings"]] == [1, 2]
    assert done["warnings"][0]["stage"] == "ocr"  # OcrStage reports pages it could not fill
    assert (settings.processed_dir / body["id"] / "pages.json").exists()


def test_metadata_and_options_are_stored(client: TestClient, make_pdf) -> None:
    response = upload(
        client,
        make_pdf(),
        metadata=json.dumps({"title": "自定义标题", "tags": ["包膜", "耐候"]}),
        doc_type="patent",
        ocr="off",
        extract_facts="false",
    )
    assert response.status_code == 202
    done = wait_until_done(client, response.json()["id"])
    assert done["title"] == "自定义标题"
    assert done["tags"] == ["包膜", "耐候"]
    assert done["doc_type"] == "patent"
    assert done["options"] == {"doc_type": "patent", "ocr": "off", "extract_facts": False}


def test_filename_path_is_stripped(client: TestClient, make_pdf) -> None:
    response = upload(client, make_pdf(), filename="C:\\Users\\me\\Desktop\\x.pdf")
    assert response.json()["filename"] == "x.pdf"


def test_duplicate_upload_returns_existing(client: TestClient, make_pdf) -> None:
    content = make_pdf(seed="dup")
    first = upload(client, content, filename="a.pdf")
    second = upload(client, content, filename="b.pdf")

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["filename"] == "a.pdf"


@pytest.mark.skipif(
    not (SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf").exists(), reason="sample PDF not present"
)
def test_sample_patent_is_parsed(settings) -> None:
    content = (SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf").read_bytes()
    roomy = settings.model_copy(update={"max_upload_mb": 50, "max_pages": 200})
    with TestClient(create_app(roomy)) as client:
        response = upload(client, content, filename="一种高耐候钛白粉及制备方法.pdf")
        assert response.status_code == 202
        done = wait_until_done(client, response.json()["id"])

    # OCR is off by default, so the image-only results table on page 6 is reported.
    assert done["status"] == "ready_with_warnings"
    assert [(w["code"], w["page"]) for w in done["warnings"]] == [("OCR_SKIPPED", 6)]
    assert done["doc_type"] == "patent"
    assert done["title"] == "一种高耐候钛白粉及制备方法"
    assert done["stats"]["pages"] == 6
    assert done["options"]["extract_facts"] is True  # default when the field is omitted

    pages = json.loads((roomy.processed_dir / done["id"] / "pages.json").read_text("utf-8"))
    large = [(p["page"], img["width"], img["height"]) for p in pages for img in p["images"] if img["large"]]
    assert large == [(6, 754, 389)]  # the image-only results table [0043]
    assert "[0043]" in pages[5]["text"]


# --- upload validation -------------------------------------------------------------


def _error(response) -> dict:
    return response.json()["error"]


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = upload(client, b"")
    assert response.status_code == 400
    assert _error(response)["code"] == "EMPTY_FILE"


def test_non_pdf_is_rejected_by_content(client: TestClient) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
    response = upload(client, png, filename="looks-like.pdf")
    assert response.status_code == 415
    assert _error(response)["code"] == "UNSUPPORTED_FILE_TYPE"
    assert _error(response)["details"]["detected"] == "image/png"


def test_too_large_file_is_rejected(client: TestClient) -> None:
    content = b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024)
    response = upload(client, content)
    assert response.status_code == 413
    assert _error(response)["code"] == "FILE_TOO_LARGE"


def test_encrypted_pdf_is_rejected(client: TestClient, make_pdf) -> None:
    response = upload(client, make_pdf(password="secret"))
    assert response.status_code == 422
    assert _error(response)["code"] == "PDF_ENCRYPTED"


def test_corrupted_pdf_is_rejected(client: TestClient) -> None:
    response = upload(client, b"%PDF-1.7\nthis is not really a pdf\n%%EOF")
    assert response.status_code == 422
    assert _error(response)["code"] == "PDF_CORRUPTED"


def test_too_many_pages_is_rejected(client: TestClient, make_pdf) -> None:
    response = upload(client, make_pdf(6))
    assert response.status_code == 422
    assert _error(response)["code"] == "TOO_MANY_PAGES"
    assert _error(response)["details"] == {"pages": 6, "max_pages": 5}


@pytest.mark.parametrize("metadata", ["{not json", '["a list"]', '{"unknown_field": 1}'])
def test_invalid_metadata_is_rejected(client: TestClient, make_pdf, metadata: str) -> None:
    response = upload(client, make_pdf(), metadata=metadata)
    assert response.status_code == 422
    assert _error(response)["code"] == "INVALID_METADATA"


@pytest.mark.parametrize("form", [{"ocr": "sometimes"}, {"extract_facts": "maybe"}])
def test_invalid_option_value_is_rejected(client: TestClient, make_pdf, form) -> None:
    response = upload(client, make_pdf(), **form)
    assert response.status_code == 422
    assert _error(response)["code"] == "VALIDATION_ERROR"


def test_extract_facts_defaults_to_true(client: TestClient, make_pdf) -> None:
    assert upload(client, make_pdf()).json()["options"]["extract_facts"] is True


def test_rejected_upload_leaves_no_files(client: TestClient, make_pdf, settings) -> None:
    upload(client, make_pdf(password="secret"))
    upload(client, b"%PDF-1.7\ngarbage")
    assert list(settings.raw_dir.iterdir()) == []
    assert list(settings.tmp_dir.iterdir()) == []


# --- query, list, delete, reindex --------------------------------------------------


def test_unknown_document_returns_404(client: TestClient) -> None:
    response = client.get(f"{UPLOAD_URL}/doc_missing")
    assert response.status_code == 404
    assert _error(response)["code"] == "DOCUMENT_NOT_FOUND"


def test_list_with_pagination_and_filter(client: TestClient, make_pdf) -> None:
    ids = [upload(client, make_pdf(seed=str(i))).json()["id"] for i in range(3)]
    for doc_id in ids:
        wait_until_done(client, doc_id)

    page = client.get(UPLOAD_URL, params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["items"][0]["id"] == ids[-1]  # newest first

    filtered = client.get(UPLOAD_URL, params={"status": "ready"}).json()
    assert filtered["total"] == 0  # blank PDFs end as ready_with_warnings


def test_delete_removes_record_and_files(client: TestClient, make_pdf, settings) -> None:
    body = upload(client, make_pdf()).json()
    wait_until_done(client, body["id"])

    response = client.delete(f"{UPLOAD_URL}/{body['id']}")
    assert response.status_code == 204
    assert client.get(f"{UPLOAD_URL}/{body['id']}").status_code == 404
    assert not (settings.raw_dir / f"{body['sha256']}.pdf").exists()
    assert not (settings.processed_dir / body["id"]).exists()
    assert client.delete(f"{UPLOAD_URL}/{body['id']}").status_code == 404


def test_reindex_reprocesses_with_new_options(client: TestClient, make_pdf) -> None:
    doc_id = upload(client, make_pdf()).json()["id"]
    wait_until_done(client, doc_id)

    response = client.post(f"{UPLOAD_URL}/{doc_id}/reindex", json={"doc_type": "paper"})
    assert response.status_code == 202
    assert response.json()["status"] == "queued"

    done = wait_until_done(client, doc_id)
    assert done["doc_type"] == "paper"
    assert done["options"]["ocr"] == "auto"  # untouched options are kept


def test_busy_document_cannot_be_deleted_or_reindexed(client: TestClient, make_pdf) -> None:
    doc_id = upload(client, make_pdf()).json()["id"]
    wait_until_done(client, doc_id)

    repo = client.app.state.document_service.repo
    repo.update(doc_id, status=DocumentStatus.PARSING)

    assert client.delete(f"{UPLOAD_URL}/{doc_id}").status_code == 409
    response = client.post(f"{UPLOAD_URL}/{doc_id}/reindex")
    assert response.status_code == 409
    assert _error(response)["code"] == "DOCUMENT_BUSY"


def test_unfinished_documents_are_recovered(client: TestClient, make_pdf) -> None:
    doc_id = upload(client, make_pdf()).json()["id"]
    wait_until_done(client, doc_id)

    service = client.app.state.document_service
    service.repo.update(doc_id, status=DocumentStatus.INDEXING)  # simulate a crash mid-run
    assert service.recover_unfinished() == 1
    assert wait_until_done(client, doc_id)["status"] == "ready_with_warnings"
