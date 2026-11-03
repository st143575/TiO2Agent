import base64

import httpx
import pytest

from ocr_fakes import P6_TABLE_HTML, FakeLayoutParsingServer, layout_parsing_body
from tio2agent.ingestion.ocr import OcrError, PaddleOcrVlClient


def make_client(handler) -> PaddleOcrVlClient:
    return PaddleOcrVlClient("http://ocr:8080/", transport=httpx.MockTransport(handler))


def test_request_payload_and_parsed_result() -> None:
    server = FakeLayoutParsingServer(
        lambda _: httpx.Response(
            200,
            json=layout_parsing_body(
                [
                    {"label": "paragraph_title", "content": "表1"},
                    {"label": "table", "content": P6_TABLE_HTML, "bbox": [1, 2, 3, 4]},
                ],
                markdown="## 表1\n\n" + P6_TABLE_HTML,
            ),
        )
    )
    client = make_client(server)

    result = client.recognize(b"\x89PNG fake bytes")

    (payload,) = server.requests
    assert base64.b64decode(payload["file"]) == b"\x89PNG fake bytes"
    assert payload["fileType"] == 1
    assert payload["visualize"] is False
    assert payload["returnMarkdownImages"] is False

    assert result.engine == "paddleocr_vl"
    assert [b.label for b in result.blocks] == ["paragraph_title", "table"]
    assert result.blocks[0].table is None
    assert result.blocks[1].bbox == [1.0, 2.0, 3.0, 4.0]
    assert result.blocks[1].order == 2
    assert result.tables[0][1] == ["老化试验色差值", "1.21", "1.35", "1.46"]
    assert result.markdown.startswith("## 表1")
    assert result.elapsed_ms >= 0


def test_health() -> None:
    assert make_client(FakeLayoutParsingServer()).health() is True


def test_service_error_is_reported() -> None:
    client = make_client(
        lambda _: httpx.Response(500, json={"logId": "x", "errorCode": 500, "errorMsg": "CUDA OOM"})
    )
    with pytest.raises(OcrError) as exc:
        client.recognize(b"img")
    assert exc.value.code == "OCR_FAILED"
    assert "CUDA OOM" in exc.value.message


def test_non_json_response() -> None:
    client = make_client(lambda _: httpx.Response(502, text="<html>Bad Gateway</html>"))
    with pytest.raises(OcrError) as exc:
        client.recognize(b"img")
    assert exc.value.details["status"] == 502


def test_connection_error_means_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OcrError) as exc:
        make_client(refuse).recognize(b"img")
    assert exc.value.code == "OCR_UNAVAILABLE"


def test_timeout_is_a_recognition_failure() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(OcrError) as exc:
        make_client(slow).recognize(b"img")
    assert exc.value.code == "OCR_FAILED"
