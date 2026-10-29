"""Client for the PaddleOCR-VL pipeline service (``POST /layout-parsing``).

We call the full pipeline service (layout detection + VLM), not the bare VLM server.
The PaddleOCR docs are explicit that sending images straight to the VLM (vLLM/SGLang)
is not equivalent and produces noticeably more hallucinated text.

The same API is served by the official Docker Compose deployment (NVIDIA GPU) and by
a manual ``paddlex --serve --pipeline PaddleOCR-VL`` (CPU, Apple Silicon), so this
client works with either.

API reference: PaddleOCR docs, "PaddleOCR-VL Usage Tutorial", section 4.3.
"""

import base64
import time
from typing import Any

import httpx

from tio2agent.ingestion.ocr.base import OcrBlock, OcrError, OcrResult
from tio2agent.ingestion.ocr.tables import parse_table

FILE_TYPE_IMAGE = 1


class PaddleOcrVlClient:
    name = "paddleocr_vl"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def recognize(self, image: bytes) -> OcrResult:
        body, elapsed_ms = self.layout_parsing(image)
        return parse_layout_parsing_result(body.get("result") or {}, elapsed_ms=elapsed_ms)

    def layout_parsing(self, image: bytes) -> tuple[dict[str, Any], int]:
        """Call /layout-parsing and return the raw response body and latency (ms)."""
        payload = {
            "file": base64.b64encode(image).decode("ascii"),
            "fileType": FILE_TYPE_IMAGE,
            "visualize": False,
            "returnMarkdownImages": False,
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/layout-parsing", json=payload)
        except httpx.TimeoutException as exc:
            raise OcrError("OCR 服务响应超时", details={"url": self.base_url}) from exc
        except httpx.HTTPError as exc:
            raise OcrError(
                f"无法连接 OCR 服务：{exc}",
                code="OCR_UNAVAILABLE",
                details={"url": self.base_url},
            ) from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        body = _json_or_none(response)
        if response.status_code != 200 or not body or body.get("errorCode", 0) != 0:
            message = (body or {}).get("errorMsg") or response.text[:200]
            raise OcrError(
                f"OCR 服务返回错误：{message}",
                details={"status": response.status_code, "errorCode": (body or {}).get("errorCode")},
            )
        return body, elapsed_ms


def parse_layout_parsing_result(result: dict[str, Any], *, elapsed_ms: int = 0) -> OcrResult:
    """Convert ``result`` of a /layout-parsing response into an OcrResult.

    For image input ``layoutParsingResults`` has one element; for robustness every
    element is read and their blocks and markdown are concatenated.
    """
    blocks: list[OcrBlock] = []
    markdown_parts: list[str] = []

    for page in result.get("layoutParsingResults") or []:
        pruned = page.get("prunedResult") or {}
        for item in pruned.get("parsing_res_list") or []:
            label = str(item.get("block_label") or "unknown")
            content = str(item.get("block_content") or "")
            bbox = item.get("block_bbox")
            blocks.append(
                OcrBlock(
                    label=label,
                    content=content,
                    bbox=[float(v) for v in bbox] if isinstance(bbox, list) else None,
                    order=item.get("block_order"),
                    table=parse_table(content) if label == "table" else None,
                )
            )
        text = (page.get("markdown") or {}).get("text")
        if text:
            markdown_parts.append(text)

    return OcrResult(
        engine=PaddleOcrVlClient.name,
        blocks=blocks,
        markdown="\n\n".join(markdown_parts),
        elapsed_ms=elapsed_ms,
    )


def _json_or_none(response: httpx.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None
