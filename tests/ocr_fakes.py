"""A fake PaddleOCR-VL pipeline service for tests.

Responses follow the documented /layout-parsing schema (PaddleOCR-VL Usage Tutorial,
section 4.3). Table content uses the HTML produced by PaddleX's ``export_to_html``:
``<table><tr><td>...</td></tr></table>`` without whitespace between tags.
"""

import base64
import io
import json
from collections.abc import Callable
from typing import Any

import httpx
from PIL import Image

P6_TABLE_HTML = (
    "<table>"
    "<tr><td>统计对象 统计项目</td><td>实施例1</td><td>实施例2</td><td>实施例3</td></tr>"
    "<tr><td>老化试验色差值</td><td>1.21</td><td>1.35</td><td>1.46</td></tr>"
    "<tr><td>耐候性能</td><td>最佳</td><td>一般</td><td>较差</td></tr>"
    "</table>"
)


def layout_parsing_body(blocks: list[dict[str, Any]], markdown: str = "") -> dict[str, Any]:
    return {
        "logId": "00000000-0000-0000-0000-000000000000",
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "page_count": None,
                        "width": 754,
                        "height": 389,
                        "model_settings": {"use_layout_detection": True},
                        "parsing_res_list": [
                            {
                                "block_label": b["label"],
                                "block_content": b["content"],
                                "block_bbox": b.get("bbox", [0, 0, 754, 389]),
                                "block_id": i,
                                "block_order": i + 1,
                            }
                            for i, b in enumerate(blocks)
                        ],
                    },
                    "markdown": {"text": markdown, "images": None},
                    "outputImages": None,
                    "inputImage": None,
                }
            ],
            "dataInfo": {"type": "image", "width": 754, "height": 389},
        },
    }


class FakeLayoutParsingServer:
    """Callable for ``httpx.MockTransport``; records the requests it receives."""

    def __init__(self, respond: Callable[[dict[str, Any]], httpx.Response] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.respond = respond or (
            lambda _: httpx.Response(
                200,
                json=layout_parsing_body(
                    [{"label": "table", "content": P6_TABLE_HTML}], markdown=P6_TABLE_HTML
                ),
            )
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"errorCode": 0, "errorMsg": "Healthy"})
        assert request.method == "POST" and request.url.path == "/layout-parsing"
        payload = json.loads(request.content)
        self.requests.append(payload)
        return self.respond(payload)

    def image_sizes(self) -> list[tuple[int, int]]:
        return [Image.open(io.BytesIO(base64.b64decode(p["file"]))).size for p in self.requests]
