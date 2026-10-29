"""Parse stage: per-page text extraction, document type and title detection.

Output: ``{work_dir}/pages.json`` and ``ctx.artifacts["pages"]``, one entry per page:

    {"page": 6, "text": "...", "char_count": 340,
     "images": [{"name": "/Im0", "width": 754, "height": 389, "large": true}, ...]}

``large`` marks images big enough to hold content (a table or figure) rather than a
logo or a decorative header strip; the OCR stage will only look at those.
"""

import json
import re

from pypdf import PageObject

from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.validation import open_pdf
from tio2agent.domain import DocType, DocumentStatus

_PATENT_MARKERS = re.compile(r"国家知识产权局|发明专利|实用新型专利|CN\s?\d{9}\s?[ABUY]")
# An image counts as "large" (worth OCR) above both thresholds, in source pixels.
LARGE_IMAGE_MIN_SIDE = 150
LARGE_IMAGE_MIN_AREA = 60000

_PATENT_TITLE = re.compile(r"[（(]54[)）]\s*(?:发明名称|实用新型名称)\s*\n\s*(\S[^\n]*)")


class ParseStage:
    status = DocumentStatus.PARSING

    def __init__(self, min_chars_per_page: int = 50) -> None:
        self.min_chars_per_page = min_chars_per_page

    def run(self, ctx: IngestionContext) -> None:
        reader = open_pdf(ctx.pdf_path)
        pages: list[dict] = []

        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                text = ""
                ctx.warn("PAGE_EXTRACT_FAILED", f"第 {number} 页文本提取失败：{exc}", page=number)

            char_count = len("".join(text.split()))
            pages.append(
                {
                    "page": number,
                    "text": text,
                    "char_count": char_count,
                    "low_text": char_count < self.min_chars_per_page,
                    "images": _list_images(page),
                }
            )

        ctx.stats.pages = len(pages)
        # Store the parsed pages in the memory for later use.
        ctx.artifacts["pages"] = pages
        # Store the parsed pages to disk.
        (ctx.work_dir / "pages.json").write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        first_page = pages[0]["text"] if pages else ""
        if ctx.record.options.doc_type is DocType.AUTO:
            ctx.record.doc_type = _detect_doc_type(first_page)
        else:
            ctx.record.doc_type = ctx.record.options.doc_type
        
        # If the user didn't provide a title, try to detect it.
        if not ctx.record.title:
            ctx.record.title = _detect_title(first_page, ctx.record.doc_type) or _stem(
                ctx.record.filename
            )


def _detect_doc_type(first_page: str) -> DocType:
    if _PATENT_MARKERS.search(first_page):
        return DocType.PATENT
    return DocType.OTHER


def _detect_title(first_page: str, doc_type: DocType | None) -> str | None:
    if doc_type is DocType.PATENT and (match := _PATENT_TITLE.search(first_page)):
        return match.group(1).strip()
    return None


def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


def _list_images(page: PageObject) -> list[dict]:
    """List image XObjects on a page, including those nested in Form XObjects."""
    images: list[dict] = []
    seen: set[int] = set()

    def walk(resources, depth: int) -> None:
        """
        Recursively walk through resources by leveraging depth-first search to find image XObjects.
        """
        if resources is None or depth > 10:
            return
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return
        for name, ref in xobjects.get_object().items():
            obj = ref.get_object()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            subtype = obj.get("/Subtype")
            if subtype == "/Image":
                width, height = int(obj.get("/Width", 0)), int(obj.get("/Height", 0))
                images.append(
                    {
                        "name": str(name),
                        "width": width,
                        "height": height,
                        "large": min(width, height) >= LARGE_IMAGE_MIN_SIDE     # threshold for large images, which will be forwarded to OCR
                        and width * height >= LARGE_IMAGE_MIN_AREA,
                    }
                )
            elif subtype == "/Form":
                walk(obj.get("/Resources"), depth + 1)

    try:
        walk(page.get("/Resources"), 0)
    except Exception:
        pass  # image listing is best-effort; text extraction already succeeded
    return images
