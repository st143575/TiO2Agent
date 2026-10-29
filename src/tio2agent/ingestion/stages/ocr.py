"""OCR stage: recognize the parts of a PDF that have no usable text layer.

The text layer extracted by ParseStage stays the primary source: it is exact, while
OCR output is a prediction. This stage only fills the gaps:

* ``ocr=auto`` (default): large embedded images (tables, figures) on any page, plus
  every non-tiny image on pages without a text layer (scans).
* ``ocr=force``: every embedded image that is not a tiny decoration.
* ``ocr=off``: nothing.

Failures never fail the document: text-layer content is still usable, so problems
are recorded as warnings and the document ends as ``ready_with_warnings``.

Output: ``{work_dir}/ocr.json`` and ``ctx.artifacts["ocr_regions"]``; each page in
``ctx.artifacts["pages"]`` gets an ``"ocr"`` list with the ids of its regions.

Limitation: pages are not rendered, only embedded images are sent. A page whose
content is vector graphics without a text layer is reported, not recognized.
"""

import hashlib
import io
import json
import logging
from collections import defaultdict
from dataclasses import dataclass

from tio2agent.domain import DocumentStatus, OcrMode
from tio2agent.ingestion.ocr import OcrEngine, OcrError, OcrResult
from tio2agent.ingestion.pipeline import IngestionContext
from tio2agent.ingestion.stages.parse import is_large_image
from tio2agent.ingestion.validation import open_pdf

logger = logging.getLogger(__name__)

# Below this (shorter side, px) an image is a bullet, logo or rule, never content.
MIN_IMAGE_SIDE = 64


@dataclass
class _Candidate:
    page: int
    index: int        # position among the page's images
    name: str
    width: int
    height: int
    reason: str       # "large_image" | "low_text_page" | "force"
    png: bytes


class OcrStage:
    status = DocumentStatus.OCR

    def __init__(self, engine: OcrEngine | None) -> None:
        self.engine = engine

    def run(self, ctx: IngestionContext) -> None:
        mode = ctx.record.options.ocr
        pages: list[dict] = ctx.artifacts.get("pages") or []
        ctx.artifacts["ocr_regions"] = []

        if mode is OcrMode.OFF:
            self._report_uncovered_pages(ctx, pages, covered=set())
            return

        candidates = self._collect(ctx, pages, mode)
        if not candidates:
            self._report_uncovered_pages(ctx, pages, covered=set())
            return

        if self.engine is None:
            self._report_skipped(ctx, candidates)
            self._report_uncovered_pages(ctx, pages, covered={c.page for c in candidates})
            return

        regions = self._recognize_all(ctx, candidates)
        covered = {r["page"] for r in regions}
        self._report_uncovered_pages(ctx, pages, covered=covered)

        by_page: dict[int, list[str]] = defaultdict(list)
        for region in regions:
            by_page[region["page"]].append(region["id"])
        for page in pages:
            page["ocr"] = by_page.get(page["page"], [])

        ctx.stats.ocr_regions = len(regions)
        ctx.artifacts["ocr_regions"] = regions
        (ctx.work_dir / "ocr.json").write_text(
            json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (ctx.work_dir / "pages.json").write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- selection -----------------------------------------------------------------

    def _collect(self, ctx: IngestionContext, pages: list[dict], mode: OcrMode) -> list[_Candidate]:
        reader = open_pdf(ctx.pdf_path)
        candidates: list[_Candidate] = []

        for info in pages:
            number = info["page"]
            try:
                images = list(reader.pages[number - 1].images)
            except Exception as exc:
                ctx.warn("OCR_IMAGE_READ_FAILED", f"第 {number} 页图片读取失败：{exc}", page=number)
                continue

            for index, image_file in enumerate(images):
                try:
                    pil = image_file.image
                    width, height = pil.size
                except Exception as exc:
                    ctx.warn(
                        "OCR_IMAGE_READ_FAILED",
                        f"第 {number} 页图片 {image_file.name} 无法解码：{exc}",
                        page=number,
                    )
                    continue

                reason = _selection_reason(mode, info.get("low_text", False), width, height)
                if reason is None:
                    continue
                candidates.append(
                    _Candidate(
                        page=number,
                        index=index,
                        name=image_file.name,
                        width=width,
                        height=height,
                        reason=reason,
                        png=_to_png(pil),
                    )
                )
        return candidates

    # --- recognition ---------------------------------------------------------------

    def _recognize_all(self, ctx: IngestionContext, candidates: list[_Candidate]) -> list[dict]:
        assert self.engine is not None
        regions: list[dict] = []
        cache: dict[str, OcrResult] = {}  # the same image reused on several pages

        for position, cand in enumerate(candidates):
            digest = hashlib.sha256(cand.png).hexdigest()
            result = cache.get(digest)
            if result is None:
                try:
                    result = self.engine.recognize(cand.png)
                except OcrError as exc:
                    ctx.warn(
                        exc.code,
                        f"第 {cand.page} 页图片 {cand.name} 识别失败：{exc.message}",
                        page=cand.page,
                    )
                    if exc.code == "OCR_UNAVAILABLE":
                        # Service is down: don't wait for a timeout on every remaining image.
                        self._report_skipped(ctx, candidates[position + 1 :])
                        break
                    continue
                cache[digest] = result

            if result.is_empty:
                ctx.warn(
                    "OCR_EMPTY",
                    f"第 {cand.page} 页图片 {cand.name} 未识别出任何内容",
                    page=cand.page,
                )
                continue

            regions.append(
                {
                    "id": f"p{cand.page}-img{cand.index}",
                    "page": cand.page,
                    "image": {"name": cand.name, "width": cand.width, "height": cand.height},
                    "reason": cand.reason,
                    "sha256": digest,
                    **result.model_dump(),
                }
            )
            logger.info(
                "document %s page %d %s: %d blocks, %d tables in %d ms",
                ctx.record.id, cand.page, cand.name,
                len(result.blocks), len(result.tables), result.elapsed_ms,
            )
        return regions

    # --- warnings ------------------------------------------------------------------

    @staticmethod
    def _report_skipped(ctx: IngestionContext, candidates: list[_Candidate]) -> None:
        per_page: dict[int, int] = defaultdict(int)
        for cand in candidates:
            per_page[cand.page] += 1
        for page, count in sorted(per_page.items()):
            ctx.warn(
                "OCR_SKIPPED",
                f"第 {page} 页有 {count} 个图片区域未识别：未配置或无法连接 OCR 服务",
                page=page,
            )

    @staticmethod
    def _report_uncovered_pages(ctx: IngestionContext, pages: list[dict], covered: set[int]) -> None:
        for info in pages:
            if info.get("low_text") and info["page"] not in covered:
                ctx.warn(
                    "LOW_TEXT_PAGE",
                    f"第 {info['page']} 页仅提取到 {info['char_count']} 个字符，且未通过 OCR 补全",
                    page=info["page"],
                )


def _selection_reason(mode: OcrMode, low_text: bool, width: int, height: int) -> str | None:
    if min(width, height) < MIN_IMAGE_SIDE:
        return None
    if mode is OcrMode.FORCE:
        return "force"
    if is_large_image(width, height):
        return "large_image"
    if low_text:
        return "low_text_page"
    return None


def _to_png(pil_image) -> bytes:
    if pil_image.mode not in ("RGB", "L", "RGBA"):
        pil_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()
