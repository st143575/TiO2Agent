"""OCR engine interface.

An engine takes one image (PNG/JPEG bytes) and returns layout blocks in reading order.
Tables are returned both as the engine's raw content and as a parsed cell grid, so
later stages can read numbers cell by cell instead of re-parsing HTML.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from tio2agent.errors import AppError


class OcrBlock(BaseModel):
    label: str                      # layout label, e.g. "text", "table", "image", "doc_title"
    content: str                    # raw content from the engine (tables: HTML or OTSL)
    bbox: list[float] | None = None  # [x1, y1, x2, y2] in source-image pixels
    order: int | None = None        # reading order, if the engine provides one
    table: list[list[str]] | None = None  # parsed cell grid for table blocks


class OcrResult(BaseModel):
    engine: str
    blocks: list[OcrBlock] = Field(default_factory=list)
    markdown: str = ""
    elapsed_ms: int = 0

    @property
    def tables(self) -> list[list[list[str]]]:
        return [b.table for b in self.blocks if b.table]

    @property
    def is_empty(self) -> bool:
        return not any(b.content.strip() for b in self.blocks) and not self.markdown.strip()


class OcrError(AppError):
    """Recognition failed for one image. ``code`` is OCR_FAILED or OCR_UNAVAILABLE."""

    code = "OCR_FAILED"
    message = "OCR 识别失败"


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: bytes) -> OcrResult:
        """Recognize one image. Raises OcrError."""
        ...

    def close(self) -> None: ...
