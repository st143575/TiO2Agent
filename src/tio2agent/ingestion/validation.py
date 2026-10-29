"""Synchronous PDF checks run during upload, before anything is queued."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from tio2agent.errors import PdfValidationError


@dataclass(frozen=True)
class PdfInfo:
    pages: int


def open_pdf(path: Path) -> PdfReader:
    """Open a PDF, transparently handling files encrypted with an empty user password.

    Raises PdfValidationError (PDF_CORRUPTED / PDF_ENCRYPTED).
    """
    try:
        reader = PdfReader(path, strict=False)
    except Exception as exc:  # pypdf raises many exception types on malformed input
        raise PdfValidationError(
            code="PDF_CORRUPTED", details={"reason": str(exc)[:200]}
        ) from exc

    if reader.is_encrypted:
        try:
            unlocked = bool(reader.decrypt(""))
        except Exception:
            unlocked = False
        if not unlocked:
            raise PdfValidationError("PDF 已加密，无法读取", code="PDF_ENCRYPTED")
    return reader


def inspect_pdf(path: Path, *, max_pages: int) -> PdfInfo:
    reader = open_pdf(path)
    try:
        pages = len(reader.pages)
    except Exception as exc:
        raise PdfValidationError(
            code="PDF_CORRUPTED", details={"reason": str(exc)[:200]}
        ) from exc

    if pages == 0:
        raise PdfValidationError("PDF 不包含任何页面", code="PDF_CORRUPTED")
    if pages > max_pages:
        raise PdfValidationError(
            f"PDF 共 {pages} 页，超过上限 {max_pages} 页",
            code="TOO_MANY_PAGES",
            details={"pages": pages, "max_pages": max_pages},
        )
    return PdfInfo(pages=pages)
