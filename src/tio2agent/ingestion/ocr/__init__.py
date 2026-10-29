from tio2agent.config import OcrBackend, Settings
from tio2agent.ingestion.ocr.base import OcrBlock, OcrEngine, OcrError, OcrResult
from tio2agent.ingestion.ocr.paddleocr_vl import PaddleOcrVlClient
from tio2agent.ingestion.ocr.tables import normalize_cell, parse_table, table_to_markdown


def build_ocr_engine(settings: Settings) -> OcrEngine | None:
    """Create the configured OCR engine, or None when OCR is disabled."""
    if settings.ocr_backend is OcrBackend.PADDLEOCR_VL:
        return PaddleOcrVlClient(settings.ocr_base_url, timeout_s=settings.ocr_timeout_s)
    return None


__all__ = [
    "OcrBlock",
    "OcrEngine",
    "OcrError",
    "OcrResult",
    "PaddleOcrVlClient",
    "build_ocr_engine",
    "normalize_cell",
    "parse_table",
    "table_to_markdown",
]
