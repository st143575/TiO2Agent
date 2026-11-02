"""Application settings, read from environment variables (prefix TIO2_) or a .env file."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OcrBackend(StrEnum):
    NONE = "none"                # no OCR service; image regions are reported as warnings
    PADDLEOCR_VL = "paddleocr_vl"  # PaddleOCR-VL pipeline service (POST /layout-parsing)


class EmbeddingBackend(StrEnum):
    NONE = "none"                            # keyword (BM25) search only
    OPENAI_COMPATIBLE = "openai_compatible"  # any /v1/embeddings endpoint


class RerankBackend(StrEnum):
    NONE = "none"                            # results ordered by fusion only
    COHERE_COMPATIBLE = "cohere_compatible"  # {base_url}/rerank, Cohere/Jina style: vLLM, SGLang, hosted APIs
    TEI = "tei"                              # Hugging Face text-embeddings-inference /rerank


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TIO2_", env_file=".env", extra="ignore")

    app_name: str = "TiO2Agent"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Storage
    data_dir: Path = Path("data")

    # Upload limits
    max_upload_mb: int = Field(50, gt=0)
    max_pages: int = Field(200, gt=0)
    upload_chunk_size: int = Field(1024 * 1024, gt=0)

    # Background ingestion
    ingest_workers: int = Field(2, ge=1)

    # OCR for image regions and scanned pages (off by default, so no GPU is needed)
    ocr_backend: OcrBackend = OcrBackend.NONE
    ocr_base_url: str = "http://localhost:8080"
    ocr_timeout_s: float = Field(300.0, gt=0)

    # Chunking and search
    chunk_max_chars: int = Field(600, ge=100)
    search_top_k: int = Field(8, ge=1, le=50)
    search_rrf_k: int = Field(60, ge=1)

    # Embeddings (optional; without them search is BM25 only)
    embedding_backend: EmbeddingBackend = EmbeddingBackend.NONE
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "bge-m3"
    embedding_api_key: str | None = None
    embedding_batch_size: int = Field(32, ge=1)
    embedding_timeout_s: float = Field(60.0, gt=0)

    # Reranking (optional; a cross-encoder re-scores the top fused candidates)
    rerank_backend: RerankBackend = RerankBackend.NONE
    rerank_base_url: str = "http://localhost:8001/v1"  # not 8000: that is this app
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_key: str | None = None
    rerank_candidates: int = Field(30, ge=2, le=200)
    rerank_max_chars: int = Field(1500, ge=100)
    rerank_timeout_s: float = Field(30.0, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def raw_dir(self) -> Path:
        """Original PDFs, stored as {sha256}.pdf."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Per-document intermediate results, one folder per document id."""
        return self.data_dir / "processed"

    @property
    def tmp_dir(self) -> Path:
        """Uploads in progress, before validation."""
        return self.data_dir / "tmp"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tio2agent.db"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.processed_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
