"""Application settings, read from environment variables (prefix TIO2_) or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
