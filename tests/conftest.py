import io
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from tio2agent.config import Settings
from tio2agent.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
UPLOAD_URL = "/api/v1/documents"


@pytest.fixture(scope="session", autouse=True)
def _isolated_env() -> Iterator[None]:
    """Tests never see TIO2_* variables exported in the shell (e.g. after sourcing
    .env), so they use fakes, not whatever OCR / embedding / rerank services are set.
    Session-scoped so that it also covers module-scoped fixtures. (Settings built in
    tests pass ``_env_file=None``, so the .env file itself is not read either.)"""
    with pytest.MonkeyPatch.context() as mp:
        for name in [n for n in os.environ if n.upper().startswith("TIO2_")]:
            mp.delenv(name)
        yield


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        max_upload_mb=1,
        max_pages=5,
        ingest_workers=1,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    """Build a small blank PDF in memory. Use ``seed`` to get distinct file contents."""

    def _make(pages: int = 1, *, password: str | None = None, seed: str = "") -> bytes:
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=595, height=842)
        writer.add_metadata({"/Title": f"test {seed}"})
        if password:
            writer.encrypt(password)
        buffer = io.BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    return _make


def upload(client: TestClient, content: bytes, filename: str = "test.pdf", **form: str):
    return client.post(
        UPLOAD_URL,
        files={"file": (filename, content, "application/pdf")},
        data=form,
    )


def wait_until_done(client: TestClient, doc_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{UPLOAD_URL}/{doc_id}").json()
        if body["status"] in {"ready", "ready_with_warnings", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"document {doc_id} did not finish within {timeout}s")
