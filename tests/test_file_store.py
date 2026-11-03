import hashlib
import io
from pathlib import Path

import pytest

from tio2agent.errors import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError
from tio2agent.storage import FileStore


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    return FileStore(tmp_path / "raw", tmp_path / "tmp", chunk_size=8)


def test_stage_hashes_and_commits(store: FileStore) -> None:
    content = b"%PDF-1.7\nhello world"
    staged = store.stage(io.BytesIO(content), max_bytes=1024)

    assert staged.sha256 == hashlib.sha256(content).hexdigest()
    assert staged.size_bytes == len(content)

    path = store.commit(staged)
    assert path.read_bytes() == content
    assert not staged.path.exists()


def test_stream_size_limit_is_enforced(store: FileStore) -> None:
    # Covers chunked uploads that the Content-Length middleware cannot see.
    with pytest.raises(FileTooLargeError):
        store.stage(io.BytesIO(b"%PDF-1.7\n" + b"x" * 100), max_bytes=50)
    assert list(store.tmp_dir.iterdir()) == []


def test_header_may_appear_after_leading_bytes(store: FileStore) -> None:
    staged = store.stage(io.BytesIO(b"\n\n  %PDF-1.4\n..."), max_bytes=1024)
    assert staged.size_bytes > 0


def test_empty_and_non_pdf(store: FileStore) -> None:
    with pytest.raises(EmptyFileError):
        store.stage(io.BytesIO(b""), max_bytes=1024)
    with pytest.raises(UnsupportedFileTypeError) as exc:
        store.stage(io.BytesIO(b"PK\x03\x04" + b"\x00" * 2000), max_bytes=4096)
    assert exc.value.details["detected"] == "application/zip"
