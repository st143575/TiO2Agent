"""Content-addressed storage for original PDF files.

Uploads are streamed to a temp file first (hashing and size-checking on the way),
then moved to ``raw/{sha256}.pdf`` once validated. The same content always maps to
the same path, which makes de-duplication and re-indexing trivial.
"""

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from tio2agent.errors import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError

PDF_MAGIC = b"%PDF-"
# The PDF spec allows the header anywhere in the first 1024 bytes.
HEADER_WINDOW = 1024

_KNOWN_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "application/zip"),
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"\xd0\xcf\x11\xe0", "application/x-ole-storage"),  # legacy .doc/.xls
    (b"{\\rtf", "application/rtf"),
)


def sniff_mime(head: bytes) -> str:
    """Guess the type of the failed file based on its header."""
    for signature, mime in _KNOWN_SIGNATURES:
        if head.startswith(signature):
            return mime
    try:
        head.decode("utf-8")
        return "text/plain"
    except UnicodeDecodeError:
        return "application/octet-stream"


@dataclass(frozen=True)
class StagedFile:
    path: Path
    sha256: str
    size_bytes: int


class FileStore:
    def __init__(self, raw_dir: Path, tmp_dir: Path, chunk_size: int = 1024 * 1024) -> None:
        self.raw_dir = raw_dir
        self.tmp_dir = tmp_dir
        self.chunk_size = chunk_size
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        """Return the path of a file with the given SHA256 for the permanent storage."""
        return self.raw_dir / f"{sha256}.pdf"

    def stage(self, stream: BinaryIO, max_bytes: int) -> StagedFile:
        """Copy an upload stream to a temp file, enforcing size and PDF signature.

        Raises EmptyFileError, FileTooLargeError or UnsupportedFileTypeError.
        The temp file is removed on any failure.
        """
        tmp_path = self.tmp_dir / f"{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        head = b""
        header_checked = False

        try:
            with tmp_path.open("wb") as out:
                while chunk := stream.read(self.chunk_size):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FileTooLargeError(
                            f"文件超过大小上限 {max_bytes // (1024 * 1024)} MB",
                            details={"max_bytes": max_bytes},
                        )
                    if not header_checked:
                        head += chunk[: HEADER_WINDOW - len(head)]
                        if len(head) >= HEADER_WINDOW:
                            self._check_header(head)
                            header_checked = True
                    digest.update(chunk)
                    out.write(chunk)

            if size == 0:
                raise EmptyFileError()
            if not header_checked:
                self._check_header(head)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        return StagedFile(path=tmp_path, sha256=digest.hexdigest(), size_bytes=size)

    def commit(self, staged: StagedFile) -> Path:
        """Move a validated temp file into permanent storage."""
        dest = self.path_for(staged.sha256)
        if dest.exists():
            staged.path.unlink(missing_ok=True)
        else:
            os.replace(staged.path, dest)
        return dest

    def discard(self, staged: StagedFile) -> None:
        """Remove a temp file."""
        staged.path.unlink(missing_ok=True)

    def delete(self, sha256: str) -> None:
        """Remove a file from permanent storage."""
        self.path_for(sha256).unlink(missing_ok=True)

    def cleanup_tmp(self) -> int:
        """Remove temp files left by uploads interrupted by a crash or restart."""
        removed = 0
        for path in self.tmp_dir.glob("*.part"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    @staticmethod
    def _check_header(head: bytes) -> None:
        if PDF_MAGIC not in head[:HEADER_WINDOW]:
            raise UnsupportedFileTypeError(details={"detected": sniff_mime(head)})


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
