"""Runs ingestion jobs on a background thread pool.

Jobs are not persisted by the pool itself; the document status in SQLite is the
source of truth. On startup, ``DocumentService.recover_unfinished`` re-submits every
document that was not in a terminal state, so a restart never loses work.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from tio2agent.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class IngestionRunner:
    def __init__(self, pipeline: IngestionPipeline, *, max_workers: int = 2) -> None:
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")
        self._inflight: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, doc_id: str) -> bool:
        """Queue a document. Returns False if it is already queued or running."""
        with self._lock:
            if doc_id in self._inflight:
                return False
            self._inflight.add(doc_id)
        self._executor.submit(self._run, doc_id)
        return True

    def is_inflight(self, doc_id: str) -> bool:
        """Check if a document is currently queued or running in IngestionRunner."""
        with self._lock:
            return doc_id in self._inflight

    def shutdown(self, *, wait: bool = False) -> None:
        """Stops the ingestion thread pool, cancelling queued jobs and optionally waiting for running tasks."""
        # Queued jobs are cancelled; they stay "queued" in the DB and are recovered on restart.
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, doc_id: str) -> None:
        try:
            self.pipeline.run(doc_id)   # 调用流水线去执行业务逻辑
        except Exception:
            logger.exception("unhandled error while ingesting %s", doc_id)
        finally:
            with self._lock:
                self._inflight.discard(doc_id)
