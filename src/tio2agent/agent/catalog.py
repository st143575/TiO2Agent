"""Which documents the Agent can use: ready documents with their display labels."""

from dataclasses import dataclass

from tio2agent.domain import DocumentStatus
from tio2agent.storage import ChunkRepository, DocumentRepository, FactRepository

READY = (DocumentStatus.READY, DocumentStatus.READY_WITH_WARNINGS)


@dataclass
class CatalogEntry:
    doc_id: str
    label: str        # publication number, else title, else file name
    title: str | None
    filename: str
    chunks: int
    facts: int


class DocumentCatalog:
    def __init__(self, documents: DocumentRepository, chunks: ChunkRepository, facts: FactRepository) -> None:
        self.documents = documents
        self.chunks = chunks
        self.facts = facts

    def entries(self, doc_ids: list[str] | None = None) -> list[CatalogEntry]:
        """Ready documents, oldest first; restricted to ``doc_ids`` if given."""
        records = []
        for status in READY:
            page, _ = self.documents.list_page(status=status, limit=10_000)
            records += page
        records.sort(key=lambda r: (r.created_at, r.id))
        labels = self.chunks.doc_labels()
        wanted = set(doc_ids) if doc_ids is not None else None
        return [
            CatalogEntry(
                doc_id=r.id,
                label=labels.get(r.id) or r.title or r.filename,
                title=r.title,
                filename=r.filename,
                chunks=r.stats.chunks,
                facts=r.stats.facts,
            )
            for r in records
            if wanted is None or r.id in wanted
        ]

    def labels(self) -> dict[str, str]:
        return {e.doc_id: e.label for e in self.entries()}
