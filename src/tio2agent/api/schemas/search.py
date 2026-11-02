"""Request/response models for search and chunk listing."""

from pydantic import BaseModel, Field

from tio2agent.domain import Chunk, ChunkKind
from tio2agent.retrieval import SearchHit, SearchMode, SearchResult


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(None, ge=1, le=50, description="默认取 TIO2_SEARCH_TOP_K")
    doc_ids: list[str] | None = None
    kinds: list[ChunkKind] | None = None
    mode: SearchMode = SearchMode.HYBRID
    rerank: bool | None = Field(None, description="默认：配置了重排序服务就重排；true/false 强制开关")


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    doc_label: str
    kind: ChunkKind
    citation: str
    context: str
    text: str
    pages: list[int]
    paragraph_ids: list[str]
    claim_numbers: list[int]
    table_ids: list[str]
    embodiment: str | None
    restates_claims: list[int]

    @classmethod
    def from_chunk(cls, c: Chunk) -> "ChunkOut":
        return cls(
            id=c.id,
            doc_id=c.doc_id,
            doc_label=c.doc_label,
            kind=c.kind,
            citation=c.citation,
            context=c.context,
            text=c.text,
            pages=c.pages,
            paragraph_ids=c.paragraph_ids,
            claim_numbers=c.claim_numbers,
            table_ids=c.table_ids,
            embodiment=c.embodiment,
            restates_claims=c.restates_claims,
        )


class SearchHitOut(BaseModel):
    rank: int
    score: float
    bm25: float | None
    vector: float | None
    bm25_rank: int | None
    vector_rank: int | None
    fused_rank: int | None
    rerank: float | None
    also_in: list[str]
    chunk: ChunkOut

    @classmethod
    def from_hit(cls, rank: int, h: SearchHit) -> "SearchHitOut":
        return cls(
            rank=rank,
            score=round(h.score, 6),
            bm25=None if h.bm25 is None else round(h.bm25, 4),
            vector=None if h.vector is None else round(h.vector, 4),
            bm25_rank=h.bm25_rank,
            vector_rank=h.vector_rank,
            fused_rank=h.fused_rank,
            rerank=None if h.rerank is None else round(h.rerank, 4),
            also_in=h.also_in,
            chunk=ChunkOut.from_chunk(h.chunk),
        )


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    mode_requested: SearchMode
    mode_used: SearchMode
    reranked: bool
    hits: list[SearchHitOut]
    warnings: list[str]

    @classmethod
    def from_result(cls, r: SearchResult) -> "SearchResponse":
        return cls(
            query=r.query,
            normalized_query=r.normalized_query,
            mode_requested=r.mode_requested,
            mode_used=r.mode_used,
            reranked=r.reranked,
            hits=[SearchHitOut.from_hit(i, h) for i, h in enumerate(r.hits, start=1)],
            warnings=r.warnings,
        )


class ChunkListOut(BaseModel):
    doc_id: str
    total: int
    items: list[ChunkOut]
