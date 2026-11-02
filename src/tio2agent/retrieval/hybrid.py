"""Hybrid search: BM25 + vector similarity, fused with Reciprocal Rank Fusion (RRF).

    BM25            exact terms: product grades, chemical names, numbers, "实施例2"
    vectors         paraphrases: "耐候性最好的是哪个" vs "色差更小，光泽度更佳"
    RRF             score = sum over methods of 1 / (k + rank); no score calibration
                    needed between the two methods (Cormack et al., 2009)

After fusion, hits that say the same thing are collapsed: a description chunk that
only restates claims and the claim chunks it restates are one piece of evidence.
The better-ranked one is kept and the other is listed in ``also_in``.

If the embedding service fails at query time, search falls back to BM25 and says so
in ``warnings`` and ``mode_used`` instead of failing the request.
"""

import threading
from dataclasses import dataclass, field
from enum import StrEnum

from tio2agent.domain import Chunk, ChunkKind
from tio2agent.ingestion.structure.normalize import normalize_text
from tio2agent.retrieval.embeddings import Embedder, EmbeddingError
from tio2agent.retrieval.keyword import Bm25Index
from tio2agent.retrieval.rerank import Reranker, RerankError
from tio2agent.retrieval.vector import VectorIndex
from tio2agent.storage.chunk_repo import ChunkRepository


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    VECTOR = "vector"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float                                   # fused RRF score (or the single method's)
    bm25: float | None = None
    vector: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    fused_rank: int | None = None                  # position after fusion, before reranking
    rerank: float | None = None                    # cross-encoder score, if reranked
    also_in: list[str] = field(default_factory=list)  # citations of collapsed duplicates


@dataclass
class SearchResult:
    query: str
    normalized_query: str
    mode_requested: SearchMode
    mode_used: SearchMode
    hits: list[SearchHit]
    warnings: list[str] = field(default_factory=list)
    reranked: bool = False


class SearchService:
    def __init__(
        self,
        repo: ChunkRepository,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        *,
        rrf_k: int = 60,  # RRF 算法的平滑常数（默认 60，学术界公认的最佳实践）。
        candidate_pool: int = 50,  # 底层单通道拉取数量（默认 50）。意思是先让 BM25 召回 50 个，让向量引擎也召回 50 个，把这 100 个混在一起重排，最后挑出 top_k（如前 8 个）给用户。
        rerank_candidates: int = 30,
        rerank_max_chars: int = 1500,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.rerank_candidates = rerank_candidates
        self.rerank_max_chars = rerank_max_chars
        self._lock = threading.Lock()
        self._generation = -1
        self._chunks: dict[str, Chunk] = {}
        self._bm25 = Bm25Index([])
        self._vectors = VectorIndex([], None)

    # --- index cache -----------------------------------------------------------

    def _refresh(self) -> None:
        with self._lock:
            if self._generation == self.repo.generation:
                return
            generation = self.repo.generation
            chunks = self.repo.all_chunks()
            self._chunks = {c.id: c for c in chunks}
            self._bm25 = Bm25Index(chunks)
            if self.embedder is not None:
                ids, matrix = self.repo.vectors(self.embedder.model)
                self._vectors = VectorIndex(ids, matrix)
            self._generation = generation

    # --- search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        doc_ids: list[str] | None = None,
        kinds: list[ChunkKind] | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        rerank: bool | None = None,
    ) -> SearchResult:
        # 刷新索引，归一化查询字符串，计算出 allowed 过滤白名单（基于文档 ID 或块类型）。
        self._refresh()
        normalized = normalize_text(query)
        warnings: list[str] = []
        allowed = self._allowed(doc_ids, kinds)

        mode_used = mode
        # 如果系统没配向量模型、或者数据库里根本没存向量，自动将请求模式降级为 BM25。
        if mode is not SearchMode.BM25 and self.embedder is None:
            mode_used = SearchMode.BM25
            if mode is SearchMode.VECTOR:
                warnings.append("未配置向量服务（TIO2_EMBEDDING_BACKEND），已改用关键词检索")
        elif mode is not SearchMode.BM25 and len(self._vectors) == 0:
            mode_used = SearchMode.BM25
            warnings.append("索引中还没有向量（文档导入时向量服务不可用？），已改用关键词检索")

        # 双路召回：根据 mode_used，分别调用 _bm25.search 和 _vectors.search，各自拉取 candidate_pool 数量的初选名单。
        bm25: list[tuple[str, float]] = []
        vector: list[tuple[str, float]] = []
        if mode_used in (SearchMode.HYBRID, SearchMode.BM25):
            bm25 = self._bm25.search(normalized, limit=self.candidate_pool, allowed=allowed)
        if mode_used in (SearchMode.HYBRID, SearchMode.VECTOR):
            try:
                assert self.embedder is not None
                query_vec = self.embedder.embed([normalized])[0]
                vector = self._vectors.search(query_vec, limit=self.candidate_pool, allowed=allowed)
            # 如果向量服务在请求时挂了（except EmbeddingError），不让系统崩溃报错，而是悄悄切换回 BM25 继续搜，并在 warnings 里记录，告诉前端展示给用户。
            except EmbeddingError as exc:
                warnings.append(f"向量检索失败，已改用关键词检索：{exc.message}")
                mode_used = SearchMode.BM25
                bm25 = bm25 or self._bm25.search(normalized, limit=self.candidate_pool, allowed=allowed)

        # 调用 _fuse 将两份名单融合为一份，然后调用 collapse_duplicates 进行去重，最后返回前 top_k 条。
        hits = self._fuse(bm25, vector)
        reranked = False
        if rerank is not False and hits:
            if self.reranker is None:
                if rerank:
                    warnings.append("未配置重排序服务（TIO2_RERANK_BACKEND），按融合结果排序")
            else:
                try:
                    hits = self._rerank(normalized, hits)
                    reranked = True
                except RerankError as exc:
                    warnings.append(f"重排序失败，按融合结果排序：{exc.message}")
        hits = collapse_duplicates(hits)[:top_k]
        return SearchResult(query, normalized, mode, mode_used, hits, warnings, reranked)

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """
        将融合后的候选文档进行重排，保持其余文档的 fused 顺序。
        """
        assert self.reranker is not None
        head, tail = hits[: self.rerank_candidates], hits[self.rerank_candidates :]
        documents = [h.chunk.search_text()[: self.rerank_max_chars] for h in head]
        scores = self.reranker.rerank(query, documents)
        for hit, score in zip(head, scores, strict=True):
            hit.rerank = score
        # 稳定排序：相等分数保持融合顺序
        return sorted(head, key=lambda h: -(h.rerank or 0.0)) + tail

    def _allowed(self, doc_ids: list[str] | None, kinds: list[ChunkKind] | None) -> set[str] | None:
        """
        计算出 allowed 过滤白名单（基于文档 ID 或块类型）。
        """
        if not doc_ids and not kinds:
            return None
        docs, wanted = set(doc_ids or []), set(kinds or [])
        return {
            cid for cid, c in self._chunks.items()
            if (not docs or c.doc_id in docs) and (not wanted or c.kind in wanted)
        }

    def _fuse(self, bm25: list[tuple[str, float]], vector: list[tuple[str, float]]) -> list[SearchHit]:
        hits: dict[str, SearchHit] = {}
        for method, ranked in (("bm25", bm25), ("vector", vector)):
            for rank, (cid, score) in enumerate(ranked, start=1):
                chunk = self._chunks.get(cid)
                if chunk is None:
                    continue
                hit = hits.setdefault(cid, SearchHit(chunk=chunk, score=0.0))
                hit.score += 1.0 / (self.rrf_k + rank)
                setattr(hit, method, score)
                setattr(hit, f"{method}_rank", rank)
        fused = sorted(hits.values(), key=lambda h: (-h.score, h.chunk.doc_id, h.chunk.order))
        for rank, hit in enumerate(fused, start=1):
            hit.fused_rank = rank
        return fused


def collapse_duplicates(hits: list[SearchHit]) -> list[SearchHit]:
    """
    Merge a restating description chunk with the claim chunks it restates.

    A claim hit is dropped when a better-ranked chunk from the same document restates it.
    A restating chunk is dropped when every claim it restates already appears above it.
    The dropped hit's citation goes to ``also_in`` of the kept one.

    说明书里有很多段落是权利要求的简单复述。如果不去重，用户搜一个东西，可能会搜出两个内容一模一样、但位置不同的 Chunk，浪费了宝贵的 LLM 上下文窗口。
    解决方案：
    - 按重排后的分数从高到低遍历 hits，逐一放入 kept（保留名单）中。
    - 规则 1（上位替代下位）：如果当前看的是一个权利要求（如权利要求 2），它去 kept 里找，有没有哪个已经排在前面的、得分更高的说明书段落已经包含了对权利要求 2 的重述？
      如果有，就把当前这个权利要求丢掉，同时把它的引文挂到那个说明书段落的 also_in 属性上。
    - 规则 2（下位替代上位）：反过来，如果当前看的是一段“复述了权利要求 1 和 2”的说明书，它去看 kept 里是不是已经同时把权利要求 1 和 2 都收录了？
      如果都收录了，就把这段说明书丢弃，它的引文挂载到排名靠前的权利要求上。
    这保证了喂给大模型的知识绝对不会发生逻辑和文本上的重复，同时引文（Citation）还能完美保留所有出现过的地方，极大地提升了最终生成的质量。
    """
    kept: list[SearchHit] = []
    for hit in hits:
        c = hit.chunk
        if c.kind is ChunkKind.CLAIM:
            covering = next(
                (k for k in kept if k.chunk.doc_id == c.doc_id and set(c.claim_numbers) <= set(k.chunk.restates_claims)),
                None,
            )
            if covering is not None:
                covering.also_in.append(c.citation)
                continue
        elif c.restates_claims:
            above = {
                n: k for k in kept if k.chunk.doc_id == c.doc_id and k.chunk.kind is ChunkKind.CLAIM
                for n in k.chunk.claim_numbers
            }
            if all(n in above for n in c.restates_claims):
                above[c.restates_claims[0]].also_in.append(c.citation)
                continue
        kept.append(hit)
    return kept
