"""SearchService (fusion, duplicates, filters, fallback) and the search/chunks API."""

import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, UPLOAD_URL, upload, wait_until_done
from retrieval_fakes import HashingEmbedder
from tio2agent.config import Settings
from tio2agent.domain import Chunk, ChunkKind
from tio2agent.main import create_app
from tio2agent.retrieval import SearchHit, SearchMode, SearchService, collapse_duplicates
from tio2agent.storage import ChunkRepository

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
SEARCH_URL = "/api/v1/search"


def chunk(cid: str, text: str, kind=ChunkKind.PARAGRAPH, doc="d1", **kw) -> Chunk:
    return Chunk(id=cid, doc_id=doc, doc_label=doc, order=int(cid.split("-")[-1]), kind=kind, text=text, **kw)


# --- duplicate collapsing -----------------------------------------------------------------


def hits(*chunks: Chunk) -> list[SearchHit]:
    return [SearchHit(chunk=c, score=1.0 / (i + 1)) for i, c in enumerate(chunks)]


def test_claim_collapses_into_better_ranked_restatement() -> None:
    para = chunk("d1-1", "步骤", restates_claims=[8], paragraph_ids=["0015"])
    claim = chunk("d1-2", "权利要求", kind=ChunkKind.CLAIM, claim_numbers=[8])
    out = collapse_duplicates(hits(para, claim))
    assert [h.chunk.id for h in out] == ["d1-1"] and out[0].also_in == ["权利要求8"]


def test_restatement_collapses_into_claims_above_it() -> None:
    c1 = chunk("d1-1", "a", kind=ChunkKind.CLAIM, claim_numbers=[1])
    c2 = chunk("d1-2", "b", kind=ChunkKind.CLAIM, claim_numbers=[2])
    para = chunk("d1-3", "ab", restates_claims=[1, 2], paragraph_ids=["0007", "0008"])
    out = collapse_duplicates(hits(c1, c2, para))
    assert [h.chunk.id for h in out] == ["d1-1", "d1-2"]
    assert out[0].also_in == ["说明书[0007]-[0008]"]


def test_partial_restatement_is_kept() -> None:
    c1 = chunk("d1-1", "a", kind=ChunkKind.CLAIM, claim_numbers=[1])
    para = chunk("d1-2", "ab", restates_claims=[1, 2], paragraph_ids=["0007"])
    assert len(collapse_duplicates(hits(c1, para))) == 2  # claim 2's text would be lost otherwise


def test_no_collapse_across_documents() -> None:
    para = chunk("d1-1", "x", restates_claims=[1], paragraph_ids=["0007"])
    claim = chunk("d2-1", "x", kind=ChunkKind.CLAIM, claim_numbers=[1], doc="d2")
    assert len(collapse_duplicates(hits(para, claim))) == 2


# --- SearchService ------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    repo = ChunkRepository(tmp_path / "db.sqlite")
    embedder = HashingEmbedder()
    chunks = [
        chunk("d1-1", "实施例1：反应釜温度65℃，转速200转/分钟", kind=ChunkKind.EMBODIMENT),
        chunk("d1-2", "实施例2：反应釜温度50℃，转速150转/分钟", kind=ChunkKind.EMBODIMENT),
        chunk("d1-3", "钛白粉的生产工艺有硫酸法和氯化法两种工艺路线"),
    ]
    repo.replace_document("d1", chunks, vectors=embedder.embed([c.search_text() for c in chunks]), model=embedder.model)
    repo.replace_document("d2", [chunk("d2-1", "氯化法钛白粉的氧化工序", doc="d2")])
    return repo, embedder


def test_hybrid_fuses_both_methods(corpus) -> None:
    repo, embedder = corpus
    result = SearchService(repo, embedder).search("实施例2反应釜温度", top_k=3)
    assert result.mode_used is SearchMode.HYBRID and result.warnings == []
    top = result.hits[0]
    assert top.chunk.id == "d1-2"
    assert top.bm25_rank == 1 and top.vector_rank == 1
    assert top.score == pytest.approx(2 / 61)  # 1/(60+1) from each method


def test_filters(corpus) -> None:
    repo, embedder = corpus
    service = SearchService(repo, embedder)
    assert {h.chunk.doc_id for h in service.search("氯化法", doc_ids=["d2"]).hits} == {"d2"}
    assert {h.chunk.kind for h in service.search("温度", kinds=[ChunkKind.EMBODIMENT]).hits} == {ChunkKind.EMBODIMENT}


def test_without_embedder_hybrid_is_bm25(corpus) -> None:
    repo, _ = corpus
    result = SearchService(repo, None).search("实施例2", mode=SearchMode.HYBRID)
    assert result.mode_used is SearchMode.BM25 and result.warnings == []
    vector_only = SearchService(repo, None).search("实施例2", mode=SearchMode.VECTOR)
    assert vector_only.mode_used is SearchMode.BM25 and vector_only.warnings


def test_embedding_failure_at_query_time_falls_back(corpus) -> None:
    repo, embedder = corpus
    embedder.fail = True
    result = SearchService(repo, embedder).search("实施例2反应釜温度")
    assert result.mode_used is SearchMode.BM25
    assert "向量检索失败" in result.warnings[0]
    assert result.hits[0].chunk.id == "d1-2"


def test_index_cache_follows_repository_changes(corpus) -> None:
    repo, embedder = corpus
    service = SearchService(repo, embedder)
    bm25 = SearchMode.BM25  # vector search always returns the nearest chunks, even unrelated ones
    assert service.search("锐钛型", mode=bm25).hits == []
    repo.replace_document("d3", [chunk("d3-1", "锐钛型钛白粉", doc="d3")])
    assert [h.chunk.id for h in service.search("锐钛型", mode=bm25).hits] == ["d3-1"]
    repo.delete_document("d3")
    assert service.search("锐钛型", mode=bm25).hits == []


def test_query_is_normalized(corpus) -> None:
    repo, _ = corpus
    result = SearchService(repo, None).search("PH 值 0 .5")
    assert result.normalized_query == "pH值0.5"


# --- API ----------------------------------------------------------------------------------


@pytest.fixture
def roomy(settings: Settings) -> Settings:
    return settings.model_copy(update={"max_upload_mb": 50, "max_pages": 200})


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")
def test_search_api_end_to_end(roomy: Settings) -> None:
    embedder = HashingEmbedder()
    with TestClient(create_app(roomy, embedder=embedder)) as client:
        doc_id = upload(client, SAMPLE.read_bytes()).json()["id"]
        done = wait_until_done(client, doc_id)
        body = client.post(SEARCH_URL, json={"query": "实施例2的反应釜温度和转速", "top_k": 3}).json()
        chunks = client.get(f"{UPLOAD_URL}/{doc_id}/chunks").json()
        dup = client.post(SEARCH_URL, json={"query": "加入锆盐后PH值调节到多少", "kinds": ["paragraph", "claim"]}).json()

        assert client.delete(f"{UPLOAD_URL}/{doc_id}").status_code == 204
        after = client.post(SEARCH_URL, json={"query": "实施例2"}).json()

    assert done["stats"]["chunks"] == 21 and done["stats"]["duplicate_chunks"] == 2
    assert len(embedder.calls[0]) == 21                    # one embedding per chunk at index time
    assert body["mode_used"] == "hybrid"
    top = body["hits"][0]
    assert top["chunk"]["citation"] == "说明书[0031]-[0034]"
    assert top["chunk"]["doc_label"] == "CN116694108A"
    assert top["bm25_rank"] is not None and top["vector_rank"] is not None
    assert chunks["total"] == 21 and chunks["items"][0]["kind"] == "bibliographic"
    first = dup["hits"][0]
    assert first["chunk"]["citation"] == "说明书[0014]-[0018]" and first["also_in"] == ["权利要求8"]
    assert after["hits"] == []                             # chunks removed with the document


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")
def test_embedding_failure_at_index_time_is_a_warning(roomy: Settings) -> None:
    embedder = HashingEmbedder()
    embedder.fail = True
    with TestClient(create_app(roomy, embedder=embedder)) as client:
        doc_id = upload(client, SAMPLE.read_bytes()).json()["id"]
        done = wait_until_done(client, doc_id)
        embedder.fail = False
        body = client.post(SEARCH_URL, json={"query": "实施例2"}).json()

    assert "EMBEDDING_FAILED" in [w["code"] for w in done["warnings"]]
    assert done["stats"]["chunks"] == 21                   # still indexed for keyword search
    assert body["mode_used"] == "bm25" and "还没有向量" in body["warnings"][0]
    assert body["hits"][0]["chunk"]["citation"] == "说明书[0031]-[0034]"


def test_search_validation_and_missing_document(client: TestClient) -> None:
    assert client.post(SEARCH_URL, json={"query": ""}).status_code == 422
    assert client.post(SEARCH_URL, json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post(SEARCH_URL, json={"query": "钛白粉"}).json()["hits"] == []
    assert client.get(f"{UPLOAD_URL}/doc_missing/chunks").status_code == 404
