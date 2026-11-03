"""Reranking: HTTP client (Cohere and TEI styles), SearchService integration, API."""

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES_DIR, upload, wait_until_done
from retrieval_fakes import FakeReranker, HashingEmbedder, rerank_server
from tio2agent.config import RerankBackend, Settings
from tio2agent.domain import Chunk, ChunkKind
from tio2agent.main import create_app
from tio2agent.retrieval import ApiStyle, HttpReranker, RerankError, SearchService, build_reranker
from tio2agent.retrieval.rerank import parse_scores
from tio2agent.storage import ChunkRepository

SAMPLE = SAMPLES_DIR / "一种高耐候钛白粉及制备方法.pdf"
SEARCH_URL = "/api/v1/search"


# --- response parsing -------------------------------------------------------------------


def test_parse_scores_accepts_both_shapes_in_any_order() -> None:
    cohere = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}
    tei = [{"index": 1, "score": 3.5}, {"index": 0, "score": -1.0}]
    assert parse_scores(cohere, 2) == [0.2, 0.9]
    assert parse_scores(tei, 2) == [-1.0, 3.5]


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"index": 0, "relevance_score": 1.0}]},          # one result missing
        [{"index": 5, "score": 1.0}, {"index": 0, "score": 1.0}],      # index out of range
        [{"idx": 0, "score": 1.0}, {"idx": 1, "score": 1.0}],          # wrong field names
        {"data": []},                                                  # unknown shape
    ],
)
def test_parse_scores_rejects_bad_responses(payload) -> None:
    with pytest.raises(RerankError):
        parse_scores(payload, 2)


# --- HTTP client ------------------------------------------------------------------------


def test_cohere_style_request() -> None:
    handler, requests = rerank_server("cohere")
    reranker = HttpReranker(
        "http://rr:8001/v1", "BAAI/bge-reranker-v2-m3", api_key="sk-x", transport=httpx.MockTransport(handler)
    )
    assert reranker.rerank("q", ["aa", "a", "aaa"]) == [2.0, 1.0, 3.0]
    (req,) = requests
    assert req["path"] == "/v1/rerank" and req["auth"] == "Bearer sk-x"
    assert req["body"] == {"model": "BAAI/bge-reranker-v2-m3", "query": "q", "documents": ["aa", "a", "aaa"], "top_n": 3}


def test_tei_style_request() -> None:
    handler, requests = rerank_server("tei")
    reranker = HttpReranker("http://tei:8080", "bge", api_style=ApiStyle.TEI, transport=httpx.MockTransport(handler))
    assert reranker.rerank("q", ["a", "aaa"]) == [1.0, 3.0]
    assert requests[0]["path"] == "/rerank"
    assert requests[0]["body"] == {"query": "q", "texts": ["a", "aaa"], "truncate": True}


@pytest.mark.parametrize(
    "respond",
    [lambda r: httpx.Response(503, text="loading"), lambda r: httpx.Response(200, text="not json")],
)
def test_client_errors(respond) -> None:
    with pytest.raises(RerankError):
        HttpReranker("http://rr/v1", "m", transport=httpx.MockTransport(respond)).rerank("q", ["a"])


def test_connection_error_and_empty_input() -> None:
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    reranker = HttpReranker("http://rr/v1", "m", transport=httpx.MockTransport(refuse))
    assert reranker.rerank("q", []) == []  # nothing to send, no request made
    with pytest.raises(RerankError):
        reranker.rerank("q", ["a"])


def test_build_reranker(settings: Settings) -> None:
    assert build_reranker(settings) is None
    tei = build_reranker(settings.model_copy(update={"rerank_backend": RerankBackend.TEI}))
    cohere = build_reranker(settings.model_copy(update={"rerank_backend": RerankBackend.COHERE_COMPATIBLE}))
    assert tei.api_style is ApiStyle.TEI and cohere.api_style is ApiStyle.COHERE
    assert cohere.model == "BAAI/bge-reranker-v2-m3"


# --- SearchService ----------------------------------------------------------------------


def chunk(cid: str, text: str, kind=ChunkKind.PARAGRAPH, **kw) -> Chunk:
    return Chunk(id=cid, doc_id="d1", doc_label="d1", order=int(cid.split("-")[1]), kind=kind, text=text, **kw)


@pytest.fixture
def repo(tmp_path) -> ChunkRepository:
    repo = ChunkRepository(tmp_path / "db.sqlite")
    repo.replace_document(
        "d1",
        [
            chunk("d1-1", "包膜温度 包膜温度 包膜温度 控制要求"),        # BM25 favourite: repeats the terms
            chunk("d1-2", "包膜时温度控制在30℃-40℃，并进行超声处理"),     # the actual answer
            chunk("d1-3", "包膜剂为氧化硅、氧化铝"),
            chunk("d1-4", "背景技术：钛白粉是一种颜料"),
        ],
    )
    return repo


def test_reranker_reorders_fused_candidates(repo) -> None:
    reranker = FakeReranker(prefer="30℃-40℃")
    result = SearchService(repo, None, reranker).search("包膜温度", top_k=3)
    assert result.reranked and result.warnings == []
    assert [h.chunk.id for h in result.hits][:2] == ["d1-2", "d1-1"]
    top = result.hits[0]
    assert top.fused_rank == 2 and top.rerank > 10   # moved up from fusion rank 2
    query, documents = reranker.calls[0]
    assert query == "包膜温度" and len(documents) == 3  # d1-4 shares no term, never a candidate


def test_only_top_candidates_are_reranked(repo) -> None:
    reranker = FakeReranker(prefer="氧化硅")
    service = SearchService(repo, None, reranker, rerank_candidates=2, rerank_max_chars=10)
    result = service.search("包膜温度 氧化硅")
    _, documents = reranker.calls[0]
    assert len(documents) == 2 and all(len(d) <= 10 for d in documents)
    assert result.hits[-1].rerank is None            # outside the candidate window: fused order


def test_rerank_switch(repo) -> None:
    reranker = FakeReranker(prefer="30℃-40℃")
    off = SearchService(repo, None, reranker).search("包膜温度", rerank=False)
    assert not off.reranked and reranker.calls == [] and off.hits[0].chunk.id == "d1-1"
    forced = SearchService(repo, None, None).search("包膜温度", rerank=True)
    assert not forced.reranked and "未配置重排序服务" in forced.warnings[0]
    default_none = SearchService(repo, None, None).search("包膜温度")
    assert default_none.warnings == []               # nothing configured, nothing asked: no warning


def test_rerank_failure_keeps_fused_order(repo) -> None:
    reranker = FakeReranker(prefer="30℃-40℃")
    reranker.fail = True
    result = SearchService(repo, None, reranker).search("包膜温度")
    assert not result.reranked
    assert result.warnings == ["重排序失败，按融合结果排序：fake reranker is down"]
    assert result.hits[0].chunk.id == "d1-1" and result.hits[0].rerank is None


def test_duplicates_collapse_after_reranking(tmp_path) -> None:
    repo = ChunkRepository(tmp_path / "db.sqlite")
    repo.replace_document("d1", [
        chunk("d1-1", "步骤：研磨15-25分钟，研磨后分离", restates_claims=[8], paragraph_ids=["0016"]),
        chunk("d1-2", "一种方法，其特征在于：研磨15-25分钟", kind=ChunkKind.CLAIM, claim_numbers=[8]),
    ])
    service = SearchService(repo, None, FakeReranker(prefer="其特征在于"))  # only the claim has it

    # Without reranking the paragraph ranks first and absorbs the claim ...
    fused = service.search("研磨 分离", rerank=False)
    assert [h.chunk.id for h in fused.hits] == ["d1-1"] and fused.hits[0].also_in == ["权利要求8"]
    # ... with reranking the claim ranks first and absorbs the paragraph instead.
    reranked = service.search("研磨 分离")
    assert [h.chunk.id for h in reranked.hits] == ["d1-2"] and reranked.hits[0].also_in == ["说明书[0016]"]


# --- API --------------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")
def test_search_api_with_reranker(settings: Settings) -> None:
    roomy = settings.model_copy(update={"max_upload_mb": 50, "max_pages": 200})
    reranker = FakeReranker(prefer="研磨25分钟")         # only 实施例3 contains this phrase
    with TestClient(create_app(roomy, embedder=HashingEmbedder(), reranker=reranker)) as client:
        doc_id = upload(client, SAMPLE.read_bytes()).json()["id"]
        wait_until_done(client, doc_id)
        on = client.post(SEARCH_URL, json={"query": "研磨时间是多少", "top_k": 3}).json()
        off = client.post(SEARCH_URL, json={"query": "研磨时间是多少", "top_k": 3, "rerank": False}).json()

    assert on["reranked"] is True and on["mode_used"] == "hybrid"
    top = on["hits"][0]
    assert top["chunk"]["citation"] == "说明书[0036]-[0039]" and top["rerank"] > 10
    assert top["fused_rank"] > 1                        # it was not first before reranking
    assert off["reranked"] is False and off["hits"][0]["rerank"] is None
    assert off["hits"][0]["chunk"]["citation"] != "说明书[0036]-[0039]"
