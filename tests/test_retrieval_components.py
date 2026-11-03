"""Tokenizer, BM25, vector index, embeddings client and chunk repository."""

import httpx
import numpy as np
import pytest

from retrieval_fakes import openai_embeddings_server
from tio2agent.domain import Chunk, ChunkKind
from tio2agent.retrieval import Bm25Index, EmbeddingError, OpenAICompatibleEmbedder, VectorIndex, tokenize
from tio2agent.storage import ChunkRepository


def chunk(cid: str, text: str, doc: str = "d1", kind: ChunkKind = ChunkKind.PARAGRAPH, **kw) -> Chunk:
    return Chunk(id=cid, doc_id=doc, doc_label=doc, order=int(cid[-1]), kind=kind, text=text, **kw)


# --- tokenizer ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "must_have", "must_not"),
    [
        ("实施例2的反应温度", {"实施例2", "反应", "温度"}, set()),
        ("根据权利要求1-7任意一项", {"权利要求1", "1-7", "1", "7"}, set()),
        ("二氧化钛45-65份", {"45-65", "45", "65", "二氧", "化钛"}, set()),
        ("温度设为50℃-80℃", {"50-80", "50", "80"}, set()),
        ("将pH值调节至6", {"ph", "6", "调节"}, set()),
        ("S2、研磨", {"s2", "研磨"}, set()),
        ("所述的一种钛白粉，其特征在于", {"钛白", "白粉"}, {"所述", "一种", "特征", "在于"}),
    ],
)
def test_tokenize(text: str, must_have: set[str], must_not: set[str]) -> None:
    tokens = set(tokenize(text))
    assert must_have <= tokens
    assert not (must_not & tokens)


# --- BM25 -----------------------------------------------------------------------------


def test_bm25_ranking_and_filter() -> None:
    chunks = [
        chunk("c1", "实施例1：反应釜温度65℃，转速200转/分钟"),
        chunk("c2", "实施例2：反应釜温度50℃，转速150转/分钟"),
        chunk("c3", "背景技术：钛白粉是一种重要的无机化工颜料"),
    ]
    index = Bm25Index(chunks)
    ranked = index.search("实施例2的反应釜温度")
    assert [cid for cid, _ in ranked][:2] == ["c2", "c1"]
    assert "c3" not in dict(ranked)                    # no shared terms -> not returned
    # with a filter only allowed chunks are scored (c1 shares the bigrams 实施/施例)
    assert [cid for cid, _ in index.search("实施例2", allowed={"c1"})] == ["c1"]
    assert index.search("") == [] and Bm25Index([]).search("钛白粉") == []


def test_bm25_number_and_range_tokens() -> None:
    index = Bm25Index([chunk("c1", "二氧化钛45-65份"), chunk("c2", "二氧化钛55份")])
    assert index.search("45-65")[0][0] == "c1"
    assert index.search("55份")[0][0] == "c2"


# --- vector index ---------------------------------------------------------------------


def test_vector_index() -> None:
    m = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    index = VectorIndex(["a", "b", "c"], m)
    assert [cid for cid, _ in index.search(np.array([1.0, 0.0]))] == ["a", "c", "b"]
    assert [cid for cid, _ in index.search(np.array([1.0, 0.0]), allowed={"b"})] == ["b"]
    assert index.search(np.array([1.0, 0.0, 0.0])) == []  # dimension mismatch
    assert VectorIndex([], None).search(np.array([1.0])) == []


# --- embeddings client ----------------------------------------------------------------


def test_openai_compatible_embedder_batches_sorts_and_normalizes() -> None:
    handler, requests = openai_embeddings_server(dim=4)
    embedder = OpenAICompatibleEmbedder(
        "http://emb/v1", "bge-m3", api_key="sk-test", batch_size=2, transport=httpx.MockTransport(handler)
    )
    vectors = embedder.embed(["a", "bbb", "cc"])
    assert len(requests) == 2 and requests[0]["body"] == {"model": "bge-m3", "input": ["a", "bbb"]}
    assert requests[0]["auth"] == "Bearer sk-test"
    assert vectors.shape == (3, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    # rows follow input order although the server answered out of order
    expected = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(vectors[0, :2], expected)
    assert vectors[1, 0] > vectors[2, 0] > vectors[0, 0]


@pytest.mark.parametrize(
    "respond",
    [
        lambda r: httpx.Response(401, json={"error": "bad key"}),
        lambda r: httpx.Response(200, json={"unexpected": True}),
        lambda r: httpx.Response(200, json={"data": []}),
    ],
)
def test_embedder_errors(respond) -> None:
    embedder = OpenAICompatibleEmbedder("http://emb/v1", "m", transport=httpx.MockTransport(respond))
    with pytest.raises(EmbeddingError):
        embedder.embed(["x"])


def test_embedder_connection_error() -> None:
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(EmbeddingError):
        OpenAICompatibleEmbedder("http://emb/v1", "m", transport=httpx.MockTransport(refuse)).embed(["x"])


# --- chunk repository -----------------------------------------------------------------


def test_chunk_repository(tmp_path) -> None:
    repo = ChunkRepository(tmp_path / "db.sqlite")
    g0 = repo.generation
    vectors = np.eye(2, dtype=np.float32)
    repo.replace_document("d1", [chunk("d1-1", "甲"), chunk("d1-2", "乙")], vectors=vectors, model="m1")
    repo.replace_document("d2", [chunk("d2-1", "丙", doc="d2")])
    assert repo.generation > g0
    assert [c.id for c in repo.list_by_doc("d1")] == ["d1-1", "d1-2"]
    assert repo.count() == 3

    ids, matrix = repo.vectors("m1")
    assert ids == ["d1-1", "d1-2"] and np.allclose(matrix, vectors)
    assert repo.vectors("other-model") == ([], None)

    repo.replace_document("d1", [chunk("d1-1", "甲改")])  # re-index replaces, no vectors now
    assert [c.text for c in repo.list_by_doc("d1")] == ["甲改"]
    assert repo.vectors("m1") == ([], None)
    assert repo.delete_document("d2") == 1 and repo.count() == 1

    with pytest.raises(ValueError):
        repo.replace_document("d3", [chunk("d3-1", "x")], vectors=vectors, model="m1")
