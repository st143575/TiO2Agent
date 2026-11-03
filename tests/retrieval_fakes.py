"""Test doubles for embeddings.

HashingEmbedder is deterministic and offline: it hashes tokens into a fixed number of
buckets. Similarity is lexical, not semantic; it exercises fusion, storage and
fallback logic, not embedding quality.
"""

import hashlib
import json

import httpx
import numpy as np

from tio2agent.retrieval import EmbeddingError
from tio2agent.retrieval.tokenize import tokenize


class HashingEmbedder:
    def __init__(self, model: str = "fake-hash-256", dim: int = 256) -> None:
        self.model = model
        self.dim = dim
        self.calls: list[list[str]] = []
        self.fail = False

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("fake embedding service is down")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def close(self) -> None:
        pass


def openai_embeddings_server(dim: int = 4):
    """MockTransport handler speaking the OpenAI /v1/embeddings protocol."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        body = json.loads(request.content)
        requests.append({"body": body, "auth": request.headers.get("authorization")})
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text)), 1.0] + [0.0] * (dim - 2)}
            for i, text in enumerate(body["input"])
        ]
        # Return out of order: clients must sort by "index".
        return httpx.Response(200, json={"object": "list", "data": list(reversed(data)), "model": body["model"]})

    return handler, requests


class FakeReranker:
    """Deterministic stand-in for a cross-encoder.

    By default it scores a document by the share of the query's tokens it contains;
    pass ``prefer`` to give documents containing that text the top score.
    """

    def __init__(self, prefer: str | None = None, model: str = "fake-reranker") -> None:
        self.model = model
        self.prefer = prefer
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = False

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        from tio2agent.retrieval import RerankError

        self.calls.append((query, list(documents)))
        if self.fail:
            raise RerankError("fake reranker is down")
        wanted = set(tokenize(query))
        scores = []
        for doc in documents:
            overlap = len(wanted & set(tokenize(doc))) / (len(wanted) or 1)
            scores.append(10.0 + overlap if self.prefer and self.prefer in doc else overlap)
        return scores

    def close(self) -> None:
        pass


def rerank_server(style: str = "cohere"):
    """MockTransport handler for /rerank in Cohere (results/relevance_score) or TEI (list/score) shape."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/rerank")
        body = json.loads(request.content)
        requests.append({"path": request.url.path, "body": body, "auth": request.headers.get("authorization")})
        docs = body["documents"] if style == "cohere" else body["texts"]
        ranked = sorted(range(len(docs)), key=lambda i: -len(docs[i]))  # longer = more relevant
        items = [{"index": i, "score": float(len(docs[i]))} for i in ranked]
        if style == "cohere":
            items = [{"index": it["index"], "relevance_score": it["score"]} for it in items]
            return httpx.Response(200, json={"id": "r1", "model": body["model"], "results": items})
        return httpx.Response(200, json=items)

    return handler, requests
