"""Cross-encoder reranking through an HTTP rerank service.

A cross-encoder reads the query and one candidate together and scores how well the
candidate answers the query. That is slower than BM25 or embeddings, so it only sees
the top candidates after fusion, and it is more precise than either.

Two request styles cover the common servers. Both POST to ``{base_url}/rerank``:

    api_style   request body                              servers
    "cohere"    {model, query, documents, top_n}          vLLM, Jina, Cohere, SGLang,
                                                          hosted APIs (base_url ".../v1")
    "tei"       {query, texts, truncate}                  Hugging Face text-embeddings-inference

Responses differ too: Cohere/Jina/vLLM return {"results": [{index, relevance_score}]},
TEI and SGLang a top-level list of {index, score}. Both are accepted.

Recommended models: BAAI/bge-reranker-v2-m3 (multilingual, Chinese works well),
Qwen/Qwen3-Reranker-0.6B / -4B / -8B.
"""

from enum import StrEnum
from typing import Any, Protocol

import httpx

from tio2agent.config import RerankBackend, Settings
from tio2agent.errors import AppError


class RerankError(AppError):
    code = "RERANK_FAILED"
    message = "重排序失败"


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """One relevance score per document, in input order (higher is better).
        Raises RerankError."""
        ...

    def close(self) -> None: ...


class ApiStyle(StrEnum):
    COHERE = "cohere"
    TEI = "tei"


class HttpReranker:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_style: ApiStyle = ApiStyle.COHERE,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.api_style = api_style
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self.api_style is ApiStyle.TEI:
            body: dict[str, Any] = {"query": query, "texts": documents, "truncate": True}
        else:
            body = {"model": self.model, "query": query, "documents": documents, "top_n": len(documents)}

        try:
            response = self._client.post("/rerank", json=body)
        except httpx.HTTPError as exc:
            raise RerankError(f"无法连接重排序服务：{exc}") from exc
        if response.status_code != 200:
            raise RerankError(
                f"重排序服务返回 {response.status_code}：{response.text[:200]}",
                details={"status": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RerankError("重排序服务返回的不是 JSON") from exc
        return parse_scores(payload, len(documents))


def parse_scores(payload: Any, n: int) -> list[float]:
    """Scores in input order from any of the supported response shapes."""
    items = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RerankError("重排序服务响应格式不正确")
    scores: list[float | None] = [None] * n
    for item in items:
        try:
            index = int(item["index"])
            score = float(item["relevance_score"] if "relevance_score" in item else item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError("重排序服务响应格式不正确") from exc
        if not 0 <= index < n:
            raise RerankError(f"重排序结果的 index {index} 超出范围")
        scores[index] = score
    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise RerankError(f"重排序服务只返回了 {n - len(missing)}/{n} 条结果")
    return [s for s in scores if s is not None]


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.rerank_backend is RerankBackend.NONE:
        return None
    style = ApiStyle.TEI if settings.rerank_backend is RerankBackend.TEI else ApiStyle.COHERE
    return HttpReranker(
        settings.rerank_base_url,
        settings.rerank_model,
        api_style=style,
        api_key=settings.rerank_api_key,
        timeout_s=settings.rerank_timeout_s,
    )
