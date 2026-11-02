"""Text embeddings through any OpenAI-compatible ``/v1/embeddings`` endpoint.

Works with local servers (Ollama, Hugging Face TEI, vLLM, Xinference) and hosted
APIs. No model is bundled: which one to use is a deployment choice. A multilingual
model with good Chinese coverage, e.g. BAAI/bge-m3, is a sensible default.

Vectors are L2-normalized here, so cosine similarity is a dot product downstream.
"""

from typing import Protocol

import httpx
import numpy as np

from tio2agent.config import EmbeddingBackend, Settings
from tio2agent.errors import AppError


class EmbeddingError(AppError):
    code = "EMBEDDING_FAILED"
    message = "向量化失败"


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        One L2-normalized float32 row per text. Raises EmbeddingError.
        输入字符串列表，返回一个 NumPy 矩阵（每一行代表一个句子的向量，类型为 float32，且必须经过 L2 归一化）。
        """
        ...

    def close(self) -> None: ...


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        batch_size: int = 32,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows += self._embed_batch(texts[start : start + self.batch_size])
        # L2-normalization
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.post("/embeddings", json={"model": self.model, "input": batch})
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"无法连接向量服务：{exc}") from exc
        if response.status_code != 200:
            raise EmbeddingError(
                f"向量服务返回 {response.status_code}：{response.text[:200]}",
                details={"status": response.status_code},
            )
        try:
            # 有些高并发的异步推理服务器，返回的向量数组顺序可能会和传入的句子顺序不一致。
            # OpenAI 标准会在返回结果里带上 index 字段。
            # 这里通过 sorted，强制将向量按 index 重新排好序，确保第一句话对应第一个向量。
            data = sorted(response.json()["data"], key=lambda d: d.get("index", 0))
            vectors = [d["embedding"] for d in data]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("向量服务响应格式不正确") from exc
        # 核对数量，防备服务端丢数据。
        if len(vectors) != len(batch):
            raise EmbeddingError(f"请求 {len(batch)} 条，返回 {len(vectors)} 条向量")
        return vectors


def build_embedder(settings: Settings) -> Embedder | None:
    """
    这是一个工厂函数（Factory Function），读取系统的全局配置对象 Settings，根据配置中的 embedding_backend 字段，返回对应的 Embedder 实例。
    如果用户在配置中选择的后端是 OPENAI_COMPATIBLE，它就提取相应的网址、密钥等参数，实例化并返回 OpenAICompatibleEmbedder。
    如果未来系统新增了直接在本地跑模型的实现（比如 SentenceTransformersEmbedder），只需要在这里加一个 elif 分支即可，其他调用它的业务代码一行都不用改。
    目前只支持 OpenAI 兼容的向量服务。
    """
    if settings.embedding_backend is EmbeddingBackend.OPENAI_COMPATIBLE:
        return OpenAICompatibleEmbedder(
            settings.embedding_base_url,
            settings.embedding_model,
            api_key=settings.embedding_api_key,
            batch_size=settings.embedding_batch_size,
            timeout_s=settings.embedding_timeout_s,
        )
    return None
