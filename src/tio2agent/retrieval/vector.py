"""Brute-force cosine search over normalized embeddings.

Exact (no approximation) and fast enough for tens of thousands of chunks: one
matrix-vector product per query. Replace with Qdrant/pgvector behind the same
``search`` signature when the corpus outgrows memory.
"""

import numpy as np


class VectorIndex:
    def __init__(self, ids: list[str], matrix: np.ndarray | None) -> None:
        self.ids = ids
        self._matrix = matrix if matrix is not None else np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        """
        获取当前索引库里一共有多少条向量记录
        """
        return len(self.ids)

    @property
    def dim(self) -> int:
        """
        返回当前模型向量的维度
        """
        return int(self._matrix.shape[1]) if self._matrix.size else 0

    def search(
        self, query: np.ndarray, *, limit: int = 50, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """
        接收用户的查询向量并返回最相似的 Chunk 列表
        """
        if not self.ids or query.size == 0 or query.shape[-1] != self.dim:
            return []
        
        # embeddings.py 中，已经对向量进行了 L2 归一化，所以这里直接做点积就是余弦相似度。
        # scores 是包含所有 Chunk 相似度打分的一维数组。
        scores = self._matrix @ query.reshape(-1).astype(np.float32)
        # 对 scores 进行降序排序，相似度最高的排在前面，获取索引。
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for i in order:
            chunk_id = self.ids[i]
            if allowed is not None and chunk_id not in allowed:
                continue
            out.append((chunk_id, float(scores[i])))
            if len(out) >= limit:
                break
        return out
