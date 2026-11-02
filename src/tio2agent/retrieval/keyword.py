"""Okapi BM25 over chunks, with a heavier breadcrumb field (BM25F-style).

A chunk's breadcrumb ("… > 具体实施方式 > 实施例2") says what the chunk is about,
while its text may only mention other things in passing ("测试本发明实施例2的…" in
the comparison section). Breadcrumb terms are therefore counted ``context_weight``
times.

Small and dependency-free: the corpus here is hundreds to a few thousand chunks, so
the index is built in memory from the chunk table and rebuilt when it changes.
"""

import math
from collections import Counter
from dataclasses import dataclass

from tio2agent.domain import Chunk
from tio2agent.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class Bm25Params:
    k1: float = 1.5
    b: float = 0.75
    context_weight: int = 2


class Bm25Index:
    def __init__(self, chunks: list[Chunk], params: Bm25Params | None = None) -> None:
        params = params or Bm25Params()
        self.params = params
        self.ids = [c.id for c in chunks]
        self._tf: list[Counter[str]] = []
        for c in chunks:
            tf = Counter(tokenize(c.text))
            for term, n in Counter(tokenize(c.context)).items():
                tf[term] += n * params.context_weight
            self._tf.append(tf)
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        # BM25+ style idf, never negative for very common terms
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, *, limit: int = 50, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """(chunk id, score) pairs, best first; only chunks with a positive score."""
        terms = tokenize(query)
        if not terms or not self.ids:
            return []
        k1, b = self.params.k1, self.params.b
        weights = Counter(terms)  # a term repeated in the query counts more
        scores: list[tuple[str, float]] = []
        for i, tf in enumerate(self._tf):
            if allowed is not None and self.ids[i] not in allowed:
                continue
            norm = k1 * (1 - b + b * self._len[i] / self._avg_len) if self._avg_len else k1
            score = 0.0
            for term, qtf in weights.items():
                f = tf.get(term)
                if f:
                    score += qtf * self._idf[term] * f * (k1 + 1) / (f + norm)
            if score > 0:
                scores.append((self.ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
