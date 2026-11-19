"""BM25 (Okapi) по леммам: небольшой корпус целиком в памяти, без внешних библиотек.

Для 7–8 тысяч пунктов индекс строится за доли секунды из сохранённых лемм, поэтому
на диске хранится только список лемм каждого куска (в `chunks.jsonl` индекса).
"""

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence


class Bm25Index:
    def __init__(self, docs: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.size = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avg_length = (sum(self.lengths) / self.size) if self.size else 0.0
        # Постинги: лемма → [(номер куска, частота)].
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            term: math.log(1 + (self.size - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(
        self, query: Sequence[str], allowed: Collection[int] | None = None
    ) -> dict[int, float]:
        """Оценка BM25 для кусков, где есть хотя бы одна лемма запроса."""
        result: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(query).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                if allowed is not None and i not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                result[i] += qtf * idf * tf * (self.k1 + 1) / (tf + norm)
        return result

    def search(
        self, query: Sequence[str], top_k: int, allowed: Collection[int] | None = None
    ) -> list[tuple[int, float]]:
        """Лучшие `top_k` кусков: [(номер, оценка)] по убыванию; при равенстве — по номеру."""
        scored = self.scores(query, allowed)
        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def rrf(rankings: Sequence[Sequence[int]], *, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: сумма 1/(k + ранг) по спискам; ранг с единицы."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
