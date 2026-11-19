"""Гибридный поиск: BM25 по леммам + векторы Chroma → RRF → реранкер → контекст.

Фильтры по метаданным: если в вопросе назван договор (номер, дата) или контрагент, поиск
идёт только по его документам. Пустой фильтр не применяется.

После реранкера к найденным пунктам договора подмешиваются пункты его доп. соглашений,
которые меняют именно эти пункты («изложить п. 2.2 в редакции …») или сами оказались среди
кандидатов: модель должна видеть действующую редакцию условия, даже если по словам ближе
старая.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docai.rag.bm25 import Bm25Index, rrf
from docai.rag.chunking import Chunk, DocMeta
from docai.rag.config import RetrievalConfig
from docai.rag.index import (
    IndexNotReadyError,
    IndexPaths,
    load_manifest,
    meta_from_json,
    open_chroma,
)
from docai.rag.lemmas import lemmas
from docai.rag.models import Embedder, Reranker
from docai.rag.query import NameMatcher, QueryInfo


@dataclass
class Hit:
    chunk: Chunk
    rrf: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank: float | None = None
    # «amendment» — пункт доп. соглашения, добавленный к найденному пункту договора.
    added: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "rrf": round(self.rrf, 5),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "rerank": None if self.rerank is None else round(self.rerank, 4),
            "added": self.added,
        }


@dataclass
class Retrieval:
    candidates: list[Hit]  # после RRF (и реранкера, если он есть), по убыванию
    context: list[Hit]  # что уходит в модель
    filters: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


class RagIndex:
    """Индекс в памяти: куски, документы, BM25; векторы — в Chroma на диске."""

    def __init__(self, root: Path) -> None:
        manifest = load_manifest(root)
        if not manifest.get("complete"):
            raise IndexNotReadyError(
                f"векторы посчитаны для {manifest.get('embedded')} из {manifest.get('chunks')} "
                "кусков; продолжите индексацию: uv run docai rag index"
            )
        paths = IndexPaths(root)
        self.paths = paths
        self.collection_name: str = manifest["collection"]
        self.docs: dict[str, DocMeta] = {}
        for line in paths.docs.read_text(encoding="utf-8").splitlines():
            meta = meta_from_json(json.loads(line))
            self.docs[meta.doc_id] = meta
        self.chunks: list[Chunk] = []
        chunk_lemmas: list[list[str]] = []
        for line in paths.chunks.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            chunk_lemmas.append(row.pop("lemmas"))
            row["amends"] = tuple(row["amends"])
            chunk = Chunk(**row)
            chunk.meta = self.docs[chunk.doc_id]
            self.chunks.append(chunk)
        self.position = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.bm25 = Bm25Index(chunk_lemmas)
        self.matcher = NameMatcher(m.counterparty for m in self.docs.values() if m.counterparty)
        # Пункты доп. соглашений по договору (без служебных).
        self.amendments: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.chunks):
            meta = self.docs[c.doc_id]
            if (
                meta.doc_kind == "amendment"
                and meta.parent_doc_id
                and c.clause
                and not c.boilerplate
            ):
                self.amendments[meta.parent_doc_id].append(i)
        self._collection: Any = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = open_chroma(self.paths.chroma, self.collection_name)
        return self._collection

    def contract_docs(self, info: QueryInfo) -> set[str]:
        """Договоры, названные в вопросе номером (и датой); с учётом контрагента."""
        found: set[str] = set()
        for ref in info.contracts:
            for meta in self.docs.values():
                if meta.doc_kind != "contract" or meta.number != ref.number:
                    continue
                if ref.date and meta.doc_date != ref.date:
                    continue
                if info.counterparties and meta.counterparty not in info.counterparties:
                    continue
                found.add(meta.doc_id)
        return found

    def allowed(self, info: QueryInfo) -> tuple[set[int] | None, dict[str, Any]]:
        """Номера кусков, разрешённых фильтром, и описание фильтра (None — без фильтра)."""
        contracts = self.contract_docs(info)
        if contracts:
            docs = {
                d
                for d, m in self.docs.items()
                if d in contracts or (m.doc_kind == "amendment" and m.parent_doc_id in contracts)
            }
            label = {"contracts": sorted(contracts)}
        elif info.counterparties:
            docs = {d for d, m in self.docs.items() if m.counterparty in info.counterparties}
            label = {"counterparties": info.counterparties}
            # «(договор аренды)» у контрагента с договорами разных видов — только этот вид.
            narrowed = {d for d in docs if self.docs[d].contract_kind in info.contract_kinds}
            if narrowed:
                docs = narrowed
                label["contract_kinds"] = sorted(info.contract_kinds)
        else:
            return None, {}
        allowed = {i for i, c in enumerate(self.chunks) if c.doc_id in docs}
        if not allowed:
            return None, {}
        label["documents"] = len(docs)
        return allowed, label


def _rank_map(items: list[int]) -> dict[int, int]:
    return {item: rank for rank, item in enumerate(items, start=1)}


class Retriever:
    def __init__(
        self,
        index: RagIndex,
        embedder: Embedder,
        reranker: Reranker | None,
        config: RetrievalConfig,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.config = config

    def vector_search(self, question: str, allowed: set[int] | None, top_k: int) -> list[int]:
        vector = self.embedder.embed_query(question)
        where = None
        if allowed is not None:
            doc_ids = sorted({self.index.chunks[i].doc_id for i in allowed})
            where = {"doc_id": {"$in": doc_ids}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}
        result = self.index.collection.query(
            query_embeddings=[vector], n_results=top_k, where=where, include=["distances"]
        )
        return [self.index.position[cid] for cid in result["ids"][0] if cid in self.index.position]

    def retrieve(self, question: str, info: QueryInfo) -> Retrieval:
        cfg = self.config
        timings: dict[str, float] = {}
        allowed, filters = self.index.allowed(info) if cfg.use_filters else (None, {})

        started = time.perf_counter()
        bm25 = [i for i, _ in self.index.bm25.search(lemmas(question), cfg.bm25_top_k, allowed)]
        timings["bm25_s"] = time.perf_counter() - started

        started = time.perf_counter()
        vector = self.vector_search(question, allowed, cfg.vector_top_k)
        timings["vector_s"] = time.perf_counter() - started

        bm25_ranks, vector_ranks = _rank_map(bm25), _rank_map(vector)
        fused = rrf([bm25, vector], k=cfg.rrf_k)
        hits = [
            Hit(self.index.chunks[i], score, bm25_ranks.get(i), vector_ranks.get(i))
            for i, score in fused
        ]

        started = time.perf_counter()
        if self.reranker is not None:
            # Пункт, который один из поисков поставил первым, а другой не нашёл вовсе, по RRF
            # уходит за 15-е место (1/61 против 2/62). Первые `per_list_guaranteed` каждого
            # списка попадают к реранкеру всегда, остальное добирается по RRF.
            guaranteed = set(bm25[: cfg.per_list_guaranteed]) | set(
                vector[: cfg.per_list_guaranteed]
            )
            chosen = [h for h in hits if self.index.position[h.chunk.chunk_id] in guaranteed]
            for h in hits:
                if len(chosen) >= max(cfg.rerank_candidates, len(guaranteed)):
                    break
                if h not in chosen:
                    chosen.append(h)
            pool = chosen
            rest = [h for h in hits if h not in pool]
            scores = self.reranker.score(question, [h.chunk.index_text() for h in pool])
            for hit, score in zip(pool, scores, strict=True):
                hit.rerank = score
            pool.sort(key=lambda h: -(h.rerank or 0.0))
            hits = pool + rest
        timings["rerank_s"] = time.perf_counter() - started

        context = self.with_amendments(
            hits[: cfg.context_top_k], hits[: max(cfg.rerank_candidates, cfg.context_top_k)]
        )
        return Retrieval(
            candidates=hits,
            context=context,
            filters=filters,
            timings={k: round(v, 3) for k, v in timings.items()},
        )

    def with_amendments(self, top: list[Hit], pool: list[Hit]) -> list[Hit]:
        """Добавить к пунктам договора пункты доп. соглашений, которые их меняют."""
        index = self.index
        chosen = {h.chunk.chunk_id for h in top}
        pool_ids = {h.chunk.chunk_id for h in pool}
        result: list[Hit] = []
        added = 0
        for hit in top:
            result.append(hit)
            chunk = hit.chunk
            meta = index.docs[chunk.doc_id]
            if meta.doc_kind != "contract":
                continue
            clauses = {h.chunk.label for h in top if h.chunk.doc_id == chunk.doc_id}
            # Реквизиты стороны — цель для «изложить банковские реквизиты …».
            if any(label.startswith("req") for label in clauses):
                clauses.add("req")
            for i in index.amendments.get(chunk.doc_id, []):
                extra = index.chunks[i]
                if extra.chunk_id in chosen or added >= self.config.max_amendment_chunks:
                    continue
                targets = set(extra.amends)
                if not targets and "реквизит" in extra.text.lower():
                    targets = {"req"}
                relevant = (targets and chunk.label in targets) or (
                    "req" in targets and chunk.label.startswith("req")
                )
                if relevant or (not targets and extra.chunk_id in pool_ids):
                    result.append(Hit(extra, added="amendment"))
                    chosen.add(extra.chunk_id)
                    added += 1
        return result
