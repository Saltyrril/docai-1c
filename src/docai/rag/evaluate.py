"""Оценка RAG на вопросах P18: попадание поиска и правильность цитат.

Попадание считается по пересечению смещений найденного пункта с фрагментом-источником
эталона (D62), а не по совпадению id: так оценка не зависит от нарезки индекса.

Цитаты ответа правильные, если:
- маршрут `rag`: ответ найден, хотя бы одна цитата пересекается с источником и все цитаты —
  из документов вопроса (`doc_ids`: договор и его соглашения, регламент); цитаты старой
  редакции (`superseded`) считаются отдельно;
- маршрут `none`: ответ «не нашёл» и цитат нет;
- маршрут `table`: запрос к реестру вернул эталонное значение (`expected_value`).
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from docai.rag.facts import keys_found


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    def overlaps(self, other: "Span") -> bool:
        return self.doc_id == other.doc_id and self.start < other.end and other.start < self.end


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def fragment_spans(fragments_path: Path) -> dict[str, Span]:
    return {
        f["fragment_id"]: Span(f["doc_id"], f["start"], f["end"])
        for f in load_jsonl(fragments_path)
    }


def source_spans(
    question: dict[str, Any], spans: dict[str, Span], key: str = "sources"
) -> list[Span]:
    return [spans[s["fragment_id"]] for s in question.get(key) or [] if s["fragment_id"] in spans]


def stratified(
    questions: Sequence[dict[str, Any]],
    per_category: dict[str, int],
    seed: int,
    exclude: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Выборка по категориям с фиксированным seed; порядок — как в наборе."""
    rng = random.Random(seed)
    skip = set(exclude)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        if q["id"] not in skip:
            by_category[q["category"]].append(q)
    chosen: set[str] = set()
    for category, n in per_category.items():
        pool = by_category.get(category, [])
        chosen.update(q["id"] for q in rng.sample(pool, min(n, len(pool))))
    return [q for q in questions if q["id"] in chosen]


def ranking_metrics(
    ranked: Sequence[Span], gold: Sequence[Span], ks: Sequence[int]
) -> dict[str, float]:
    """hit@k (нашёлся хотя бы один источник), recall@k (доля источников), MRR."""
    result: dict[str, float] = {}
    first = next((i for i, s in enumerate(ranked, 1) if any(s.overlaps(g) for g in gold)), None)
    result["mrr"] = 1.0 / first if first else 0.0
    for k in ks:
        top = ranked[:k]
        found = [g for g in gold if any(s.overlaps(g) for s in top)]
        result[f"hit@{k}"] = 1.0 if found else 0.0
        result[f"recall@{k}"] = len(found) / len(gold) if gold else 0.0
    return result


def _number(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def value_found(expected: Any, rows: Sequence[Sequence[Any]] | None) -> bool:
    """Эталонное значение есть среди ячеек результата (числа — с точностью до копейки)."""
    target = _number(expected)
    for row in rows or []:
        for cell in row:
            if str(cell) == str(expected):
                return True
            value = _number(cell)
            if target is not None and value is not None and abs(value - target) <= Decimal("0.01"):
                return True
    return False


def judge(
    question: dict[str, Any], answer: dict[str, Any], spans: dict[str, Span]
) -> dict[str, Any]:
    """Правильность ответа и цитат на одном вопросе (см. описание модуля)."""
    route = question["route"]
    citations = answer.get("citations") or []
    cited = [Span(c["doc_id"], c["start"], c["end"]) for c in citations]
    verdict: dict[str, Any] = {"route_ok": None, "citations_ok": False, "keys_ok": None}
    if route == "table":
        verdict["route_ok"] = answer.get("route") == "sql"
        verdict["citations_ok"] = verdict["route_ok"] and value_found(
            question.get("expected_value"), answer.get("sql_rows")
        )
        return verdict
    verdict["route_ok"] = answer.get("route") == "rag"
    if route == "none":
        verdict["citations_ok"] = not answer.get("found") and not citations
        return verdict
    gold = source_spans(question, spans)
    old = source_spans(question, spans, "superseded")
    hits = [s for s in cited if any(s.overlaps(g) for g in gold)]
    allowed_docs = set(question.get("doc_ids") or []) | {g.doc_id for g in gold}
    verdict.update(
        cited=len(cited),
        cited_gold=len(hits),
        sources_cited=sum(1 for g in gold if any(s.overlaps(g) for s in cited)),
        sources=len(gold),
        cited_superseded=sum(1 for s in cited if any(s.overlaps(o) for o in old)),
        foreign_docs=sorted({s.doc_id for s in cited} - allowed_docs),
    )
    # Старая редакция рядом с действующей («было 30 дней, соглашением сокращено до 20»)
    # допустима и считается отдельно (`cited_superseded`); ошибка — если процитирована
    # только она: тогда среди цитат нет источника.
    verdict["citations_ok"] = bool(answer.get("found") and hits and not verdict["foreign_docs"])
    verdict["keys_ok"] = keys_found(answer.get("answer") or "", question.get("keys") or [])
    return verdict
