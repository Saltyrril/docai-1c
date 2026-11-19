"""Вопросы владельца (30 штук своими словами): проверка и приведение к формату вопросов.

Владелец пишет вопрос, ответ и дословную цитату из документа (`doc_id` + `quote`), а
id фрагмента находится сам: цитата ищется в тексте документа, фрагмент — тот, внутри
которого она лежит. Пустые заготовки (без текста вопроса) пропускаются.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from docai.rag.corpus import Corpus
from docai.rag.document import CorpusDoc
from docai.rag.questions import CATEGORIES

_ROUTES = {"registry_aggregate": "table", "no_answer": "none"}


@dataclass(slots=True)
class ManualResult:
    total: int
    questions: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip()


def locate(doc: CorpusDoc, quote: str) -> list[str]:
    """id фрагментов, в которые попадает цитата (без учёта пробелов, переносов и «ё»)."""
    target = _squash(quote).lower()
    if not target:
        return []
    found = []
    for fragment in doc.fragments:
        if target in _squash(fragment.text).lower():
            found.append(fragment.fragment_id)
    return found


def check_manual(path: Path, corpus: Corpus) -> ManualResult:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("questions") or []
    docs = {d.doc_id: d for d in corpus.documents}
    result = ManualResult(total=len(items))
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        qid = str(item.get("id") or f"M{index:02d}")
        question = (item.get("question") or "").strip()
        if not question:
            continue
        where = f"{qid}:"
        if qid in seen:
            result.problems.append(f"{where} id повторяется")
        seen.add(qid)
        category = item.get("category")
        if category not in CATEGORIES:
            result.problems.append(
                f"{where} категория «{category}» не из списка: {', '.join(CATEGORIES)}"
            )
            continue
        answer = (item.get("answer") or "").strip()
        if not answer:
            result.problems.append(f"{where} нет эталонного ответа")
        sources: list[dict[str, str]] = []
        for source in item.get("sources") or []:
            doc_id = (source or {}).get("doc_id", "")
            quote = (source or {}).get("quote", "")
            doc = docs.get(doc_id)
            if doc is None:
                result.problems.append(f"{where} документа {doc_id!r} нет в корпусе")
                continue
            fragments = locate(doc, quote)
            if not fragments:
                result.problems.append(
                    f"{where} цитата не найдена в {doc_id}: «{quote[:60]}» "
                    f"(скопируйте фразу из файла документа дословно, внутри одного пункта)"
                )
                continue
            sources += [{"fragment_id": f, "doc_id": doc_id, "quote": quote} for f in fragments]
        if category == "no_answer" and sources:
            result.problems.append(f"{where} у вопроса без ответа не должно быть источников")
        if (
            category not in ("no_answer", "registry_aggregate")
            and not sources
            and not any(p.startswith(where) for p in result.problems)
        ):
            result.problems.append(f"{where} нужен хотя бы один источник (doc_id и quote)")
        if category == "registry_aggregate" and not (item.get("sql") or item.get("expected_value")):
            result.problems.append(
                f"{where} для агрегата по реестру укажите sql или expected_value"
            )
        result.questions.append(
            {
                "id": qid,
                "category": category,
                "category_name": CATEGORIES[category],
                "question": question,
                "answer": answer,
                "keys": [str(k) for k in item.get("keys") or []],
                "route": _ROUTES.get(category, "rag"),
                "sources": sources,
                "superseded": [],
                "wrong_answers": [str(x) for x in item.get("wrong_answers") or []],
                "doc_ids": sorted({s["doc_id"] for s in sources}),
                "registry_rows": [],
                "sql": item.get("sql"),
                "expected_value": item.get("expected_value"),
                "template": "manual",
                "author": "owner",
                "note": item.get("note") or "",
            }
        )
    return result
