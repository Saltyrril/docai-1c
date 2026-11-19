"""Маршрутизатор вопроса: агрегат по реестру первичных документов → SQL, остальное → RAG.

Правила, а не модель: признаки агрегата узнаются по словам («сколько УПД», «на какую
сумму … получено», «последний документ от …»), решение мгновенное и объяснимое, а модели
на GPU не приходится переключаться между задачами. Точность правил меряется на 200
вопросах P18 (тест `test_router_on_p18_questions`).
"""

import re
from typing import Literal

Route = Literal["sql", "rag"]

_DOC = (
    r"(?:документ\w*|сч[её]т\w*|упд|акт(?:ов|ы|а)?|накладн\w*|торг-?12|"
    r"сч[её]т\w*-фактур\w*|отгрузочн\w*)"
)
# «Сколько УПД», «сколько накладных ТОРГ-12», «сколько документов»: считаем документы.
_COUNT_DOCS = re.compile(rf"\bсколько\s+(?:[а-яё-]+\s+)?{_DOC}\b", re.IGNORECASE)
# Документ получен/выставлен/прислан — событие реестра, а не условие договора.
_RECEIVED = re.compile(
    r"\b(?:получен[оаы]?|получили|прислал\w*|выставил\w*|выставлен\w*|предъявил\w*|"
    r"зарегистрирован\w*|поступил\w*|в\s+реестре)\b",
    re.IGNORECASE,
)
_AGGREGATE = re.compile(
    r"\b(?:сколько|сумм\w*|итог\w*|крупн\w*|последн\w*|больше\s+всего|чаще\s+всего|"
    r"всего|общ\w+|средн\w*|максимальн\w*)\b",
    re.IGNORECASE,
)
_DOC_OR_VAT = re.compile(rf"\b(?:{_DOC}|ндс|контрагент\w*)\b", re.IGNORECASE)


# «Сколько счетов можно оплатить без договора?» — правило регламента, а не подсчёт.
_RULE = re.compile(r"\b(?:можно|нужно|надо|должн\w*|разреш\w*|допуска\w*|положено)\b", re.I)


def route_question(question: str) -> Route:
    text = question.replace("ё", "е").replace("Ё", "Е")
    if _RULE.search(text):
        return "rag"
    if _COUNT_DOCS.search(text):
        return "sql"
    if _RECEIVED.search(text) and _AGGREGATE.search(text) and _DOC_OR_VAT.search(text):
        return "sql"
    return "rag"
