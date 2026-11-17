"""Расстояние Левенштейна и ANLS для текстовых полей.

ANLS (Average Normalized Levenshtein Similarity) — Biten et al., «Scene Text Visual
Question Answering», ICCV 2019, https://arxiv.org/abs/1905.13648:

    NL(a, b) = Lev(a, b) / max(|a|, |b|)
    s(a, b)  = 1 − NL(a, b), если NL(a, b) < τ, иначе 0      (τ = 0,5)
    ANLS     = среднее s по всем эталонным значениям поля

Строки сравниваются после нормализации FIELD_SPEC (регистр, ё→е, кавычки, пробелы).
Пустое предсказание при непустом эталоне даёт 0, два пустых — 1.
"""


def levenshtein(a: str, b: str) -> int:
    """Число вставок, удалений и замен символов, переводящих a в b."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalized_distance(a: str, b: str) -> float:
    """Lev(a, b) / max(|a|, |b|); для двух пустых строк — 0."""
    longest = max(len(a), len(b))
    return levenshtein(a, b) / longest if longest else 0.0


def similarity(a: str | None, b: str | None) -> float:
    """1 − NL без порога (для сопоставления строк таблицы). None — как пустая строка."""
    return 1.0 - normalized_distance(a or "", b or "")


def anls_score(pred: str | None, ref: str | None, threshold: float = 0.5) -> float:
    """Оценка одного значения по ANLS: 1 − NL, если NL < τ, иначе 0."""
    nl = normalized_distance(pred or "", ref or "")
    return 1.0 - nl if nl < threshold else 0.0
