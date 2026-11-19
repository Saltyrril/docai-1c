"""Слова и леммы для BM25 и поиска названий: pymorphy3, кэш по словоформе.

«Вектору» → «вектор», «просрочки» → «просрочка». Числа и номера («П-668/23», «30»)
остаются как есть: по ним ищут номер договора и сроки. Слово через дефис даёт и целую
форму, и части: «Балтика-Торг» → «балтика-торг», «балтика», «торг».
"""

import re
from functools import lru_cache

import pymorphy3

# Слово: буквы/цифры, внутри — дефис, слеш или точка между цифрами («0,05» режется на числа).
_WORD = re.compile(r"[0-9a-zа-яё]+(?:[-/][0-9a-zа-яё]+)*", re.IGNORECASE)

# Служебные слова, которые ничего не дают поиску. Отрицание «не» оставлено: «не входят».
_STOPWORDS_TEXT = """
и в во на с со к ко по о об от до за из у же ли бы то а но или либо что чем как так
это этот эта эти тот та те для при под над без через про между также который которая
которые которого которой каждый каждой каждого его ее её их им ими она они оно он мы
нам нас наш наша наши вы вам ваш быть был была были есть является являются
какой какая какие каков какова каковы где когда сколько нужно надо можно
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


_morph: pymorphy3.MorphAnalyzer | None = None


def _analyzer() -> pymorphy3.MorphAnalyzer:
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def words(text: str) -> list[str]:
    """Слова текста в нижнем регистре, «ё» → «е»."""
    return [m.group(0).lower().replace("ё", "е") for m in _WORD.finditer(text)]


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    """Нормальная форма слова; числа и слова латиницей — как есть."""
    if not re.search("[а-яе]", word) or word[0].isdigit():
        return word
    return _analyzer().parse(word)[0].normal_form.replace("ё", "е")


def lemmas(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Леммы для BM25: составное слово даёт себя целиком и свои части."""
    result: list[str] = []
    for word in words(text):
        parts = re.split(r"[-/]", word)
        if len(parts) > 1:
            result.append(word)
            result.extend(lemma(p) for p in parts if p and not (drop_stopwords and p in STOPWORDS))
            continue
        if drop_stopwords and word in STOPWORDS:
            continue
        result.append(lemma(word))
    return result


def name_lemmas(text: str) -> list[str]:
    """Леммы частей названия по порядку («Балтика-Торг» → «балтика», «торг»)."""
    return [lemma(p) for w in words(text) for p in re.split(r"[-/]", w) if p]
