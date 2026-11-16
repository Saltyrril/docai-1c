"""Дословное вхождение значения в текст (текстовый слой PDF или OCR)."""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from docai.schemas.fields import NormKind
from docai.validation.normalize import normalize_for_compare, normalize_text, quantize_money

SourceKind = Literal["text_layer", "ocr"]

_MONTHS_GEN = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]
# Пробел-разделитель тысяч между цифрами: «12 200,50» → «12200,50».
_THOUSANDS_RE = re.compile(r"(?<=\d)[\s  ](?=\d{3}(?!\d))")


@dataclass(frozen=True, slots=True)
class SourceText:
    """Текст-источник: исходный вид, «плотный» (без разделителей тысяч) и для сравнения."""

    kind: SourceKind
    raw: str
    dense: str
    plain: str

    @classmethod
    def build(cls, kind: SourceKind, text: str) -> "SourceText":
        raw = text.replace("|", " ")
        return cls(kind, raw, _THOUSANDS_RE.sub("", raw), normalize_text(raw))


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def _number_forms(value: Decimal, with_kopecks: bool) -> list[str]:
    if with_kopecks:
        money = quantize_money(value)
        forms = [f"{money:.2f}".replace(".", "[.,]")]
        if money == money.to_integral_value():
            forms.append(re.escape(str(int(money))))
        return forms
    plain = f"{value.normalize():f}"
    if "." in plain:
        return [plain.replace(".", "[.,]")]
    return [re.escape(plain), re.escape(plain) + r"[.,]0+"]


def _in_number(value: Decimal, sources: list[SourceText], with_kopecks: bool) -> bool:
    for form in _number_forms(value, with_kopecks):
        pattern = rf"(?<![\d.,]){form}(?!\d|[.,]\d)"
        if any(_has(pattern, s.dense) for s in sources):
            return True
    return False


def _in_digits(digits: str, sources: list[SourceText]) -> bool:
    pattern = rf"(?<!\d){re.escape(digits)}(?!\d)"
    return any(_has(pattern, s.dense) or _has(pattern, s.raw) for s in sources)


def _in_date(value: date, sources: list[SourceText]) -> bool:
    year_short = f"{value.year % 100:02d}"
    forms = [
        rf"(?<!\d){value.day:02d}[./\-]{value.month:02d}[./\-](?:{value.year}|{year_short})",
        rf"(?<!\d){value.day}[./\-]{value.month:02d}[./\-]{value.year}",
        rf"{value.year}-{value.month:02d}-{value.day:02d}",
        rf"(?<!\d)0?{value.day}\s+{_MONTHS_GEN[value.month - 1]}\s+{value.year}",
    ]
    return any(_has(f, s.plain) for s in sources for f in forms)


def _in_text(text: str, sources: list[SourceText]) -> bool:
    flat = normalize_text(text).replace("№", "").replace(" ", "")
    if not flat:
        return False
    return any(flat in s.plain.replace("№", "").replace(" ", "") for s in sources)


def verbatim(norm: NormKind, value: Any, sources: list[SourceText]) -> tuple[bool, bool] | None:
    """(найдено, сильное_ли_свидетельство) или None, если для поля это не определено.

    Короткие значения («2», «шт») встречаются в тексте случайно, поэтому их вхождение
    считается слабым свидетельством.
    """
    if not sources or value is None:
        return None
    canon = normalize_for_compare(norm, value)
    if canon is None:
        return None
    match norm:
        case NormKind.INN | NormKind.DIGITS | NormKind.OKEI:
            digits = str(canon)
            if not digits.isdigit():
                return False, False
            return _in_digits(digits, sources), len(digits) >= 6
        case NormKind.KPP:
            text = str(canon)
            found = _in_digits(text, sources) if text.isdigit() else _in_text(text, sources)
            return found, True
        case NormKind.MONEY | NormKind.QUANTITY:
            if not isinstance(canon, Decimal):
                return False, False
            strong = len(f"{abs(canon).normalize():f}".replace(".", "")) >= 4
            return _in_number(canon, sources, norm is NormKind.MONEY), strong
        case NormKind.DATE:
            return (_in_date(canon, sources) if isinstance(canon, date) else False), True
        case NormKind.TEXT | NormKind.DOC_NUMBER:
            text = str(canon)
            return _in_text(text, sources), len(text) >= 5
    return None
