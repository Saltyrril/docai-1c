"""Нормализаторы: строки из документа и из ответа модели → канонические значения.

Функции `parse_*` бросают `ValueError` с понятным сообщением, если строку разобрать
нельзя (их можно ставить в Pydantic как BeforeValidator). Функции `normalize_*`
для реквизитов не бросают исключений: они только чистят строку.
`normalize_for_compare` приводит значение поля из FIELD_SPEC к виду для сравнения
и никогда не бросает исключений: неразбираемое значение остаётся строкой.
"""

import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from docai.schemas.common import VatRate
from docai.schemas.fields import NormKind

# Пробелы внутри чисел: обычный, неразрывный, узкий неразрывный, тонкий; апостроф.
_SPACES_RE = re.compile(r"[\s   ']+")
_MINUS_CHARS = "-−–"  # дефис, минус, короткое тире

# «12 200 руб. 00 коп.», «12200 р. 5 к.»
_RUB_KOP_RE = re.compile(
    r"^(?P<sign>[-−–]?)\s*(?P<rub>\d[\d\s   ]*)\s*"
    r"(?:руб(?:лей|ля|ль)?|р)\.?\s*(?P<kop>\d{1,2})\s*(?:коп(?:еек|ейки|ейка)?|к)\.?$",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"(руб(?:лей|ля|ль)?\.?|р\.|₽|rub\.?|rur\.?)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# «12 200-00»: копейки через дефис, как пишут от руки и в некоторых бланках.
_DASH_KOP_RE = re.compile(r"^(-?\d+)[-–](\d{2})$")


def parse_decimal(value: str | int | float | Decimal) -> Decimal:
    """Число из строки: «12 200,00», «12200.00», «1 234 567,5», «1,234.50», «12 200-00».

    Запятая и точка обе считаются десятичным разделителем. Если в строке есть оба,
    десятичный — последний. Если один знак повторяется («1.234.567»), это разделитель
    тысяч. Знак рубля и слово «руб.» отбрасываются.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("ожидалось число, а не логическое значение")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    text = _CURRENCY_RE.sub("", value.strip())
    text = _SPACES_RE.sub("", text)
    if text[:1] in _MINUS_CHARS:
        text = "-" + text[1:]
    text = _DASH_KOP_RE.sub(r"\1.\2", text)
    commas, dots = text.count(","), text.count(".")
    if commas and dots:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif commas > 1 or dots > 1:
        text = text.replace(",", "").replace(".", "")
    else:
        text = text.replace(",", ".")
    if not _NUMBER_RE.fullmatch(text):
        raise ValueError(f"не удалось разобрать число «{value}»")
    try:
        return Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover — regex уже отсёк мусор
        raise ValueError(f"не удалось разобрать число «{value}»") from exc


def parse_money(value: str | int | float | Decimal) -> Decimal:
    """Сумма в рублях: всё, что понимает `parse_decimal`, плюс «12 200 руб. 00 коп.»."""
    if isinstance(value, str):
        match = _RUB_KOP_RE.match(value.strip())
        if match:
            rub = int(_SPACES_RE.sub("", match["rub"]))
            kop = int(match["kop"])
            if kop > 99:  # pragma: no cover — regex пропускает не больше двух цифр
                raise ValueError(f"копеек больше 99 в «{value}»")
            amount = Decimal(rub) + Decimal(kop) / 100
            return -amount if match["sign"] else amount
    return parse_decimal(value)


def quantize_money(value: Decimal) -> Decimal:
    """Округлить до копеек (половина — вверх по модулю, как в бухгалтерии)."""
    return value.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")


# Месяцы: по первым трём буквам (родительный и именительный падеж, сокращения).
_MONTHS = {
    "янв": 1,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "мая": 5,
    "май": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}
_NUMERIC_DATE_RE = re.compile(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{4}|\d{2})")
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_TEXT_DATE_RE = re.compile(r"(\d{1,2})\s+([а-яё]+)\.?\s+(\d{4})", re.IGNORECASE)


def _make_date(year: int, month: int, day: int, source: str) -> dt.date:
    try:
        return dt.date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"несуществующая дата «{source}»") from exc


def parse_date(value: str | dt.date) -> dt.date:
    """Дата из «14.11.2026», «14.11.26», «2026-11-14», «14 ноября 2026 г.», «"14" нояб. 2026».

    Двузначный год считается 20xx.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = re.sub(r"[«»\"'“”„]", "", value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*(?:г\.?|года?)$", "", text, flags=re.IGNORECASE).strip()
    if match := _ISO_DATE_RE.fullmatch(text):
        year, month, day = (int(g) for g in match.groups())
        return _make_date(year, month, day, value)
    if match := _NUMERIC_DATE_RE.fullmatch(text):
        day, month = int(match[1]), int(match[2])
        year = int(match[3]) + (2000 if len(match[3]) == 2 else 0)
        return _make_date(year, month, day, value)
    if match := _TEXT_DATE_RE.fullmatch(text):
        month = _MONTHS.get(match[2].lower().replace("ё", "е")[:3])
        if month is None:
            raise ValueError(f"неизвестный месяц «{match[2]}» в дате «{value}»")
        return _make_date(int(match[3]), month, int(match[1]), value)
    raise ValueError(f"не удалось разобрать дату «{value}»")


def normalize_inn(value: str) -> str:
    """ИНН: убрать пробелы, дефисы и подпись «ИНН»."""
    text = re.sub(r"^\s*ИНН[:\s]*", "", value, flags=re.IGNORECASE)
    return re.sub(r"[\s \-]", "", text)


# Кириллические буквы, похожие на латинские (частая ошибка OCR в КПП).
_CYR_TO_LAT = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")


def normalize_kpp(value: str) -> str:
    """КПП: убрать пробелы и подпись «КПП», буквы — в латиницу верхнего регистра."""
    text = re.sub(r"^\s*КПП[:\s]*", "", value, flags=re.IGNORECASE)
    return re.sub(r"[\s \-]", "", text).upper().translate(_CYR_TO_LAT)


def split_inn_kpp(value: str) -> tuple[str, str | None]:
    """«ИНН/КПП 7707083893/773601001» → ("7707083893", "773601001"); без КПП — (ИНН, None)."""
    text = re.sub(r"^\s*ИНН\s*/\s*КПП[:\s]*", "", value, flags=re.IGNORECASE)
    inn, _, kpp = text.partition("/")
    return normalize_inn(inn), (normalize_kpp(kpp) or None)


def normalize_digits(value: str) -> str:
    """БИК, счета: оставить только цифры."""
    return re.sub(r"\D", "", value)


def normalize_okei(value: str) -> str:
    """Код ОКЕИ: только цифры, дополнить нулями слева до 3 знаков («6» → «006»)."""
    digits = normalize_digits(value)
    return digits.zfill(3) if digits else ""


_QUOTES_RE = re.compile(r"[«»\"“”„'`]")


def normalize_text(value: str) -> str:
    """Текст для сравнения: нижний регистр, ё→е, без кавычек, один пробел между словами."""
    text = _QUOTES_RE.sub("", value).replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_doc_number(value: str) -> str:
    """Номер документа: как текст, без «№» и без пробелов («№ 12 / 3» → «12/3»)."""
    return re.sub(r"[\s№]", "", normalize_text(value))


_VAT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:%|/\s*(\d+))?")


def parse_vat_rate(value: str | VatRate) -> VatRate:
    """Ставка НДС: «22%», «22 %», «НДС 22», «22/122», «Без НДС», «без налога (НДС)»."""
    if isinstance(value, VatRate):
        return value
    text = value.strip().lower()
    if "без" in text:
        return VatRate.NO_VAT
    match = _VAT_RE.search(text)
    if match:
        number = parse_decimal(match[1])
        if number < 1 and match[2] is None and "%" not in text and number > 0:
            number *= 100  # 0.22 → 22
        key = f"{number.normalize():f}"
        candidate = f"{key}/{match[2]}" if match[2] else f"{key}%"
        try:
            return VatRate(candidate)
        except ValueError:
            pass
    raise ValueError(f"неизвестная ставка НДС «{value}»")


def normalize_for_compare(kind: NormKind, value: Any) -> Any:
    """Привести значение поля к виду для сравнения при подсчёте метрик.

    None и пустая строка → None. Если значение не разбирается (модель вернула
    мусор), возвращается нормализованная строка — она просто не совпадёт с эталоном.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        match kind:
            case NormKind.TEXT:
                return normalize_text(str(value))
            case NormKind.DOC_NUMBER:
                return normalize_doc_number(str(value))
            case NormKind.DATE:
                return parse_date(value)
            case NormKind.MONEY:
                return quantize_money(parse_money(value))
            case NormKind.QUANTITY:
                return parse_decimal(value).normalize()
            case NormKind.INN:
                return normalize_inn(str(value))
            case NormKind.KPP:
                return normalize_kpp(str(value))
            case NormKind.DIGITS:
                return normalize_digits(str(value))
            case NormKind.OKEI:
                return normalize_okei(str(value))
            case NormKind.VAT_RATE:
                return parse_vat_rate(value)
            case NormKind.EXACT:
                return str(value).strip()
    except (ValueError, TypeError):
        return normalize_text(str(value))
    raise AssertionError(f"неизвестный вид нормализации {kind}")  # pragma: no cover
