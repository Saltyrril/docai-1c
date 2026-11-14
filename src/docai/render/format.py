"""Форматирование значений для печатных форм: суммы, количества, даты, сумма прописью."""

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal

NBSP = " "

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


def _group(int_part: str) -> str:
    """Разбить целую часть на тройки через неразрывный пробел."""
    out: list[str] = []
    while len(int_part) > 3:
        out.insert(0, int_part[-3:])
        int_part = int_part[:-3]
    out.insert(0, int_part)
    return NBSP.join(out)


def money(value: Decimal | None) -> str:
    """12200 → «12 200,00» (пробел неразрывный)."""
    if value is None:
        return ""
    q = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    sign = "-" if q < 0 else ""
    int_part, frac = f"{abs(q):f}".split(".")
    return f"{sign}{_group(int_part)},{frac}"


def price(value: Decimal | None) -> str:
    """Цена: две копейки обязательно, больше знаков — только если они значимы."""
    if value is None:
        return ""
    exponent = value.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        text = f"{abs(value):f}".rstrip("0")
        int_part, frac = text.split(".")
        frac = frac.ljust(2, "0")
        sign = "-" if value < 0 else ""
        return f"{sign}{_group(int_part)},{frac}"
    return money(value)


def quantity(value: Decimal | None) -> str:
    """Количество без хвостовых нулей: 44 → «44», 1.500 → «1,5»."""
    if value is None:
        return ""
    text = f"{value.normalize():f}"
    int_part, _, frac = text.partition(".")
    sign = "-" if int_part.startswith("-") else ""
    int_part = int_part.lstrip("-")
    return f"{sign}{_group(int_part)}" + (f",{frac}" if frac else "")


def date_num(value: dt.date | None) -> str:
    """14.11.2026."""
    return "" if value is None else value.strftime("%d.%m.%Y")


def date_long(value: dt.date | None) -> str:
    """«14» ноября 2026 г."""
    if value is None:
        return ""
    return f"«{value.day:02d}» {_MONTHS_GEN[value.month - 1]} {value.year} г."


def vat_rate(value: object) -> str:
    """Ставка НДС как в документе: «22%», «22/122», «без НДС»."""
    if value is None:
        return ""
    return str(getattr(value, "value", value))


_ONES_M = ("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_ONES_F = ("", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_TEENS = [
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
]
_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    ("восемьдесят"),
    "девяносто",
)
_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    return forms[1] if n % 10 in (2, 3, 4) else forms[0] if n % 10 == 1 else forms[2]


def _triplet(n: int, feminine: bool) -> list[str]:
    words = [_HUNDREDS[n // 100]]
    rest = n % 100
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        words.append(_TENS[rest // 10])
        words.append((_ONES_F if feminine else _ONES_M)[rest % 10])
    return [w for w in words if w]


def amount_in_words(value: Decimal | None) -> str:
    """12200.50 → «Двенадцать тысяч двести рублей 50 копеек»."""
    if value is None:
        return ""
    q = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    rub, kop = divmod(int(abs(q) * 100), 100)
    groups = (
        (rub % 1000, False, ("", "", "")),
        ((rub // 1000) % 1000, True, ("тысяча", "тысячи", "тысяч")),
        ((rub // 10**6) % 1000, False, ("миллион", "миллиона", "миллионов")),
        ((rub // 10**9) % 1000, False, ("миллиард", "миллиарда", "миллиардов")),
    )
    parts: list[str] = []
    for number, feminine, forms in reversed(groups):
        if number:
            parts += _triplet(number, feminine)
            if forms[0]:
                parts.append(_plural(number, forms))
    if not parts:
        parts = ["ноль"]
    text = " ".join(parts) + " " + _plural(rub, ("рубль", "рубля", "рублей"))
    text += f" {kop:02d} " + _plural(kop, ("копейка", "копейки", "копеек"))
    return ("минус " if q < 0 else "") + text[0].upper() + text[1:]
