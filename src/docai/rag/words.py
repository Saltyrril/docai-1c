"""Числа прописью, склонение по числу, даты и суммы — для текста договоров и эталонных ответов.

Юридический стиль требует «в течение 10 (десяти) рабочих дней» и
«85 000 (восемьдесят пять тысяч) рублей 00 копеек», поэтому здесь есть и именительный,
и родительный падеж числительных. Числа больше 999 в родительном падеже не нужны.
"""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

_UNITS_M = ("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_UNITS_F = ("", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)  # fmt: skip
_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
    "восемьдесят", "девяносто",
)  # fmt: skip
_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот",
    "девятьсот",
)  # fmt: skip

_GEN_UNITS = (
    "", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти",
)  # fmt: skip
_GEN_TEENS = (
    "десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати",
    "шестнадцати", "семнадцати", "восемнадцати", "девятнадцати",
)  # fmt: skip
_GEN_TENS = (
    "", "", "двадцати", "тридцати", "сорока", "пятидесяти", "шестидесяти", "семидесяти",
    "восьмидесяти", "девяноста",
)  # fmt: skip
_GEN_HUNDREDS = (
    "", "ста", "двухсот", "трёхсот", "четырёхсот", "пятисот", "шестисот", "семисот",
    "восьмисот", "девятисот",
)  # fmt: skip

MONTHS_GEN = (
    "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
    "октября", "ноября", "декабря",
)  # fmt: skip


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Форма слова после числа: (1 день, 2 дня, 5 дней)."""
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _triad(n: int, feminine: bool) -> list[str]:
    words = [_HUNDREDS[n // 100]]
    rest = n % 100
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        words.append(_TENS[rest // 10])
        words.append((_UNITS_F if feminine else _UNITS_M)[rest % 10])
    return [w for w in words if w]


def number_words(n: int, *, feminine: bool = False) -> str:
    """Число прописью в именительном падеже: 1250000 → «один миллион двести пятьдесят тысяч»."""
    if n < 0:
        raise ValueError("отрицательные числа не нужны")
    if n == 0:
        return "ноль"
    groups = (
        (10**9, False, ("миллиард", "миллиарда", "миллиардов")),
        (10**6, False, ("миллион", "миллиона", "миллионов")),
        (10**3, True, ("тысяча", "тысячи", "тысяч")),
    )
    words: list[str] = []
    for size, fem, forms in groups:
        part = n // size % 1000
        if part:
            words += _triad(part, fem)
            words.append(plural(part, forms))
    words += _triad(n % 1000, feminine)
    return " ".join(words)


def number_words_genitive(n: int) -> str:
    """Число прописью в родительном падеже (до 999): 45 → «сорока пяти»."""
    if not 1 <= n <= 999:
        raise ValueError("в родительном падеже поддерживаются числа 1..999")
    words = [_GEN_HUNDREDS[n // 100]]
    rest = n % 100
    if 10 <= rest <= 19:
        words.append(_GEN_TEENS[rest - 10])
    else:
        words.append(_GEN_TENS[rest // 10])
        words.append(_GEN_UNITS[rest % 10])
    return " ".join(w for w in words if w)


# --- Дни ---------------------------------------------------------------------------------

# Прилагательное к «день»: именительный ед., родительный мн./им. мн. после 2–4, родительный ед.
DAY_KINDS = {
    "calendar": ("календарный", "календарных", "календарного"),
    "working": ("рабочий", "рабочих", "рабочего"),
    "banking": ("банковский", "банковских", "банковского"),
}


def days_nom(n: int, kind: str | None = "calendar") -> str:
    """«30 календарных дней», «3 рабочих дня», «1 рабочий день» — для ответов."""
    noun = plural(n, ("день", "дня", "дней"))
    if kind is None:
        return f"{n} {noun}"
    sg, pl, _ = DAY_KINDS[kind]
    adj = sg if noun == "день" else pl
    return f"{n} {adj} {noun}"


def days_gen(n: int, kind: str | None = "calendar", *, words: bool = False) -> str:
    """Родительный падеж после «в течение»: «30 (тридцати) календарных дней»."""
    single = n % 10 == 1 and n % 100 != 11
    noun = "дня" if single else "дней"
    number = f"{n} ({number_words_genitive(n)})" if words else str(n)
    if kind is None:
        return f"{number} {noun}"
    _, pl, gen_sg = DAY_KINDS[kind]
    return f"{number} {gen_sg if single else pl} {noun}"


def months_nom(n: int) -> str:
    return f"{n} {plural(n, ('месяц', 'месяца', 'месяцев'))}"


def months_words(n: int) -> str:
    """«12 (двенадцать) месяцев»."""
    return f"{n} ({number_words(n)}) {plural(n, ('месяц', 'месяца', 'месяцев'))}"


# --- Проценты ----------------------------------------------------------------------------

_FRACTION_WORDS = {
    Decimal("0.01"): "одна сотая",
    Decimal("0.02"): "две сотых",
    Decimal("0.03"): "три сотых",
    Decimal("0.05"): "пять сотых",
    Decimal("0.1"): "одна десятая",
    Decimal("0.2"): "две десятых",
    Decimal("0.3"): "три десятых",
    Decimal("0.5"): "пять десятых",
}


def decimal_text(value: Decimal) -> str:
    """Десятичное число с запятой и без лишних нулей: 0.10 → «0,1», 54.30 → «54,3»."""
    text = format(value.normalize(), "f")
    return text.replace(".", ",")


def percent_words(p: int) -> str:
    """«тридцать процентов»."""
    return f"{number_words(p)} {plural(p, ('процент', 'процента', 'процентов'))}"


def percent_text(p: int, style: int) -> str:
    """Процент в одном из стилей: «30%», «30% (тридцать процентов)», «30 (тридцать) процентов»."""
    if style == 0:
        return f"{p}%"
    if style == 1:
        return f"{p}% ({percent_words(p)})"
    return f"{p} ({number_words(p)}) {plural(p, ('процент', 'процента', 'процентов'))}"


def rate_text(rate: Decimal, style: int) -> str:
    """Дробный процент неустойки.

    «0,1%», «0,1 (ноль целых одна десятая) процента» или «0,1 процента».
    """
    number = decimal_text(rate)
    if style == 0:
        return f"{number}%"
    if style == 1 and rate in _FRACTION_WORDS:
        return f"{number} (ноль целых {_FRACTION_WORDS[rate]}) процента"
    return f"{number} процента"


# --- Деньги ------------------------------------------------------------------------------

_CENT = Decimal("0.01")


def money_digits(amount: Decimal, *, kopecks: bool = True) -> str:
    """«1 250 000,00» (пробел между разрядами)."""
    amount = amount.quantize(_CENT, rounding=ROUND_HALF_UP)
    rubles = int(amount)
    groups = f"{rubles:,}".replace(",", " ")
    if not kopecks:
        return groups
    cents = int((amount - rubles) * 100)
    return f"{groups},{cents:02d}"


def rubles_text(amount: Decimal, style: int) -> str:
    """Сумма в одном из стилей.

    0 — «85 000,00 руб.»; 1 — «85 000 (восемьдесят пять тысяч) рублей 00 копеек»;
    2 — «85 000 рублей» (если без копеек; иначе как 0).
    """
    amount = amount.quantize(_CENT, rounding=ROUND_HALF_UP)
    rubles = int(amount)
    cents = int((amount - rubles) * 100)
    if style == 1:
        rub_word = plural(rubles, ("рубль", "рубля", "рублей"))
        kop_word = plural(cents, ("копейка", "копейки", "копеек"))
        words = number_words(rubles)
        return f"{money_digits(amount, kopecks=False)} ({words}) {rub_word} {cents:02d} {kop_word}"
    if style == 2 and cents == 0:
        return (
            f"{money_digits(amount, kopecks=False)} {plural(rubles, ('рубль', 'рубля', 'рублей'))}"
        )
    return f"{money_digits(amount)} руб."


def money_answer(amount: Decimal) -> str:
    """Сумма для эталонного ответа: «85 000,00 руб.»."""
    return f"{money_digits(amount)} руб."


# --- Даты --------------------------------------------------------------------------------


def date_text(d: date, style: int) -> str:
    """0 — «12.03.2025»; 1 — «12» марта 2025 г.; 2 — «12 марта 2025 года»."""
    if style == 0:
        return d.strftime("%d.%m.%Y")
    if style == 1:
        return f"«{d.day:02d}» {MONTHS_GEN[d.month - 1]} {d.year} г."
    return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year} года"


def date_short(d: date) -> str:
    """Дата для вопросов и ответов: «12.03.2025»."""
    return d.strftime("%d.%m.%Y")
