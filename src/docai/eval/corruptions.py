"""Намеренные порчи документа: так ошибается чтение (LLM, OCR), а не человек.

Нужны, чтобы измерить, какую долю ошибок ловят проверки (P10). Каждая порча меняет ОДНО
место в верном документе и возвращает True, если применима (например, «порча БИК» не
применима к документу без банковских реквизитов).

Две группы:
- `rule` — ошибки, которые правила обязаны ловить: арифметика, контрольные цифры,
  ставка НДС на дату, форма КПП;
- `outside` — ошибки, которые правила поймать не могут по смыслу (опечатка в наименовании,
  номер документа, КПП той же формы): остаётся уверенность по тексту и другому пути.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Literal

from docai.schemas.common import VatRate

Group = Literal["rule", "outside"]
Data = dict[str, Any]
Apply = Callable[[random.Random, Data], bool]

_RATES = (VatRate.R0, VatRate.R10, VatRate.R20, VatRate.R22)
_MAIN = (VatRate.R20, VatRate.R22)


@dataclass(frozen=True, slots=True)
class Corruption:
    name: str
    group: Group
    title_ru: str
    apply: Apply


# --- Операции над числами и строками -----------------------------------------------------


def change_digit(rng: random.Random, text: str) -> str:
    """Заменить одну цифру другой (первую цифру не обнуляем)."""
    spots = [i for i, ch in enumerate(text) if ch.isdigit()]
    i = rng.choice(spots)
    options = [d for d in "0123456789" if d != text[i] and not (i == spots[0] and d == "0")]
    return text[:i] + rng.choice(options) + text[i + 1 :]


def drop_digit(rng: random.Random, text: str) -> str:
    spots = [i for i, ch in enumerate(text) if ch.isdigit()]
    i = rng.choice(spots)
    return text[:i] + text[i + 1 :]


def swap_adjacent_digits(rng: random.Random, text: str) -> str | None:
    pairs = [
        i
        for i in range(len(text) - 1)
        if text[i].isdigit() and text[i + 1].isdigit() and text[i] != text[i + 1]
    ]
    if not pairs:
        return None
    i = rng.choice(pairs)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mutate_decimal(rng: random.Random, value: Decimal, op: Callable[[str], str | None]) -> Decimal:
    new = op(_decimal_text(abs(value)))
    if new is None or not new.replace(".", "").isdigit():
        return value
    return Decimal(new).copy_sign(value)


def _alter_decimal(data: Data, path: tuple[Any, ...], func: Callable[[Decimal], Decimal]) -> bool:
    node = data
    for key in path[:-1]:
        node = node[key]
    value = node[path[-1]]
    if value is None or value == 0:
        return False
    new = func(value)
    if new == value:
        return False
    node[path[-1]] = new
    return True


def _pick_line(rng: random.Random, data: Data, key: str) -> int | None:
    indexes = [i for i, ln in enumerate(data["lines"]) if ln.get(key) not in (None, 0)]
    return rng.choice(indexes) if indexes else None


# --- Порчи правил ------------------------------------------------------------------------


def _line_digit(key: str) -> Apply:
    def apply(rng: random.Random, data: Data) -> bool:
        i = _pick_line(rng, data, key)
        if i is None:
            return False
        return _alter_decimal(
            data,
            ("lines", i, key),
            lambda v: _mutate_decimal(rng, v, lambda t: change_digit(rng, t)),
        )

    return apply


def _line_drop_digit(rng: random.Random, data: Data) -> bool:
    candidates = [
        i
        for i, ln in enumerate(data["lines"])
        if ln["amount_without_vat"] is not None and ln["amount_without_vat"] >= 100
    ]
    if not candidates:
        return False
    i = rng.choice(candidates)
    return _alter_decimal(
        data,
        ("lines", i, "amount_without_vat"),
        lambda v: _mutate_decimal(rng, v, lambda t: drop_digit(rng, t)),
    )


def _line_shift(rng: random.Random, data: Data) -> bool:
    i = _pick_line(rng, data, "amount_without_vat")
    if i is None:
        return False
    factor = rng.choice((Decimal(10), Decimal("0.1")))
    return _alter_decimal(
        data, ("lines", i, "amount_without_vat"), lambda v: (v * factor).quantize(Decimal("0.01"))
    )


def _total_digit(key: str) -> Apply:
    def apply(rng: random.Random, data: Data) -> bool:
        return _alter_decimal(
            data,
            ("totals", key),
            lambda v: _mutate_decimal(rng, v, lambda t: change_digit(rng, t)),
        )

    return apply


def _total_transpose(rng: random.Random, data: Data) -> bool:
    return _alter_decimal(
        data,
        ("totals", "amount_with_vat"),
        lambda v: _mutate_decimal(rng, v, lambda t: swap_adjacent_digits(rng, t)),
    )


def _quantity_wrong(rng: random.Random, data: Data) -> bool:
    i = _pick_line(rng, data, "quantity")
    if i is None:
        return False
    return _alter_decimal(
        data, ("lines", i, "quantity"), lambda v: v + rng.choice((Decimal(1), Decimal(2)))
    )


def _vat_rate_wrong(rng: random.Random, data: Data) -> bool:
    indexes = [i for i, ln in enumerate(data["lines"]) if ln["vat_rate"] in _RATES]
    if not indexes:
        return False
    i = rng.choice(indexes)
    current = data["lines"][i]["vat_rate"]
    data["lines"][i]["vat_rate"] = rng.choice([r for r in _RATES if r != current])
    return True


def _date_across_regime(rng: random.Random, data: Data) -> bool:
    """Год даты ушёл за границу 01.01.2026 при ставке 20% или 22%."""
    if not any(ln["vat_rate"] in _MAIN for ln in data["lines"]):
        return False
    current: date = data["date"]
    data["date"] = (
        date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
        if current.year < 2026
        else date(2025, 1, 1) + timedelta(days=rng.randint(0, 300))
    )
    return True


def _inn_digit(rng: random.Random, data: Data) -> bool:
    party = data[rng.choice(("seller", "buyer"))]
    if not party["inn"]:
        return False
    party["inn"] = change_digit(rng, party["inn"])
    return True


def _bank_digit(key: str) -> Apply:
    def apply(rng: random.Random, data: Data) -> bool:
        bank = data.get("seller_bank")
        if not bank or not bank.get(key):
            return False
        bank[key] = change_digit(rng, bank[key])
        return True

    return apply


def _kpp_drop(rng: random.Random, data: Data) -> bool:
    parties = [p for p in (data["seller"], data["buyer"]) if p["kpp"]]
    if not parties:
        return False
    party = rng.choice(parties)
    party["kpp"] = drop_digit(rng, party["kpp"])
    return True


def _line_dropped(rng: random.Random, data: Data) -> bool:
    if len(data["lines"]) < 2:
        return False
    del data["lines"][rng.randrange(len(data["lines"]))]
    return True


def _line_duplicated(rng: random.Random, data: Data) -> bool:
    i = rng.randrange(len(data["lines"]))
    data["lines"].insert(i, dict(data["lines"][i]))
    return True


# --- Порчи вне правил --------------------------------------------------------------------


def _typo(rng: random.Random, text: str) -> str:
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    i = rng.choice(letters)
    pool = "аеиорстнлвк"
    return text[:i] + rng.choice([c for c in pool if c != text[i].lower()]) + text[i + 1 :]


def _text_typo(path: tuple[Any, ...]) -> Apply:
    def apply(rng: random.Random, data: Data) -> bool:
        node: Any = data
        for key in path[:-1]:
            node = node[key]
        text = node[path[-1]]
        if not text or not any(ch.isalpha() for ch in text):
            return False
        node[path[-1]] = _typo(rng, text)
        return True

    return apply


def _line_name_typo(rng: random.Random, data: Data) -> bool:
    return _text_typo(("lines", rng.randrange(len(data["lines"])), "name"))(rng, data)


def _number_digit(rng: random.Random, data: Data) -> bool:
    if not any(ch.isdigit() for ch in data["number"]):
        return False
    data["number"] = change_digit(rng, data["number"])
    return True


def _kpp_digit(rng: random.Random, data: Data) -> bool:
    """Цифра КПП заменена на другую цифру: форма верна, контрольной суммы у КПП нет."""
    parties = [p for p in (data["seller"], data["buyer"]) if p["kpp"]]
    if not parties:
        return False
    party = rng.choice(parties)
    kpp = party["kpp"]
    digits = [i for i, ch in enumerate(kpp) if ch.isdigit() and i < 4]
    i = rng.choice(digits)
    party["kpp"] = kpp[:i] + rng.choice([d for d in "0123456789" if d != kpp[i]]) + kpp[i + 1 :]
    return True


def _date_same_regime(rng: random.Random, data: Data) -> bool:
    """Дата сдвинута внутри того же года: ставки НДС прежние, проверка молчит."""
    current: date = data["date"]
    boundary = date(2026, 1, 1)
    for _ in range(20):
        new = current + timedelta(days=rng.randint(-60, 60))
        if new != current and (new >= boundary) == (current >= boundary):
            data["date"] = new
            return True
    return False


def _basis_number_digit(rng: random.Random, data: Data) -> bool:
    basis = data.get("basis")
    if not basis or not basis.get("number") or not any(c.isdigit() for c in basis["number"]):
        return False
    basis["number"] = change_digit(rng, basis["number"])
    return True


CORRUPTIONS: tuple[Corruption, ...] = (
    Corruption(
        "line_amount_digit", "rule", "цифра в сумме строки", _line_digit("amount_without_vat")
    ),
    Corruption(
        "line_with_vat_digit", "rule", "цифра в сумме строки с НДС", _line_digit("amount_with_vat")
    ),
    Corruption("line_vat_digit", "rule", "цифра в НДС строки", _line_digit("vat_amount")),
    Corruption("line_price_digit", "rule", "цифра в цене", _line_digit("price")),
    Corruption("line_amount_drop_digit", "rule", "потеряна цифра в сумме строки", _line_drop_digit),
    Corruption("line_amount_shift", "rule", "запятая сдвинута в сумме строки", _line_shift),
    Corruption("line_quantity_wrong", "rule", "неверное количество", _quantity_wrong),
    Corruption(
        "total_without_digit", "rule", "цифра в итоге без НДС", _total_digit("amount_without_vat")
    ),
    Corruption("total_vat_digit", "rule", "цифра в итоге НДС", _total_digit("vat_amount")),
    Corruption("total_with_digit", "rule", "цифра в итоге с НДС", _total_digit("amount_with_vat")),
    Corruption("total_with_transpose", "rule", "две цифры итога переставлены", _total_transpose),
    Corruption("vat_rate_wrong", "rule", "неверная ставка НДС в строке", _vat_rate_wrong),
    Corruption("date_across_regime", "rule", "дата за границей смены ставки", _date_across_regime),
    Corruption("inn_digit", "rule", "цифра в ИНН", _inn_digit),
    Corruption("bik_digit", "rule", "цифра в БИК", _bank_digit("bik")),
    Corruption("account_digit", "rule", "цифра в расчётном счёте", _bank_digit("account")),
    Corruption("corr_account_digit", "rule", "цифра в корр. счёте", _bank_digit("corr_account")),
    Corruption("kpp_dropped_char", "rule", "потерян знак КПП", _kpp_drop),
    Corruption("line_dropped", "rule", "пропущена строка таблицы", _line_dropped),
    Corruption("line_duplicated", "rule", "строка таблицы прочитана дважды", _line_duplicated),
    Corruption(
        "seller_name_typo",
        "outside",
        "опечатка в наименовании продавца",
        _text_typo(("seller", "name")),
    ),
    Corruption(
        "buyer_address_typo",
        "outside",
        "опечатка в адресе покупателя",
        _text_typo(("buyer", "address")),
    ),
    Corruption("line_name_typo", "outside", "опечатка в наименовании товара", _line_name_typo),
    Corruption("number_digit", "outside", "цифра в номере документа", _number_digit),
    Corruption("kpp_digit", "outside", "цифра в КПП (форма верна)", _kpp_digit),
    Corruption("date_same_regime", "outside", "дата сдвинута в пределах года", _date_same_regime),
    Corruption("basis_number_digit", "outside", "цифра в номере договора", _basis_number_digit),
)
CORRUPTIONS_BY_NAME = {c.name: c for c in CORRUPTIONS}
