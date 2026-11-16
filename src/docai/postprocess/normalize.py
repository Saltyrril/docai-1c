"""Нормализация значений после LLM (правила P2): ИНН, КПП, БИК, счета, ОКЕИ, пробелы.

Меняется только запись значения, не его смысл. Каждое изменение попадает в журнал.
"""

import re
from collections.abc import Callable

from docai.postprocess.models import Correction
from docai.schemas.common import BaseDocument
from docai.validation.normalize import (
    normalize_digits,
    normalize_inn,
    normalize_kpp,
    normalize_okei,
    split_inn_kpp,
)

_SPACES_RE = re.compile(r"\s+")
_NUMBER_PREFIX_RE = re.compile(r"^\s*(?:№|N|No)\s*", re.IGNORECASE)

type Rule = tuple[str, Callable[[str], str], str]


def clean_text(value: str) -> str:
    """Один пробел между словами, неразрывные пробелы — обычные."""
    return _SPACES_RE.sub(" ", value).strip()


def clean_number(value: str) -> str:
    """Номер документа: без подписи «№» в начале."""
    return clean_text(_NUMBER_PREFIX_RE.sub("", value))


# (атрибут, функция, название правила)
_PARTY_RULES: tuple[Rule, ...] = (
    ("inn", normalize_inn, "inn_digits"),
    ("kpp", normalize_kpp, "kpp_chars"),
    ("name", clean_text, "text_spaces"),
    ("address", clean_text, "text_spaces"),
)
_BANK_RULES: tuple[Rule, ...] = (
    ("bik", normalize_digits, "digits_only"),
    ("account", normalize_digits, "digits_only"),
    ("corr_account", normalize_digits, "digits_only"),
    ("bank_name", clean_text, "text_spaces"),
)
_LINE_RULES: tuple[Rule, ...] = (
    ("unit_code", normalize_okei, "okei_3_digits"),
    ("unit_name", clean_text, "text_spaces"),
    ("name", clean_text, "text_spaces"),
)
_BASIS_RULES: tuple[Rule, ...] = (
    ("name", clean_text, "text_spaces"),
    ("number", clean_number, "doc_number"),
)
_KEEP_EMPTY = frozenset({"name"})  # обязательные строки остаются пустыми, а не None


def _apply(obj: object, rules: tuple[Rule, ...], prefix: str, out: list[Correction]) -> None:
    for attr, func, rule in rules:
        value = getattr(obj, attr, None)
        if not isinstance(value, str):
            continue
        new: str | None = func(value)
        if new == "" and attr not in _KEEP_EMPTY:
            new = None
        if new != value:
            setattr(obj, attr, new)
            out.append(
                Correction(
                    kind="normalize",
                    field=f"{prefix}{attr}",
                    before=value,
                    after=new,
                    rule=rule,
                    reason="запись значения приведена к каноническому виду",
                )
            )


def _split_inn_kpp(party: object, prefix: str, out: list[Correction]) -> None:
    """«ИНН/КПП» в поле ИНН → ИНН и КПП раздельно (КПП не затирается, если уже есть)."""
    inn = getattr(party, "inn", None)
    if not isinstance(inn, str) or "/" not in inn:
        return
    new_inn, new_kpp = split_inn_kpp(inn)
    party.inn = new_inn  # type: ignore[attr-defined]
    reason = "в поле ИНН записаны ИНН и КПП через «/»"
    out.append(
        Correction(
            kind="normalize",
            field=f"{prefix}inn",
            before=inn,
            after=new_inn,
            rule="split_inn_kpp",
            reason=reason,
        )
    )
    if new_kpp and not getattr(party, "kpp", None):
        party.kpp = new_kpp  # type: ignore[attr-defined]
        out.append(
            Correction(
                kind="normalize",
                field=f"{prefix}kpp",
                before=None,
                after=new_kpp,
                rule="split_inn_kpp",
                reason=reason,
            )
        )


def normalize_document[D: BaseDocument](doc: D) -> tuple[D, list[Correction]]:
    """Копия документа с нормализованными строковыми полями и журнал изменений."""
    result = doc.model_copy(deep=True)
    out: list[Correction] = []
    old_number = result.number
    result.number = clean_number(old_number)
    if result.number != old_number:
        out.append(
            Correction(
                kind="normalize",
                field="number",
                before=old_number,
                after=result.number,
                rule="doc_number",
                reason="запись значения приведена к каноническому виду",
            )
        )
    _split_inn_kpp(result.seller, "seller.", out)
    _split_inn_kpp(result.buyer, "buyer.", out)
    _apply(result.seller, _PARTY_RULES, "seller.", out)
    _apply(result.buyer, _PARTY_RULES, "buyer.", out)
    bank = getattr(result, "seller_bank", None)
    if bank is not None:
        _apply(bank, _BANK_RULES, "seller_bank.", out)
    basis = getattr(result, "basis", None)
    if basis is not None:
        _apply(basis, _BASIS_RULES, "basis.", out)
    for index, line in enumerate(result.lines):
        _apply(line, _LINE_RULES, f"lines[{index}].", out)
    return result, out
