"""Плоский список извлекаемых полей `FIELD_SPEC` — основа метрик извлечения.

Имя поля — путь в схеме документа: `seller.inn`, `totals.vat_amount`. Поля строк
табличной части начинаются с `lines[].`: `lines[].quantity`.

Как сравнивать значения, задаёт `norm` (вид нормализации); сами нормализаторы
лежат в `docai.validation.normalize` (функция `normalize_for_compare`).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from docai.schemas.common import BaseDocument, DocType


class ValueType(StrEnum):
    """Тип значения поля в схеме."""

    STR = "str"
    INT = "int"
    DATE = "date"
    DECIMAL = "decimal"
    ENUM = "enum"


class NormKind(StrEnum):
    """Как привести значение к каноническому виду перед сравнением."""

    TEXT = "text"  # регистр, ё→е, кавычки, лишние пробелы
    DOC_NUMBER = "doc_number"  # как TEXT, плюс без «№» и пробелов
    DATE = "date"  # «14.11.2026», «14 ноября 2026 г.» → date
    MONEY = "money"  # «12 200,00», «12 200 руб. 00 коп.» → Decimal с 2 знаками
    QUANTITY = "quantity"  # «1 000,500» → Decimal без хвостовых нулей
    INN = "inn"  # только цифры
    KPP = "kpp"  # без пробелов, латиница в верхнем регистре
    DIGITS = "digits"  # БИК, счета: только цифры
    OKEI = "okei"  # код ОКЕИ из 3 цифр с ведущими нулями
    VAT_RATE = "vat_rate"  # «НДС 22 %», «Без налога (НДС)» → VatRate
    EXACT = "exact"  # строка как есть, без пробелов по краям (тип документа, статус УПД)


ALL_TYPES = frozenset(DocType)
_TAX = frozenset({DocType.UPD, DocType.VAT_INVOICE})  # документы с ролью счёта-фактуры
_GOODS = frozenset({DocType.UPD, DocType.VAT_INVOICE, DocType.TORG12})  # колонки без/с НДС
_WITH_BANK = frozenset({DocType.BILL, DocType.ACT, DocType.TORG12})
_WITH_BASIS = frozenset({DocType.UPD, DocType.BILL, DocType.ACT, DocType.TORG12})
_BILL = frozenset({DocType.BILL})

LINE_PREFIX = "lines[]."


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Описание одного извлекаемого поля."""

    name: str
    value_type: ValueType
    norm: NormKind
    doc_types: frozenset[DocType]
    required_in: frozenset[DocType] = field(default_factory=frozenset)
    title_ru: str = ""

    def __post_init__(self) -> None:
        if not self.required_in <= self.doc_types:
            raise ValueError(f"{self.name}: required_in должен входить в doc_types")

    @property
    def is_line_field(self) -> bool:
        return self.name.startswith(LINE_PREFIX)

    @property
    def path(self) -> tuple[str, ...]:
        """Путь внутри документа (для строк — внутри одной строки)."""
        return tuple(self.name.removeprefix(LINE_PREFIX).split("."))

    def applies_to(self, doc_type: DocType) -> bool:
        return doc_type in self.doc_types

    def is_required(self, doc_type: DocType) -> bool:
        return doc_type in self.required_in


def _f(
    name: str,
    value_type: ValueType,
    norm: NormKind,
    title_ru: str,
    doc_types: frozenset[DocType] = ALL_TYPES,
    required_in: frozenset[DocType] = frozenset(),
) -> FieldSpec:
    return FieldSpec(name, value_type, norm, doc_types, required_in, title_ru)


_S, _I, _D, _N, _E = ValueType.STR, ValueType.INT, ValueType.DATE, ValueType.DECIMAL, ValueType.ENUM

FIELD_SPEC: tuple[FieldSpec, ...] = (
    # Шапка
    _f("doc_type", _E, NormKind.EXACT, "Тип документа", required_in=ALL_TYPES),
    _f("number", _S, NormKind.DOC_NUMBER, "Номер документа", required_in=ALL_TYPES),
    _f("date", _D, NormKind.DATE, "Дата документа", required_in=ALL_TYPES),
    _f("upd_status", _I, NormKind.EXACT, "Статус УПД", frozenset({DocType.UPD}),
       frozenset({DocType.UPD})),
    # Продавец
    _f("seller.name", _S, NormKind.TEXT, "Продавец: наименование", required_in=ALL_TYPES),
    _f("seller.inn", _S, NormKind.INN, "Продавец: ИНН", required_in=ALL_TYPES),
    _f("seller.kpp", _S, NormKind.KPP, "Продавец: КПП"),
    _f("seller.address", _S, NormKind.TEXT, "Продавец: адрес", required_in=_TAX),
    # Покупатель
    _f("buyer.name", _S, NormKind.TEXT, "Покупатель: наименование", required_in=ALL_TYPES),
    _f("buyer.inn", _S, NormKind.INN, "Покупатель: ИНН", required_in=_TAX),
    _f("buyer.kpp", _S, NormKind.KPP, "Покупатель: КПП"),
    _f("buyer.address", _S, NormKind.TEXT, "Покупатель: адрес", required_in=_TAX),
    # Банковские реквизиты продавца
    _f("seller_bank.bank_name", _S, NormKind.TEXT, "Банк продавца", _WITH_BANK, _BILL),
    _f("seller_bank.bik", _S, NormKind.DIGITS, "БИК", _WITH_BANK, _BILL),
    _f("seller_bank.corr_account", _S, NormKind.DIGITS, "Корр. счёт", _WITH_BANK, _BILL),
    _f("seller_bank.account", _S, NormKind.DIGITS, "Расчётный счёт", _WITH_BANK, _BILL),
    # Основание
    _f("basis.name", _S, NormKind.TEXT, "Основание: вид документа", _WITH_BASIS),
    _f("basis.number", _S, NormKind.DOC_NUMBER, "Основание: номер", _WITH_BASIS),
    _f("basis.date", _D, NormKind.DATE, "Основание: дата", _WITH_BASIS),
    # Строки
    _f("lines[].name", _S, NormKind.TEXT, "Строка: наименование", required_in=ALL_TYPES),
    _f("lines[].unit_name", _S, NormKind.TEXT, "Строка: единица измерения"),
    _f("lines[].unit_code", _S, NormKind.OKEI, "Строка: код ОКЕИ"),
    _f("lines[].quantity", _N, NormKind.QUANTITY, "Строка: количество", required_in=ALL_TYPES),
    _f("lines[].price", _N, NormKind.MONEY, "Строка: цена", required_in=ALL_TYPES),
    _f("lines[].amount_without_vat", _N, NormKind.MONEY, "Строка: сумма без НДС",
       required_in=_GOODS),
    _f("lines[].vat_rate", _E, NormKind.VAT_RATE, "Строка: ставка НДС", required_in=_GOODS),
    _f("lines[].vat_amount", _N, NormKind.MONEY, "Строка: сумма НДС"),
    _f("lines[].amount_with_vat", _N, NormKind.MONEY, "Строка: сумма с НДС", required_in=_GOODS),
    # Итоги
    _f("totals.amount_without_vat", _N, NormKind.MONEY, "Итого без НДС", required_in=_GOODS),
    _f("totals.vat_amount", _N, NormKind.MONEY, "Итого НДС"),
    _f("totals.amount_with_vat", _N, NormKind.MONEY, "Всего с НДС", required_in=ALL_TYPES),
)  # fmt: skip

FIELDS_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPEC}


def fields_for(doc_type: DocType) -> tuple[FieldSpec, ...]:
    """Поля, которые бывают у документа данного типа."""
    return tuple(spec for spec in FIELD_SPEC if spec.applies_to(doc_type))


def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def get_field_value(doc: BaseDocument, spec: FieldSpec) -> Any:
    """Значение поля в документе.

    Для поля шапки — одно значение (или None), для поля строки — список значений
    по строкам в их порядке. Для поля, которого у типа документа нет, — None.
    """
    if not spec.applies_to(doc.doc_type):
        return None
    if spec.is_line_field:
        return [_get_path(line, spec.path) for line in doc.lines]
    return _get_path(doc, spec.path)
