"""Преобразование «форма разметки ↔ документ нашей схемы».

Форма — это набор строк (то, что человек напечатал), а документ — модель Pydantic.
Здесь нет Gradio, поэтому всё проверяется обычными тестами. Поля берутся из FIELD_SPEC,
так что форма и метрики извлечения всегда говорят об одном и том же наборе полей.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from docai.schemas import FIELD_SPEC, DocType, FieldSpec, NormKind, VatRate, get_field_value
from docai.schemas.documents import AnyDocument, parse_document
from docai.schemas.fields import LINE_PREFIX
from docai.validation import ValidationIssue, validate_document
from docai.validation.normalize import (
    normalize_digits,
    normalize_inn,
    normalize_kpp,
    normalize_okei,
    parse_date,
    parse_decimal,
    parse_money,
    parse_vat_rate,
)

# Поля шапки и итогов (без типа документа: он выбирается отдельно) и колонки таблицы.
HEADER_SPECS: tuple[FieldSpec, ...] = tuple(
    spec for spec in FIELD_SPEC if not spec.is_line_field and spec.name != "doc_type"
)
LINE_SPECS: tuple[FieldSpec, ...] = tuple(spec for spec in FIELD_SPEC if spec.is_line_field)
LINE_COLUMNS: tuple[str, ...] = tuple(spec.name.removeprefix(LINE_PREFIX) for spec in LINE_SPECS)
LINE_HEADERS: tuple[str, ...] = (
    "Наименование",
    "Ед. изм.",
    "Код ОКЕИ",
    "Кол-во",
    "Цена",
    "Сумма без НДС",
    "Ставка НДС",
    "Сумма НДС",
    "Сумма с НДС",
)
DOC_DATE_FORMAT = "%d.%m.%Y"


class FormError(ValueError):
    """В форме есть значения, из которых нельзя собрать документ (список сообщений)."""

    def __init__(self, messages: Sequence[str]) -> None:
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


def empty_lines(count: int = 1) -> list[list[str]]:
    """Пустая таблица строк нужной ширины."""
    return [["" for _ in LINE_COLUMNS] for _ in range(count)]


def _clean(value: Any) -> str:
    """Строка из ячейки формы: None и NaN из таблицы Gradio становятся пустой строкой."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return " ".join(text.split())


def _convert(spec: FieldSpec, text: str) -> str | int:
    """Напечатанное → каноническое значение для схемы. Бросает ValueError с понятным текстом."""
    match spec.norm:
        case NormKind.DATE:
            return parse_date(text).isoformat()
        case NormKind.MONEY:
            amount = parse_money(text)  # понимает и «12 200 руб. 50 коп.»
            if amount.as_tuple().exponent > -2:  # type: ignore[operator]
                amount = amount.quantize(
                    Decimal("0.01")
                )  # «100» → «100.00»; цену с 4 знаками не трогаем
            return format(amount, "f")
        case NormKind.QUANTITY:
            return format(parse_decimal(text), "f")
        case NormKind.INN:
            return normalize_inn(text)
        case NormKind.KPP:
            return normalize_kpp(text)
        case NormKind.DIGITS:
            return normalize_digits(text)
        case NormKind.OKEI:
            return normalize_okei(text)
        case NormKind.VAT_RATE:
            return parse_vat_rate(text).value
        case NormKind.EXACT if spec.name == "upd_status":
            if text not in {"1", "2"}:
                raise ValueError("статус УПД — 1 или 2")
            return int(text)
        case _:
            return text


def _short(spec: FieldSpec) -> str:
    return spec.title_ru.removeprefix("Строка: ")


def _put(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for part in path[:-1]:
        target = target.setdefault(part, {})
    target[path[-1]] = value


def build_document(
    doc_type: str,
    header: Mapping[str, Any],
    lines: Sequence[Sequence[Any]],
) -> AnyDocument:
    """Собрать документ из значений формы.

    `header` — словарь «имя поля из FIELD_SPEC → текст»; `lines` — строки таблицы в порядке
    `LINE_COLUMNS`. Пустые необязательные блоки (основание, реквизиты банка) отбрасываются,
    полностью пустые строки таблицы пропускаются. Бросает `FormError`.
    """
    try:
        kind = DocType(doc_type)
    except ValueError:
        raise FormError(["Не выбран тип документа"]) from None

    errors: list[str] = []
    data: dict[str, Any] = {"doc_type": kind.value}
    for spec in HEADER_SPECS:
        if not spec.applies_to(kind):
            continue
        text = _clean(header.get(spec.name))
        value: str | int | None = None
        if text:
            try:
                value = _convert(spec, text)
            except ValueError as exc:
                errors.append(f"«{spec.title_ru}»: {exc}")
        _put(data, spec.path, value)

    for block in ("basis", "seller_bank"):
        keep = block == "seller_bank" and kind is DocType.BILL  # в счёте блок обязателен
        if block in data and not keep and all(v is None for v in data[block].values()):
            del data[block]

    rows: list[dict[str, Any]] = []
    for number, row in enumerate(lines, start=1):
        cells = [_clean(cell) for cell in row]
        cells += [""] * (len(LINE_SPECS) - len(cells))
        if not any(cells):
            continue
        line: dict[str, Any] = {}
        for spec, text in zip(LINE_SPECS, cells, strict=False):
            line_value: str | int | None = None
            if text:
                try:
                    line_value = _convert(spec, text)
                except ValueError as exc:
                    errors.append(f"Строка {number}, «{_short(spec)}»: {exc}")
            _put(line, spec.path, line_value)
        rows.append(line)
    data["lines"] = rows

    if errors:
        raise FormError(errors)
    try:
        return parse_document(data)
    except ValidationError as exc:
        raise FormError(_describe_schema_errors(exc, kind)) from exc


def _describe_schema_errors(exc: ValidationError, kind: DocType) -> list[str]:
    """Ошибки Pydantic по-русски: «Не заполнено поле «Номер документа»»."""
    titles = {spec.name: spec.title_ru for spec in FIELD_SPEC}
    messages: list[str] = []
    for item in exc.errors():
        loc = [str(part) for part in item["loc"] if str(part) != kind.value]
        row = ""
        if len(loc) >= 3 and loc[0] == "lines" and loc[1].isdigit():
            row = f"Строка {int(loc[1]) + 1}, "
            name = f"{LINE_PREFIX}{loc[2]}"
        else:
            name = ".".join(loc)
        empty = item["type"] in {"missing", "string_type"} or item.get("input") is None
        reason = "не заполнено" if empty else item["msg"]
        title = titles.get(name, name or "документ").removeprefix("Строка: ")
        messages.append(f"{row}«{title}»: {reason}")
    return messages


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.date):
        return value.strftime(DOC_DATE_FORMAT)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, VatRate):
        return value.value
    return str(value)


def document_to_form(doc: AnyDocument) -> tuple[str, dict[str, str], list[list[str]]]:
    """Документ → (тип, значения шапки по именам FIELD_SPEC, строки таблицы)."""
    header = {
        spec.name: _format_value(get_field_value(doc, spec))
        for spec in HEADER_SPECS
        if spec.applies_to(doc.doc_type)
    }
    lines = [
        [_format_value(getattr(line, column)) for column in LINE_COLUMNS] for line in doc.lines
    ]
    return doc.doc_type.value, header, lines or empty_lines()


def check_document(doc: AnyDocument) -> list[ValidationIssue]:
    """Проверка валидаторами проекта (ИНН, счета, НДС, арифметика)."""
    return validate_document(doc)


def format_issues(issues: Sequence[ValidationIssue]) -> str:
    """Список проверок как markdown для показа рядом с формой."""
    if not issues:
        return "✅ Все проверки пройдены."
    out = []
    for item in issues:
        mark = "❌" if item.severity.value == "error" else "⚠️"
        where = f" `{item.field}`" if item.field else ""
        out.append(f"- {mark}{where} {item.message}")
    return "\n".join(out)
