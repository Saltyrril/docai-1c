"""Черновик документа: то, что заполняет LLM, и его перевод в нашу схему `Document`.

Почему не сама схема `Document`:
- в ней есть дискриминированное объединение, Decimal с регулярным выражением и поля,
  которые бывают не у всех типов; грамматике Ollama проще одна плоская схема;
- в черновике все ключи обязательны, но допускают null: модель выписывает каждый ключ,
  а «не нашла» выглядит как null, а не как пропуск.

Черновик переводится в `Document`: лишние для типа блоки отбрасываются, обязательные,
но не найденные значения заменяются пустыми с предупреждением.
"""

import copy
import datetime as dt
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, WithJsonSchema

from docai.schemas.common import DocType, VatRate
from docai.schemas.documents import AnyDocument, parse_document
from docai.validation.normalize import parse_date

# Число в схеме для модели — JSON number; в Python — Decimal (ответ читается с parse_float=Decimal).
Num = Annotated[Decimal, WithJsonSchema({"type": "number"})]
Date = Annotated[dt.date, WithJsonSchema({"type": "string", "format": "date"})]

# Как модель записывает даты (D47):
# iso — грамматика Ollama требует ГГГГ-ММ-ДД (`format: date`); так было в P9–P11;
# printed — дата свободной строкой, как напечатана («05.02.2026», «5 февраля 2026 г.»),
#   в дату её переводит `parse_date`. Модель 2B под грамматикой ГГГГ-ММ-ДД переписывает
#   цифры ДД.ММ.ГГГГ подряд («05.02.2026» → «0502-02-20»), см. reports/2026-11-17_p11_baseline.
DateFormat = Literal["iso", "printed"]
DATE_PATHS: tuple[tuple[str, ...], ...] = (("date",), ("basis", "date"))


class _Draft(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class DraftParty(_Draft):
    name: str | None
    inn: str | None
    kpp: str | None
    address: str | None


class DraftBank(_Draft):
    bank_name: str | None
    bik: str | None
    corr_account: str | None
    account: str | None


class DraftBasis(_Draft):
    name: str | None
    number: str | None
    date: Date | None


class DraftLine(_Draft):
    name: str | None
    unit_name: str | None
    unit_code: str | None
    quantity: Num | None
    price: Num | None
    amount_without_vat: Num | None
    vat_rate: VatRate | None
    vat_amount: Num | None
    amount_with_vat: Num | None


class DraftTotals(_Draft):
    amount_without_vat: Num | None
    vat_amount: Num | None
    amount_with_vat: Num | None


class Draft(_Draft):
    """Плоская схема для LLM: все блоки всех пяти типов документов."""

    doc_type: DocType
    upd_status: Literal[1, 2] | None
    number: str | None
    date: Date | None
    seller: DraftParty
    buyer: DraftParty
    seller_bank: DraftBank | None
    basis: DraftBasis | None
    lines: list[DraftLine]
    totals: DraftTotals


def _strip_notes(node: Any) -> Any:
    """Убрать `title` и `description`: грамматике они не нужны, а запрос короче."""
    if isinstance(node, dict):
        return {k: _strip_notes(v) for k, v in node.items() if k not in ("title", "description")}
    if isinstance(node, list):
        return [_strip_notes(v) for v in node]
    return node


def _drop_date_format(node: Any) -> Any:
    """Убрать `format: date`: дата остаётся строкой без шаблона."""
    if isinstance(node, dict):
        return {
            k: _drop_date_format(v) for k, v in node.items() if not (k == "format" and v == "date")
        }
    if isinstance(node, list):
        return [_drop_date_format(v) for v in node]
    return node


def draft_json_schema(date_format: DateFormat = "iso") -> dict[str, Any]:
    """JSON Schema черновика для параметра `format` Ollama."""
    schema = _strip_notes(Draft.model_json_schema())
    return _drop_date_format(schema) if date_format == "printed" else schema


def _printed_dates(raw: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Даты, записанные как в документе, → ГГГГ-ММ-ДД; неразборчивые → null с предупреждением."""
    data = copy.deepcopy(raw)
    for path in DATE_PATHS:
        node = data
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            continue
        value = node.get(path[-1])
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            node[path[-1]] = parse_date(value).isoformat()
        except ValueError as exc:
            node[path[-1]] = None
            warnings.append(f"{'.'.join(path)}: {exc} → null")
    return data


def _set_none(data: Any, loc: tuple[int | str, ...]) -> bool:
    """Заменить значение по пути на None. False, если пути нет."""
    node = data
    for key in loc[:-1]:
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return False
    try:
        if node[loc[-1]] is None:
            return False
        node[loc[-1]] = None
    except (KeyError, IndexError, TypeError):
        return False
    return True


def parse_draft(raw: dict[str, Any], date_format: DateFormat = "iso") -> tuple[Draft, list[str]]:
    """Черновик из ответа модели. Невалидные значения (например, 30 февраля) → null.

    Грамматика Ollama гарантирует форму JSON, но не смысл: дата «2025-02-30» ей
    подходит. Такие значения обнуляются с предупреждением, остальное сохраняется.
    При `date_format="printed"` даты сначала разбираются из вида, как в документе.
    """
    warnings: list[str] = []
    if date_format == "printed":
        raw = _printed_dates(raw, warnings)
    data = raw
    for _ in range(50):
        try:
            return Draft.model_validate(data), warnings
        except ValidationError as exc:
            if data is raw:
                data = copy.deepcopy(raw)
            changed = False
            for error in exc.errors():
                loc = tuple(error["loc"])
                # Pydantic добавляет к пути имя варианта объединения (например, «date»); отрезаем.
                while loc and not _path_exists(data, loc):
                    loc = loc[:-1]
                if loc and _set_none(data, loc):
                    warnings.append(f"{'.'.join(map(str, loc))}: {error['msg']} → null")
                    changed = True
            if not changed:
                raise
    raise ValueError("не удалось привести ответ модели к черновику")


def _path_exists(data: Any, loc: tuple[int | str, ...]) -> bool:
    node = data
    for key in loc:
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return False
    return True


def _first(values: list[Any]) -> Any:
    return next((v for v in values if v is not None), None)


def _merge_objects(objects: list[_Draft | None], *, last: bool = False) -> dict[str, Any] | None:
    """Слить блоки по полям: первое (или последнее) непустое значение каждого поля."""
    present = [o for o in objects if o is not None]
    if not present:
        return None
    if last:
        present = present[::-1]
    fields = type(present[0]).model_fields
    return {name: _first([getattr(o, name) for o in present]) for name in fields}


def merge_drafts(drafts: list[Draft]) -> Draft:
    """Черновики страниц → один документ.

    Шапка (номер, дата, стороны, банк, основание) — первое непустое значение по
    страницам; строки таблицы — подряд со всех страниц; итоги — последнее непустое
    значение (итоговая строка обычно на последней странице). Тип — с первой страницы.
    """
    if len(drafts) == 1:
        return drafts[0]
    return Draft.model_validate(
        {
            "doc_type": drafts[0].doc_type,
            "upd_status": _first([d.upd_status for d in drafts]),
            "number": _first([d.number for d in drafts]),
            "date": _first([d.date for d in drafts]),
            "seller": _merge_objects([d.seller for d in drafts]),
            "buyer": _merge_objects([d.buyer for d in drafts]),
            "seller_bank": _merge_objects([d.seller_bank for d in drafts]),
            "basis": _merge_objects([d.basis for d in drafts]),
            "lines": [line.model_dump() for d in drafts for line in d.lines],
            "totals": _merge_objects([d.totals for d in drafts], last=True),
        }
    )


# Какие необязательные блоки есть у какого типа (см. docai.schemas.documents).
_HAS_BANK = {DocType.BILL, DocType.ACT, DocType.TORG12}
_HAS_BASIS = {DocType.UPD, DocType.BILL, DocType.ACT, DocType.TORG12}


def _party(party: DraftParty, role: str, warnings: list[str]) -> dict[str, Any]:
    data = party.model_dump()
    if not data["name"]:
        warnings.append(f"{role}.name не найдено → пустая строка")
        data["name"] = ""
    return data


def draft_to_document(draft: Draft) -> tuple[AnyDocument | None, list[str], list[str]]:
    """Черновик → документ нашей схемы. Возвращает (документ или None, предупреждения, ошибки).

    Документ не собирается только без даты: дата обязательна, а придумать её нельзя.
    """
    warnings: list[str] = []
    errors: list[str] = []
    if draft.date is None:
        errors.append("date не найдена: без даты документ не собрать")
        return None, warnings, errors

    data: dict[str, Any] = {
        "doc_type": draft.doc_type,
        "number": draft.number or "",
        "date": draft.date,
        "seller": _party(draft.seller, "seller", warnings),
        "buyer": _party(draft.buyer, "buyer", warnings),
        "lines": [],
        "totals": draft.totals.model_dump(),
    }
    if not draft.number:
        warnings.append("number не найден → пустая строка")
    for index, line in enumerate(draft.lines):
        item = line.model_dump()
        if not item["name"]:
            warnings.append(f"lines.{index}.name не найдено → пустая строка")
            item["name"] = ""
        data["lines"].append(item)

    doc_type = draft.doc_type
    if doc_type is DocType.UPD:
        if draft.upd_status is None:
            warnings.append("upd_status не найден → 1")
        data["upd_status"] = draft.upd_status or 1
    if doc_type in _HAS_BANK:
        bank = draft.seller_bank.model_dump() if draft.seller_bank else None
        if doc_type is DocType.BILL and bank is None:
            warnings.append("seller_bank не найден → пустые реквизиты")
            bank = {}
        data["seller_bank"] = bank
    if doc_type in _HAS_BASIS:
        data["basis"] = draft.basis.model_dump() if draft.basis else None

    try:
        return parse_document(data), warnings, errors
    except ValidationError as exc:
        errors.extend(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return None, warnings, errors
