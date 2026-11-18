"""Тип документа по заголовку в тексте-источнике (P13, D50).

Модель 2B на пути text в двух третях случаев называет УПД счётом-фактурой: шапка УПД
содержит строку «Счёт-фактура № …», а признак УПД — «Статус: 1» — стоит рядом и теряется.
Тип и статус УПД однозначно видны в заголовке, поэтому их можно взять из текста без LLM.

Признаки (по первым строкам первой страницы):
- УПД — «Универсальный передаточный документ» или «УПД», либо поле «Статус: 1|2» рядом
  со «Счёт-фактура». Форма УПД — счёт-фактура с полем «Статус» (рекомендована письмом ФНС
  России от 21.10.2013 № ММВ-20-3/96@, приложение 1); статус 1 — счёт-фактура и передаточный
  документ, 2 — только передаточный документ;
- счёт-фактура — «Счёт-фактура» без признаков УПД;
- счёт на оплату — строка, начинающаяся со «Счёт на оплату» или «Счёт №»;
- акт — строка, начинающаяся с «Акт»;
- ТОРГ-12 — «ТОРГ-12», «Товарная накладная» или код формы по ОКУД 0330212.

Если в заголовке найден ровно один тип и он расходится с ответом модели, документ
переводится в найденный тип: поля, которых у нового типа нет, отбрасываются, недостающие
обязательные блоки (банк у счёта на оплату) создаются пустыми. Каждое изменение пишется
в журнал исправлений. Если признаков нет или они противоречат друг другу, ничего не меняется.
"""

import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from docai.postprocess.models import Correction
from docai.schemas.common import BaseDocument, DocType
from docai.schemas.documents import parse_document

# Сколько символов начала текста считать заголовком. В счёте на оплату заголовку
# предшествуют предупреждение и таблица банковских реквизитов (≈ 400 символов).
TITLE_CHARS = 800

_UPD = re.compile(r"универсальн\w*\s+передаточн\w*\s+документ|\bУПД\b", re.IGNORECASE)
_STATUS = re.compile(r"\bСтатус\s*:?\s*([12])\b", re.IGNORECASE)
_VAT_INVOICE = re.compile(r"сч[её]т[-\s]*фактур", re.IGNORECASE)
_BILL = re.compile(r"^\s*сч[её]т\s*(?:на\s+оплату|№)", re.IGNORECASE | re.MULTILINE)
_ACT = re.compile(r"^\s*акт\b", re.IGNORECASE | re.MULTILINE)
# 0330212 — код формы ТОРГ-12 по ОКУД (постановление Госкомстата России от 25.12.1998 № 132).
_TORG12 = re.compile(r"ТОРГ\s*-?\s*12|товарн\w*\s+накладн|\b0330212\b", re.IGNORECASE)

# Блоки, которые есть только у части типов (см. docai.schemas.documents).
_HAS_BANK = {DocType.BILL, DocType.ACT, DocType.TORG12}
_HAS_BASIS = {DocType.UPD, DocType.BILL, DocType.ACT, DocType.TORG12}


def title_type(text: str) -> tuple[DocType, int | None] | None:
    """Тип документа и статус УПД по заголовку; None — признаков нет или они противоречат."""
    title = text[:TITLE_CHARS]
    status_match = _STATUS.search(title)
    status = int(status_match.group(1)) if status_match else None
    found: set[DocType] = set()
    if _UPD.search(title) or (status is not None and _VAT_INVOICE.search(title)):
        found.add(DocType.UPD)
    elif _VAT_INVOICE.search(title):
        found.add(DocType.VAT_INVOICE)
    if _BILL.search(title):
        found.add(DocType.BILL)
    if _ACT.search(title):
        found.add(DocType.ACT)
    if _TORG12.search(title):
        found.add(DocType.TORG12)
    if len(found) != 1:
        return None
    (doc_type,) = found
    return doc_type, status if doc_type is DocType.UPD else None


def _convert(doc: BaseDocument, doc_type: DocType, status: int | None) -> dict[str, Any]:
    data = doc.model_dump(mode="json")
    data["doc_type"] = doc_type.value
    data.pop("upd_status", None)
    if doc_type is DocType.UPD:
        data["upd_status"] = status or 1
    if doc_type not in _HAS_BANK:
        data.pop("seller_bank", None)
    elif doc_type is DocType.BILL and not data.get("seller_bank"):
        data["seller_bank"] = {}
    if doc_type not in _HAS_BASIS:
        data.pop("basis", None)
    return data


def retype_document(
    doc: BaseDocument, texts: Sequence[str]
) -> tuple[BaseDocument, list[Correction]]:
    """Перевести документ в тип из заголовка, если модель назвала другой (или другой статус)."""
    if not texts:
        return doc, []
    found = title_type(texts[0])
    if found is None:
        return doc, []
    doc_type, status = found
    current_status = getattr(doc, "upd_status", None)
    same_type = doc.doc_type == doc_type
    if same_type and (doc_type is not DocType.UPD or status in (None, current_status)):
        return doc, []
    try:
        new_doc = parse_document(_convert(doc, doc_type, status or current_status))
    except ValidationError:
        return doc, []
    corrections = []
    if not same_type:
        corrections.append(
            Correction(
                kind="autofix",
                field="doc_type",
                before=str(doc.doc_type),
                after=str(doc_type),
                rule="doc_type_by_title",
                reason="тип документа взят из заголовка текста",
            )
        )
    new_status = getattr(new_doc, "upd_status", None)
    if new_status != current_status:
        corrections.append(
            Correction(
                kind="autofix",
                field="upd_status",
                before=None if current_status is None else str(current_status),
                after=None if new_status is None else str(new_status),
                rule="doc_type_by_title",
                reason="статус УПД взят из заголовка текста"
                if status is not None
                else "тип УПД из заголовка, статус не найден → 1",
            )
        )
    return new_doc, corrections
