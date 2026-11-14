"""Пять типов документов и общий тип `Document` с выбором класса по `doc_type`.

Какие блоки есть у какого типа (по печатным формам):
- банковские реквизиты продавца — в счёте на оплату (обязательно), в акте и ТОРГ-12;
  в формах УПД и счёта-фактуры их нет;
- основание (договор) — в УПД, счёте, акте, ТОРГ-12; в счёте-фактуре его нет.
"""

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter

from docai.schemas.common import BankDetails, BaseDocument, Basis, DocType


class Upd(BaseDocument):
    """Универсальный передаточный документ.

    Статус 1 — счёт-фактура и передаточный документ одновременно;
    статус 2 — только передаточный (первичный) документ.
    """

    doc_type: Literal[DocType.UPD] = DocType.UPD
    upd_status: Literal[1, 2] = Field(description="Статус УПД: 1 или 2")
    basis: Basis | None = Field(default=None, description="Основание передачи (сдачи)")


class Bill(BaseDocument):
    """Счёт на оплату."""

    doc_type: Literal[DocType.BILL] = DocType.BILL
    seller_bank: BankDetails = Field(description="Банковские реквизиты продавца")
    basis: Basis | None = Field(default=None, description="Основание (договор)")


class Act(BaseDocument):
    """Акт выполненных работ (оказанных услуг)."""

    doc_type: Literal[DocType.ACT] = DocType.ACT
    seller_bank: BankDetails | None = Field(default=None, description="Реквизиты исполнителя")
    basis: Basis | None = Field(default=None, description="Основание (договор)")


class Torg12(BaseDocument):
    """Товарная накладная ТОРГ-12."""

    doc_type: Literal[DocType.TORG12] = DocType.TORG12
    seller_bank: BankDetails | None = Field(default=None, description="Реквизиты поставщика")
    basis: Basis | None = Field(default=None, description="Основание (договор)")


class VatInvoice(BaseDocument):
    """Счёт-фактура."""

    doc_type: Literal[DocType.VAT_INVOICE] = DocType.VAT_INVOICE


type AnyDocument = Upd | Bill | Act | Torg12 | VatInvoice

Document = Annotated[Upd | Bill | Act | Torg12 | VatInvoice, Field(discriminator="doc_type")]
"""Любой из пяти документов; класс выбирается по полю `doc_type`."""

DOCUMENT_CLASSES: dict[DocType, type[BaseDocument]] = {
    DocType.UPD: Upd,
    DocType.BILL: Bill,
    DocType.ACT: Act,
    DocType.TORG12: Torg12,
    DocType.VAT_INVOICE: VatInvoice,
}

_document_adapter: TypeAdapter[AnyDocument] = TypeAdapter(Document)


def parse_document(data: dict[str, Any] | str | bytes) -> AnyDocument:
    """Разобрать документ из словаря или JSON; класс выбирается по `doc_type`."""
    if isinstance(data, str | bytes):
        return _document_adapter.validate_json(data)
    return _document_adapter.validate_python(data)
