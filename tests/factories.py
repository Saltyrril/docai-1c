"""Общие тестовые данные: корректные документы с публичными реквизитами банков.

ИНН, КПП, БИК и корр. счета — публичные реквизиты крупных банков. Расчётный счёт
продавца синтетический, его ключ посчитан по БИК (см. test_requisites.py).
"""

from datetime import date
from typing import Any

SBER = {"name": "ПАО Сбербанк", "inn": "7707083893", "kpp": "773601001", "address": "Москва"}
TBANK = {"name": "АО «ТБанк»", "inn": "7710140679", "kpp": "771301001", "address": "Москва"}
SELLER_BANK = {
    "bank_name": "ПАО Сбербанк",
    "bik": "044525225",
    "corr_account": "30101810400000000225",
    "account": "40702810938000000001",
}
INDIVIDUAL_INN = "500100732259"


def line(
    qty: str,
    price: str,
    without: str | None,
    rate: str | None,
    vat: str | None,
    with_vat: str | None,
    name: str = "Товар",
) -> dict[str, Any]:
    return {
        "name": name,
        "unit_name": "шт",
        "unit_code": "796",
        "quantity": qty,
        "price": price,
        "amount_without_vat": without,
        "vat_rate": rate,
        "vat_amount": vat,
        "amount_with_vat": with_vat,
    }


def doc_data(doc_type: str = "upd", **overrides: Any) -> dict[str, Any]:
    """Корректный документ от 14.11.2026 со ставкой 22% и двумя строками."""
    data: dict[str, Any] = {
        "doc_type": doc_type,
        "number": "123",
        "date": date(2026, 11, 14),
        "seller": dict(SBER),
        "buyer": dict(TBANK),
        "lines": [
            line("2", "1000.00", "2000.00", "22%", "440.00", "2440.00", "Бумага"),
            line("3", "33.33", "99.99", "22%", "22.00", "121.99", "Ручка"),
        ],
        "totals": {
            "amount_without_vat": "2099.99",
            "vat_amount": "462.00",
            "amount_with_vat": "2561.99",
        },
    }
    if doc_type == "upd":
        data["upd_status"] = 1
    if doc_type in ("bill", "act", "torg12"):
        data["seller_bank"] = dict(SELLER_BANK)
    if doc_type != "vat_invoice":
        data["basis"] = {"name": "Договор поставки", "number": "Д-1", "date": date(2026, 1, 15)}
    data.update(overrides)
    return data


ALL_DOC_TYPES = ["upd", "bill", "act", "torg12", "vat_invoice"]
