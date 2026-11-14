"""Общие части документов: типы, ставки НДС, стороны, банк, строки, итоги.

Схемы описывают документ «как он написан», поэтому они нестрогие: ИНН с неверной
контрольной цифрой или неверная сумма сюда помещаются. Ошибки ищет пакет
`docai.validation` и возвращает их списком, а не исключением.
"""

import datetime as dt
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocType(StrEnum):
    """Тип первичного документа."""

    UPD = "upd"  # универсальный передаточный документ (статус 1 или 2)
    BILL = "bill"  # счёт на оплату
    ACT = "act"  # акт выполненных работ (оказанных услуг)
    TORG12 = "torg12"  # товарная накладная ТОРГ-12
    VAT_INVOICE = "vat_invoice"  # счёт-фактура

    @property
    def title_ru(self) -> str:
        return _DOC_TYPE_TITLES[self]


_DOC_TYPE_TITLES = {
    DocType.UPD: "УПД",
    DocType.BILL: "счёт на оплату",
    DocType.ACT: "акт выполненных работ",
    DocType.TORG12: "товарная накладная ТОРГ-12",
    DocType.VAT_INVOICE: "счёт-фактура",
}


class VatRate(StrEnum):
    """Ставка НДС так, как она пишется в документе (ст. 164 НК РФ).

    Процентные ставки применяются к сумме без НДС. Расчётные ставки (п. 4 ст. 164,
    например 22/122 с аванса) применяются к сумме с НДС. Допустимость ставки на
    дату документа проверяет `docai.validation.vat`.
    """

    NO_VAT = "без НДС"
    R0 = "0%"
    R5 = "5%"
    R7 = "7%"
    R10 = "10%"
    R18 = "18%"
    R20 = "20%"
    R22 = "22%"
    R5_105 = "5/105"
    R7_107 = "7/107"
    R10_110 = "10/110"
    R18_118 = "18/118"
    R20_120 = "20/120"
    R22_122 = "22/122"

    @property
    def percent(self) -> Decimal | None:
        """Число процентов: 22 для «22%» и для «22/122»; None для «без НДС»."""
        if self is VatRate.NO_VAT:
            return None
        return Decimal(self.value.split("%")[0].split("/")[0])

    @property
    def is_calculated(self) -> bool:
        """Расчётная ставка вида 22/122."""
        return "/" in self.value

    @property
    def fraction(self) -> Decimal:
        """Доля налога от базы: 0.22 для «22%», 22/122 для «22/122», 0 для «без НДС»."""
        percent = self.percent
        if percent is None:
            return Decimal(0)
        if self.is_calculated:
            return percent / (100 + percent)
        return percent / 100


class _Model(BaseModel):
    """Базовая модель: лишние поля запрещены, пробелы по краям строк убираются."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Party(_Model):
    """Сторона сделки: продавец (исполнитель, поставщик) или покупатель (заказчик)."""

    name: str = Field(description="Наименование организации или ФИО ИП")
    inn: str | None = Field(default=None, description="ИНН: 10 цифр у юрлица, 12 у ИП")
    kpp: str | None = Field(default=None, description="КПП, 9 символов; у ИП отсутствует")
    address: str | None = Field(default=None, description="Адрес")


class BankDetails(_Model):
    """Банковские реквизиты продавца."""

    bank_name: str | None = Field(default=None, description="Наименование банка")
    bik: str | None = Field(default=None, description="БИК, 9 цифр")
    corr_account: str | None = Field(default=None, description="Корреспондентский счёт банка")
    account: str | None = Field(default=None, description="Расчётный счёт, 20 цифр")


class Basis(_Model):
    """Основание: договор (или другой документ), по которому составлен документ."""

    name: str | None = Field(default=None, description="Вид документа, например «Договор поставки»")
    number: str | None = Field(default=None, description="Номер договора")
    date: dt.date | None = Field(default=None, description="Дата договора")


class LineItem(_Model):
    """Строка табличной части.

    Суммы без НДС, НДС и с НДС необязательны: в счёте на оплату часто печатается
    только одна «Сумма», а НДС — только в итогах.
    """

    name: str = Field(description="Наименование товара, работы, услуги")
    unit_name: str | None = Field(default=None, description="Единица измерения, например «шт»")
    unit_code: str | None = Field(default=None, description="Код единицы по ОКЕИ, например «796»")
    quantity: Decimal | None = Field(default=None, description="Количество")
    price: Decimal | None = Field(default=None, description="Цена за единицу")
    amount_without_vat: Decimal | None = Field(default=None, description="Сумма без НДС")
    vat_rate: VatRate | None = Field(default=None, description="Ставка НДС")
    vat_amount: Decimal | None = Field(default=None, description="Сумма НДС")
    amount_with_vat: Decimal | None = Field(default=None, description="Сумма с НДС")


class Totals(_Model):
    """Итоги документа."""

    amount_without_vat: Decimal | None = Field(default=None, description="Итого без НДС")
    vat_amount: Decimal | None = Field(default=None, description="Итого НДС")
    amount_with_vat: Decimal | None = Field(default=None, description="Всего с НДС (к оплате)")


class BaseDocument(_Model):
    """Поля, общие для всех пяти типов документов."""

    doc_type: DocType
    number: str = Field(description="Номер документа")
    date: dt.date = Field(description="Дата документа")
    seller: Party = Field(description="Продавец / исполнитель / поставщик")
    buyer: Party = Field(description="Покупатель / заказчик / плательщик")
    lines: list[LineItem] = Field(default_factory=list, description="Табличная часть")
    totals: Totals = Field(default_factory=Totals, description="Итоги")
