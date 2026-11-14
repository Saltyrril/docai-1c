"""Генератор: наш `Upd` → XML счёта-фактуры и документа об отгрузке (формат ФНС 5.03).

Что кладётся в XML и откуда (подробности и допущения — D30 в docs/DECISIONS.md):
- статус УПД 1 → `Функция="СЧФДОП"`, статус 2 → `Функция="ДОП"`;
- юрлицо (ИНН из 10 цифр) → `СвЮЛУч`, ИП (12 цифр, имя вида «ИП Фамилия Имя Отчество»)
  → `СвИП`; адрес — одной строкой в `АдрИнф` (страна 643), потому что в наших документах
  адрес и есть строка, а разбор её на элементы `АдрРФ` был бы угадыванием;
- основание передачи (`basis`) → `СвПродПер/СвПер/ОснПер`; нет основания → `БезДокОснПер`;
- строка → `СведТов`, итоги → `ВсегоОпл`; «нет суммы НДС» → `БезНДС`.

Если документ нельзя записать без потерь (нет ставки НДС, неполное основание, ИНН не
10 и не 12 цифр, имя ИП не раскладывается на ФИО), генератор поднимает `EdoError`,
а не выдумывает значения. Готовый XML всегда проверяется по XSD.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from lxml import etree

from docai import __version__
from docai.edo.errors import EdoError
from docai.edo.schema import ENCODING, FILE_PREFIX, FORMAT_VERSION, KND, validate_xml
from docai.schemas.common import Basis, LineItem, Party, Totals
from docai.schemas.documents import Upd

COUNTRY_CODE = "643"
COUNTRY_NAME = "Россия"
CURRENCY_CODE = "643"
CURRENCY_NAME = "Российский рубль"

_FUNCTION_BY_STATUS = {1: "СЧФДОП", 2: "ДОП"}
_DOC_TITLE = (
    "Документ об отгрузке товаров (выполнении работ), передаче имущественных прав "
    "(документ об оказании услуг)"
)
_OPERATION = "Товары (работы, услуги) переданы"
_INFO_TIME = "12.00.00"  # время формирования файла: фиксированное, чтобы XML был воспроизводим
_SIGNER_BY_POSITION = "1"  # СпосПодтПолном: полномочия видны из данных электронной подписи
_NO_VAT_TEXT = "без НДС"

# Пространство имён для идентификаторов в имени файла: одни и те же данные дают один и тот же GUID.
_GUID_NAMESPACE = uuid.UUID("6f1b8f0e-3c52-5d0a-9a57-0c7d3b1a2e44")
_IP_NAME = re.compile(r"^ИП\s+(\S+)\s+(\S+)(?:\s+(\S.*))?$")
_TWO_PLACES = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Signer:
    """Лицо, подписывающее документ со стороны продавца (в нашем JSON его нет)."""

    last_name: str
    first_name: str
    middle_name: str | None
    position: str


DEMO_SIGNER = Signer("Иванов", "Иван", "Иванович", "Генеральный директор")
"""Вымышленный подписант для синтетических документов."""


def _date(value: date) -> str:
    return f"{value:%d.%m.%Y}"


def _number(value: Decimal) -> str:
    return format(value, "f")


def _money(value: Decimal) -> str:
    """Сумма с двумя знаками, если это не меняет значения (28169.4 → 28169.40)."""
    rounded = value.quantize(_TWO_PLACES)
    return _number(rounded if rounded == value else value)


def _set(el: etree._Element, name: str, value: str | None) -> None:
    if value is not None:
        el.set(name, value)


def _set_money(el: etree._Element, name: str, value: Decimal | None) -> None:
    if value is not None:
        el.set(name, _money(value))


def _fio(parent: etree._Element, last: str, first: str, middle: str | None) -> None:
    fio = etree.SubElement(parent, "ФИО")
    fio.set("Фамилия", last)
    fio.set("Имя", first)
    _set(fio, "Отчество", middle)


def _party(parent: etree._Element, tag: str, party: Party, role: str) -> None:
    el = etree.SubElement(parent, tag)
    ident = etree.SubElement(el, "ИдСв")
    inn = party.inn or ""
    if len(inn) == 10 and inn.isdigit():
        legal = etree.SubElement(ident, "СвЮЛУч")
        legal.set("НаимОрг", party.name)
        legal.set("ИННЮЛ", inn)
        _set(legal, "КПП", party.kpp)
    elif len(inn) == 12 and inn.isdigit():
        match = _IP_NAME.match(party.name)
        if match is None or party.kpp is not None:
            raise EdoError(
                f"{role}: ИП с ИНН {inn} должен называться «ИП Фамилия Имя [Отчество]» "
                f"и не иметь КПП, а здесь «{party.name}», КПП {party.kpp}"
            )
        last, first, middle = match.groups()
        if party.name != " ".join(("ИП", last, first, *([middle] if middle else []))):
            raise EdoError(f"{role}: имя «{party.name}» не восстановить из ФИО без потерь")
        ip = etree.SubElement(ident, "СвИП")
        ip.set("ИННФЛ", inn)
        _fio(ip, last, first, middle)
    else:
        raise EdoError(f"{role}: ИНН «{party.inn}» не из 10 и не из 12 цифр")
    if party.address is not None:
        address = etree.SubElement(el, "Адрес")
        info = etree.SubElement(address, "АдрИнф")
        info.set("КодСтр", COUNTRY_CODE)
        info.set("НаимСтран", COUNTRY_NAME)
        info.set("АдрТекст", party.address)


def _basis(parent: etree._Element, basis: Basis | None) -> None:
    if basis is None or (basis.name, basis.number, basis.date) == (None, None, None):
        etree.SubElement(parent, "БезДокОснПер").text = "1"
        return
    if None in (basis.name, basis.number, basis.date):
        raise EdoError(
            "основание передачи заполнено не полностью: формат требует вид документа, "
            f"номер и дату вместе ({basis.name!r}, {basis.number!r}, {basis.date!r})"
        )
    assert basis.name is not None and basis.number is not None and basis.date is not None
    el = etree.SubElement(parent, "ОснПер")
    el.set("РеквНаимДок", basis.name)
    el.set("РеквНомерДок", basis.number)
    el.set("РеквДатаДок", _date(basis.date))


def _vat_sum(parent: etree._Element, tag: str, value: Decimal | None) -> None:
    """Сумма налога (`СумНДСТип`): число или «без НДС»; tag — `СумНал` или `СумНалВсего`."""
    holder = etree.SubElement(parent, tag)
    if value is None:
        etree.SubElement(holder, "БезНДС").text = _NO_VAT_TEXT
    else:
        etree.SubElement(holder, "СумНал").text = _money(value)


def _line(parent: etree._Element, number: int, line: LineItem) -> None:
    if line.vat_rate is None:
        raise EdoError(f"строка {number}: нет ставки НДС, а формат её требует")
    el = etree.SubElement(parent, "СведТов")
    el.set("НомСтр", str(number))
    el.set("НаимТов", line.name)
    _set(el, "ОКЕИ_Тов", line.unit_code)
    _set(el, "НаимЕдИзм", line.unit_name)
    if line.quantity is not None:
        el.set("КолТов", _number(line.quantity))
    if line.price is not None:
        el.set("ЦенаТов", _number(line.price))
    _set_money(el, "СтТовБезНДС", line.amount_without_vat)
    el.set("НалСт", line.vat_rate.value)
    _set_money(el, "СтТовУчНал", line.amount_with_vat)
    _vat_sum(el, "СумНал", line.vat_amount)


def _table(root: etree._Element, lines: list[LineItem], totals: Totals) -> None:
    if not lines:
        if totals.model_dump(exclude_none=True):
            raise EdoError("итоги есть, а строк нет: в формате итоги живут в таблице строк")
        return
    table = etree.SubElement(root, "ТаблСчФакт")
    for number, line in enumerate(lines, start=1):
        _line(table, number, line)
    total = etree.SubElement(table, "ВсегоОпл")
    _set_money(total, "СтТовБезНДСВсего", totals.amount_without_vat)
    _set_money(total, "СтТовУчНалВсего", totals.amount_with_vat)
    _vat_sum(total, "СумНалВсего", totals.vat_amount)


def file_id(doc: Upd) -> str:
    """Имя файла обмена без расширения (ИдФайл), п. 4 раздела II приказа.

    Файл идёт не через оператора ЭДО, поэтому у получателя и отправителя префикс «000»,
    а за ним GUID участника. GUID считаются из реквизитов (uuid5), поэтому имя и сам XML
    воспроизводимы. Признаки прослеживаемости и маркировки (N2–N6) — «0», N7 — «00».
    """
    buyer = uuid.uuid5(_GUID_NAMESPACE, f"{doc.buyer.inn}/{doc.buyer.kpp or ''}")
    seller = uuid.uuid5(_GUID_NAMESPACE, f"{doc.seller.inn}/{doc.seller.kpp or ''}")
    this = uuid.uuid5(_GUID_NAMESPACE, f"{doc.seller.inn}/{doc.number}/{doc.date.isoformat()}")
    return f"{FILE_PREFIX}_000{buyer}_000{seller}_{doc.date:%Y%m%d}_{this}_0_0_0_0_0_00"


def upd_to_element(doc: Upd, signer: Signer) -> etree._Element:
    """Собрать дерево XML (без проверки по XSD)."""
    root = etree.Element("Файл")
    root.set("ИдФайл", file_id(doc))
    root.set("ВерсФорм", FORMAT_VERSION)
    root.set("ВерсПрог", f"docai-1c {__version__}")

    document = etree.SubElement(root, "Документ")
    document.set("КНД", KND)
    document.set("Функция", _FUNCTION_BY_STATUS[doc.upd_status])
    document.set("ПоФактХЖ", _DOC_TITLE)
    document.set("НаимДокОпр", "Универсальный передаточный документ")
    document.set("ДатаИнфПр", _date(doc.date))
    document.set("ВремИнфПр", _INFO_TIME)
    seller_ids = f"{doc.seller.inn}/{doc.seller.kpp}" if doc.seller.kpp else doc.seller.inn
    label = "ИНН/КПП" if doc.seller.kpp else "ИНН"
    document.set("НаимЭконСубСост", f"{doc.seller.name}, {label} {seller_ids}")

    invoice = etree.SubElement(document, "СвСчФакт")
    invoice.set("НомерДок", doc.number)
    invoice.set("ДатаДок", _date(doc.date))
    _party(invoice, "СвПрод", doc.seller, "продавец")
    consignor = etree.SubElement(etree.SubElement(invoice, "ГрузОт"), "ОнЖе")
    consignor.text = "он же"
    _party(invoice, "СвПокуп", doc.buyer, "покупатель")
    currency = etree.SubElement(invoice, "ДенИзм")
    currency.set("КодОКВ", CURRENCY_CODE)
    currency.set("НаимОКВ", CURRENCY_NAME)

    _table(document, doc.lines, doc.totals)

    transfer = etree.SubElement(document, "СвПродПер")
    operation = etree.SubElement(transfer, "СвПер")
    operation.set("СодОпер", _OPERATION)
    operation.set("ДатаПер", _date(doc.date))
    _basis(operation, doc.basis)

    sign = etree.SubElement(document, "Подписант")
    sign.set("Должн", signer.position)
    sign.set("СпосПодтПолном", _SIGNER_BY_POSITION)
    _fio(sign, signer.last_name, signer.first_name, signer.middle_name)
    return root


def serialize(root: etree._Element) -> bytes:
    """XML в windows-1251 с заголовком, который требует формат (первая строка файла)."""
    body = etree.tostring(root, encoding=ENCODING, pretty_print=True, xml_declaration=False)
    return f'<?xml version="1.0" encoding="{ENCODING}"?>\n'.encode(ENCODING) + body


def upd_to_xml(doc: Upd, signer: Signer = DEMO_SIGNER, *, validate: bool = True) -> bytes:
    """УПД → байты XML в windows-1251; по умолчанию проверяется по XSD."""
    data = serialize(upd_to_element(doc, signer))
    if validate:
        validate_xml(data)
    return data
