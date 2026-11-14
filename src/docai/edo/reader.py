"""Парсер: XML счёта-фактуры и документа об отгрузке (формат ФНС 5.03) → наш `Upd`.

Работает без LLM. Принимает файлы с функциями `СЧФДОП` (статус УПД 1) и `ДОП`
(статус 2); `СЧФ`, `СвРК`, `СвЗК` — другие документы, для них поднимается `EdoError`.

Из файла берутся только поля нашей схемы (FIELD_SPEC). Всё остальное (грузополучатель,
подписанты, сведения о прослеживаемости и т. д.) игнорируется. Если в файле несколько
продавцов, покупателей или оснований передачи, берётся первое.
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation

from lxml import etree

from docai.edo.errors import EdoError
from docai.edo.schema import FORMAT_VERSION, parse_xml, validate_xml
from docai.schemas.common import Basis, LineItem, Party, Totals, VatRate
from docai.schemas.documents import Upd

_STATUS_BY_FUNCTION = {"СЧФДОП": 1, "ДОП": 2}


def _date(value: str | None, where: str):
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%d.%m.%Y").date()
    except ValueError as exc:
        raise EdoError(f"{where}: «{value}» — не дата ДД.ММ.ГГГГ") from exc


def _decimal(value: str | None, where: str) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise EdoError(f"{where}: «{value}» — не число") from exc


def _required(el: etree._Element, name: str) -> str:
    value = el.get(name)
    if value is None:
        raise EdoError(f"у <{el.tag}> нет атрибута {name}")
    return value


def _child(el: etree._Element, name: str) -> etree._Element | None:
    return el.find(name)


def _fio(el: etree._Element | None) -> str:
    if el is None:
        return ""
    parts = (el.get("Фамилия"), el.get("Имя"), el.get("Отчество"))
    return " ".join(p for p in parts if p)


def _address(el: etree._Element | None) -> str | None:
    """Адрес одной строкой. `АдрИнф` — как есть; `АдрРФ` и `АдрГАР` — склейка частей."""
    if el is None:
        return None
    address = next(iter(el), None)
    if address is None:
        return None
    if address.tag == "АдрИнф":
        return address.get("АдрТекст")
    if address.tag == "АдрРФ":
        names = ("Индекс", "НаимРегион", "Район", "Город", "НаселПункт", "Улица", "Дом")
        names += ("Корпус", "Кварт", "ИныеСвед")
        parts = [address.get(name) for name in names]
    else:  # АдрГАР: регион и вложенные элементы вида «тип + наименование/номер»
        parts = [address.get("Индекс"), address.findtext("НаимРегион")]
        for part in address:
            if part.tag in ("Регион", "НаимРегион"):
                continue
            label = part.get("Вид") or part.get("Тип") or ""
            value = part.get("Наим") or part.get("Номер") or part.text
            parts.append(" ".join(p for p in (label, value) if p) or None)
    text = ", ".join(p for p in parts if p)
    return text or None


def _party(el: etree._Element, role: str) -> Party:
    ident = _child(el, "ИдСв")
    kind = next(iter(ident), None) if ident is not None else None
    if kind is None:
        raise EdoError(f"{role}: нет сведений об участнике (ИдСв)")
    name: str
    inn: str | None = None
    kpp: str | None = None
    if kind.tag == "СвЮЛУч":
        name, inn, kpp = _required(kind, "НаимОрг"), _required(kind, "ИННЮЛ"), kind.get("КПП")
    elif kind.tag == "СвИП":
        name, inn = f"ИП {_fio(_child(kind, 'ФИО'))}", _required(kind, "ИННФЛ")
    elif kind.tag == "СвИнНеУч":
        name = _required(kind, "Наим")  # иностранный участник: ИНН у него нет
    else:  # СвФЛУч: физлицо
        name, inn = _fio(_child(kind, "ФИО")), kind.get("ИННФЛ")
        if not name:
            raise EdoError(f"{role}: у физического лица нет ФИО")
    return Party(name=name, inn=inn, kpp=kpp, address=_address(_child(el, "Адрес")))


def _basis(transfer: etree._Element | None) -> Basis | None:
    if transfer is None:
        return None
    base = transfer.find("СвПер/ОснПер")
    if base is None:
        return None  # БезДокОснПер
    return Basis(
        name=base.get("РеквНаимДок"),
        number=base.get("РеквНомерДок"),
        date=_date(base.get("РеквДатаДок"), "ОснПер/РеквДатаДок"),
    )


def _vat_sum(el: etree._Element, tag: str) -> Decimal | None:
    holder = _child(el, tag)
    if holder is None:
        return None
    return _decimal(holder.findtext("СумНал"), tag)  # «без НДС» → None


def _line(el: etree._Element) -> LineItem:
    where = f"строка {el.get('НомСтр')}"
    return LineItem(
        name=_required(el, "НаимТов"),
        unit_name=el.get("НаимЕдИзм"),
        unit_code=el.get("ОКЕИ_Тов"),
        quantity=_decimal(el.get("КолТов"), f"{where}: КолТов"),
        price=_decimal(el.get("ЦенаТов"), f"{where}: ЦенаТов"),
        amount_without_vat=_decimal(el.get("СтТовБезНДС"), f"{where}: СтТовБезНДС"),
        vat_rate=VatRate(_required(el, "НалСт")),
        vat_amount=_vat_sum(el, "СумНал"),
        amount_with_vat=_decimal(el.get("СтТовУчНал"), f"{where}: СтТовУчНал"),
    )


def _totals(table: etree._Element | None) -> Totals:
    total = table.find("ВсегоОпл") if table is not None else None
    if total is None:
        return Totals()
    return Totals(
        amount_without_vat=_decimal(total.get("СтТовБезНДСВсего"), "СтТовБезНДСВсего"),
        vat_amount=_vat_sum(total, "СумНалВсего"),
        amount_with_vat=_decimal(total.get("СтТовУчНалВсего"), "СтТовУчНалВсего"),
    )


def element_to_upd(root: etree._Element) -> Upd:
    """Дерево XML (уже разобранное) → `Upd`. Проверку по XSD делает `xml_to_upd`."""
    if root.tag != "Файл":
        raise EdoError(f"корневой элемент <{root.tag}>, а ожидался <Файл>")
    version = root.get("ВерсФорм")
    if version != FORMAT_VERSION:
        raise EdoError(f"версия формата {version}, поддерживается только {FORMAT_VERSION}")
    document = _child(root, "Документ")
    if document is None:
        raise EdoError("в файле нет элемента <Документ>")
    function = document.get("Функция")
    status = _STATUS_BY_FUNCTION.get(function or "")
    if status is None:
        raise EdoError(f"функция «{function}» — это не УПД (ожидалась СЧФДОП или ДОП)")

    invoice = _child(document, "СвСчФакт")
    if invoice is None:
        raise EdoError("в документе нет <СвСчФакт>")
    seller = _child(invoice, "СвПрод")
    buyer = _child(invoice, "СвПокуп")
    if seller is None or buyer is None:
        raise EdoError("в документе нет продавца (СвПрод) или покупателя (СвПокуп)")
    table = _child(document, "ТаблСчФакт")
    doc_date = _date(_required(invoice, "ДатаДок"), "ДатаДок")
    assert doc_date is not None
    return Upd(
        number=_required(invoice, "НомерДок"),
        date=doc_date,
        upd_status=status,
        seller=_party(seller, "продавец"),
        buyer=_party(buyer, "покупатель"),
        basis=_basis(_child(document, "СвПродПер")),
        lines=[_line(el) for el in table.findall("СведТов")] if table is not None else [],
        totals=_totals(table),
    )


def xml_to_upd(data: bytes, *, validate: bool = True) -> Upd:
    """Байты XML → `Upd`; по умолчанию сначала проверяет файл по XSD."""
    root = parse_xml(data)
    if validate:
        validate_xml(root)
    return element_to_upd(root)
