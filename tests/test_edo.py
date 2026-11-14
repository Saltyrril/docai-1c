"""Тесты P7: УПД из ЭДО — XML формата ФНС 5.03, проверка по XSD, «туда и обратно»."""

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from lxml import etree
from typer.testing import CliRunner

from docai.cli import app
from docai.edo import (
    DEMO_SIGNER,
    EdoError,
    EdoValidationError,
    Signer,
    file_id,
    upd_to_xml,
    validate_xml,
    xml_to_upd,
)
from docai.edo.reader import element_to_upd
from docai.edo.schema import XSD_PATH, parse_xml, xsd_errors
from docai.edo.writer import upd_to_element
from docai.schemas import FIELD_SPEC, DocType, VatRate
from docai.schemas.common import BaseDocument
from docai.schemas.documents import Upd, parse_document
from docai.schemas.fields import get_field_value
from docai.synth.generator import generate_document
from docai.validation.normalize import normalize_for_compare

runner = CliRunner()

# SHA-256 XSD ФНС, скачанной 14.11.2026 (D30). Изменился файл — перечитать D30.
XSD_SHA256 = "9bd06a0c6eaffd33e011efc0a7bf96346bcbf14ea1da9b88049c96a30f1b8221"
CORPUS_SIZE = 100
CORPUS_SEED = 20261114


def _make_corpus(n: int = CORPUS_SIZE, seed: int = CORPUS_SEED) -> list[Upd]:
    """n корректных синтетических УПД (без намеренных ошибок) подряд из серии seed."""
    docs: list[Upd] = []
    index = 0
    while len(docs) < n:
        index += 1
        document = generate_document(seed, index, error_rate=0.0).document
        if isinstance(document, Upd):
            docs.append(document)
    return docs


@pytest.fixture(scope="module")
def corpus() -> list[Upd]:
    return _make_corpus()


def upd(**changes: Any) -> Upd:
    """Небольшой УПД с точечными заменами: `upd(upd_status=2)`, `upd(basis=None)`."""
    data: dict[str, Any] = {
        "doc_type": "upd",
        "number": "17",
        "date": "2026-03-10",
        "upd_status": 1,
        "seller": {
            "name": "ООО «Ромашка»",
            "inn": "7707083893",
            "kpp": "773601001",
            "address": "119019, г. Москва, ул. Воздвиженка, д. 10",
        },
        "buyer": {
            "name": "АО «Лютик»",
            "inn": "7710140679",
            "kpp": "771301001",
            "address": "127287, г. Москва, ул. Хуторская 2-я, д. 38А",
        },
        "lines": [
            {
                "name": "Бумага А4",
                "unit_name": "пач",
                "unit_code": "778",
                "quantity": "10",
                "price": "122.00",
                "amount_without_vat": "1000.00",
                "vat_rate": "22%",
                "vat_amount": "220.00",
                "amount_with_vat": "1220.00",
            }
        ],
        "totals": {
            "amount_without_vat": "1000.00",
            "vat_amount": "220.00",
            "amount_with_vat": "1220.00",
        },
        "basis": {"name": "Договор поставки", "number": "Д-5", "date": "2026-01-09"},
    }
    data.update(changes)
    document = parse_document(data)
    assert isinstance(document, Upd)
    return document


def with_rate(rate: str, vat: str | None) -> Upd:
    line = dict(upd().lines[0].model_dump(mode="json"), vat_rate=rate, vat_amount=vat)
    return upd(lines=[line])


# ---------------------------------------------------------------- схема


def test_xsd_is_the_official_file() -> None:
    assert hashlib.sha256(XSD_PATH.read_bytes()).hexdigest() == XSD_SHA256


def test_vat_rates_match_xsd_enumeration() -> None:
    """Список ставок в VatRate совпадает с перечнем атрибута НалСт в XSD (D31)."""
    schema = etree.parse(str(XSD_PATH))
    xs = "{http://www.w3.org/2001/XMLSchema}"
    attribute = schema.find(f".//{xs}attribute[@name='НалСт']")
    assert attribute is not None
    in_xsd = {e.get("value") for e in attribute.iter(f"{xs}enumeration")}
    assert in_xsd == {rate.value for rate in VatRate}


@pytest.mark.parametrize("rate", list(VatRate))
def test_every_vat_rate_is_accepted_by_xsd_and_round_trips(rate: VatRate) -> None:
    vat = None if rate.is_untaxed else "1.00"
    document = with_rate(rate.value, vat)
    assert xml_to_upd(upd_to_xml(document)) == document


# ------------------------------------------------- 100 документов туда и обратно


def test_corpus_is_diverse(corpus: list[Upd]) -> None:
    """Без этого проверка на 100 документах могла бы пройти на однообразных данных."""
    assert len(corpus) == CORPUS_SIZE
    assert {d.upd_status for d in corpus} == {1, 2}
    assert any(len(d.seller.inn or "") == 12 for d in corpus)  # ИП-продавец
    assert any(len(d.buyer.inn or "") == 12 for d in corpus)  # ИП-покупатель
    assert any(d.basis is None for d in corpus)
    assert any(len(d.lines) > 5 for d in corpus)
    rates = Counter(line.vat_rate for d in corpus for line in d.lines)
    assert {VatRate.R20, VatRate.R22, VatRate.NO_VAT} <= set(rates)
    assert any(line.vat_amount is None for d in corpus for line in d.lines)


def test_100_xml_pass_xsd(corpus: list[Upd]) -> None:
    for document in corpus:
        data = upd_to_xml(document, validate=False)
        assert xsd_errors(parse_xml(data)) == [], f"{document.number} {document.date}"


def test_100_round_trip_is_lossless(corpus: list[Upd]) -> None:
    for document in corpus:
        assert xml_to_upd(upd_to_xml(document)) == document


def test_100_round_trip_by_field_spec(corpus: list[Upd]) -> None:
    """Каждое поле FIELD_SPEC после XML совпадает с исходным: как есть и после нормализации."""
    compared = 0
    for document in corpus:
        back = xml_to_upd(upd_to_xml(document))
        for spec in FIELD_SPEC:
            if not spec.applies_to(DocType.UPD):
                continue
            before, after = get_field_value(document, spec), get_field_value(back, spec)
            assert before == after, f"{spec.name}: {before!r} → {after!r}"
            if spec.is_line_field:
                norm_before = [normalize_for_compare(spec.norm, v) for v in before]
                norm_after = [normalize_for_compare(spec.norm, v) for v in after]
            else:
                norm_before = normalize_for_compare(spec.norm, before)
                norm_after = normalize_for_compare(spec.norm, after)
            assert norm_before == norm_after, spec.name
            compared += 1
    assert compared > 20 * CORPUS_SIZE


def test_xml_is_deterministic(corpus: list[Upd]) -> None:
    assert upd_to_xml(corpus[0]) == upd_to_xml(corpus[0])
    assert len({file_id(d) for d in corpus}) == len(corpus)


# ------------------------------------------------------------ формат файла


def test_file_is_windows_1251_with_required_first_line() -> None:
    data = upd_to_xml(upd())
    assert data.startswith(b'<?xml version="1.0" encoding="windows-1251"?>\n')
    assert not data.startswith(b"\xef\xbb\xbf")  # без BOM
    text = data.decode("windows-1251")
    assert "ООО «Ромашка»" in text and "Договор поставки" in text
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")


def test_characters_outside_windows_1251_survive() -> None:
    document = upd(
        lines=[dict(upd().lines[0].model_dump(mode="json"), name="Резистор 5 Ω ≥ 1 кОм")]
    )
    data = upd_to_xml(document)
    assert xml_to_upd(data).lines[0].name == "Резистор 5 Ω ≥ 1 кОм"


def test_file_id_follows_fns_pattern() -> None:
    value = file_id(upd())
    guid = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    assert re.fullmatch(rf"ON_NSCHFDOPPR_000{guid}_000{guid}_20260310_{guid}_0_0_0_0_0_00", value)
    root = parse_xml(upd_to_xml(upd()))
    assert root.get("ИдФайл") == value
    assert root.get("ВерсФорм") == "5.03"


@pytest.mark.parametrize(("status", "function"), [(1, "СЧФДОП"), (2, "ДОП")])
def test_function_follows_upd_status(status: int, function: str) -> None:
    root = parse_xml(upd_to_xml(upd(upd_status=status)))
    document = root.find("Документ")
    assert document is not None and document.get("Функция") == function
    assert xml_to_upd(upd_to_xml(upd(upd_status=status))).upd_status == status


def test_signer_is_written() -> None:
    signer = Signer("Петров", "Пётр", None, "Директор")
    root = parse_xml(upd_to_xml(upd(), signer))
    sign = root.find("Документ/Подписант")
    assert sign is not None and sign.get("Должн") == "Директор"
    fio = sign.find("ФИО")
    assert fio is not None and fio.get("Фамилия") == "Петров" and fio.get("Отчество") is None
    assert DEMO_SIGNER.position == "Генеральный директор"


def test_individual_entrepreneur_becomes_sv_ip() -> None:
    document = upd(
        seller={
            "name": "ИП Блинова Жанна Альбертовна",
            "inn": "612025664948",
            "kpp": None,
            "address": "839076, г. Билибино",
        }
    )
    root = parse_xml(upd_to_xml(document))
    ip = root.find("Документ/СвСчФакт/СвПрод/ИдСв/СвИП")
    assert ip is not None and ip.get("ИННФЛ") == "612025664948"
    fio = ip.find("ФИО")
    assert fio is not None and fio.get("Имя") == "Жанна"
    assert xml_to_upd(upd_to_xml(document)) == document


def test_no_basis_uses_bez_dok_osn_per() -> None:
    document = upd(basis=None)
    root = parse_xml(upd_to_xml(document))
    assert root.find("Документ/СвПродПер/СвПер/БезДокОснПер") is not None
    assert root.find("Документ/СвПродПер/СвПер/ОснПер") is None
    assert xml_to_upd(upd_to_xml(document)).basis is None


def test_money_is_written_with_two_places_when_value_allows() -> None:
    line = dict(
        upd().lines[0].model_dump(mode="json"), amount_without_vat="1000", vat_amount="220.5"
    )
    root = parse_xml(upd_to_xml(upd(lines=[line])))
    row = root.find("Документ/ТаблСчФакт/СведТов")
    assert row is not None and row.get("СтТовБезНДС") == "1000.00"
    assert row.findtext("СумНал/СумНал") == "220.50"


def test_quantity_keeps_fraction() -> None:
    line = dict(upd().lines[0].model_dump(mode="json"), quantity="2.375")
    document = upd(lines=[line])
    assert xml_to_upd(upd_to_xml(document)).lines[0].quantity == Decimal("2.375")


# --------------------------------------------------------- проверка по XSD


def _tree(document: Upd) -> etree._Element:
    return upd_to_element(document, DEMO_SIGNER)


def test_xsd_rejects_unknown_rate() -> None:
    root = _tree(upd())
    row = root.find("Документ/ТаблСчФакт/СведТов")
    assert row is not None
    row.set("НалСт", "15%")
    errors = xsd_errors(root)
    assert errors and "НалСт" in errors[0]
    with pytest.raises(EdoValidationError, match="НалСт"):
        validate_xml(root)


def test_xsd_rejects_bad_inn_format() -> None:
    root = _tree(upd())
    legal = root.find("Документ/СвСчФакт/СвПрод/ИдСв/СвЮЛУч")
    assert legal is not None
    legal.set("ИННЮЛ", "12345")
    assert any("ИННЮЛ" in e for e in xsd_errors(root))


def test_xsd_rejects_missing_required_attribute() -> None:
    root = _tree(upd())
    invoice = root.find("Документ/СвСчФакт")
    assert invoice is not None
    del invoice.attrib["НомерДок"]
    assert any("НомерДок" in e for e in xsd_errors(root))


def test_xsd_rejects_wrong_date_and_wrong_version() -> None:
    root = _tree(upd())
    invoice = root.find("Документ/СвСчФакт")
    assert invoice is not None
    invoice.set("ДатаДок", "31.02.2026")
    root.set("ВерсФорм", "5.02")
    errors = xsd_errors(root)
    assert any("ДатаДок" in e for e in errors) and any("ВерсФорм" in e for e in errors)


def test_xsd_rejects_money_with_three_decimals() -> None:
    line = dict(upd().lines[0].model_dump(mode="json"), amount_without_vat="1000.001")
    with pytest.raises(EdoValidationError, match="СтТовБезНДС"):
        upd_to_xml(upd(lines=[line]))


def test_validation_can_be_switched_off() -> None:
    line = dict(upd().lines[0].model_dump(mode="json"), amount_without_vat="1000.001")
    assert upd_to_xml(upd(lines=[line]), validate=False)


# ------------------------------------------- что генератор не умеет записать


def test_writer_refuses_missing_vat_rate() -> None:
    line = dict(upd().lines[0].model_dump(mode="json"), vat_rate=None)
    with pytest.raises(EdoError, match="ставк"):
        upd_to_xml(upd(lines=[line]))


def test_writer_refuses_partial_basis() -> None:
    with pytest.raises(EdoError, match="основани"):
        upd_to_xml(upd(basis={"name": "Договор", "number": None, "date": None}))


@pytest.mark.parametrize("inn", ["123", "77070838931", None])
def test_writer_refuses_bad_inn_length(inn: str | None) -> None:
    seller = dict(upd().seller.model_dump(), inn=inn)
    with pytest.raises(EdoError, match="ИНН"):
        upd_to_xml(upd(seller=seller))


@pytest.mark.parametrize("name", ["Блинова Жанна Альбертовна", "ИП Блинова", "ИП  Блинова  Жанна"])
def test_writer_refuses_ip_name_it_cannot_restore(name: str) -> None:
    seller = {"name": name, "inn": "612025664948", "kpp": None, "address": None}
    with pytest.raises(EdoError):
        upd_to_xml(upd(seller=seller))


def test_writer_refuses_totals_without_lines() -> None:
    with pytest.raises(EdoError, match="итоги"):
        upd_to_xml(upd(lines=[]))


# ------------------------------------------------------------- чужие файлы


def _replace(data: bytes, old: str, new: str) -> bytes:
    text = data.decode("windows-1251")
    assert old in text
    return text.replace(old, new, 1).encode("windows-1251")


def test_reader_refuses_other_functions() -> None:
    data = _replace(upd_to_xml(upd()), 'Функция="СЧФДОП"', 'Функция="СЧФ"')
    with pytest.raises(EdoError, match="СЧФ"):
        xml_to_upd(data)


def test_reader_refuses_other_version() -> None:
    data = _replace(upd_to_xml(upd()), 'ВерсФорм="5.03"', 'ВерсФорм="5.02"')
    with pytest.raises(EdoValidationError):
        xml_to_upd(data)
    with pytest.raises(EdoError, match="версия"):
        xml_to_upd(data, validate=False)


def test_reader_refuses_non_xml_and_foreign_root() -> None:
    with pytest.raises(EdoError, match="не XML"):
        xml_to_upd(b"%PDF-1.7 not xml")
    with pytest.raises(EdoError, match="Файл"):
        xml_to_upd(b"<?xml version='1.0'?><Other/>", validate=False)


def test_reader_does_not_resolve_external_entities(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("СЕКРЕТНОЕ-СОДЕРЖИМОЕ", encoding="utf-8")
    evil = (
        f'<?xml version="1.0" encoding="windows-1251"?>\n'
        f'<!DOCTYPE x [<!ENTITY e SYSTEM "{secret.as_uri()}">]>\n<x>&e;</x>'
    ).encode("windows-1251")
    root = parse_xml(evil)
    assert "СЕКРЕТНОЕ" not in etree.tostring(root, encoding="unicode", method="text")
    assert "СЕКРЕТНОЕ" not in (root.text or "")


def _real_world_party(address_xml: str, ident_xml: str) -> bytes:
    """Файл от «чужого» ЭДО: покупатель с нужным адресом и видом идентификатора."""
    root = _tree(upd())
    buyer = root.find("Документ/СвСчФакт/СвПокуп")
    assert buyer is not None
    for child in list(buyer):
        buyer.remove(child)
    buyer.append(etree.fromstring(f"<ИдСв>{ident_xml}</ИдСв>"))
    if address_xml:
        buyer.append(etree.fromstring(f"<Адрес>{address_xml}</Адрес>"))
    return etree.tostring(root, encoding="windows-1251", xml_declaration=True)


def test_reader_builds_address_from_adr_rf() -> None:
    adr = (
        '<АдрРФ Индекс="123112" КодРегион="77" НаимРегион="г Москва" Город="Москва" '
        'Улица="ул Тестовая" Дом="д 5" Корпус="к 2" Кварт="кв 7"/>'
    )
    ident = '<СвЮЛУч НаимОрг="ООО Тест" ИННЮЛ="7707083893" КПП="773601001"/>'
    document = xml_to_upd(_real_world_party(adr, ident))
    assert document.buyer.address == "123112, г Москва, Москва, ул Тестовая, д 5, к 2, кв 7"
    assert document.buyer.kpp == "773601001"


def test_reader_builds_address_from_adr_gar() -> None:
    adr = (
        '<АдрГАР ИдНом="11111111-1111-1111-1111-111111111111" Индекс="123112">'
        "<Регион>77</Регион><НаимРегион>г Москва</НаимРегион>"
        '<ЭлУлДорСети Тип="улица" Наим="Тестовая"/><Здание Тип="дом" Номер="5"/>'
        "</АдрГАР>"
    )
    ident = '<СвЮЛУч НаимОрг="ООО Тест" ИННЮЛ="7707083893"/>'
    document = xml_to_upd(_real_world_party(adr, ident))
    assert document.buyer.address == "123112, г Москва, улица Тестовая, дом 5"


def test_reader_handles_foreign_party_and_individual() -> None:
    foreign = '<СвИнНеУч ИдСтат="ИО" КодСтр="398" НаимСтран="Казахстан" Наим="Некая Компания"/>'
    party = xml_to_upd(_real_world_party("", foreign)).buyer
    assert (party.name, party.inn, party.kpp, party.address) == ("Некая Компания", None, None, None)

    person = '<СвФЛУч ИННФЛ="500100732259"><ФИО Фамилия="Сидоров" Имя="Сидор"/></СвФЛУч>'
    party = xml_to_upd(_real_world_party("", person)).buyer
    assert (party.name, party.inn) == ("Сидоров Сидор", "500100732259")


def test_reader_ignores_extra_blocks() -> None:
    """Грузополучатель, второй подписант и т. п. не мешают: берём только поля схемы."""
    root = _tree(upd())
    invoice = root.find("Документ/СвСчФакт")
    assert invoice is not None
    consignee = etree.fromstring(
        '<ГрузПолуч><ИдСв><СвЮЛУч НаимОрг="Склад" ИННЮЛ="7707083893"/></ИдСв></ГрузПолуч>'
    )
    invoice.insert(2, consignee)  # после СвПрод и ГрузОт, до СвПокуп
    validate_xml(root)
    assert element_to_upd(root) == upd()


# ------------------------------------------------------------------ CLI


def test_cli_export_and_parse_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "json"
    source.mkdir()
    documents = _make_corpus(5, seed=7)
    for i, document in enumerate(documents, start=1):
        payload = document.model_dump(mode="json")
        (source / f"{i:05d}_upd.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    other = generate_non_upd_json()
    (source / "00099_other.json").write_text(
        json.dumps(other.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )

    xml_dir, back_dir = tmp_path / "xml", tmp_path / "back"
    result = runner.invoke(app, ["edo", "export", "--in", str(source), "--out", str(xml_dir)])
    assert result.exit_code == 0, result.output
    assert len(list(xml_dir.glob("*.xml"))) == 5
    result = runner.invoke(app, ["edo", "parse", "--in", str(xml_dir), "--out", str(back_dir)])
    assert result.exit_code == 0, result.output
    for i, document in enumerate(documents, start=1):
        text = (back_dir / f"{i:05d}_upd.json").read_text(encoding="utf-8")
        assert parse_document(text) == document


def generate_non_upd_json() -> BaseDocument:
    index = 0
    while True:
        index += 1
        document = generate_document(7, index, error_rate=0.0).document
        if not isinstance(document, Upd):
            return document


def test_cli_export_reports_failure(tmp_path: Path) -> None:
    source = tmp_path / "json"
    source.mkdir()
    bad = upd().model_dump(mode="json")
    bad["lines"][0]["vat_rate"] = None
    (source / "00001_upd.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    result = runner.invoke(
        app, ["edo", "export", "--in", str(source), "--out", str(tmp_path / "x")]
    )
    assert result.exit_code == 1
    assert "ставк" in result.output
