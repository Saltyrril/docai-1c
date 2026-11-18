"""P13: анализ ошибок по причинам, тип документа по заголовку, таблица «графа: значение»."""

import re
from pathlib import Path

import pytest

from docai.eval.compare import compare_documents
from docai.eval.config import EvalConfig, ExtractSection
from docai.eval.errors import (
    CAUSES,
    PROMPT_EXAMPLE_VALUES,
    classify_document,
    summarize,
)
from docai.eval.predict import fingerprint_config
from docai.extract import ExtractConfig
from docai.extract.pdftext import render_table, table_to_labeled
from docai.extract.prompts import load_prompt
from docai.postprocess import PostprocessConfig, postprocess
from docai.postprocess.doctype import retype_document, title_type
from docai.schemas.common import DocType
from docai.schemas.documents import parse_document
from factories import SELLER_BANK, doc_data, line

# ---------- анализ ошибок ----------


def _doc(doc_type: str = "upd", **overrides):
    data = doc_data(doc_type, **overrides)
    if doc_type == "upd":
        data.setdefault("upd_status", 1)
    if doc_type == "bill":
        data.setdefault("seller_bank", dict(SELLER_BANK))
    return parse_document(data)


def _causes(truth, pred, text: str = "") -> dict[str, str]:
    cmp = compare_documents(truth, pred)
    items = classify_document("d", cmp, truth, text)
    return {(e.field if e.row is None else f"{e.field}#{e.row}"): e.cause for e in items}


def test_correct_document_has_no_errors() -> None:
    truth = _doc()
    assert classify_document("d", compare_documents(truth, truth), truth, "") == []


def test_wrong_doc_type_and_its_consequences() -> None:
    truth = _doc("upd")
    pred = _doc("vat_invoice")
    causes = _causes(truth, pred)
    assert causes["doc_type"] == "doc_type"
    assert causes["upd_status"] == "doc_type"


def test_postcode_as_inn_is_neighbour_field_and_address_lacks_postcode() -> None:
    seller = {"name": "ООО «Ромашка»", "inn": "7707083893", "kpp": None,
              "address": "123456, г. Москва, ул. Ленина, д. 1"}  # fmt: skip
    truth = _doc("upd", seller=seller)
    wrong = dict(seller, inn="123456", address="г. Москва, ул. Ленина, д. 1")
    pred = _doc("upd", seller=wrong)
    text = "ООО «Ромашка» 123456, г. Москва, ул. Ленина, д. 1 7707083893"
    causes = _causes(truth, pred, text)
    assert causes["seller.inn"] == "neighbour_field"
    assert causes["seller.address"] == "text_postcode"


def test_corr_account_as_account_and_missing_corr() -> None:
    truth = _doc("bill")
    bank = dict(SELLER_BANK, account=SELLER_BANK["corr_account"], corr_account=None)
    pred = _doc("bill", seller_bank=bank)
    causes = _causes(truth, pred, "Сч. № 30101810400000000225 Сч. № 40702810938000000001")
    assert causes["seller_bank.account"] == "neighbour_field"
    assert causes["seller_bank.corr_account"] == "missing"


def test_party_swap() -> None:
    truth = _doc()
    seller, buyer = truth.seller.model_dump(), truth.buyer.model_dump()
    pred = _doc(seller=dict(seller, name=buyer["name"]))
    assert _causes(truth, pred)["seller.name"] == "party_swap"


def test_prompt_example_leak() -> None:
    assert "7707083893" in PROMPT_EXAMPLE_VALUES
    seller = {"name": "ИП Иванов", "inn": "500100732259", "kpp": None, "address": "Москва"}
    truth = _doc(seller=seller)
    pred = _doc(seller=dict(seller, kpp="773601001"))
    assert _causes(truth, pred)["seller.kpp"] == "prompt_leak"


def test_table_causes_shift_fraction_and_rows() -> None:
    lines = [
        line("12.5", "100.00", "1250.00", "22%", "275.00", "1525.00", "Бумага"),
        line("2", "50.00", "100.00", "22%", "22.00", "122.00", "Ручка"),
    ]
    totals = {"amount_without_vat": "1350.00", "vat_amount": "297.00",
              "amount_with_vat": "1647.00"}  # fmt: skip
    truth = _doc(lines=lines, totals=totals)
    shifted = [
        line("12", "100.00", "1250.00", "22%", "275.00", "1525.00", "Бумага"),  # 12,5 → 12
        line("2", "100.00", "100.00", "22%", "22.00", "122.00", "Ручка"),  # цена = сумма
    ]
    pred = _doc(lines=shifted, totals=totals)
    text = "Бумага 12,5 100,00 1 250,00 Ручка 2 50,00 100,00"
    causes = _causes(truth, pred, text)
    assert causes["lines[].quantity#0"] == "decimal_lost"
    assert causes["lines[].price#1"] == "column_shift"
    lost = _doc(lines=lines[:1], totals=totals)
    assert _causes(truth, lost, text)["lines"] == "rows_lost"


def test_row_number_as_quantity_is_column_shift() -> None:
    truth = _doc()
    pred_lines = [line("1", "1000.00", "2000.00", "22%", "440.00", "2440.00", "Бумага"),
                  truth.lines[1].model_dump(mode="json")]  # fmt: skip
    pred = _doc(lines=pred_lines)
    assert _causes(truth, pred, "1 Бумага 2 1 000,00")["lines[].quantity#0"] == "column_shift"


def test_hallucinated_number_not_in_text() -> None:
    truth = _doc()
    lines = [line("2", "1000.00", "2000.00", "22%", "440.00", "2440.00", "Бумага"),
             line("3", "77.77", "99.99", "22%", "22.00", "121.99", "Ручка")]  # fmt: skip
    pred = _doc(lines=lines)
    text = "Бумага 2 1 000,00 2 000,00 Ручка 3 33,33 99,99"
    assert _causes(truth, pred, text)["lines[].price#1"] == "hallucinated"


def test_autofix_breaking_correct_answer_is_separate_cause() -> None:
    truth = _doc()
    raw = compare_documents(truth, truth)
    bad = _doc(lines=[line("2", "1000.00", "2000.00", "22%", "440.00", "2440.00", "Бумага"),
                      line("3", "33.33", "99.99", "22%", "22.00", "122.00", "Ручка")])  # fmt: skip
    items = classify_document("d", compare_documents(truth, bad), truth, "", raw)
    assert {e.cause for e in items} == {"autofix"}


def test_not_extracted() -> None:
    truth = _doc()
    items = classify_document("d", compare_documents(truth, None), truth, "")
    assert [e.cause for e in items] == ["not_extracted"]


def test_summarize_shares_and_ceiling() -> None:
    truth = _doc("upd")
    items = classify_document("a", compare_documents(truth, _doc("vat_invoice")), truth, "")
    seller = truth.seller.model_dump()
    pred_b = _doc("upd", seller=dict(seller, name=truth.buyer.name))
    items += classify_document("b", compare_documents(truth, pred_b), truth, "")
    s = summarize(items, ["a", "b", "c"], ["c"])
    by = {c["cause"]: c for c in s["causes"]}
    assert s["failed"] == 2 and s["ok"] == 1
    assert by["doc_type"]["ceiling_docs"] == 1
    assert by["party_swap"]["ceiling_docs"] == 1
    assert sum(c["share_all"] for c in s["causes"]) == pytest.approx(1.0)
    assert all(len(c["examples"]) <= 3 for c in s["causes"])
    assert set(by) <= set(CAUSES)


# ---------- тип документа по заголовку ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("УПД № 484 от 22.11.2026 Статус: 1\nСчёт-фактура № 484 от 22.11.2026 (1)", ("upd", 1)),
        ("Приложение № 1\nСтатус: 2 к письму ФНС России\nСчёт-фактура № 6510 от 06.02.2025",
         ("upd", 2)),
        ("Универсальный передаточный документ\nСчёт-фактура № 1", ("upd", None)),
        ("Счёт-фактура № 8403 от 09.06.2025 Исправление № -- от --", ("vat_invoice", None)),
        ("Внимание! Оплата счёта...\nСчёт на оплату № КС-1 от 03.06.2025", ("bill", None)),
        ("СЧЁТ № 499\nот 04.03.2026", ("bill", None)),
        ("АКТ № Г-8739\nвыполненных работ (оказанных услуг)", ("act", None)),
        ("Товарная накладная ТОРГ-12 № 2869 от 27.06.2026", ("torg12", None)),
        ("| 0330212 |\n\nГрузоотправитель ООО «Мастер»", ("torg12", None)),
        ("Договор поставки № 1", None),
        ("Счёт на оплату № 1\nАкт № 2", None),  # противоречие — не трогаем
    ],
)  # fmt: skip
def test_title_type(text: str, expected) -> None:
    found = title_type(text)
    if expected is None:
        assert found is None
    else:
        assert (str(found[0]), found[1]) == expected


def test_title_type_looks_only_at_title() -> None:
    assert title_type("x" * 1000 + "\nАкт № 1") is None


def test_retype_vat_invoice_to_upd_keeps_values() -> None:
    doc = _doc("vat_invoice")
    new, corrections = retype_document(doc, ["УПД № 1 от 14.11.2026 Статус: 2\nСчёт-фактура № 1"])
    assert new.doc_type is DocType.UPD and new.upd_status == 2
    assert new.seller == doc.seller and new.lines == doc.lines
    assert [c.field for c in corrections] == ["doc_type", "upd_status"]


def test_retype_upd_to_bill_creates_empty_bank() -> None:
    new, corrections = retype_document(_doc("upd"), ["Счёт на оплату № 1 от 14.11.2026"])
    assert new.doc_type is DocType.BILL
    assert new.seller_bank is not None
    assert corrections[0].rule == "doc_type_by_title"


def test_retype_fixes_upd_status_only() -> None:
    new, corrections = retype_document(_doc("upd"), ["УПД № 1 Статус: 2"])
    assert new.upd_status == 2 and [c.field for c in corrections] == ["upd_status"]


def test_retype_keeps_document_without_title() -> None:
    doc = _doc("upd")
    assert retype_document(doc, ["Договор"]) == (doc, [])
    assert retype_document(doc, []) == (doc, [])


def test_postprocess_title_rule_is_default_and_can_be_off() -> None:
    doc = _doc("vat_invoice")
    texts = [("text_layer", "УПД № 123 от 14.11.2026 Статус: 1\nСчёт-фактура № 123")]
    off = postprocess(doc, route="text", texts=texts,
                      config=PostprocessConfig(doc_type_by_title=False))  # fmt: skip
    on = postprocess(doc, route="text", texts=texts)
    assert off.document.doc_type is DocType.VAT_INVOICE
    assert on.document.doc_type is DocType.UPD
    assert any(c.rule == "doc_type_by_title" for c in on.report.corrections)
    xml = postprocess(doc, route="xml", texts=texts,
                      config=PostprocessConfig(doc_type_by_title=True))  # fmt: skip
    assert xml.document.doc_type is DocType.VAT_INVOICE


# ---------- таблица «графа: значение» ----------

UPD_TABLE = [
    ["№ п/п", "Наименование товара", "Единица измерения", None, "Количе ство (объём)", "Цена"],
    [None, None, "код", "условное обозначени е (нац.)", None, None],
    ["А", "1", "2", "2а", "3", "4"],
    ["1", "Герметик", "796", "шт", "25", "295,49"],
    ["2", "Папка", "--", "шт", "172", "11,36"],
    ["Всего к оплате (9)", None, None, None, None, "306,85"],
]


def test_labeled_table_merges_header_and_drops_column_numbers() -> None:
    text = table_to_labeled(UPD_TABLE)
    assert text is not None
    lines = text.splitlines()
    assert lines[0] == (
        "- № п/п: 1; Наименование товара: Герметик; Единица измерения / код: 796; "
        "Единица измерения / условное обозначени е: шт; Количе ство: 25; Цена: 295,49"
    )
    assert "--" not in lines[1] and "Единица измерения / код" not in lines[1]
    assert lines[2] == "- № п/п: Всего к оплате (9); Цена: 306,85"
    assert not any(re.search(r"\b1а\b|: А\b", ln) for ln in lines)


def test_labeled_falls_back_to_grid_without_header() -> None:
    bank = [["Банк получателя АО «Банк»", None, "БИК", "049331500"],
            [None, None, "Сч. №", "30101810100000000500"]]  # fmt: skip
    assert table_to_labeled(bank) is None
    assert render_table(bank, "labeled") == render_table(bank, "grid")
    assert render_table(UPD_TABLE, "grid").startswith("| № п/п |")


# ---------- конфиги и промпт ----------


def test_new_options_keep_old_cache_fingerprint() -> None:
    base = fingerprint_config(ExtractConfig())
    assert "text_tables" not in base
    assert "text_tables" in fingerprint_config(ExtractConfig(text_tables="labeled"))
    with_rule = ExtractConfig(postprocess=PostprocessConfig(doc_type_by_title=True))
    assert fingerprint_config(with_rule) == base


def test_eval_config_overrides_postprocess_and_tables() -> None:
    section = ExtractSection(config=None, postprocess={"doc_type_by_title": True},
                             text_tables="labeled")  # fmt: skip
    built = section.build()
    assert built.postprocess.doc_type_by_title is True
    assert built.postprocess.autofix is True
    assert built.text_tables == "labeled"
    EvalConfig(name="x", extract=section)


def test_text_v3_has_no_real_requisites_in_examples() -> None:
    prompt = load_prompt("text_v3")
    assert not re.search(r"\d{9,}", prompt)
    for marker in ("{source}", "{source_title}", "{document_text}", "30101", "почтовый индекс"):
        assert marker in prompt


def test_eval_configs_p13_parse() -> None:
    for path in sorted(Path("configs/eval").glob("p13_*.yaml")):
        from docai.eval.config import load_eval_config

        cfg = load_eval_config(path)
        assert cfg.data.split in ("val", "test"), path


def test_labeled_v2_marks_positions_and_totals() -> None:
    text = render_table(UPD_TABLE, "labeled_v2")
    lines = text.splitlines()
    assert lines[0].startswith("Позиция 1: Наименование товара: Герметик;")
    assert lines[1].startswith("Позиция 2: ")
    assert lines[2] == "Итоговая строка «Всего к оплате (9)»: Цена: 306,85"
    assert not any(line.startswith("- ") for line in lines)
