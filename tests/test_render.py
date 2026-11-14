"""Тесты печатных форм (P4): PDF, разметка bbox, многостраничность, CLI."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from docai.render.engine import (
    VARIANTS,
    Decor,
    Renderer,
    RenderError,
    choose_look,
    expected_fields,
    normalize_text,
    pdf_text,
)
from docai.render.format import (
    amount_in_words,
    date_long,
    date_num,
    money,
    price,
    quantity,
)
from docai.schemas.common import DocType
from docai.schemas.documents import parse_document
from docai.synth.generator import generate_document

ALL_DECOR = Decor(
    logo={"shape": 0, "color": "#1f6feb"},
    stamp={"angle": 5, "color": "#1a3a9c"},
    sign={"kind": 0},
)
NO_DECOR = Decor(logo=None, stamp=None, sign=None)


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as r:
        yield r


def _doc_of_type(doc_type: DocType, min_lines: int = 1, max_lines: int = 99):
    """Первый синтетический документ нужного типа с подходящим числом строк."""
    for index in range(1, 1000):
        item = generate_document(42, index, error_rate=0.0)
        doc = item.document
        if doc.doc_type == doc_type and min_lines <= len(doc.lines) <= max_lines:
            return doc
    raise AssertionError(f"нет документа {doc_type} с {min_lines}..{max_lines} строками")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("12200"), "12 200,00"),
        (Decimal("0.5"), "0,50"),
        (Decimal("1234567.891"), "1 234 567,89"),
        (Decimal("-5"), "-5,00"),
    ],
)
def test_money(value, expected):
    assert money(value) == expected


def test_money_none_and_price_quantity_dates():
    import datetime as dt

    assert money(None) == ""
    assert price(Decimal("46.63")) == "46,63"
    assert price(Decimal("100")) == "100,00"
    assert price(Decimal("1.2345")) == "1,2345"
    assert quantity(Decimal("44")) == "44"
    assert quantity(Decimal("1.500")) == "1,5"
    assert quantity(Decimal("1000")) == "1 000"
    assert quantity(Decimal("100")) == "100"
    assert date_num(dt.date(2026, 1, 5)) == "05.01.2026"
    assert date_long(dt.date(2026, 11, 14)) == "«14» ноября 2026 г."


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0"), "Ноль рублей 00 копеек"),
        (Decimal("1"), "Один рубль 00 копеек"),
        (Decimal("2.01"), "Два рубля 01 копейка"),
        (Decimal("1000"), "Одна тысяча рублей 00 копеек"),
        (Decimal("2000.50"), "Две тысячи рублей 50 копеек"),
        (Decimal("11011"), "Одиннадцать тысяч одиннадцать рублей 00 копеек"),
        (Decimal("3353.64"), "Три тысячи триста пятьдесят три рубля 64 копейки"),
        (Decimal("1000000"), "Один миллион рублей 00 копеек"),
        (Decimal("21000000.12"), "Двадцать один миллион рублей 12 копеек"),
    ],
)
def test_amount_in_words(value, expected):
    assert amount_in_words(value) == expected


def test_choose_look_is_deterministic_and_respects_logo_slots():
    assert choose_look(DocType.BILL, "a", 1) == choose_look(DocType.BILL, "a", 1)
    seen = {choose_look(DocType.UPD, f"k{i}", 3)[0] for i in range(40)}
    assert seen == set(VARIANTS)
    for i in range(60):
        variant, decor = choose_look(DocType.UPD, f"k{i}", 3)
        if variant == 1:
            assert decor.logo is None  # в официальной форме места под логотип нет
    variant, _ = choose_look(DocType.BILL, "x", 1, variant=2)
    assert variant == 2


@pytest.mark.parametrize("doc_type", list(DocType))
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_template_renders_with_all_fields(renderer, doc_type, variant):
    """15 шаблонов: каждый печатает все поля, и их текст есть в текстовом слое PDF."""
    doc = _doc_of_type(doc_type, 2, 6)
    for decor in (ALL_DECOR, NO_DECOR):
        result = renderer.render(doc, variant=variant, decor=decor, source="t.json")
        assert result.pdf.startswith(b"%PDF")
        printed = {e["field"] for e in result.layout["fields"]}
        assert expected_fields(doc) <= printed
        assert result.layout["variant"] == variant
        assert result.layout["decor"]["stamp"] is (decor is ALL_DECOR)


def test_text_layer_has_formatted_values(renderer):
    doc = _doc_of_type(DocType.UPD, 2, 5)
    result = renderer.render(doc, variant=1, decor=NO_DECOR)
    text = normalize_text(" ".join(pdf_text(result.pdf)))
    assert normalize_text(money(doc.totals.amount_with_vat)) in text
    assert doc.seller.inn in text
    assert doc.number in text
    for line in doc.lines:
        assert normalize_text(line.name) in text
        assert normalize_text(money(line.amount_with_vat)) in text
        assert normalize_text(quantity(line.quantity)) in text


def test_bbox_points_to_the_printed_value(renderer):
    """bbox поля, вырезанный из PDF, содержит именно напечатанный там текст."""
    doc = _doc_of_type(DocType.BILL, 2, 5)
    result = renderer.render(doc, variant=2, decor=NO_DECOR)
    width, height = result.layout["page_size_pt"]
    assert (round(width), round(height)) == (595, 842)  # A4 книжная
    pdf = pdfium.PdfDocument(result.pdf)
    checked = 0
    for entry in result.layout["fields"]:
        if not entry["field"].startswith(("lines[", "totals.", "number", "seller.inn")):
            continue
        x0, y0, x1, y1 = entry["bbox"]
        page = pdf[entry["page"] - 1]
        # PDFium: начало внизу слева, ось Y вверх; наши bbox — сверху вниз
        got = page.get_textpage().get_text_bounded(x0 - 1, height - y1 - 1, x1 + 1, height - y0 + 1)
        assert normalize_text(entry["text"]) in normalize_text(got), entry
        checked += 1
    assert checked > 10


def test_landscape_forms_are_landscape(renderer):
    result = renderer.render(_doc_of_type(DocType.TORG12), variant=1, decor=NO_DECOR)
    width, height = result.layout["page_size_pt"]
    assert width > height
    assert (round(width), round(height)) == (842, 595)


@pytest.mark.parametrize("doc_type", list(DocType))
def test_long_table_spans_pages_without_losing_rows(renderer, doc_type):
    """Длинная таблица (20+ строк) идёт на нескольких страницах, ни одна строка не теряется."""
    doc = _doc_of_type(doc_type, 8)
    doc = doc.model_copy(update={"lines": doc.lines * 5})  # 40+ строк: точно больше страницы
    result = renderer.render(doc, variant=2, decor=ALL_DECOR)
    layout = result.layout
    assert layout["page_count"] >= 2
    assert len(pdfium.PdfDocument(result.pdf)) == layout["page_count"]
    names = {e["field"]: e for e in layout["fields"] if e["field"].endswith("].name")}
    assert len(names) == len(doc.lines)
    pages_with_rows = {e["page"] for e in names.values()}
    assert pages_with_rows == set(range(1, max(pages_with_rows) + 1))
    # порядок строк сохраняется: страница и вертикальная позиция не убывают
    order = [
        (names[f"lines[{i}].name"]["page"], names[f"lines[{i}].name"]["bbox"][1])
        for i in range(len(doc.lines))
    ]
    assert order == sorted(order)
    # итоги — на последней странице или раньше, но после последней строки
    totals_page = max(e["page"] for e in layout["fields"] if e["field"] == "totals.amount_with_vat")
    assert totals_page >= order[-1][0]


def test_row_is_not_clipped_by_page_bottom(renderer):
    """Нижняя граница каждого поля лежит внутри страницы (строки таблицы не обрезаются)."""
    doc = _doc_of_type(DocType.UPD, 8)
    doc = doc.model_copy(update={"lines": doc.lines * 4})
    result = renderer.render(doc, variant=1, decor=NO_DECOR)
    _, height = result.layout["page_size_pt"]
    assert all(e["bbox"][3] <= height for e in result.layout["fields"])


def test_missing_field_is_reported(renderer, monkeypatch):
    """Если шаблон не напечатал поле, рендер падает с понятной ошибкой, а не молчит."""
    from docai.render import engine

    doc = _doc_of_type(DocType.ACT, 1, 3)
    real = engine.expected_fields
    monkeypatch.setattr(engine, "expected_fields", lambda d: real(d) | {"seller.fax"})
    with pytest.raises(RenderError, match=r"seller.fax"):
        renderer.render(doc, variant=1, decor=NO_DECOR, source="x.json")


def test_unknown_variant_rejected(renderer):
    with pytest.raises(RenderError):
        renderer.render(_doc_of_type(DocType.ACT), variant=9, decor=NO_DECOR)


def test_missing_seller_bank_and_basis_still_render(renderer):
    """Опциональные блоки (реквизиты банка, основание, КПП ИП) не ломают вёрстку."""
    for doc_type in (DocType.ACT, DocType.TORG12):
        doc = _doc_of_type(doc_type, 1, 3)
        doc = doc.model_copy(update={"seller_bank": None, "basis": None})
        for variant in VARIANTS:
            renderer.render(doc, variant=variant, decor=NO_DECOR)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """CLI запускается отдельным процессом: sync-API Playwright не уживается с чужим event loop."""
    return subprocess.run(
        [sys.executable, "-m", "docai.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_cli_render_writes_pdf_and_layout(tmp_path: Path):
    src = tmp_path / "json"
    src.mkdir()
    for index in (1, 2, 3):
        item = generate_document(42, index, error_rate=0.0)
        (src / item.file_name).write_text(item.document.model_dump_json(), encoding="utf-8")
    (src / "_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    out = tmp_path / "pdf"
    result = _run_cli("render", "--in", str(src), "--out", str(out), "--seed", "5")
    assert result.returncode == 0, result.stdout + result.stderr
    pdfs = sorted(out.glob("*.pdf"))
    assert len(pdfs) == 3
    for pdf in pdfs:
        layout = json.loads(pdf.with_suffix(".layout.json").read_text(encoding="utf-8"))
        assert layout["source"] == pdf.stem + ".json"
        assert layout["page_count"] == len(pdfium.PdfDocument(pdf.read_bytes()))
        assert layout["fields"]
        parse_document((src / layout["source"]).read_text(encoding="utf-8"))


def test_cli_render_empty_folder_fails(tmp_path: Path):
    result = _run_cli("render", "--in", str(tmp_path), "--out", str(tmp_path / "o"))
    assert result.returncode == 1
