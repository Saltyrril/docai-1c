"""Тесты P8: форма разметки, хранилище эталонов, логика интерфейса, обезличивание."""

import json
import os
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from docai.cli import app as cli_app
from docai.labeling.app import (
    LabelingApp,
    build_demo,
    disable_telemetry,
    section_of,
)
from docai.labeling.form import (
    HEADER_SPECS,
    LINE_COLUMNS,
    LINE_HEADERS,
    FormError,
    build_document,
    document_to_form,
    empty_lines,
)
from docai.labeling.prefill import PrefillUnavailable, prefill_document
from docai.labeling.pseudonymize import (
    LeakError,
    Pseudonymizer,
    load_salt,
    pseudonymize_dir,
)
from docai.labeling.store import LabelStore, StoreError
from docai.schemas import FIELD_SPEC, DocType
from docai.schemas.documents import parse_document
from docai.synth.generator import generate_document
from docai.validation import has_errors, validate_document

runner = CliRunner()
SALT = "unit-test-salt-0123456789abcdef"
PDF_SAMPLE = Path("reports/2026-11-14_p4_render_examples/00023_bill.pdf")


def one_of_each_type(seed: int = 5) -> dict[DocType, object]:
    found: dict[DocType, object] = {}
    index = 0
    while len(found) < len(DocType):
        index += 1
        document = generate_document(seed, index, error_rate=0.0).document
        found.setdefault(document.doc_type, document)
    return found


def fill_form(doc):
    kind, header, lines = document_to_form(doc)
    return kind, header, lines


# ---- форма -----------------------------------------------------------------------------


def test_form_covers_every_field_spec() -> None:
    """Форма и таблица строятся из FIELD_SPEC: ни одно поле не потеряно."""
    names = {spec.name for spec in HEADER_SPECS} | {f"lines[].{c}" for c in LINE_COLUMNS}
    assert names | {"doc_type"} == {spec.name for spec in FIELD_SPEC}
    assert len(LINE_HEADERS) == len(LINE_COLUMNS)


@pytest.mark.parametrize("kind", list(DocType))
def test_round_trip_document_form_document(kind: DocType) -> None:
    original = one_of_each_type()[kind]
    rebuilt = build_document(*fill_form(original))
    assert rebuilt.model_dump(mode="json") == original.model_dump(mode="json")


def test_build_document_accepts_human_formatting() -> None:
    original = one_of_each_type()[DocType.BILL]
    kind, header, lines = fill_form(original)
    header["seller.inn"] = "ИНН " + header["seller.inn"][:4] + " " + header["seller.inn"][4:]
    header["date"] = "3 июня 2025 г."
    header["totals.amount_with_vat"] = "12 200 руб. 50 коп."
    rebuilt = build_document(kind, header, lines)
    assert rebuilt.seller.inn == original.seller.inn
    assert rebuilt.date.isoformat() == "2025-06-03"
    assert str(rebuilt.totals.amount_with_vat) == "12200.50"


def test_build_document_drops_empty_optional_blocks_and_rows() -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.ACT])
    for name in header:
        if name.startswith(("basis.", "seller_bank.")):
            header[name] = ""
    rebuilt = build_document(kind, header, [*lines, *empty_lines(2)])
    assert rebuilt.basis is None
    assert rebuilt.seller_bank is None
    assert len(rebuilt.lines) == len(lines)


def test_build_document_bill_keeps_bank_block() -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.BILL])
    for name in header:
        if name.startswith("seller_bank."):
            header[name] = ""
    assert build_document(kind, header, lines).seller_bank is not None


def test_build_document_reports_all_problems_in_russian() -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.UPD])
    header["date"] = "тридцатое февраля"
    header["totals.vat_amount"] = "много"
    lines[0][LINE_COLUMNS.index("vat_rate")] = "семь"
    with pytest.raises(FormError) as caught:
        build_document(kind, header, lines)
    text = "\n".join(caught.value.messages)
    assert "Дата документа" in text
    assert "Итого НДС" in text
    assert "Строка 1" in text
    assert "ставка НДС" in text


def test_build_document_reports_missing_required_fields() -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.UPD])
    header["number"] = ""
    header["seller.name"] = ""
    header["upd_status"] = ""
    with pytest.raises(FormError) as caught:
        build_document(kind, header, lines)
    text = "\n".join(caught.value.messages)
    assert "«Номер документа»: не заполнено" in text
    assert "«Продавец: наименование»: не заполнено" in text
    assert "«Статус УПД»: не заполнено" in text


def test_build_document_needs_doc_type() -> None:
    with pytest.raises(FormError, match="тип документа"):
        build_document("", {}, [])


def test_nan_cells_from_table_are_empty() -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.TORG12])
    lines[0][LINE_COLUMNS.index("unit_name")] = float("nan")
    lines[0][LINE_COLUMNS.index("unit_code")] = None
    rebuilt = build_document(kind, header, lines)
    assert rebuilt.lines[0].unit_name is None
    assert rebuilt.lines[0].unit_code is None


def test_sections_cover_all_header_fields() -> None:
    titles = {section_of(spec) for spec in HEADER_SPECS}
    assert titles == {"Шапка", "Продавец", "Покупатель", "Банк продавца", "Основание", "Итоги"}


# ---- хранилище -------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    files = tmp_path / "files"
    files.mkdir()
    Image.new("RGB", (300, 200), "white").save(files / "photo_1.jpg")
    pages = files / "multi"
    pages.mkdir()
    for n in (1, 2, 10):
        Image.new("RGB", (100, 100), "white").save(pages / f"{n}.png")
    return tmp_path


@pytest.fixture
def store(root: Path) -> LabelStore:
    return LabelStore(root)


def test_list_sources_kinds_and_pages(store: LabelStore) -> None:
    sources = {s.id: s for s in store.list_sources()}
    assert set(sources) == {"photo_1", "multi"}
    assert sources["photo_1"].kind == "image"
    assert sources["multi"].kind == "folder"
    assert store.page_count(sources["multi"]) == 3
    assert store.render_page(sources["multi"], 2).size == (100, 100)


@pytest.mark.skipif(not PDF_SAMPLE.is_file(), reason="нет примера PDF из P4")
def test_pdf_source_is_rendered(store: LabelStore) -> None:
    store.add_files([PDF_SAMPLE])
    source = store.get_source("00023_bill")
    assert source.kind == "pdf"
    assert store.page_count(source) == 1
    assert max(store.render_page(source, 0).size) > 1000


def test_exif_rotation_is_applied(root: Path, store: LabelStore) -> None:
    image = Image.new("RGB", (300, 200), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # повернуть на 90° по часовой
    image.save(root / "files" / "rot.jpg", exif=exif)
    assert store.render_page(store.get_source("rot")).size == (200, 300)


def test_duplicate_source_names_are_rejected(root: Path, store: LabelStore) -> None:
    Image.new("RGB", (10, 10)).save(root / "files" / "photo_1.png")
    with pytest.raises(StoreError, match="одним именем"):
        store.list_sources()


def test_add_files_never_overwrites(tmp_path: Path, store: LabelStore) -> None:
    incoming = tmp_path / "incoming" / "photo_1.jpg"
    incoming.parent.mkdir()
    Image.new("RGB", (10, 10), "black").save(incoming)
    (tmp_path / "incoming" / "notes.txt").write_text("не документ", encoding="utf-8")
    added = store.add_files([incoming, incoming.with_name("notes.txt")])
    assert added == ["photo_1_2"]
    assert store.render_page(store.get_source("photo_1")).size == (300, 200)  # не затёрт
    assert store.render_page(store.get_source("photo_1_2")).size == (10, 10)


def test_save_and_load_label(store: LabelStore) -> None:
    document = one_of_each_type()[DocType.UPD]
    path = store.save("photo_1", document, {"capture": "скан", "seconds": 12.5})
    assert path == store.labels_dir / "photo_1.json"
    assert parse_document(path.read_text(encoding="utf-8")) == document
    assert store.load_label("photo_1") == document
    assert store.load_meta("photo_1")["capture"] == "скан"
    assert store.is_labeled("photo_1")
    assert store.label_ids() == ["photo_1"]
    assert store.progress() == (1, 2)
    # В эталоне только поля схемы: служебное лежит рядом.
    assert "capture" not in path.read_text(encoding="utf-8")
    assert not list(store.labels_dir.glob(".*.tmp"))


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "_meta"])
def test_unsafe_ids_are_rejected(store: LabelStore, bad: str) -> None:
    with pytest.raises(StoreError):
        store.is_labeled(bad)


# ---- логика интерфейса -----------------------------------------------------------------


@pytest.fixture
def lapp(store: LabelStore) -> LabelingApp:
    return LabelingApp(store)


def test_choices_mark_labeled(lapp: LabelingApp, store: LabelStore) -> None:
    store.save("photo_1", one_of_each_type()[DocType.ACT])
    assert lapp.choices() == [("• multi", "multi"), ("✓ photo_1", "photo_1")]
    assert lapp.progress_text() == "Размечено **1** из **2**"


def test_navigation(lapp: LabelingApp, store: LabelStore) -> None:
    assert lapp.next_unlabeled(None) == "multi"
    assert lapp.neighbour("multi", +1) == "photo_1"
    assert lapp.neighbour("photo_1", +1) == "multi"
    store.save("multi", one_of_each_type()[DocType.ACT])
    assert lapp.next_unlabeled("multi") == "photo_1"
    store.save("photo_1", one_of_each_type()[DocType.ACT])
    assert lapp.next_unlabeled("multi") is None


def test_load_fresh_and_labeled(lapp: LabelingApp) -> None:
    fresh = lapp.load("photo_1")
    assert fresh.doc_type is None
    assert fresh.lines == empty_lines()
    assert fresh.pages == 1
    document = one_of_each_type()[DocType.BILL]
    kind, header, lines = fill_form(document)
    assert lapp.save("photo_1", kind, "фото", "криво", False, header, lines, 90).ok
    again = lapp.load("photo_1")
    assert again.doc_type == "bill"
    assert again.header["number"] == document.number
    assert again.capture == "фото"
    assert again.note == "криво"
    assert again.seconds_before == 90


def test_save_blocked_by_schema_errors(lapp: LabelingApp, store: LabelStore) -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.BILL])
    header["number"] = ""
    result = lapp.save("photo_1", kind, None, "", False, header, lines)
    assert not result.ok
    assert "Номер документа" in result.message
    assert not store.is_labeled("photo_1")


def test_save_blocked_by_validation_errors_unless_accepted(
    lapp: LabelingApp, store: LabelStore
) -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.BILL])
    header["seller.inn"] = "7707083894"  # неверная контрольная цифра
    blocked = lapp.save("photo_1", kind, None, "", False, header, lines)
    assert not blocked.ok
    assert "Так в документе" in blocked.message
    assert not store.is_labeled("photo_1")

    saved = lapp.save("photo_1", kind, None, "", True, header, lines, 30)
    assert saved.ok
    meta = store.load_meta("photo_1")
    assert meta["accepted_errors"] is True
    assert any("INN_CHECKSUM" in item for item in meta["issues"])
    # Эталон остался ровно таким, как напечатано в документе.
    assert store.load_label("photo_1").seller.inn == "7707083894"
    assert has_errors(validate_document(store.load_label("photo_1")))


def test_save_accumulates_seconds(lapp: LabelingApp, store: LabelStore) -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.ACT])
    lapp.save("photo_1", kind, None, "", False, header, lines, 100)
    lapp.save("photo_1", kind, None, "", False, header, lines, 20)
    assert store.load_meta("photo_1")["seconds"] == 120


def test_save_without_document_selected(lapp: LabelingApp) -> None:
    assert not lapp.save(None, "bill", None, "", False, {}, []).ok


def test_check_does_not_save(lapp: LabelingApp, store: LabelStore) -> None:
    kind, header, lines = fill_form(one_of_each_type()[DocType.TORG12])
    assert "✅" in lapp.check(kind, header, lines)
    assert "❌" in lapp.check("", header, lines)
    assert not store.is_labeled("photo_1")


def test_prefill_is_stub_until_p9(lapp: LabelingApp) -> None:
    loaded, message = lapp.prefill("photo_1")
    assert loaded is None
    assert "P9" in message
    with pytest.raises(PrefillUnavailable):
        prefill_document([])


def test_demo_builds_and_telemetry_is_off(lapp: LabelingApp) -> None:
    disable_telemetry()
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert build_demo(lapp) is not None


# ---- обезличивание ---------------------------------------------------------------------


def real_like_bill():
    return parse_document(
        {
            "doc_type": "bill",
            "number": "17",
            "date": "2026-03-04",
            "seller": {
                "name": "ООО «Ромашка»",
                "inn": "7707083893",
                "kpp": "770701001",
                "address": "г. Москва, ул. Тверская, д. 1",
            },
            "buyer": {"name": "ИП Петров Пётр Петрович", "inn": "500100732259"},
            "seller_bank": {
                "bank_name": "ПАО Сбербанк",
                "bik": "044525225",
                "corr_account": "30101810400000000225",
                "account": "40702810138250123017",
            },
            "basis": {"name": "Договор", "number": "Д-2026/55", "date": "2026-01-10"},
            "lines": [
                {
                    "name": "Услуга",
                    "quantity": "1",
                    "price": "100.00",
                    "amount_with_vat": "100.00",
                }
            ],
            "totals": {"amount_with_vat": "100.00"},
        }
    )


def test_pseudonyms_are_stable_and_hide_originals() -> None:
    doc = real_like_bill()
    first = Pseudonymizer(SALT).document(doc)
    second = Pseudonymizer(SALT).document(doc)
    assert first == second
    other_salt = Pseudonymizer("another-salt-0123456789abcdef").document(doc)
    assert other_salt.seller.inn != first.seller.inn

    text = json.dumps(first.model_dump(mode="json"), ensure_ascii=False)
    for secret in (
        "Ромашка",
        "7707083893",
        "770701001",
        "Тверская",
        "Петров",
        "500100732259",
        "044525225",
        "40702810138250123017",
        "30101810400000000225",
        "Сбербанк",
        "Д-2026/55",
    ):
        assert secret not in text


def test_pseudonyms_keep_shape_and_stay_valid() -> None:
    result = Pseudonymizer(SALT).document(real_like_bill())
    assert result.seller.name.startswith("ООО «Контрагент-")
    assert result.buyer.name.startswith("ИП Контрагент-")
    assert len(result.seller.inn) == 10
    assert len(result.buyer.inn) == 12
    assert len(result.seller.kpp) == 9
    # Сумма, даты, номер и строки не тронуты.
    assert result.number == "17"
    assert result.date.isoformat() == "2026-03-04"
    assert result.totals.amount_with_vat == real_like_bill().totals.amount_with_vat
    # ИНН, БИК, счета остались настоящими по контрольным цифрам.
    codes = {i.code.value for i in validate_document(result)}
    assert not codes & {"INN_CHECKSUM", "INN_LENGTH", "ACCOUNT_KEY", "BIK_FORMAT"}
    assert "CORR_ACCOUNT_KEY" not in codes
    assert "CORR_ACCOUNT_BIK_MISMATCH" not in codes


def test_same_party_gets_same_pseudonym_in_every_document() -> None:
    pseudo = Pseudonymizer(SALT)
    a = pseudo.document(real_like_bill())
    b = pseudo.document(real_like_bill())
    assert a.seller == b.seller
    assert a.seller.inn != a.buyer.inn


def test_leak_is_detected() -> None:
    doc = real_like_bill()
    data = doc.model_dump(mode="json")
    data["lines"][0]["name"] = "Услуга для ООО «Ромашка»"
    with pytest.raises(LeakError):
        Pseudonymizer(SALT).document(parse_document(data))


def test_salt_is_created_once(tmp_path: Path) -> None:
    first = load_salt(tmp_path)
    assert len(first) >= 32
    assert load_salt(tmp_path) == first
    assert (tmp_path / ".pseudo_salt").is_file()


def test_pseudonymize_dir_hides_file_names(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    doc = real_like_bill()
    (labels / "счёт_Ромашка_март.json").write_text(
        json.dumps(doc.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    (labels / "_meta").mkdir()
    out = tmp_path / "anon"
    written, problems = pseudonymize_dir(labels, out, SALT)
    assert problems == []
    assert len(written) == 1
    assert written[0].name.startswith("bill_")
    assert "Ромашка" not in written[0].name
    again, problems = pseudonymize_dir(labels, out, SALT)
    assert again == [] and "уже есть" in problems[0]
    forced, problems = pseudonymize_dir(labels, out, SALT, force=True)
    assert len(forced) == 1 and problems == []


def test_pseudonymize_dir_reports_leaks_without_writing(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    data = real_like_bill().model_dump(mode="json")
    data["lines"][0]["name"] = "Работы для 7707083893"
    (labels / "a.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    written, problems = pseudonymize_dir(labels, tmp_path / "anon", SALT)
    assert written == []
    assert len(problems) == 1
    assert "7707083893" not in problems[0]


# ---- командная строка ------------------------------------------------------------------


def test_cli_help_lists_label_command() -> None:
    result = runner.invoke(cli_app, ["label", "--help"])
    assert result.exit_code == 0
    assert "anonymize" in result.output


def test_cli_anonymize(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "x.json").write_text(
        json.dumps(real_like_bill().model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    out = tmp_path / "anon"
    result = runner.invoke(
        cli_app,
        ["label", "anonymize", "--in", str(labels), "--out", str(out), "--root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    (file,) = out.glob("*.json")
    assert "Ромашка" not in file.read_text(encoding="utf-8")


def test_cli_anonymize_missing_folder(tmp_path: Path) -> None:
    result = runner.invoke(
        cli_app, ["label", "anonymize", "--in", str(tmp_path / "нет"), "--root", str(tmp_path)]
    )
    assert result.exit_code == 1
