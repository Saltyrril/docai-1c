"""P9: маршрутизация, черновик для LLM, конвейер извлечения (Ollama подменена), CLI."""

import copy
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from docai.cli import app
from docai.edo import upd_to_xml
from docai.extract import ExtractConfig, Mode, Route, RoutingError, choose_route, extract
from docai.extract import ocr as ocr_module
from docai.extract import pipeline as pipeline_module
from docai.extract.draft import (
    Draft,
    draft_json_schema,
    draft_to_document,
    merge_drafts,
    parse_draft,
)
from docai.extract.images import fit_long_side
from docai.extract.ollama import ChatResult, build_chat_request
from docai.extract.pdftext import pdf_pages_text, table_to_text
from docai.extract.prompts import load_prompt, render_prompt
from docai.extract.router import InputKind, route_inputs, sniff_kind
from docai.extract.vram import VramMonitor
from docai.schemas import DocType
from docai.schemas.documents import Upd, parse_document
from factories import doc_data

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "reports" / "2026-11-14_p4_render_examples"
TEXT_PDF = EXAMPLES / "00023_bill.pdf"  # PDF из Playwright: текстовый слой есть
TWO_PAGE_PDF = EXAMPLES / "00080_torg12.pdf"


# ---------- данные для тестов ----------


def draft_data(**changes: Any) -> dict[str, Any]:
    """Ответ модели в формате черновика: все ключи на месте, отсутствующее — null."""
    party = {"name": "ПАО Сбербанк", "inn": "7707083893", "kpp": "773601001", "address": "Москва"}
    data: dict[str, Any] = {
        "doc_type": "upd",
        "upd_status": 1,
        "number": "123",
        "date": "2026-11-14",
        "seller": party,
        "buyer": {**party, "name": "АО «ТБанк»", "inn": "7710140679", "kpp": "771301001"},
        "seller_bank": None,
        "basis": {"name": "Договор", "number": "7", "date": "2026-01-10"},
        "lines": [
            {
                "name": "Бумага",
                "unit_name": "шт",
                "unit_code": "796",
                "quantity": 2,
                "price": 1000.00,
                "amount_without_vat": 2000.00,
                "vat_rate": "22%",
                "vat_amount": 440.00,
                "amount_with_vat": 2440.00,
            }
        ],
        "totals": {"amount_without_vat": 2000.00, "vat_amount": 440.00, "amount_with_vat": 2440.0},
    }
    data.update(changes)
    return data


def answer(content: dict[str, Any] | str, **changes: Any) -> ChatResult:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    fields: dict[str, Any] = {
        "content": text,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "done_reason": "stop",
        "total_s": 1.0,
        "load_s": 0.1,
        "prompt_eval_s": 0.2,
        "eval_s": 0.7,
    }
    fields.update(changes)
    return ChatResult(**fields)


class FakeOllama:
    """Подмена `chat`: запоминает запросы и отдаёт заготовленные ответы по очереди."""

    def __init__(self, *answers: ChatResult) -> None:
        self.answers = list(answers)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, config: ExtractConfig, body: dict[str, Any]) -> ChatResult:
        self.bodies.append(body)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """В тестах nvidia-smi не вызывается: результат не зависит от машины."""
    monkeypatch.setattr("docai.extract.vram.shutil.which", lambda name: None)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    fake = FakeOllama(answer(draft_data()))
    monkeypatch.setattr(pipeline_module, "chat", fake)
    return fake


@pytest.fixture
def scan_pdf(tmp_path: Path) -> Path:
    """PDF без текстового слоя: две страницы-картинки."""
    pages = [Image.new("RGB", (1654, 2339), "white"), Image.new("RGB", (1654, 2339), "white")]
    path = tmp_path / "scan.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=200)
    return path


@pytest.fixture
def png(tmp_path: Path) -> Path:
    path = tmp_path / "photo.png"
    Image.new("RGB", (3000, 2000), "white").save(path)
    return path


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    upd = parse_document(doc_data("upd", upd_status=1))
    assert isinstance(upd, Upd)
    path = tmp_path / "upd.xml"
    path.write_bytes(upd_to_xml(upd))
    return path


# ---------- определение типа файла ----------


@pytest.mark.parametrize(
    ("head", "kind"),
    [
        (b"%PDF-1.7\n", InputKind.PDF),
        (b"\x89PNG\r\n\x1a\n\x00\x00", InputKind.IMAGE),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", InputKind.IMAGE),
        (b"II*\x00\x08\x00", InputKind.IMAGE),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", InputKind.IMAGE),
        (b'<?xml version="1.0" encoding="windows-1251"?>', InputKind.XML),
        (b"\xef\xbb\xbf  <?xml version='1.0'?>", InputKind.XML),
        ("<Файл ИдФайл=".encode("cp1251"), InputKind.XML),
    ],
)
def test_sniff_kind(head: bytes, kind: InputKind) -> None:
    assert sniff_kind(head) is kind


def test_sniff_kind_rejects_unknown() -> None:
    with pytest.raises(RoutingError, match="неизвестный формат"):
        sniff_kind(b"PK\x03\x04 zip")


# ---------- решение о маршруте (чистая функция) ----------


@pytest.mark.parametrize(
    ("kind", "mode", "chars", "route"),
    [
        (InputKind.XML, Mode.AUTO, (), Route.XML),
        (InputKind.PDF, Mode.AUTO, (500, 800), Route.TEXT),
        (InputKind.PDF, Mode.AUTO, (500, 0), Route.VISION),  # одна страница — скан
        (InputKind.PDF, Mode.AUTO, (0,), Route.VISION),
        (InputKind.PDF, Mode.AUTO, (29,), Route.VISION),  # ниже порога 30
        (InputKind.PDF, Mode.AUTO, (30,), Route.TEXT),
        (InputKind.IMAGE, Mode.AUTO, (), Route.VISION),
        (InputKind.PDF, Mode.TEXT, (500,), Route.TEXT),
        (InputKind.PDF, Mode.VISION, (500,), Route.VISION),
        (InputKind.PDF, Mode.OCR, (500,), Route.OCR),
        (InputKind.PDF, Mode.OCR, (0,), Route.OCR),
        (InputKind.IMAGE, Mode.VISION, (), Route.VISION),
        (InputKind.IMAGE, Mode.OCR, (), Route.OCR),
    ],
)
def test_choose_route(kind: InputKind, mode: Mode, chars: tuple[int, ...], route: Route) -> None:
    decision = choose_route(kind, mode, chars, min_text_chars=30)
    assert decision.route is route
    assert decision.kind is kind
    assert decision.reason


@pytest.mark.parametrize(
    ("kind", "mode", "chars", "message"),
    [
        (InputKind.XML, Mode.TEXT, (), "парсером ЭДО"),
        (InputKind.XML, Mode.VISION, (), "парсером ЭДО"),
        (InputKind.XML, Mode.OCR, (), "парсером ЭДО"),
        (InputKind.IMAGE, Mode.TEXT, (), "только с PDF"),
        (InputKind.PDF, Mode.TEXT, (0, 0), "нет текстового слоя"),
        (InputKind.PDF, Mode.TEXT, (), "нет текстового слоя"),
    ],
)
def test_choose_route_rejects(
    kind: InputKind, mode: Mode, chars: tuple[int, ...], message: str
) -> None:
    with pytest.raises(RoutingError, match=message):
        choose_route(kind, mode, chars)


# ---------- маршрутизация настоящих файлов ----------


def test_route_text_pdf() -> None:
    decision = route_inputs([TEXT_PDF], Mode.AUTO)
    assert decision.route is Route.TEXT
    assert all(c > 100 for c in decision.text_chars)


def test_route_scan_pdf(scan_pdf: Path) -> None:
    decision = route_inputs([scan_pdf], Mode.AUTO)
    assert decision.route is Route.VISION
    assert decision.text_chars == (0, 0)


def test_route_image(png: Path) -> None:
    assert route_inputs([png], Mode.AUTO).route is Route.VISION


def test_route_xml(xml_file: Path) -> None:
    assert route_inputs([xml_file], Mode.AUTO).route is Route.XML


def test_route_ignores_extension(tmp_path: Path) -> None:
    """JPEG с расширением .pdf идёт как картинка: тип определяется по содержимому."""
    path = tmp_path / "scan.pdf"
    Image.new("RGB", (100, 100), "white").save(path, format="JPEG")
    assert route_inputs([path], Mode.AUTO).route is Route.VISION


def test_route_several_images(png: Path, tmp_path: Path) -> None:
    second = tmp_path / "p2.jpg"
    Image.new("RGB", (100, 100), "white").save(second)
    assert route_inputs([png, second], Mode.AUTO).route is Route.VISION


def test_route_rejects_pdf_among_pages(png: Path) -> None:
    with pytest.raises(RoutingError, match="только как картинки"):
        route_inputs([png, TEXT_PDF], Mode.AUTO)


def test_route_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RoutingError, match="не найден"):
        route_inputs([tmp_path / "nope.pdf"], Mode.AUTO)


# ---------- черновик для LLM ----------


def _walk(node: Any) -> list[dict[str, Any]]:
    found = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_walk(value))
    return found


def test_draft_schema_is_grammar_friendly() -> None:
    """Каждый объект требует все свои ключи; без title/description и регулярных выражений."""
    schema = draft_json_schema()
    objects = [n for n in _walk(schema) if n.get("type") == "object"]
    assert objects
    for node in objects:
        assert set(node["required"]) == set(node["properties"])
    text = json.dumps(schema)
    assert '"pattern"' not in text
    assert '"title"' not in text
    assert '"description"' not in text


def test_parse_draft_reads_decimals_exactly() -> None:
    raw = json.loads(json.dumps(draft_data()), parse_float=Decimal)
    draft, warnings = parse_draft(raw)
    assert warnings == []
    assert draft.totals.vat_amount == Decimal("440.0")


def test_parse_draft_nulls_impossible_values() -> None:
    raw = draft_data(date="2025-02-30", upd_status=3)
    original = copy.deepcopy(raw)
    draft, warnings = parse_draft(raw)
    assert draft.date is None
    assert draft.upd_status is None
    assert len(warnings) == 2
    assert raw == original  # исходный ответ не меняется


def test_merge_drafts_across_pages() -> None:
    first = Draft.model_validate(
        draft_data(totals={"amount_without_vat": None, "vat_amount": None, "amount_with_vat": None})
    )
    second_data = draft_data(number=None, date="2026-12-31", doc_type="bill")
    second_data["seller"] = {"name": None, "inn": None, "kpp": None, "address": "Казань"}
    second_data["lines"][0]["name"] = "Ручка"
    second_data["totals"]["amount_with_vat"] = 9999
    merged = merge_drafts([first, Draft.model_validate(second_data)])
    assert merged.doc_type is DocType.UPD  # тип — с первой страницы
    assert merged.number == "123"
    assert str(merged.date) == "2026-11-14"  # шапка — первое непустое
    assert merged.seller.name == "ПАО Сбербанк"
    assert [line.name for line in merged.lines] == ["Бумага", "Ручка"]
    assert merged.totals.amount_with_vat == 9999  # итоги — последнее непустое


def test_draft_to_document_upd() -> None:
    draft = Draft.model_validate(
        draft_data(
            seller_bank={"bank_name": "Банк", "bik": None, "corr_account": None, "account": None}
        )
    )
    document, warnings, errors = draft_to_document(draft)
    assert errors == []
    assert warnings == []
    assert isinstance(document, Upd)
    assert not hasattr(document, "seller_bank")  # у УПД нет банковского блока


def test_draft_to_document_fills_required_gaps() -> None:
    data = draft_data(doc_type="bill", upd_status=None, number=None, basis=None)
    data["seller"]["name"] = None
    document, warnings, errors = draft_to_document(Draft.model_validate(data))
    assert errors == []
    assert document is not None
    assert document.doc_type is DocType.BILL
    assert document.number == ""
    assert len(warnings) == 3  # номер, имя продавца, пустой банк


def test_draft_to_document_vat_invoice_drops_basis() -> None:
    document, _, errors = draft_to_document(
        Draft.model_validate(draft_data(doc_type="vat_invoice"))
    )
    assert errors == []
    assert document is not None
    assert "basis" not in document.model_dump()


def test_draft_to_document_without_date() -> None:
    document, _, errors = draft_to_document(Draft.model_validate(draft_data(date=None)))
    assert document is None
    assert "date" in errors[0]


# ---------- запрос к Ollama, промпты, вспомогательное ----------


def test_chat_request_is_deterministic_and_structured() -> None:
    config = ExtractConfig()
    schema = draft_json_schema()
    body = build_chat_request(config, "промпт", schema, ["aGVsbG8="])
    assert body["format"] == schema
    assert body["think"] is False
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["seed"] == 42
    assert body["options"]["presence_penalty"] == 0
    assert body["messages"][0]["images"] == ["aGVsbG8="]


def test_prompts_are_versioned_files() -> None:
    assert "JSON" in load_prompt("text_v1")
    assert "{page_info}" in load_prompt("vision_v1")
    rendered = render_prompt("text_v1", source="s", source_title="T", document_text="ИНН 1")
    assert "{document_text}" not in rendered
    assert "ИНН 1" in rendered
    with pytest.raises(ValueError, match="версия"):
        load_prompt("../secret")
    with pytest.raises(FileNotFoundError):
        load_prompt("text_v999")


def test_fit_long_side() -> None:
    assert fit_long_side(Image.new("RGB", (2560, 1810)), 1280).size == (1280, 905)
    assert fit_long_side(Image.new("RGB", (1000, 3000)), 1280).size == (427, 1280)
    small = Image.new("RGB", (800, 600))
    assert fit_long_side(small, 1280) is small  # не увеличиваем


def test_table_to_text() -> None:
    rows = [["№", "Товар\nи услуги", None], [None, None, None], ["1", "Ручка | синяя", "10,00"]]
    assert table_to_text(rows) == "| № | Товар и услуги |  |\n| 1 | Ручка / синяя | 10,00 |"


def test_pdf_text_keeps_tables() -> None:
    pages = pdf_pages_text(TEXT_PDF)
    assert len(pages) == 1
    assert "| " in pages[0]  # таблицы оформлены строками
    assert "Счёт на оплату" in pages[0]


def test_vram_monitor_without_nvidia_smi() -> None:
    with VramMonitor(0.01) as monitor:
        pass
    stats = monitor.stats()
    assert not stats.available
    assert stats.peak_mib is None


def test_vram_monitor_tracks_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([100, 300, 200] + [150] * 1000)
    monkeypatch.setattr("docai.extract.vram._query", lambda cmd: next(values))
    with VramMonitor(0.01, nvidia_smi="nvidia-smi") as monitor:
        import time

        time.sleep(0.1)
    stats = monitor.stats()
    assert stats.baseline_mib == 100
    assert stats.peak_mib == 300
    assert stats.samples >= 3


# ---------- конвейер целиком (Ollama подменена) ----------


def test_extract_text_route(fake: FakeOllama) -> None:
    result = extract([TEXT_PDF])
    assert result.valid
    assert result.meta.route == "text"
    assert result.meta.prompt_version == "text_v1"
    assert result.meta.tokens.prompt == 100
    assert {"route", "pdf_text", "llm", "assemble", "total"} <= set(result.meta.timings_s)
    assert len(fake.bodies) == 1
    prompt = fake.bodies[0]["messages"][0]["content"]
    assert "Счёт на оплату" in prompt  # текст PDF попал в промпт
    assert "images" not in fake.bodies[0]["messages"][0]
    parse_document(result.document)  # JSON валиден по схеме


def test_extract_vision_route_per_page(fake: FakeOllama, scan_pdf: Path) -> None:
    result = extract([scan_pdf])
    assert result.valid
    assert result.meta.route == "vision"
    assert result.meta.prompt_version == "vision_v1"
    assert result.meta.pages == 2
    assert len(fake.bodies) == 2
    assert [c.part for c in result.meta.llm_calls] == ["page 1/2", "page 2/2"]
    for call in result.meta.llm_calls:
        assert call.image_size is not None
        assert max(call.image_size) <= 1280
    assert "страница 2 из 2" in fake.bodies[1]["messages"][0]["content"]
    assert len(result.document["lines"]) == 2  # строки двух страниц объединены
    assert result.meta.tokens.prompt == 200


def test_extract_image_is_downscaled(fake: FakeOllama, png: Path) -> None:
    result = extract([png])
    assert result.meta.llm_calls[0].image_size == (1280, 853)
    assert "единственная страница" in fake.bodies[0]["messages"][0]["content"]


def test_extract_xml_route_without_llm(fake: FakeOllama, xml_file: Path) -> None:
    result = extract([xml_file])
    assert result.valid
    assert result.meta.route == "xml"
    assert result.meta.model is None
    assert fake.bodies == []
    assert result.document["number"] == "123"


def test_extract_reports_broken_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module, "chat", FakeOllama(answer('{"doc_type": ', done_reason="length"))
    )
    result = extract([TEXT_PDF])
    assert not result.valid
    assert result.document is None
    assert any("обрезан" in e for e in result.errors)


def test_extract_ocr_without_tesseract(monkeypatch: pytest.MonkeyPatch, png: Path) -> None:
    monkeypatch.setattr(ocr_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_module, "WINDOWS_DEFAULT", Path("Z:/нет/tesseract.exe"))
    with pytest.raises(ocr_module.OcrError, match="winget install"):
        extract([png], Mode.OCR)


def test_extract_ocr_route(monkeypatch: pytest.MonkeyPatch, fake: FakeOllama, png: Path) -> None:
    monkeypatch.setattr(pipeline_module, "find_tesseract", lambda cmd: Path("tesseract.exe"))
    monkeypatch.setattr(pipeline_module, "check_languages", lambda cmd, lang: None)
    monkeypatch.setattr(pipeline_module, "tesseract_version", lambda cmd: "tesseract 5.5.0")
    monkeypatch.setattr(pipeline_module, "ocr_image", lambda *a: "Счёт № 123 от 14.11.2026")
    result = extract([png], Mode.OCR)
    assert result.valid
    assert result.meta.route == "ocr"
    assert result.meta.ocr_engine.startswith("tesseract 5.5.0")
    assert "Счёт № 123" in fake.bodies[0]["messages"][0]["content"]
    assert "ocr" in result.meta.timings_s


# ---------- CLI ----------


def test_cli_extract_writes_json(fake: FakeOllama, tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    result = CliRunner().invoke(app, ["extract", str(TEXT_PDF), "--out", str(out), "-q"])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["valid"]
    assert data["meta"]["route"] == "text"
    parse_document(data["document"])


def test_cli_extract_routing_error(png: Path) -> None:
    result = CliRunner().invoke(app, ["extract", str(png), "--mode", "text", "-q"])
    assert result.exit_code == 2
    assert "только с PDF" in result.output


def test_config_file_matches_defaults() -> None:
    """configs/extract.yaml и умолчания ExtractConfig не расходятся."""
    from docai.extract import load_config

    assert load_config(ROOT / "configs" / "extract.yaml") == ExtractConfig()


def test_truncated_whole_text_retries_per_page(
    monkeypatch: pytest.MonkeyPatch, scan_pdf: Path
) -> None:
    """Ответ на весь двухстраничный текст обрезан → два запроса по страницам, документ собран."""
    fake = FakeOllama(answer('{"doc_type": ', done_reason="length"), answer(draft_data()))
    monkeypatch.setattr(pipeline_module, "chat", fake)
    monkeypatch.setattr(pipeline_module, "find_tesseract", lambda cmd: Path("tesseract.exe"))
    monkeypatch.setattr(pipeline_module, "check_languages", lambda cmd, lang: None)
    monkeypatch.setattr(pipeline_module, "tesseract_version", lambda cmd: "tesseract 5.4.0")
    monkeypatch.setattr(pipeline_module, "ocr_image", lambda *a: "Счёт-фактура № 1")
    result = extract([scan_pdf], Mode.OCR)
    assert result.valid
    assert result.errors == []
    assert [c.part for c in result.meta.llm_calls] == ["document", "page 1/2", "page 2/2"]
    assert any("повтор по страницам" in w for w in result.warnings)
    assert "Страница 2 из 2" in fake.bodies[2]["messages"][0]["content"]
    assert len(result.document["lines"]) == 2


def test_truncated_single_page_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """У одностраничного документа повторять нечего: ошибка остаётся."""
    fake = FakeOllama(answer('{"doc_type": ', done_reason="length"))
    monkeypatch.setattr(pipeline_module, "chat", fake)
    result = extract([TEXT_PDF])
    assert not result.valid
    assert len(fake.bodies) == 1
