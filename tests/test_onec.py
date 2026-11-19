"""Тесты P15: обмен файлами с 1С (способ Б), исходники внешней обработки, журнал 1С.

Сама 1С в тестах не запускается (её нет в CI); проверка с настоящей платформой —
`scripts/onec_smoke.ps1`, см. onec/README.md.
"""

import importlib.util
import json
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typer.testing import CliRunner

from docai.cli import app as cli_app
from docai.service.app import create_app
from docai.service.exchange import ExchangeError, FileExchange
from docai.service.settings import ServiceSettings
from test_service import BASE, pdf_bytes, stub_extractor, stub_probe

ROOT = Path(__file__).resolve().parents[1]
ONEC_SRC = ROOT / "onec" / "src"
MODULE = ONEC_SRC / "DocaiCheck" / "Forms" / "Form" / "Ext" / "Form" / "Module.bsl"
FORM = ONEC_SRC / "DocaiCheck" / "Forms" / "Form" / "Ext" / "Form.xml"


# --- обмен файлами: модуль отдельно ---------------------------------------------------------


class Health(BaseModel):
    status: str = "ok"


class Job(BaseModel):
    id: str
    status: str


class FakeService:
    """Очередь вместо настоящей: задания, их статусы и принятые файлы."""

    def __init__(self) -> None:
        self.jobs: dict[str, str] = {}
        self.received: list[tuple[str, bytes]] = []
        self.refuse: str | None = None

    def enqueue(self, path: Path, filename: str) -> str:
        if self.refuse:
            raise ExchangeError(self.refuse)
        job_id = f"job{len(self.jobs) + 1}"
        self.jobs[job_id] = "queued"
        self.received.append((filename, path.read_bytes()))
        return job_id

    def view(self, job_id: str) -> Job | None:
        status = self.jobs.get(job_id)
        return None if status is None else Job(id=job_id, status=status)


@pytest.fixture
def fake() -> FakeService:
    return FakeService()


@pytest.fixture
def exchange(tmp_path: Path, fake: FakeService) -> FileExchange:
    ex = FileExchange(
        tmp_path / "exchange", health=Health, enqueue=fake.enqueue, job_view=fake.view
    )
    for folder in (ex.inbox, ex.outbox, ex.pending, ex.rejected):
        folder.mkdir(parents=True)
    return ex


def answer(ex: FileExchange, request_id: str) -> dict:
    return json.loads((ex.outbox / f"{request_id}.json").read_text(encoding="utf-8"))


def test_health_request_is_answered_and_removed(exchange: FileExchange) -> None:
    (exchange.inbox / "abc123.json").write_text('{"method": "health"}', encoding="utf-8")
    exchange.poll_once()
    assert answer(exchange, "abc123") == {"status": "ok"}
    assert list(exchange.inbox.iterdir()) == []


def test_request_with_bom_from_1c_is_accepted(exchange: FileExchange) -> None:
    # ЗаписьТекста в 1С с КодировкаТекста.UTF8 пишет BOM
    (exchange.inbox / "r1.json").write_bytes(b"\xef\xbb\xbf" + b'{"method": "health"}')
    exchange.poll_once()
    assert answer(exchange, "r1")["status"] == "ok"


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (b'{"method": "drop_tables"}', "неизвестный метод"),
        (b"[1, 2]", "неизвестный метод"),
        (b"not json", "не JSON"),
        (b"\xff\xfe{}", "не JSON"),
    ],
)
def test_bad_request_gets_detail(exchange: FileExchange, body: bytes, fragment: str) -> None:
    (exchange.inbox / "bad.json").write_bytes(body)
    exchange.poll_once()
    assert fragment in answer(exchange, "bad")["detail"]
    assert not (exchange.inbox / "bad.json").exists()


def test_too_large_request_is_refused(exchange: FileExchange) -> None:
    (exchange.inbox / "big.json").write_bytes(b" " * (64 * 1024 + 1))
    exchange.poll_once()
    assert "64 КБ" in answer(exchange, "big")["detail"]


def test_tmp_and_unknown_files_are_left_alone(exchange: FileExchange) -> None:
    # Недописанный запрос (1С ещё не переименовала .tmp) и посторонние файлы не трогаются
    for name in ("half.tmp", "notes.txt", "x.json.tmp"):
        (exchange.inbox / name).write_text("{}", encoding="utf-8")
    exchange.poll_once()
    assert sorted(p.name for p in exchange.inbox.iterdir()) == [
        "half.tmp",
        "notes.txt",
        "x.json.tmp",
    ]
    assert list(exchange.outbox.iterdir()) == []


@pytest.mark.parametrize("name", ["a b.json", "..json", "смета.pdf", "x" * 65 + ".json"])
def test_unsafe_name_goes_to_rejected(exchange: FileExchange, name: str) -> None:
    (exchange.inbox / name).write_text('{"method": "health"}', encoding="utf-8")
    exchange.poll_once()
    assert (exchange.rejected / name).exists()
    assert list(exchange.outbox.iterdir()) == []


def test_document_is_answered_when_job_finishes(exchange: FileExchange, fake: FakeService) -> None:
    (exchange.inbox / "doc1.pdf").write_bytes(b"%PDF-1.4 test")
    exchange.poll_once()
    assert fake.received == [("doc1.pdf", b"%PDF-1.4 test")]
    assert not (exchange.inbox / "doc1.pdf").exists()
    assert (exchange.pending / "doc1.txt").read_text(encoding="ascii") == "job1"
    assert list(exchange.outbox.iterdir()) == []  # задание ещё не выполнено

    fake.jobs["job1"] = "running"
    exchange.poll_once()
    assert list(exchange.outbox.iterdir()) == []

    fake.jobs["job1"] = "done"
    exchange.poll_once()
    assert answer(exchange, "doc1") == {"id": "job1", "status": "done"}
    assert list(exchange.pending.iterdir()) == []


def test_failed_job_is_answered_too(exchange: FileExchange, fake: FakeService) -> None:
    (exchange.inbox / "doc2.png").write_bytes(b"png")
    exchange.poll_once()
    fake.jobs["job1"] = "failed"
    exchange.poll_once()
    assert answer(exchange, "doc2")["status"] == "failed"


def test_pending_survives_restart(
    tmp_path: Path, exchange: FileExchange, fake: FakeService
) -> None:
    (exchange.inbox / "doc3.xml").write_bytes(b"<x/>")
    exchange.poll_once()
    fake.jobs["job1"] = "done"
    restarted = FileExchange(exchange.root, health=Health, enqueue=fake.enqueue, job_view=fake.view)
    restarted.poll_once()
    assert answer(restarted, "doc3")["status"] == "done"


def test_unknown_job_in_pending_is_reported(exchange: FileExchange) -> None:
    (exchange.pending / "lost.txt").write_text("nojob", encoding="ascii")
    exchange.poll_once()
    assert answer(exchange, "lost") == {"detail": "задание не найдено"}


def test_refused_document_gets_detail(exchange: FileExchange, fake: FakeService) -> None:
    fake.refuse = "очередь заполнена, повторите позже"
    (exchange.inbox / "doc4.pdf").write_bytes(b"%PDF")
    exchange.poll_once()
    assert answer(exchange, "doc4") == {"detail": "очередь заполнена, повторите позже"}
    assert not (exchange.inbox / "doc4.pdf").exists()


def test_answer_is_written_atomically(exchange: FileExchange) -> None:
    (exchange.inbox / "r2.json").write_text('{"method": "health"}', encoding="utf-8")
    exchange.poll_once()
    # Временных файлов не остаётся, ответ — целый JSON
    assert [p.name for p in exchange.outbox.iterdir()] == ["r2.json"]


def test_loop_survives_errors(tmp_path: Path) -> None:
    calls = 0

    def broken_health() -> BaseModel:
        nonlocal calls
        calls += 1
        raise RuntimeError("сбой")

    ex = FileExchange(
        tmp_path / "ex",
        health=broken_health,
        enqueue=lambda p, n: "j",
        job_view=lambda j: None,
        poll_interval_s=0.01,
    )
    ex.start()
    try:
        for i in range(2):
            (ex.inbox / f"r{i}.json").write_text('{"method": "health"}', encoding="utf-8")
        deadline = time.monotonic() + 5
        while calls < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ex.alive
        assert calls >= 2
    finally:
        ex.stop()
    assert not ex.alive


# --- обмен файлами: в составе сервиса -------------------------------------------------------


@pytest.fixture
def service(tmp_path: Path) -> Iterator[TestClient]:
    # Фоновый проход — раз в час: в тестах папки просматриваются вручную (poll_once).
    settings = ServiceSettings(data_dir=tmp_path / "svc", exchange_poll_s=3600)
    application = create_app(settings, extractor=stub_extractor, health_probe=stub_probe)
    with TestClient(application, base_url=BASE) as client:
        yield client


def test_service_answers_health_through_files(service: TestClient) -> None:
    ex: FileExchange = service.app.state.exchange
    assert ex.alive
    assert ex.root == service.app.state.settings.data_dir / "exchange"
    (ex.inbox / "h1.json").write_text('{"method": "health"}', encoding="utf-8")
    ex.poll_once()
    by_file = answer(ex, "h1")
    by_http = service.get("/v1/health").json()
    assert by_file == by_http
    assert by_file["status"] == "ok"


def test_service_extracts_document_through_files(service: TestClient) -> None:
    ex: FileExchange = service.app.state.exchange
    (ex.inbox / "d1.pdf").write_bytes(pdf_bytes())
    ex.poll_once()
    job_id = (ex.pending / "d1.txt").read_text(encoding="ascii")
    deadline = time.monotonic() + 10
    while service.get(f"/v1/jobs/{job_id}").json()["status"] != "done":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    ex.poll_once()
    by_file = answer(ex, "d1")
    assert by_file == service.get(f"/v1/jobs/{job_id}").json()
    assert by_file["filename"] == "d1.pdf"
    assert by_file["fields"]


def test_service_refuses_wrong_file_type_through_files(service: TestClient) -> None:
    ex: FileExchange = service.app.state.exchange
    (ex.inbox / "w1.pdf").write_bytes(b"this is not a pdf")
    ex.poll_once()
    assert "ожидаются PDF" in answer(ex, "w1")["detail"]


def test_exchange_can_be_disabled(tmp_path: Path) -> None:
    settings = ServiceSettings(data_dir=tmp_path, exchange_enabled=False)
    application = create_app(settings, extractor=stub_extractor, health_probe=stub_probe)
    with TestClient(application, base_url=BASE):
        assert application.state.exchange is None
    assert not (tmp_path / "exchange").exists()


def test_serve_passes_exchange_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    apps: list = []
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: apps.append(application))
    runner = CliRunner()
    custom = tmp_path / "custom"
    result = runner.invoke(
        cli_app,
        ["serve", "--data-dir", str(tmp_path / "d"), "--exchange-dir", str(custom)],
    )
    assert result.exit_code == 0, result.output
    assert "Обмен файлами с 1С" in result.output
    assert apps[0].state.exchange.root == custom

    result = runner.invoke(cli_app, ["serve", "--data-dir", str(tmp_path / "d"), "--no-exchange"])
    assert result.exit_code == 0, result.output
    assert apps[1].state.exchange is None


# --- исходники внешней обработки ------------------------------------------------------------


def test_onec_sources_are_wellformed_xml_of_format_2_1() -> None:
    files = sorted(ONEC_SRC.rglob("*.xml"))
    # DocaiCheck (P15): 3 файла, DocaiLoad (P16): описание и три формы — 7 файлов
    assert len(files) == 10
    for path in files:
        root = ET.parse(path).getroot()
        # Учебная платформа 8.3.8 принимает выгрузку формата не новее 2.1 (D54)
        assert root.get("version") == "2.1", path


def test_onec_sources_are_stored_as_designer_writes_them() -> None:
    # UTF-8 с BOM и CRLF: иначе повторная выгрузка из Конфигуратора даёт ложные изменения
    for path in [*ONEC_SRC.rglob("*.xml"), *ONEC_SRC.rglob("*.bsl")]:
        data = path.read_bytes()
        assert data.startswith(b"\xef\xbb\xbf"), path
        assert b"\n" not in data.replace(b"\r\n", b""), path


def test_form_handlers_exist_in_module() -> None:
    ns = {"f": "http://v8.1c.ru/8.3/xcf/logform"}
    form = ET.parse(FORM).getroot()
    handlers = {e.text for e in form.findall(".//f:Events/f:Event", ns)}
    handlers |= {e.text for e in form.findall(".//f:Commands/f:Command/f:Action", ns)}
    assert handlers == {"ПриОткрытии", "ПроверитьСвязь", "ПроверитьЧерезФайлы"}
    code = MODULE.read_text(encoding="utf-8-sig")
    for name in handlers:
        assert re.search(rf"^Процедура {name}\(", code, re.MULTILINE), name


def test_module_matches_service_contract() -> None:
    code = MODULE.read_text(encoding="utf-8-sig")
    assert '"/v1/health"' in code
    assert '"127.0.0.1"' in code and "8000" in code
    # Способ Б: запрос пишется под .tmp и переименовывается в .json, ответ — в outbox
    assert '".tmp"' in code and "ПереместитьФайл" in code
    assert '{""method"": ""health""}' in code
    assert '"\\outbox\\"' in code
    # Ключи ответа, которые читает обработка, есть в HealthView (разбор — в Структуру)
    assert "Новый ЧтениеJSON" in code
    from docai.service.models import GpuHealth, HealthView, OllamaHealth

    for key in re.findall(r"Данные\.([a-z_]+)", code):
        assert key in HealthView.model_fields, key
    for key in re.findall(r"Ollama\.([a-z_]+)", code):
        assert key in OllamaHealth.model_fields, key
    for key in re.findall(r"GPU\.([a-z_]+)", code):
        assert key in GpuHealth.model_fields, key


def test_module_avoids_constructs_rejected_by_8_3_8() -> None:
    # Вызов метода у выражения в скобках — «(Новый Файл(Путь)).Существует()» — платформа
    # 8.3.8 не компилирует («Ожидается ключевое слово 'Тогда'»), найдено при P15.
    code = MODULE.read_text(encoding="utf-8-sig")
    assert not re.search(r"\(Новый [^()]*\([^()]*\)\)\.", code)


@pytest.mark.parametrize(
    "name",
    [
        "СтрСоединить",
        "СтрНачинаетсяС",
        "СтрЗаканчиваетсяНа",
        "СтрРазделить",
        "СтрНайти",
        "СтрШаблон",
        "ПрочитатьJSON",
        "ЗаписатьJSON",
    ],
)
def test_module_works_in_compatibility_mode_8_2_16(name: str) -> None:
    # Глобальные функции 8.3.6+ недоступны, если конфигурация (например, старая
    # Бухгалтерия) работает в режиме совместимости 8.3.5 и ниже: модуль не компилируется.
    # Проверено на 8.3.8 с режимом Version8_2_16 (D54). Упоминания в комментариях допустимы.
    code_lines = [
        line.split("//", 1)[0] for line in MODULE.read_text(encoding="utf-8-sig").splitlines()
    ]
    assert not any(re.search(rf"\b{name}\(", line) for line in code_lines)


# --- журнал регистрации 1С -----------------------------------------------------------------


def load_errors_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "onec_errors", ROOT / "scripts" / "onec_errors.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_log(infobase: Path, rows: list[tuple[int, int, str]]) -> None:
    folder = infobase / "1Cv8Log"
    folder.mkdir(parents=True)
    db = sqlite3.connect(folder / "1Cv8.lgd")
    db.execute(
        "CREATE TABLE EventLog"
        " (rowID INTEGER PRIMARY KEY, date INTEGER, severity INTEGER, comment TEXT)"
    )
    db.executemany("INSERT INTO EventLog (date, severity, comment) VALUES (?, ?, ?)", rows)
    db.commit()
    db.close()


def test_onec_log_ticks_roundtrip() -> None:
    errors = load_errors_script()
    # Запись из настоящего журнала 8.3.8: ошибка в 20:24:56 по Москве = 17:24:56 UTC
    moment = errors.ticks_to_datetime(639306194963870)
    assert moment == datetime(2026, 11, 18, 17, 24, 56, 387000, tzinfo=UTC)
    assert errors.datetime_to_ticks(moment) == 639306194963870


def test_onec_log_reports_recent_errors_only(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    errors = load_errors_script()
    now = datetime.now(UTC)
    make_log(
        tmp_path,
        [
            (errors.datetime_to_ticks(now - timedelta(hours=2)), 3, "старая ошибка"),
            (errors.datetime_to_ticks(now - timedelta(minutes=1)), 1, "сеанс начат"),
            (errors.datetime_to_ticks(now - timedelta(minutes=1)), 3, "Ожидается 'Тогда'"),
        ],
    )
    assert errors.main([str(tmp_path), "--minutes", "10"]) == 1
    out = capsys.readouterr().out
    assert "Ожидается 'Тогда'" in out
    assert "старая ошибка" not in out and "сеанс начат" not in out


def test_onec_log_missing_is_not_an_error(tmp_path: Path) -> None:
    assert load_errors_script().main([str(tmp_path)]) == 0
