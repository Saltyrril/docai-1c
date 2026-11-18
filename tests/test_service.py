"""Тесты HTTP-сервиса (P14). Вместо модели — заглушка, Ollama и GPU не нужны."""

import io
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from docai import __version__
from docai.cli import app as cli_app
from docai.extract import ExtractionResult, Mode, RawExtraction, finish
from docai.extract.ollama import OllamaError
from docai.extract.result import Meta
from docai.schemas.documents import parse_document
from docai.service.app import _clean_filename, create_app
from docai.service.models import GpuHealth, OllamaHealth
from docai.service.settings import HOST, ServiceSettings
from docai.service.store import Store
from docai.service.views import extracted_values, number_str, same_value
from docai.service.worker import Worker
from factories import doc_data

BASE = "http://127.0.0.1"


def stub_result(route: str = "xml") -> ExtractionResult:
    """Результат «модели»: корректный УПД, прошедший настоящую постобработку."""
    document = parse_document(doc_data("upd"))
    meta = Meta(
        docai_version=__version__,
        started_at="2026-11-18T10:00:00+00:00",
        source=["stub"],
        mode="auto",
        route=route,
        pages=1,
        model="stub-model",
        timings_s={"total": 1.5},
    )
    raw = ExtractionResult(document=document.model_dump(mode="json"), valid=True, meta=meta)
    return finish(RawExtraction(result=raw))


def stub_extractor(path: Path, mode: Mode) -> ExtractionResult:
    return stub_result()


def stub_probe() -> tuple[OllamaHealth, GpuHealth]:
    return (
        OllamaHealth(reachable=True, model="stub-model", model_available=True, model_loaded=True),
        GpuHealth(available=True, vram_total_mib=4096, vram_free_mib=3000),
    )


def pdf_bytes(pages: int = 1) -> bytes:
    images = [Image.new("RGB", (300, 400), "white") for _ in range(pages)]
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, "PNG")
    return buffer.getvalue()


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, "JPEG")
    return buffer.getvalue()


XML = '<?xml version="1.0" encoding="windows-1251"?><Файл/>'.encode("cp1251")


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(
        extractor: Callable[[Path, Mode], ExtractionResult] = stub_extractor, **settings: Any
    ) -> TestClient:
        service = create_app(
            ServiceSettings(data_dir=tmp_path / "svc", **settings),
            extractor=extractor,
            health_probe=stub_probe,
        )
        client = TestClient(service, base_url=BASE)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def upload(
    client: TestClient, content: bytes, name: str = "doc.pdf", **form: str
) -> dict[str, Any]:
    response = client.post("/v1/extract", files={"file": (name, content)}, data=form)
    assert response.status_code == 202, response.text
    return response.json()


def wait_for(client: TestClient, job_id: str, *states: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"задание {job_id} не дошло до {states}")


# --- приём файлов --------------------------------------------------------------------------


def test_upload_pdf_returns_job_id_and_result(client: TestClient) -> None:
    created = upload(client, pdf_bytes())
    assert created["status"] == "queued"
    assert len(created["id"]) == 32
    job = wait_for(client, created["id"], "done")
    assert job["document_status"] == "upload"
    assert job["document_status_ru"] == "можно загружать"
    assert job["valid"] is True
    assert job["doc_type"] == "upd"
    assert job["upd_status"] == "1"
    assert job["filename"] == "doc.pdf"
    assert job["source_kind"] == "pdf"
    assert job["route"] == "xml"
    assert job["model"] == "stub-model"
    assert job["seconds"] == "1.5"
    assert job["error"] is None


@pytest.mark.parametrize(
    ("content", "name", "kind"),
    [
        (pdf_bytes(), "a.pdf", "pdf"),
        (png_bytes(), "a.png", "image"),
        (jpeg_bytes(), "a.jpg", "image"),
        (XML, "a.xml", "xml"),
    ],
)
def test_allowed_types_are_detected_by_content(
    client: TestClient, content: bytes, name: str, kind: str
) -> None:
    job = wait_for(client, upload(client, content, name)["id"], "done")
    assert job["source_kind"] == kind


def test_type_is_checked_by_content_not_by_name(client: TestClient) -> None:
    job = wait_for(client, upload(client, pdf_bytes(), "скан.jpg")["id"], "done")
    assert job["source_kind"] == "pdf"


@pytest.mark.parametrize(
    "content",
    [
        b"GIF89a....",
        b"PK\x03\x04zipfile",
        b"MZ\x90\x00 exe",
        b"just text, not a document",
        b"BM" + b"\x00" * 60,  # BMP — картинка, но не PNG/JPEG
    ],
)
def test_unsupported_type_is_rejected(client: TestClient, content: bytes) -> None:
    response = client.post("/v1/extract", files={"file": ("x.pdf", content)})
    assert response.status_code == 415


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/extract", files={"file": ("x.pdf", b"")})
    assert response.status_code == 400


def test_file_without_form_field_is_422(client: TestClient) -> None:
    assert client.post("/v1/extract").status_code == 422


def test_file_over_limit_is_rejected_after_reading(make_client: Callable[..., TestClient]) -> None:
    client = make_client(max_upload_bytes=1000)
    response = client.post("/v1/extract", files={"file": ("x.pdf", b"%PDF-1.4\n" + b"x" * 2000)})
    assert response.status_code == 413
    assert not list((client.app.state.settings.uploads_dir).iterdir())  # type: ignore[attr-defined]


def test_huge_body_is_cut_by_middleware(make_client: Callable[..., TestClient]) -> None:
    client = make_client(max_upload_bytes=1000)
    response = client.post("/v1/extract", files={"file": ("x.pdf", b"%PDF-1.4\n" + b"x" * 200_000)})
    assert response.status_code == 413


def test_body_without_content_length_is_cut(make_client: Callable[..., TestClient]) -> None:
    client = make_client(max_upload_bytes=1000)

    crlf = b"\r\n"
    head = (
        b"--b" + crlf + b'Content-Disposition: form-data; name="file"; filename="x.pdf"' + crlf * 2
    ) + b"%PDF-1.4\n"

    def chunks() -> Iterator[bytes]:
        yield head
        for _ in range(200):
            yield b"x" * 1024
        yield crlf + b"--b--" + crlf

    response = client.post(
        "/v1/extract",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=b"},
    )
    assert response.status_code == 413


def test_default_limit_is_20_mib() -> None:
    assert ServiceSettings().max_upload_bytes == 20 * 1024 * 1024


def test_unknown_mode_is_422(client: TestClient) -> None:
    response = client.post(
        "/v1/extract", files={"file": ("x.pdf", pdf_bytes())}, data={"mode": "magic"}
    )
    assert response.status_code == 422


def test_mode_is_passed_to_extractor(make_client: Callable[..., TestClient]) -> None:
    seen: list[Mode] = []

    def extractor(path: Path, mode: Mode) -> ExtractionResult:
        seen.append(mode)
        return stub_result()

    client = make_client(extractor)
    wait_for(client, upload(client, pdf_bytes(), mode="ocr")["id"], "done")
    wait_for(client, upload(client, pdf_bytes())["id"], "done")
    assert seen == [Mode.OCR, Mode.AUTO]


def test_filename_is_sanitized(client: TestClient) -> None:
    bs = chr(92)  # обратная косая черта
    name = f"..{bs}..{bs}etc/passwd.pdf"
    job = wait_for(client, upload(client, pdf_bytes(), name)["id"], "done")
    assert job["filename"] == "passwd.pdf"


def test_clean_filename_drops_path_and_control_characters() -> None:
    bs, nul, nl = chr(92), chr(0), chr(10)
    assert _clean_filename(f"C:{bs}Users{bs}a{bs}акт{nul}.pdf") == "акт.pdf"
    assert _clean_filename(f"../../x{nl}.pdf") == "x.pdf"
    assert _clean_filename(None) == "upload"
    assert _clean_filename("") == "upload"
    assert len(_clean_filename("я" * 1000)) == 255


def test_queue_full_is_429(make_client: Callable[..., TestClient]) -> None:
    gate = threading.Event()

    def blocked(path: Path, mode: Mode) -> ExtractionResult:
        gate.wait(10)
        return stub_result()

    client = make_client(blocked, max_queued=1)
    first = upload(client, pdf_bytes())
    wait_for(client, first["id"], "running")
    upload(client, pdf_bytes())  # одно ждёт — это предел
    response = client.post("/v1/extract", files={"file": ("x.pdf", pdf_bytes())})
    gate.set()
    assert response.status_code == 429


# --- очередь -------------------------------------------------------------------------------


def test_jobs_run_one_at_a_time_in_order(make_client: Callable[..., TestClient]) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0
    order: list[str] = []

    def extractor(path: Path, mode: Mode) -> ExtractionResult:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            order.append(path.parent.name)
        time.sleep(0.05)
        with lock:
            active -= 1
        return stub_result()

    client = make_client(extractor)
    ids = [upload(client, pdf_bytes())["id"] for _ in range(4)]
    for job_id in ids:
        wait_for(client, job_id, "done")
    assert peak == 1
    assert order == ids


def test_queue_position_is_reported(make_client: Callable[..., TestClient]) -> None:
    gate = threading.Event()

    def blocked(path: Path, mode: Mode) -> ExtractionResult:
        gate.wait(10)
        return stub_result()

    client = make_client(blocked)
    first = upload(client, pdf_bytes())
    wait_for(client, first["id"], "running")
    second = upload(client, pdf_bytes())
    third = upload(client, pdf_bytes())
    assert (second["queue_ahead"], third["queue_ahead"]) == (1, 2)
    assert client.get(f"/v1/jobs/{third['id']}").json()["queue_ahead"] == 2
    assert client.get(f"/v1/jobs/{first['id']}").json()["queue_ahead"] is None
    gate.set()
    wait_for(client, third["id"], "done")


def test_failed_job_reports_error_and_queue_continues(
    make_client: Callable[..., TestClient],
) -> None:
    calls = iter([OllamaError("Ollama недоступна"), RuntimeError("секретная трассировка"), None])

    def extractor(path: Path, mode: Mode) -> ExtractionResult:
        error = next(calls)
        if error is not None:
            raise error
        return stub_result()

    client = make_client(extractor)
    ids = [upload(client, pdf_bytes())["id"] for _ in range(3)]
    jobs = [wait_for(client, i, "done", "failed") for i in ids]
    assert [j["status"] for j in jobs] == ["failed", "failed", "done"]
    assert jobs[0]["error"] == "Ollama недоступна"
    assert "секретная" not in jobs[1]["error"]  # внутренние детали наружу не уходят
    assert jobs[0]["fields"] == []
    assert jobs[0]["document_status"] is None


def test_running_jobs_return_to_queue_after_restart(tmp_path: Path) -> None:
    settings = ServiceSettings(data_dir=tmp_path / "svc")
    settings.job_dir("a" * 32).mkdir(parents=True)
    store = Store(settings.db_path)
    store.create_job(
        "a" * 32, filename="x.pdf", stored_name="source.pdf", source_kind="pdf", mode="auto",
        size_bytes=1,
    )  # fmt: skip
    (settings.job_dir("a" * 32) / "source.pdf").write_bytes(pdf_bytes())
    assert store.claim_next() is not None  # «выполнялось» в момент остановки
    assert store.count("running") == 1

    worker = Worker(store, stub_extractor, settings)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while store.count("done") == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert store.count("done") == 1
    store.close()


# --- результат в формате для 1С ------------------------------------------------------------


def test_result_is_flat_strings_for_1c(client: TestClient) -> None:
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    fields = {f["name"]: f for f in job["fields"]}
    assert fields["date"]["value"] == "2026-11-14"  # дата ISO
    assert fields["totals.amount_with_vat"]["value"] == "2561.99"  # число строкой с точкой
    assert fields["seller.inn"]["value"] == "7707083893"
    assert fields["lines[0].price"]["value"] == "1000.00"
    assert fields["lines[1].vat_rate"]["value"] == "22%"
    assert fields["basis.date"]["value"] == "2026-01-15"
    for item in job["fields"]:
        assert isinstance(item["confidence"], str)
        assert 0 <= float(item["confidence"]) <= 1
        assert isinstance(item["needs_review"], bool)
    assert isinstance(job["mean_confidence"], str)


def test_lines_are_listed_with_confidence(client: TestClient) -> None:
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    assert [line["n"] for line in job["lines"]] == [1, 2]
    first = job["lines"][0]
    assert first["name"] == "Бумага"
    assert first["quantity"] == "2"
    assert first["amount_with_vat"] == "2440.00"
    assert first["vat_rate"] == "22%"
    assert isinstance(first["min_confidence"], str)
    assert first["needs_review"] is False


def test_validation_issues_and_review_status(make_client: Callable[..., TestClient]) -> None:
    def extractor(path: Path, mode: Mode) -> ExtractionResult:
        document = parse_document(doc_data("upd", seller={"name": "ООО", "inn": "7707083890"}))
        meta = Meta(docai_version="t", started_at="t", source=["s"], mode="auto", route="xml")
        raw = ExtractionResult(document=document.model_dump(mode="json"), valid=True, meta=meta)
        return finish(RawExtraction(result=raw))

    client = make_client(extractor)
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    assert job["document_status"] in {"review", "reject"}
    assert any(
        i["code"] == "INN_CHECKSUM" and i["field"] == "seller.inn" and i["severity"] == "error"
        for i in job["issues"]
    )
    assert job["status_reasons"]
    assert any(f["name"] == "seller.inn" and f["needs_review"] for f in job["fields"])


def test_job_view_has_stable_keys_in_every_state(make_client: Callable[..., TestClient]) -> None:
    gate = threading.Event()

    def blocked(path: Path, mode: Mode) -> ExtractionResult:
        gate.wait(10)
        return stub_result()

    client = make_client(blocked)
    created = upload(client, pdf_bytes())
    queued_or_running = client.get(f"/v1/jobs/{created['id']}").json()
    gate.set()
    done = wait_for(client, created["id"], "done")
    assert set(queued_or_running) == set(done)


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/" + "0" * 32).status_code == 404
    assert client.get("/v1/jobs/../../etc/passwd").status_code == 404
    assert client.get("/v1/jobs/" + "0" * 32 + "/preview/1").status_code == 404
    assert (
        client.post("/v1/jobs/" + "0" * 32 + "/feedback", json={"fields": {"a": "1"}}).status_code
        == 404
    )


# --- страницы ------------------------------------------------------------------------------


def test_preview_pdf_pages(client: TestClient) -> None:
    job = wait_for(client, upload(client, pdf_bytes(pages=2))["id"], "done")
    assert job["preview_pages"] == 2
    for page in (1, 2):
        response = client.get(f"/v1/jobs/{job['id']}/preview/{page}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
    assert client.get(f"/v1/jobs/{job['id']}/preview/3").status_code == 404
    assert client.get(f"/v1/jobs/{job['id']}/preview/0").status_code == 404


def test_preview_image_is_resized(make_client: Callable[..., TestClient]) -> None:
    client = make_client(preview_max_side=100)
    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), "white").save(buffer, "PNG")
    job = wait_for(client, upload(client, buffer.getvalue(), "s.png")["id"], "done")
    response = client.get(f"/v1/jobs/{job['id']}/preview/1")
    assert Image.open(io.BytesIO(response.content)).size == (100, 50)


def test_preview_pages_are_capped(make_client: Callable[..., TestClient]) -> None:
    client = make_client(preview_max_pages=2)
    job = wait_for(client, upload(client, pdf_bytes(pages=4))["id"], "done")
    assert job["preview_pages"] == 2


def test_xml_has_no_preview(client: TestClient) -> None:
    job = wait_for(client, upload(client, XML, "d.xml")["id"], "done")
    assert job["preview_pages"] == 0
    assert client.get(f"/v1/jobs/{job['id']}/preview/1").status_code == 404


def test_preview_of_unfinished_job_is_409(make_client: Callable[..., TestClient]) -> None:
    gate = threading.Event()

    def blocked(path: Path, mode: Mode) -> ExtractionResult:
        gate.wait(10)
        return stub_result()

    client = make_client(blocked)
    created = upload(client, pdf_bytes())
    try:
        assert client.get(f"/v1/jobs/{created['id']}/preview/1").status_code == 409
    finally:
        gate.set()


def test_broken_pdf_does_not_fail_job_because_of_preview(client: TestClient) -> None:
    job = wait_for(client, upload(client, b"%PDF-1.4\nthis is not a pdf")["id"], "done")
    assert job["status"] == "done"
    assert job["preview_pages"] == 0


# --- правки --------------------------------------------------------------------------------


def test_feedback_counts_changed_values(client: TestClient) -> None:
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    response = client.post(
        f"/v1/jobs/{job['id']}/feedback",
        json={
            "fields": {
                "number": "123",  # подтверждено
                "seller.inn": "7710140679",  # исправлено
                "totals.amount_with_vat": "2561.990",  # то же число, другая запись
                "buyer.kpp": None,  # человек стёр значение
                "lines[0].name": "Бумага А4",  # исправлено
            },
            "author": "Иванов",
            "comment": "проверил глазами",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["changed"]) == (5, 3)
    assert body["job_id"] == job["id"]
    assert client.get(f"/v1/jobs/{job['id']}").json()["feedback_count"] == 1
    second = client.post(f"/v1/jobs/{job['id']}/feedback", json={"fields": {"number": "124"}})
    assert second.json()["feedback_id"] > body["feedback_id"]
    assert client.get(f"/v1/jobs/{job['id']}").json()["feedback_count"] == 2


def test_feedback_is_stored_with_before_and_after(client: TestClient) -> None:
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    client.post(f"/v1/jobs/{job['id']}/feedback", json={"fields": {"seller.name": "ООО «Ромашка»"}})
    store: Store = client.app.state.store  # type: ignore[attr-defined]
    row = store._db.execute("SELECT * FROM feedback").fetchone()
    assert row["changed"] == 1
    assert "ПАО Сбербанк" in row["corrections_json"]
    assert "Ромашка" in row["corrections_json"]


@pytest.mark.parametrize(
    "fields",
    [{}, {"Seller.INN": "1"}, {"lines[x].price": "1"}, {"a b": "1"}, {"number": 5}],
)
def test_feedback_validates_body(client: TestClient, fields: dict) -> None:
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    response = client.post(f"/v1/jobs/{job['id']}/feedback", json={"fields": fields})
    assert response.status_code == 422


def test_feedback_before_job_is_done_is_409(make_client: Callable[..., TestClient]) -> None:
    gate = threading.Event()

    def blocked(path: Path, mode: Mode) -> ExtractionResult:
        gate.wait(10)
        return stub_result()

    client = make_client(blocked)
    created = upload(client, pdf_bytes())
    try:
        response = client.post(f"/v1/jobs/{created['id']}/feedback", json={"fields": {"a": "1"}})
        assert response.status_code == 409
    finally:
        gate.set()


# --- состояние и безопасность --------------------------------------------------------------


def test_health(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["worker_alive"] is True
    assert body["ollama"] == {
        "reachable": True,
        "model": "stub-model",
        "model_available": True,
        "model_loaded": True,
        "error": None,
    }
    assert body["gpu"]["vram_free_mib"] == 3000
    assert (body["queued"], body["running"]) == (0, 0)


def test_health_is_degraded_when_ollama_is_down(tmp_path: Path) -> None:
    def down() -> tuple[OllamaHealth, GpuHealth]:
        return (
            OllamaHealth(reachable=False, model="m", error="Ollama недоступна"),
            GpuHealth(available=False, error="нет nvidia-smi"),
        )

    service = create_app(
        ServiceSettings(data_dir=tmp_path), extractor=stub_extractor, health_probe=down
    )
    with TestClient(service, base_url=BASE) as client:
        body = client.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert body["gpu"]["available"] is False


def test_foreign_host_header_is_rejected(client: TestClient) -> None:
    assert client.get("/v1/health", headers={"host": "evil.example.com"}).status_code == 400
    assert client.get("/v1/health", headers={"host": "localhost:8000"}).status_code == 200


def test_swagger_and_openapi_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/v1/extract",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/preview/{page}",
        "/v1/jobs/{job_id}/feedback",
        "/v1/health",
    } <= set(paths)
    assert client.get("/", follow_redirects=False).status_code in {302, 307}


def test_serve_listens_only_on_loopback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(
        cli_app, ["serve", "--port", "8123", "--data-dir", str(tmp_path / "d")]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["host"] == HOST == "127.0.0.1"
    assert calls[0]["port"] == 8123


def test_serve_has_no_host_option() -> None:
    result = CliRunner().invoke(cli_app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code != 0


# --- вспомогательное -----------------------------------------------------------------------


def test_number_str_has_no_exponent() -> None:
    from decimal import Decimal

    assert number_str(Decimal("1E+2")) == "100"
    assert number_str(Decimal("2440.00")) == "2440.00"
    assert number_str(1.5) == "1.5"


@pytest.mark.parametrize(
    ("before", "after", "same"),
    [
        ("10", "10.00", True),
        ("abc", " abc ", True),
        (None, "", True),
        (None, "x", False),
        ("10", "11", False),
        ("ООО А", "ООО Б", False),
    ],
)
def test_same_value(before: str | None, after: str | None, same: bool) -> None:
    assert same_value(before, after) is same


def test_extracted_values_of_failed_result_is_empty() -> None:
    meta = Meta(docai_version="t", started_at="t", source=["s"], mode="auto")
    assert extracted_values(ExtractionResult(document=None, valid=False, meta=meta)) == {}
    assert extracted_values(None) == {}
