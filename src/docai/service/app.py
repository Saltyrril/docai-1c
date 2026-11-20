"""FastAPI-приложение: приём файлов, очередь, результаты, страницы, правки, состояние."""

import json
import logging
import re
import shutil
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from docai import __version__
from docai.extract import ExtractionResult, Mode
from docai.extract.ollama import OllamaError
from docai.extract.router import InputKind, RoutingError, sniff_kind
from docai.rag.index import IndexNotReadyError
from docai.rag.pipeline import AskResult, RagPipeline
from docai.service.exchange import ExchangeError, FileExchange
from docai.service.health import HealthProbe, default_probe
from docai.service.models import (
    FeedbackIn,
    FeedbackOut,
    HealthView,
    JobCreated,
    JobStatus,
    JobView,
)
from docai.service.preview import preview_path
from docai.service.settings import ServiceSettings
from docai.service.store import JobRow, Store
from docai.service.views import build_job_view, extracted_values, same_value
from docai.service.worker import Extractor, Worker, make_extractor

log = logging.getLogger("docai.service")

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"
TOO_LARGE = "файл больше допустимого размера"
_CHUNK = 1024 * 1024
# Запас на служебные части multipart сверх размера файла.
_BODY_OVERHEAD = 64 * 1024
_FILENAME_BAD = re.compile(r"[\x00-\x1f\x7f]")


class BodyLimitMiddleware:
    """Обрывает запрос, тело которого больше лимита, не дожидаясь конца загрузки.

    Starlette разбирает multipart целиком и складывает файл во временный файл, поэтому
    без этой защиты 20 МБ проверялись бы только после того, как всё уже записано на диск.
    Если Content-Length есть, запрос отклоняется сразу; иначе — как только набралось
    больше лимита (HTTPException из `receive` FastAPI пробрасывает как есть, не в 400).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            await self._reject(send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(413, TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps({"detail": TOO_LARGE}, ensure_ascii=False).encode()
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="Вопрос на русском")
    debug: bool = Field(False, description="Вернуть кандидатов поиска и контекст модели")


RagFactory = Callable[[], RagPipeline]


def default_rag_factory(settings: ServiceSettings) -> RagFactory:
    def make() -> RagPipeline:
        from docai.rag.config import load_rag_config

        return RagPipeline(load_rag_config(settings.rag_config))

    return make


class Utf8JsonMiddleware:
    """Дописывает `charset=utf-8` к `application/json` во всех ответах, включая ошибки.

    Без него Windows PowerShell 5.1 (`Invoke-RestMethod`) читает тело как ISO-8859-1 и
    показывает кириллицу кракозябрами (D75). JSON и так всегда в UTF-8 (RFC 8259).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_charset(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [
                    (k, b"application/json; charset=utf-8")
                    if k.lower() == b"content-type" and v == b"application/json"
                    else (k, v)
                    for k, v in message.get("headers", [])
                ]
            await send(message)

        await self.app(scope, receive, send_with_charset)


def detect_upload(head: bytes) -> tuple[InputKind, str]:
    """Тип и расширение по первым байтам. Допустимы PDF, PNG, JPEG и XML."""
    try:
        kind = sniff_kind(head)
    except RoutingError as exc:
        raise HTTPException(415, "ожидаются PDF, PNG, JPEG или XML") from exc
    if kind is InputKind.IMAGE:
        if head.startswith(_PNG):
            return kind, ".png"
        if head.startswith(_JPEG):
            return kind, ".jpg"
        raise HTTPException(415, "из картинок принимаются только PNG и JPEG")
    return kind, ".pdf" if kind is InputKind.PDF else ".xml"


def _clean_filename(name: str | None) -> str:
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = _FILENAME_BAD.sub("", base).strip()
    return base[:255] or "upload"


def create_app(
    settings: ServiceSettings | None = None,
    *,
    extractor: Extractor | None = None,
    health_probe: HealthProbe | None = None,
    rag_factory: RagFactory | None = None,
) -> FastAPI:
    """Собрать приложение. `extractor`, `health_probe`, `rag_factory` подменяются в тестах."""
    settings = settings or ServiceSettings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.db_path)
    worker = Worker(store, extractor or make_extractor(settings.extract), settings)
    probe = health_probe or default_probe(settings.extract)

    def load_result(row: JobRow) -> ExtractionResult | None:
        if row.result_json is None:
            return None
        return ExtractionResult.model_validate_json(row.result_json)

    def enqueue(stream: BinaryIO, filename: str | None, mode: Mode) -> JobCreated:
        """Сохранить файл и поставить в очередь. Общая часть HTTP и обмена файлами."""
        if store.count("queued") >= settings.max_queued:
            raise HTTPException(429, "очередь заполнена, повторите позже")
        head = stream.read(64)
        if not head:
            raise HTTPException(400, "файл пустой")
        kind, ext = detect_upload(head)

        job_id = uuid.uuid4().hex
        job_dir = settings.job_dir(job_id)
        job_dir.mkdir(parents=True)
        stored_name = f"source{ext}"
        size = len(head)
        try:
            with (job_dir / stored_name).open("wb") as out:
                out.write(head)
                while chunk := stream.read(_CHUNK):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, TOO_LARGE)
                    out.write(chunk)
            store.create_job(
                job_id,
                filename=_clean_filename(filename),
                stored_name=stored_name,
                source_kind=kind.value,
                mode=mode.value,
                size_bytes=size,
            )
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        worker.notify()
        status: JobStatus = "queued"
        return JobCreated(id=job_id, status=status, queue_ahead=store.queue_ahead(job_id) or 0)

    def job_view(job_id: str) -> JobView | None:
        row = store.get_job(job_id)
        if row is None:
            return None
        return build_job_view(
            row,
            load_result(row),
            queue_ahead=store.queue_ahead(job_id),
            feedback_count=store.feedback_count(job_id),
        )

    def health_view() -> HealthView:
        ollama, gpu = probe()
        ok = worker.alive and ollama.reachable and ollama.model_available
        return HealthView(
            status="ok" if ok else "degraded",
            version=__version__,
            worker_alive=worker.alive,
            queued=store.count("queued"),
            running=store.count("running"),
            ollama=ollama,
            gpu=gpu,
        )

    def enqueue_file(path: Path, filename: str) -> str:
        """Для обмена файлами: ошибки приёма превращаются в текст ответа."""
        try:
            with path.open("rb") as stream:
                return enqueue(stream, filename, Mode.AUTO).id
        except HTTPException as exc:
            raise ExchangeError(str(exc.detail)) from exc

    exchange = (
        FileExchange(
            settings.exchange_dir,
            health=health_view,
            enqueue=enqueue_file,
            job_view=job_view,
            poll_interval_s=settings.exchange_poll_s,
        )
        if settings.exchange_enabled
        else None
    )

    make_rag = rag_factory or default_rag_factory(settings)
    rag_lock = threading.Lock()
    rag_state: dict[str, RagPipeline] = {}

    def get_rag() -> RagPipeline:
        """Конвейер RAG создаётся один раз, при первом вопросе (или прогреве)."""
        with rag_lock:
            if "pipeline" not in rag_state:
                rag_state["pipeline"] = make_rag()
            return rag_state["pipeline"]

    def warm_up_rag() -> None:
        try:
            get_rag().warm_up()
            log.info("RAG: модели загружены")
        except Exception as exc:  # прогрев необязателен: ошибка придёт с первым вопросом
            log.warning("RAG: прогрев не удался: %s", exc)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        worker.start()
        if settings.rag_enabled and settings.rag_warm_up:
            threading.Thread(target=warm_up_rag, name="rag-warm-up", daemon=True).start()
        if exchange is not None:
            exchange.start()
        try:
            yield
        finally:
            if exchange is not None:
                exchange.stop()
            worker.stop()
            store.close()

    app = FastAPI(
        title="docai-1c",
        version=__version__,
        description=(
            "Локальный сервис извлечения реквизитов из первичных документов для 1С. "
            "Слушает только 127.0.0.1. Контракт и примеры — docs/API.md."
        ),
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.worker = worker
    app.state.settings = settings
    app.state.exchange = exchange
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + _BODY_OVERHEAD)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(Utf8JsonMiddleware)

    def get_row(job_id: str) -> JobRow:
        row = store.get_job(job_id)
        if row is None:
            raise HTTPException(404, "задание не найдено")
        return row

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.post("/v1/extract", status_code=202, response_model=JobCreated, tags=["Задания"])
    def create_job(
        file: Annotated[UploadFile, File(description="PDF, PNG, JPEG или XML, до 20 МБ")],
        mode: Annotated[Mode, Form(description="Путь извлечения; auto выбирает сам")] = Mode.AUTO,
    ) -> JobCreated:
        """Принять файл и поставить его в очередь. Результат — `GET /v1/jobs/{id}`."""
        return enqueue(file.file, file.filename, mode)

    @app.get("/v1/jobs/{job_id}", response_model=JobView, tags=["Задания"])
    def get_job(job_id: str) -> JobView:
        """Состояние задания; при `status = done` — поля, строки, уверенность и ошибки проверок."""
        view = job_view(job_id)
        if view is None:
            raise HTTPException(404, "задание не найдено")
        return view

    @app.get(
        "/v1/jobs/{job_id}/preview/{page}",
        response_class=FileResponse,
        responses={200: {"content": {"image/png": {}}}},
        tags=["Задания"],
    )
    def get_preview(job_id: str, page: int) -> FileResponse:
        """Страница документа как PNG (нумерация с 1) для формы проверки."""
        row = get_row(job_id)
        if row.status != "done":
            raise HTTPException(409, "задание ещё не выполнено")
        if not 1 <= page <= row.preview_pages:
            raise HTTPException(404, "такой страницы нет")
        path = preview_path(settings.job_dir(job_id), page)
        if not path.is_file():
            raise HTTPException(404, "картинка страницы не найдена")
        return FileResponse(path, media_type="image/png")

    @app.post("/v1/jobs/{job_id}/feedback", response_model=FeedbackOut, tags=["Задания"])
    def post_feedback(job_id: str, body: FeedbackIn) -> FeedbackOut:
        """Поля, которые человек проверил или исправил: данные для дообучения и метрика правок."""
        row = get_row(job_id)
        if row.status != "done":
            raise HTTPException(409, "правки принимаются только у выполненных заданий")
        before = extracted_values(load_result(row))
        corrections = []
        for name, after in body.fields.items():
            old = before.get(name)
            corrections.append(
                {
                    "field": name,
                    "before": old,
                    "after": after,
                    "changed": not same_value(old, after),
                }
            )
        feedback_id, changed = store.add_feedback(
            job_id, author=body.author, comment=body.comment, corrections=corrections
        )
        return FeedbackOut(
            job_id=job_id, feedback_id=feedback_id, total=len(corrections), changed=changed
        )

    @app.post("/v1/rag/ask", response_model=AskResult, tags=["RAG"])
    def rag_ask(body: AskIn) -> AskResult:
        """Вопрос по договорам, доп. соглашениям, регламентам или реестру документов.

        Ответ — текст со ссылками [n] и цитаты (id пункта, документ, полный текст пункта).
        Вопросы-агрегаты по реестру («сколько УПД получено от … за 2025 год») решаются
        запросом SQL к реестру; запрос возвращается в поле `sql`. Если ответа в документах
        нет — `found = false` и «В документах не нашёл.».
        """
        if not settings.rag_enabled:
            raise HTTPException(404, "RAG выключен в настройках сервиса")
        try:
            pipeline = get_rag()
        except IndexNotReadyError as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            return pipeline.ask(body.question, debug=body.debug)
        except IndexNotReadyError as exc:
            raise HTTPException(503, str(exc)) from exc
        except OllamaError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/v1/health", response_model=HealthView, tags=["Служебное"])
    def health() -> HealthView:
        """Жива ли очередь, доступна ли Ollama и загружена ли модель, сколько VRAM свободно."""
        return health_view()

    return app
