"""Конвейер извлечения: маршрут → подготовка входа → LLM (или парсер XML) → документ.

Ошибки окружения (нет Ollama, нет Tesseract, файл не подходит под режим) поднимаются
исключением. Ошибки содержания (модель вернула не то, документ не собрался) попадают
в `ExtractionResult.errors`, а `valid` становится False.
"""

import json
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from docai import __version__
from docai.edo.errors import EdoError
from docai.edo.reader import xml_to_upd
from docai.extract.config import ExtractConfig
from docai.extract.draft import (
    Draft,
    draft_json_schema,
    draft_to_document,
    merge_drafts,
    parse_draft,
)
from docai.extract.images import (
    fit_long_side,
    image_files_pages,
    pdf_pages_fit,
    pdf_pages_images,
    to_base64_png,
)
from docai.extract.ocr import check_languages, find_tesseract, ocr_image, tesseract_version
from docai.extract.ollama import build_chat_request, chat
from docai.extract.pdftext import pdf_pages_text
from docai.extract.prompts import render_prompt
from docai.extract.result import ExtractionResult, LlmCall, Meta
from docai.extract.router import InputKind, Mode, Route, route_inputs
from docai.extract.vram import VramMonitor
from docai.postprocess import PostprocessConfig, SourceKind, postprocess
from docai.schemas.documents import AnyDocument, parse_document

log = logging.getLogger("docai.extract")


class _Run:
    """Состояние одного запуска: замеры этапов, вызовы LLM, предупреждения и ошибки."""

    def __init__(self, config: ExtractConfig, meta: Meta) -> None:
        self.config = config
        self.meta = meta
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.schema = draft_json_schema(config.date_format)
        # Сообщение об обрезанном ответе последнего запроса (None, если ответ полный).
        self.truncated: str | None = None
        # Тексты-источники (текстовый слой, OCR) для проверки «значение дословно есть».
        self.texts: list[tuple[SourceKind, str]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Замер этапа; повторные этапы с тем же именем суммируются."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            timings = self.meta.timings_s
            timings[name] = round(timings.get(name, 0.0) + elapsed, 3)
            log.info("этап %-8s %7.2f с", name, elapsed)

    def ask(
        self,
        prompt: str,
        part: str,
        image: Image.Image | None = None,
        text_chars: int | None = None,
    ) -> Draft | None:
        """Один запрос к модели → черновик (или None с ошибкой в списке)."""
        images = [to_base64_png(image)] if image is not None else None
        body = build_chat_request(self.config, prompt, self.schema, images)
        with self.stage("llm"):
            answer = chat(self.config, body)
        call = LlmCall(
            part=part,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            done_reason=answer.done_reason,
            total_s=answer.total_s,
            load_s=answer.load_s,
            prompt_eval_s=answer.prompt_eval_s,
            eval_s=answer.eval_s,
            image_size=image.size if image is not None else None,
            text_chars=text_chars,
        )
        self.meta.llm_calls.append(call)
        self.meta.tokens.prompt += call.prompt_tokens
        self.meta.tokens.completion += call.completion_tokens
        log.info(
            "LLM %s: токенов %d → %d, %.2f с (загрузка %.2f с), done_reason=%s",
            part,
            call.prompt_tokens,
            call.completion_tokens,
            call.total_s,
            call.load_s,
            call.done_reason,
        )
        if answer.thinking:
            self.warnings.append(f"{part}: модель вернула рассуждения, хотя think=false")
        self.truncated = None
        if answer.done_reason == "length":
            self.truncated = (
                f"{part}: ответ обрезан по num_predict={self.config.options.num_predict}"
            )
            self.errors.append(self.truncated)
            return None
        try:
            raw = answer.parse_json()
        except json.JSONDecodeError as exc:
            self.errors.append(f"{part}: ответ модели — не JSON ({exc})")
            return None
        try:
            draft, notes = parse_draft(raw, self.config.date_format)
        except (ValidationError, ValueError) as exc:
            self.errors.append(f"{part}: ответ не по схеме ({exc})")
            return None
        self.warnings.extend(f"{part}: {note}" for note in notes)
        return draft


def _page_parts(pages: list[str]) -> list[tuple[str, str]]:
    """Текст по страницам, каждая с пометкой «Страница N из M»."""
    total = len(pages)
    return [
        (f"page {i}/{total}", f"--- Страница {i} из {total} ---\n{text}")
        for i, text in enumerate(pages, 1)
    ]


def _text_parts(pages: list[str], max_chars: int) -> list[tuple[str, str]]:
    """Текст документа одним куском или, если он слишком длинный, по страницам."""
    if len(pages) == 1:
        return [("document", pages[0])]
    by_page = _page_parts(pages)
    whole = "\n\n".join(text for _, text in by_page)
    if len(whole) <= max_chars:
        return [("document", whole)]
    return by_page


def _ask_texts(run: _Run, parts: list[tuple[str, str]], source: str, title: str) -> list[Draft]:
    drafts = []
    for part, text in parts:
        prompt = render_prompt(
            run.config.text_prompt, source=source, source_title=title, document_text=text
        )
        draft = run.ask(prompt, part, text_chars=len(text))
        if draft is not None:
            drafts.append(draft)
    return drafts


def _drafts_from_texts(run: _Run, pages: list[str], source: str, title: str) -> list[Draft]:
    """Весь текст одним запросом; если ответ на него обрезан, а страниц несколько, —
    повтор по страницам (у каждой страницы своя таблица строк, ответ короче)."""
    parts = _text_parts(pages, run.config.max_text_chars)
    drafts = _ask_texts(run, parts, source, title)
    if not drafts and len(parts) == 1 and len(pages) > 1 and run.truncated:
        run.errors.remove(run.truncated)
        run.warnings.append(f"{run.truncated} → повтор по страницам")
        drafts = _ask_texts(run, _page_parts(pages), source, title)
    return drafts


def _page_info(index: int, total: int) -> str:
    return "единственная страница" if total == 1 else f"страница {index} из {total}"


def _load_images(run: _Run, paths: Sequence[Path], kind: InputKind, dpi: int | None) -> list:
    """Страницы как картинки. dpi=None — сразу под длинную сторону для vision-модели."""
    with run.stage("render"):
        if kind is InputKind.PDF:
            if dpi is None:
                return pdf_pages_fit(paths[0], run.config.max_image_side)
            return pdf_pages_images(paths[0], dpi)
        return image_files_pages(paths)


def _run_text(run: _Run, path: Path) -> list[Draft]:
    run.meta.prompt_version = run.config.text_prompt
    with run.stage("pdf_text"):
        pages = pdf_pages_text(path)
    run.meta.pages = len(pages)
    run.texts.append(("text_layer", "\n".join(pages)))
    return _drafts_from_texts(run, pages, "текстовый слой PDF", "Текст документа")


def _run_vision(run: _Run, paths: Sequence[Path], kind: InputKind) -> list[Draft]:
    run.meta.prompt_version = run.config.vision_prompt
    images = _load_images(run, paths, kind, dpi=None)
    run.meta.pages = len(images)
    drafts = []
    for index, image in enumerate(images, 1):
        with run.stage("resize"):
            fitted = fit_long_side(image, run.config.max_image_side)
        prompt = render_prompt(run.config.vision_prompt, page_info=_page_info(index, len(images)))
        draft = run.ask(prompt, f"page {index}/{len(images)}", image=fitted)
        if draft is not None:
            drafts.append(draft)
    return drafts


def _run_ocr(run: _Run, paths: Sequence[Path], kind: InputKind) -> list[Draft]:
    config = run.config
    run.meta.prompt_version = config.text_prompt
    with run.stage("ocr_init"):
        cmd = find_tesseract(config.tesseract_cmd)
        check_languages(cmd, config.ocr_lang)
        run.meta.ocr_engine = (
            f"{tesseract_version(cmd)}; lang={config.ocr_lang}; psm={config.ocr_psm}"
        )
    images = _load_images(run, paths, kind, dpi=config.ocr_dpi)
    run.meta.pages = len(images)
    texts = []
    for image in images:
        with run.stage("ocr"):
            texts.append(ocr_image(cmd, image, config.ocr_lang, config.ocr_psm))
    run.texts.append(("ocr", "\n".join(texts)))
    source = "текст, распознанный OCR; в нём возможны ошибки распознавания"
    return _drafts_from_texts(run, texts, source, "Распознанный текст")


def _run_xml(run: _Run, path: Path) -> AnyDocument | None:
    run.meta.pages = None
    try:
        with run.stage("xml"):
            return xml_to_upd(path.read_bytes())
    except EdoError as exc:
        run.errors.append(f"XML: {exc}")
        return None


def _extract_raw(
    paths: Sequence[Path], mode: Mode, config: ExtractConfig
) -> tuple[ExtractionResult, AnyDocument | None, _Run]:
    """Один путь извлечения без постобработки."""
    meta = Meta(
        docai_version=__version__,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=[p.name for p in paths],
        mode=str(mode),
        config=config.model_dump(),
    )
    run = _Run(config, meta)
    started = time.perf_counter()
    document: AnyDocument | None = None

    with VramMonitor(config.vram_poll_s) as vram:
        try:
            with run.stage("route"):
                decision = route_inputs(paths, mode, config.min_text_chars_per_page)
            meta.route = str(decision.route)
            meta.route_reason = decision.reason
            meta.input_kind = str(decision.kind)
            log.info("маршрут: %s (%s)", decision.route, decision.reason)

            if decision.route is Route.XML:
                document = _run_xml(run, paths[0])
            else:
                meta.model = config.model
                if decision.route is Route.TEXT:
                    drafts = _run_text(run, paths[0])
                elif decision.route is Route.VISION:
                    drafts = _run_vision(run, paths, decision.kind)
                else:
                    drafts = _run_ocr(run, paths, decision.kind)
                if drafts:
                    with run.stage("assemble"):
                        document, notes, problems = draft_to_document(merge_drafts(drafts))
                    run.warnings.extend(notes)
                    run.errors.extend(problems)
                else:
                    run.errors.append("ни один запрос к модели не дал черновика")
        finally:
            meta.timings_s["total"] = round(time.perf_counter() - started, 3)
    meta.vram = vram.stats()

    peak = meta.vram.peak_gib
    if peak is not None:
        log.info("пиковая VRAM: %d МиБ (%.2f ГиБ)", meta.vram.peak_mib, peak)
        if peak > config.vram_limit_gib:
            run.warnings.append(f"пиковая VRAM {peak:.2f} ГиБ выше порога {config.vram_limit_gib}")

    result = ExtractionResult(
        document=document.model_dump(mode="json") if document is not None else None,
        valid=document is not None,
        warnings=run.warnings,
        errors=run.errors,
        meta=meta,
    )
    return result, document, run


class RawExtraction(BaseModel):
    """Извлечение без постобработки: «сырой» ответ модели и всё, что нужно постобработке.

    Нужен оценке (P11): ответы модели кэшируются, а постобработка с любыми константами
    (D42) пересчитывается по кэшу без повторного запуска LLM.
    """

    # Результат основного пути; `document` в нём — документ до постобработки.
    result: ExtractionResult
    # Тексты-источники (текстовый слой, OCR) основного и второго путей.
    texts: list[tuple[SourceKind, str]] = Field(default_factory=list)
    # Документы второго пути (--cross-check) в JSON.
    alternates: list[dict[str, Any]] = Field(default_factory=list)


def extract_raw(
    paths: Sequence[Path],
    mode: Mode = Mode.AUTO,
    config: ExtractConfig | None = None,
    cross_check: Mode | None = None,
) -> RawExtraction:
    """Основной путь извлечения и (если задан) второй путь для сверки, без постобработки."""
    config = config or ExtractConfig()
    result, _, run = _extract_raw(paths, mode, config)
    route = result.meta.route or "vision"
    texts = list(run.texts)
    alternates: list[dict[str, Any]] = []
    if cross_check is not None and route != str(Route.XML):
        second, second_doc, second_run = _extract_raw(paths, cross_check, config)
        result.meta.cross_check = str(second.meta.route)
        result.meta.cross_check_total_s = second.meta.timings_s.get("total")
        result.warnings.extend(f"cross-check: {w}" for w in second.warnings)
        result.errors.extend(f"cross-check: {e}" for e in second.errors)
        texts += second_run.texts
        if second_doc is not None:
            alternates.append(second_doc.model_dump(mode="json"))
    return RawExtraction(result=result, texts=texts, alternates=alternates)


def finish(raw: RawExtraction, config: PostprocessConfig | None = None) -> ExtractionResult:
    """Постобработка «сырого» извлечения: нормализация, исправления, уверенность, статус.

    Исходный `raw` не меняется.
    """
    result = raw.result.model_copy(deep=True)
    document = parse_document(result.document) if result.document is not None else None
    alternates = [parse_document(alt) for alt in raw.alternates]
    started = time.perf_counter()
    done = postprocess(
        document,
        route=result.meta.route or "vision",
        texts=raw.texts,
        alternates=alternates,
        config=config,
    )
    result.meta.timings_s["postprocess"] = round(time.perf_counter() - started, 3)
    result.postprocess = done.report
    if done.document is not None:
        result.document = done.document.model_dump(mode="json")
    return result


def extract(
    paths: Sequence[Path],
    mode: Mode = Mode.AUTO,
    config: ExtractConfig | None = None,
    cross_check: Mode | None = None,
) -> ExtractionResult:
    """Извлечь документ из файла (или нескольких картинок-страниц одного документа).

    После LLM документ нормализуется, проверяется, при однозначной ошибке исправляется,
    по каждому полю считается уверенность и выставляется статус (`result.postprocess`, D41–D43).
    `cross_check` — второй путь для сверки (например, ocr при основном vision): значения,
    совпавшие на двух путях, получают больше уверенности, расхождения — меньше.
    """
    config = config or ExtractConfig()
    return finish(extract_raw(paths, mode, config, cross_check), config.postprocess)
