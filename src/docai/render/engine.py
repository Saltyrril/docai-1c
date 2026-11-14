"""Рендер документа в PDF через Chromium (Playwright) и разметка расположения полей.

Схема работы: документ → контекст Jinja2 → HTML → страница Chromium → JS режет таблицу по
страницам и собирает bbox всех элементов с `data-field` → печать в PDF. Координаты bbox —
пункты PDF (1/72 дюйма), начало в левом верхнем углу страницы, ось Y вниз.
"""

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import jinja2
import pypdfium2 as pdfium
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from docai.render import format as fmt
from docai.schemas.common import BaseDocument, DocType

TEMPLATES_DIR = Path(__file__).parent / "templates"
VARIANTS = (1, 2, 3)
LAYOUT_SCHEMA_VERSION = 1

# Ориентация листа: УПД, ТОРГ-12 и счёт-фактура печатаются альбомной, счёт и акт — книжной.
ORIENTATION = {
    DocType.UPD: "landscape",
    DocType.TORG12: "landscape",
    DocType.VAT_INVOICE: "landscape",
    DocType.BILL: "portrait",
    DocType.ACT: "portrait",
}

# Где в шаблоне есть место под логотип (в официальных формах первого варианта его нет).
LOGO_SLOTS: dict[DocType, frozenset[int]] = {
    DocType.UPD: frozenset({2, 3}),
    DocType.TORG12: frozenset({2, 3}),
    DocType.VAT_INVOICE: frozenset({2, 3}),
    DocType.BILL: frozenset({1, 2, 3}),
    DocType.ACT: frozenset({1, 2, 3}),
}

# Поля, которых нет в печатной форме данного типа: в форме просто нет такой графы.
NOT_PRINTED: dict[DocType, frozenset[str]] = {
    DocType.BILL: frozenset({"lines[].unit_code"}),
    DocType.ACT: frozenset({"lines[].unit_code"}),
}

_LOGO_COLORS = ("#1f6feb", "#c0392b", "#16825d", "#8e44ad", "#d35400", "#2c3e50")
_STAMP_COLORS = ("#1a3a9c", "#27408b", "#2b4eae")


class RenderError(RuntimeError):
    """Документ не удалось отрисовать или в PDF не попали все поля."""


@dataclass(frozen=True)
class Decor:
    """Декоративные заглушки документа: логотип, печать, подпись (None — их нет)."""

    logo: dict[str, Any] | None
    stamp: dict[str, Any] | None
    sign: dict[str, Any] | None


def choose_look(
    doc_type: DocType, key: str, seed: int, variant: int | None = None
) -> tuple[int, Decor]:
    """Детерминированно выбрать вариант вёрстки и заглушки по (seed, ключ документа)."""
    rng = random.Random(f"{seed}:{key}")
    chosen = variant if variant is not None else rng.choice(VARIANTS)
    has_logo = rng.random() < 0.5 and chosen in LOGO_SLOTS[doc_type]
    has_stamp = rng.random() < 0.7
    has_sign = rng.random() < 0.8
    logo = {"shape": rng.randrange(3), "color": rng.choice(_LOGO_COLORS)}
    stamp = {"angle": rng.randint(-14, 14), "color": rng.choice(_STAMP_COLORS)}
    sign = {"kind": rng.randrange(3)}
    return chosen, Decor(
        logo=logo if has_logo else None,
        stamp=stamp if has_stamp else None,
        sign=sign if has_sign else None,
    )


def _make_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        undefined=jinja2.Undefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        money=fmt.money,
        price=fmt.price,
        qty=fmt.quantity,
        dnum=fmt.date_num,
        dlong=fmt.date_long,
        vat=fmt.vat_rate,
        words=fmt.amount_in_words,
    )
    return env


def build_html(doc: BaseDocument, variant: int, decor: Decor) -> str:
    """Собрать HTML документа из шаблона `<тип>/v<вариант>.html.j2`."""
    template = _make_env().get_template(f"{doc.doc_type.value}/v{variant}.html.j2")
    return template.render(
        d=doc,
        v=variant,
        deco=decor,
        orientation=ORIENTATION[doc.doc_type],
        title=f"{doc.doc_type.title_ru} № {doc.number}",
        base_css=(TEMPLATES_DIR / "_base.css").read_text(encoding="utf-8")
        + (TEMPLATES_DIR / "_look.css").read_text(encoding="utf-8"),
        paginate_js=(TEMPLATES_DIR / "_paginate.js").read_text(encoding="utf-8"),
    )


def expected_fields(doc: BaseDocument) -> set[str]:
    """Пути всех непустых полей документа, которые должны быть напечатаны в форме."""
    skip = NOT_PRINTED.get(doc.doc_type, frozenset())
    found: set[str] = set()

    def walk(node: Any, path: str, generic: str) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if not path and key == "doc_type":
                    continue
                walk(
                    value, f"{path}.{key}" if path else key, f"{generic}.{key}" if generic else key
                )
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]", f"{generic}[]")
        elif generic not in skip:
            found.add(path)

    walk(doc.model_dump(mode="json"), "", "")
    return found


def pdf_text(pdf: bytes) -> list[str]:
    """Текстовый слой PDF по страницам."""
    document = pdfium.PdfDocument(pdf)
    try:
        return [document[i].get_textpage().get_text_range() for i in range(len(document))]
    finally:
        document.close()


def normalize_text(text: str) -> str:
    """Свернуть пробелы (в том числе неразрывные) для сравнения текста из PDF."""
    return " ".join(text.replace(" ", " ").replace(" ", " ").split())


@dataclass
class RenderResult:
    """Готовый PDF и разметка расположения полей."""

    pdf: bytes
    layout: dict[str, Any]


class Renderer:
    """Контекстный менеджер: один запуск Chromium на серию документов."""

    def __init__(self) -> None:
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def __enter__(self) -> Self:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page()
        self._page.emulate_media(media="print")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def render(
        self,
        doc: BaseDocument,
        *,
        variant: int,
        decor: Decor,
        source: str = "",
        check: bool = True,
    ) -> RenderResult:
        """Отрисовать документ. При `check` убедиться, что все поля напечатаны и есть в PDF."""
        if self._page is None:
            raise RenderError("Renderer нужно использовать через `with`.")
        if variant not in VARIANTS:
            raise RenderError(f"Вариант вёрстки {variant} не существует, есть {VARIANTS}.")
        page = self._page
        page.set_content(build_html(doc, variant, decor), wait_until="load")
        measured = page.evaluate("docaiLayout()")
        pdf = page.pdf(prefer_css_page_size=True, print_background=True)

        pdf_doc = pdfium.PdfDocument(pdf)
        pages_in_pdf = len(pdf_doc)
        real_size = [round(v, 2) for v in pdf_doc[0].get_size()]
        pdf_doc.close()
        if pages_in_pdf != measured["page_count"]:
            raise RenderError(
                f"{source}: в PDF {pages_in_pdf} стр., а разметка рассчитана на "
                f"{measured['page_count']}: страницы разъехались."
            )
        layout: dict[str, Any] = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "source": source,
            "doc_type": doc.doc_type.value,
            "variant": variant,
            "orientation": ORIENTATION[doc.doc_type],
            "decor": {k: v is not None for k, v in asdict(decor).items()},
            "page_count": measured["page_count"],
            "page_size_pt": real_size,
            "fields": measured["fields"],
        }
        if check:
            problems = check_render(doc, pdf, layout)
            if problems:
                raise RenderError(f"{source}: " + "; ".join(problems[:5]))
        return RenderResult(pdf=pdf, layout=layout)


def check_render(doc: BaseDocument, pdf: bytes, layout: dict[str, Any]) -> list[str]:
    """Проверки результата: все поля имеют bbox, их текст есть в текстовом слое PDF,
    строки таблицы не обрезаны (bbox внутри страницы)."""
    problems: list[str] = []
    printed = {entry["field"] for entry in layout["fields"]}
    missing = sorted(expected_fields(doc) - printed)
    if missing:
        problems.append(f"поля не напечатаны: {', '.join(missing[:8])}")
    width, height = layout["page_size_pt"]
    text = normalize_text(" ".join(pdf_text(pdf)))
    for entry in layout["fields"]:
        x0, y0, x1, y1 = entry["bbox"]
        if x0 < -0.5 or y0 < -0.5 or x1 > width + 0.5 or y1 > height + 0.5:
            problems.append(f"{entry['field']}: bbox вышел за страницу")
        if normalize_text(entry["text"]) not in text:
            problems.append(f"{entry['field']}: «{entry['text'][:30]}» нет в текстовом слое PDF")
    return problems
