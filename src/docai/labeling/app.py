"""Интерфейс разметки на Gradio: слева документ, справа форма по FIELD_SPEC и таблица строк.

Логика вынесена в `LabelingApp` (без Gradio-объектов в аргументах), чтобы её можно было
тестировать обычными вызовами; `build_demo` лишь связывает методы с виджетами.

Приватность: сервер слушает только 127.0.0.1, телеметрия Gradio выключена, картинки
передаются в браузер в памяти, а не через раздачу папок (`allowed_paths` не используется).
"""

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from docai.labeling.form import (
    HEADER_SPECS,
    LINE_COLUMNS,
    LINE_HEADERS,
    FormError,
    build_document,
    check_document,
    document_to_form,
    empty_lines,
    format_issues,
)
from docai.labeling.prefill import PrefillUnavailable, prefill_document
from docai.labeling.store import CAPTURE_KINDS, DEFAULT_ROOT, LabelStore, Source, StoreError
from docai.schemas import DocType, FieldSpec
from docai.validation import has_errors

if TYPE_CHECKING:
    import gradio as gr

HOST = "127.0.0.1"
PORT = 7860

# Разделы формы: (заголовок, префикс имени поля; пустой — поля шапки).
SECTIONS: tuple[tuple[str, str], ...] = (
    ("Шапка", ""),
    ("Продавец", "seller."),
    ("Покупатель", "buyer."),
    ("Банк продавца", "seller_bank."),
    ("Основание", "basis."),
    ("Итоги", "totals."),
)
# Документ слева «прилипает» к верху окна, пока заполняется длинная форма справа.
CSS = "#doc-col { position: sticky; top: 8px; align-self: flex-start; }"


def disable_telemetry() -> None:
    """Выключить телеметрию и проверку обновлений Gradio. Вызывать до `import gradio`."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def section_of(spec: FieldSpec) -> str:
    """Заголовок раздела формы, в котором лежит поле."""
    for title, prefix in SECTIONS[1:]:
        if spec.name.startswith(prefix):
            return title
    return SECTIONS[0][0]


@dataclass(slots=True)
class LoadedDocument:
    """Всё, что нужно показать при открытии документа."""

    doc_id: str
    page: Image.Image | None
    pages: int
    doc_type: str | None = None
    capture: str | None = None
    header: dict[str, str] = field(default_factory=dict)
    lines: list[list[str]] = field(default_factory=empty_lines)
    note: str = ""
    accepted_errors: bool = False
    seconds_before: float = 0.0
    status: str = ""


@dataclass(frozen=True, slots=True)
class SaveResult:
    ok: bool
    message: str


class LabelingApp:
    """Действия интерфейса: открыть документ, сохранить эталон, перейти к следующему."""

    def __init__(self, store: LabelStore) -> None:
        self.store = store

    # ---- список ------------------------------------------------------------------------

    def choices(self) -> list[tuple[str, str]]:
        """(подпись, id) для выпадающего списка; ✓ — уже размечен."""
        return [
            (("✓ " if self.store.is_labeled(s.id) else "• ") + s.id, s.id)
            for s in self.store.list_sources()
        ]

    def progress_text(self) -> str:
        done, total = self.store.progress()
        return f"Размечено **{done}** из **{total}**"

    def next_unlabeled(self, current: str | None) -> str | None:
        """Следующий неразмеченный документ после текущего (по кругу) или None."""
        sources = self.store.list_sources()
        ids = [s.id for s in sources]
        start = ids.index(current) + 1 if current in ids else 0
        for source in sources[start:] + sources[:start]:
            if source.id != current and not self.store.is_labeled(source.id):
                return source.id
        return None

    def neighbour(self, current: str | None, step: int) -> str | None:
        ids = [s.id for s in self.store.list_sources()]
        if not ids:
            return None
        if current not in ids:
            return ids[0]
        return ids[(ids.index(current) + step) % len(ids)]

    # ---- открыть -----------------------------------------------------------------------

    def render_page(self, doc_id: str, page: int) -> Image.Image:
        source = self.store.get_source(doc_id)
        count = self.store.page_count(source)
        return self.store.render_page(source, min(max(page, 1), count) - 1)

    def load(self, doc_id: str) -> LoadedDocument:
        source: Source = self.store.get_source(doc_id)
        loaded = LoadedDocument(
            doc_id=doc_id,
            page=self.store.render_page(source, 0),
            pages=self.store.page_count(source),
        )
        meta = self.store.load_meta(doc_id)
        loaded.capture = meta.get("capture")
        loaded.note = meta.get("note", "")
        loaded.accepted_errors = bool(meta.get("accepted_errors", False))
        loaded.seconds_before = float(meta.get("seconds", 0.0))
        document = self.store.load_label(doc_id)
        if document is not None:
            loaded.doc_type, loaded.header, loaded.lines = document_to_form(document)
            loaded.status = "Эталон уже сохранён; можно править и сохранить заново."
        return loaded

    # ---- сохранить ---------------------------------------------------------------------

    def check(
        self, doc_type: str | None, header: Mapping[str, Any], lines: Sequence[Sequence[Any]]
    ) -> str:
        """Проверка без сохранения (markdown)."""
        try:
            document = build_document(doc_type or "", header, lines)
        except FormError as exc:
            return "**Документ ещё нельзя собрать:**\n" + "\n".join(
                f"- ❌ {m}" for m in exc.messages
            )
        return format_issues(check_document(document))

    def save(
        self,
        doc_id: str | None,
        doc_type: str | None,
        capture: str | None,
        note: str,
        accepted_errors: bool,
        header: Mapping[str, Any],
        lines: Sequence[Sequence[Any]],
        elapsed_seconds: float = 0.0,
    ) -> SaveResult:
        """Собрать документ, проверить валидаторами и записать эталон.

        Ошибки схемы (нет номера, непонятная дата) сохранение блокируют всегда.
        Ошибки проверок (неверная контрольная цифра ИНН, не сходится сумма) блокируют, пока не
        отмечено «так напечатано в документе»: настоящие документы бывают неверными, и
        в тестовом наборе такой документ должен остаться с тем, что в нём написано.
        """
        if not doc_id:
            return SaveResult(False, "❌ Не выбран документ.")
        try:
            document = build_document(doc_type or "", header, lines)
        except FormError as exc:
            body = "\n".join(f"- ❌ {m}" for m in exc.messages)
            return SaveResult(False, f"**Не сохранено.** Исправьте:\n{body}")
        issues = check_document(document)
        if has_errors(issues) and not accepted_errors:
            return SaveResult(
                False,
                "**Не сохранено:** проверки нашли ошибки. Если в самом документе так напечатано, "
                "отметьте «Так в документе» и сохраните ещё раз.\n\n" + format_issues(issues),
            )
        previous = self.store.load_meta(doc_id)
        meta = {
            "source": self.store.get_source(doc_id).path.name,
            "capture": capture,
            "note": note.strip(),
            "accepted_errors": bool(accepted_errors and has_errors(issues)),
            "issues": [f"{i.severity.value}:{i.code.value}" for i in issues],
            "seconds": round(float(previous.get("seconds", 0.0)) + elapsed_seconds, 1),
        }
        path = self.store.save(doc_id, document, meta)
        return SaveResult(True, f"✅ Сохранено: `{path.name}`\n\n{format_issues(issues)}")

    def prefill(self, doc_id: str) -> tuple[LoadedDocument | None, str]:
        """Предзаполнение моделью (после P9). Возвращает форму или сообщение для пользователя."""
        source = self.store.get_source(doc_id)
        pages = [self.store.render_page(source, n) for n in range(self.store.page_count(source))]
        try:
            document = prefill_document(pages)
        except PrefillUnavailable as exc:
            return None, f"ℹ️ {exc}"
        loaded = LoadedDocument(doc_id=doc_id, page=None, pages=len(pages))
        loaded.doc_type, loaded.header, loaded.lines = document_to_form(document)
        return loaded, "Поля предзаполнены моделью: проверьте каждое глазами."


# ---- Gradio ----------------------------------------------------------------------------


def build_demo(app: LabelingApp) -> "gr.Blocks":
    """Собрать интерфейс. `import gradio` — здесь, после `disable_telemetry()`."""
    import gradio as gr

    type_choices = [(t.title_ru, t.value) for t in DocType]
    specs_by_section = {
        title: [s for s in HEADER_SPECS if section_of(s) == title] for title, _ in SECTIONS
    }
    status_hint = "Выберите документ слева сверху."

    with gr.Blocks(title="Разметка документов", analytics_enabled=False) as demo:
        started_state = gr.State(0.0)

        gr.Markdown("## Разметка реальных документов\n_Данные не покидают этот компьютер._")
        with gr.Row(equal_height=False):
            # ---------- слева: документ ----------
            with gr.Column(scale=5, elem_id="doc-col"):
                with gr.Row():
                    doc_select = gr.Dropdown(
                        choices=app.choices(),
                        value=None,
                        label="Документ",
                        scale=4,
                        filterable=True,
                    )
                    prev_btn = gr.Button("◀", scale=1, min_width=40)
                    next_btn = gr.Button("▶", scale=1, min_width=40)
                    next_todo_btn = gr.Button("След. неразмеченный ▶▶", scale=2)
                progress = gr.Markdown(app.progress_text())
                page_view = gr.Image(
                    label="Документ", interactive=False, height=820, buttons=["fullscreen"]
                )
                page_slider = gr.Slider(1, 2, value=1, step=1, label="Страница", visible=False)
                upload = gr.UploadButton(
                    "Добавить файлы (PDF, JPG, PNG)", file_count="multiple", size="sm"
                )

            # ---------- справа: форма ----------
            with gr.Column(scale=6):
                with gr.Row():
                    doc_type = gr.Dropdown(type_choices, value=None, label="Тип документа", scale=3)
                    capture = gr.Dropdown(
                        list(CAPTURE_KINDS), value=None, label="Как получен", scale=2
                    )
                    prefill_btn = gr.Button("Предзаполнить", scale=1)

                fields: dict[str, Any] = {}
                groups: dict[str, Any] = {}
                for title, _ in SECTIONS:
                    specs = specs_by_section[title]
                    if not specs:
                        continue
                    with gr.Group(visible=True) as group:
                        gr.Markdown(f"**{title}**")
                        for i in range(0, len(specs), 3):
                            with gr.Row():
                                for spec in specs[i : i + 3]:
                                    if spec.name == "upd_status":
                                        fields[spec.name] = gr.Dropdown(
                                            ["1", "2"], value=None, label=spec.title_ru
                                        )
                                    else:
                                        fields[spec.name] = gr.Textbox(
                                            label=spec.title_ru,
                                            placeholder="дд.мм.гггг"
                                            if spec.name.endswith("date")
                                            else None,
                                            max_lines=3,
                                        )
                    groups[title] = group

                gr.Markdown("**Таблица строк**")
                lines_table = gr.Dataframe(
                    headers=list(LINE_HEADERS),
                    value=empty_lines(),
                    datatype="str",
                    type="array",
                    row_count=1,
                    column_count=len(LINE_COLUMNS),
                    interactive=True,
                    wrap=True,
                    max_height=360,
                )
                with gr.Row():
                    add_row_btn = gr.Button("＋ Строка", size="sm")
                    del_row_btn = gr.Button("− Последняя строка", size="sm")
                note = gr.Textbox(label="Замечание (необязательно)", lines=1)
                accepted = gr.Checkbox(
                    label="Так в документе (сохранить, даже если проверки нашли ошибки)"
                )
                with gr.Row():
                    check_btn = gr.Button("Проверить")
                    save_btn = gr.Button("Сохранить")
                    save_next_btn = gr.Button("Сохранить и далее ▶▶", variant="primary")
                result = gr.Markdown(status_hint)

        field_names = list(fields)
        field_components = [fields[n] for n in field_names]
        form_outputs = [doc_type, *field_components, lines_table]

        # ---------- видимость полей по типу документа ----------
        def visibility(kind: str | None) -> list[Any]:
            try:
                current = DocType(kind) if kind else None
            except ValueError:
                current = None
            updates: list[Any] = []
            for name in field_names:
                spec = next(s for s in HEADER_SPECS if s.name == name)
                updates.append(gr.update(visible=current is None or spec.applies_to(current)))
            for title in groups:
                shown = current is None or any(
                    s.applies_to(current) for s in specs_by_section[title]
                )
                updates.append(gr.update(visible=shown))
            return updates

        doc_type.change(
            visibility, doc_type, [*field_components, *groups.values()], show_progress="hidden"
        )

        def form_values(loaded: LoadedDocument) -> list[Any]:
            values: list[Any] = [loaded.doc_type]
            values += [loaded.header.get(n, "") or None for n in field_names]
            values.append(loaded.lines)
            return values

        # ---------- открыть документ ----------
        def on_select(doc_id: str | None):
            if not doc_id:
                blank = LoadedDocument(doc_id="", page=None, pages=1)
                return (
                    None, gr.update(visible=False), time.monotonic(), blank.capture, "",
                    False, status_hint, *form_values(blank),
                )  # fmt: skip
            try:
                loaded = app.load(doc_id)
            except StoreError as exc:
                raise gr.Error(str(exc)) from exc
            slider = gr.update(
                minimum=1, maximum=max(loaded.pages, 2), value=1, visible=loaded.pages > 1
            )
            return (
                loaded.page, slider, time.monotonic(), loaded.capture,
                loaded.note, loaded.accepted_errors, loaded.status or status_hint,
                *form_values(loaded),
            )  # fmt: skip

        doc_select.change(
            on_select,
            doc_select,
            [page_view, page_slider, started_state, capture, note, accepted, result,
             *form_outputs],
        )  # fmt: skip

        page_slider.release(
            lambda doc_id, n: app.render_page(doc_id, int(n)) if doc_id else None,
            [doc_select, page_slider],
            page_view,
            show_progress="hidden",
        )

        # ---------- строки таблицы ----------
        def add_row(rows):
            return [*rows, *empty_lines()]

        def drop_row(rows):
            return rows[:-1] if len(rows) > 1 else rows

        add_row_btn.click(add_row, lines_table, lines_table, show_progress="hidden")
        del_row_btn.click(drop_row, lines_table, lines_table, show_progress="hidden")

        # ---------- навигация ----------
        prev_btn.click(lambda d: app.neighbour(d, -1), doc_select, doc_select)
        next_btn.click(lambda d: app.neighbour(d, +1), doc_select, doc_select)

        def go_next_todo(doc_id: str | None):
            target = app.next_unlabeled(doc_id)
            if target is None:
                gr.Info("Неразмеченных документов не осталось.")
                return gr.update()
            return target

        next_todo_btn.click(go_next_todo, doc_select, doc_select)

        # ---------- проверка и сохранение ----------
        def collect(values: Sequence[Any]) -> tuple[dict[str, Any], Any]:
            return dict(zip(field_names, values[:-1], strict=True)), values[-1]

        def do_check(kind, *values):
            header, lines = collect(values)
            return app.check(kind, header, lines)

        check_btn.click(do_check, [doc_type, *field_components, lines_table], result)

        def do_save(doc_id, kind, cap, note_text, accept, started, *values):
            header, lines = collect(values)
            elapsed = max(time.monotonic() - float(started or 0.0), 0.0) if started else 0.0
            outcome = app.save(doc_id, kind, cap, note_text, accept, header, lines, elapsed)
            fresh = time.monotonic() if outcome.ok else started
            return (
                outcome.message, app.progress_text(),
                gr.update(choices=app.choices(), value=doc_id), fresh, outcome.ok,
            )  # fmt: skip

        saved_ok = gr.State(False)
        save_inputs = [doc_select, doc_type, capture, note, accepted, started_state,
                       *field_components, lines_table]  # fmt: skip
        save_outputs = [result, progress, doc_select, started_state, saved_ok]
        save_btn.click(do_save, save_inputs, save_outputs)
        save_next_btn.click(do_save, save_inputs, save_outputs).then(
            lambda ok, d: go_next_todo(d) if ok else gr.update(), [saved_ok, doc_select], doc_select
        )

        # ---------- предзаполнение и загрузка файлов ----------
        def do_prefill(doc_id: str | None):
            if not doc_id:
                return (gr.update(), *[gr.update()] * (len(form_outputs)))
            loaded, message = app.prefill(doc_id)
            if loaded is None:
                return (message, *[gr.update()] * len(form_outputs))
            return (message, *form_values(loaded))

        prefill_btn.click(do_prefill, doc_select, [result, *form_outputs])

        def do_upload(files: list[str] | None):
            added = app.store.add_files([Path(f) for f in files or []])
            message = f"Добавлено файлов: {len(added)}." if added else "Подходящих файлов нет."
            return message, gr.update(choices=app.choices()), app.progress_text()

        def first_document() -> str | None:
            """При открытии страницы — первый неразмеченный документ (или первый вообще)."""
            ids = [s.id for s in app.store.list_sources()]
            return app.next_unlabeled(None) or (ids[0] if ids else None)

        demo.load(first_document, None, doc_select)
        upload.upload(do_upload, upload, [result, doc_select, progress])

    return demo


def run(root: Path = DEFAULT_ROOT, port: int = PORT, open_browser: bool = True) -> None:
    """Запустить интерфейс разметки на 127.0.0.1."""
    disable_telemetry()
    app = LabelingApp(LabelStore(root))
    app.store.files_dir.mkdir(parents=True, exist_ok=True)
    app.store.labels_dir.mkdir(parents=True, exist_ok=True)
    demo = build_demo(app)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=HOST,
        server_port=port,
        inbrowser=open_browser,
        share=False,
        footer_links=[],
        css=CSS,
    )
