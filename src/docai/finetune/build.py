"""Сборка SFT-датасета для дообучения Qwen3.5-2B в Colab (шаг P21, D77).

Пример — ровно один запрос конвейера извлечения к модели и эталонный ответ на него:
- vision: картинка страницы уровня clean/scan/photo, уменьшенная до 1280 px по длинной
  стороне, как в конвейере, и промпт `vision_v2` со «страница N из M»;
- text: текстовый слой PDF (pdfplumber, таблицы `grid`) и промпт `text_v3`; документ
  целиком или по страницам — по тому же правилу, что в конвейере (`_text_parts`).
Ответ — черновик `Draft` из того, что напечатано на этой странице (`targets.py`).

Формат — то, что принимает `UnslothVisionDataCollator`: строка JSONL с ключом `messages`
(user: картинка, затем текст — в этом порядке картинки ставит и рендерер Ollama для
qwen3.5; assistant: текст ответа). Картинка — путь относительно папки датасета.
Части: train — из train P6, val — из val P6; test P6 не используется.
"""

import json
import random
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docai.extract.images import fit_long_side, image_files_pages, pdf_pages_fit
from docai.extract.pdftext import pdf_pages_text
from docai.extract.pipeline import _page_info, _text_parts
from docai.extract.prompts import render_prompt
from docai.extract.router import InputKind, read_kind
from docai.finetune.config import KINDS, Kind, SftConfig
from docai.finetune.feedback import FeedbackReport, collect_feedback, printed_view
from docai.finetune.targets import draft_target, to_json, visible_fields

TEXT_SOURCE = "текстовый слой PDF"  # как в `_run_text` конвейера
TEXT_TITLE = "Текст документа"
MIN_TEXT_CHARS_PER_PAGE = 30  # как `min_text_chars_per_page` конвейера

PdfText = Callable[[Path, str], list[str]]


class SftBuildError(RuntimeError):
    pass


@dataclass
class Example:
    id: str
    split: str
    kind: Kind
    doc_id: str
    doc_type: str
    part: str  # document | page i/N
    prompt: str
    target: str
    source: str = "synth"  # synth | feedback
    has_errors: bool = False
    image: str | None = None  # путь относительно папки датасета
    image_size: tuple[int, int] | None = None

    def record(self) -> dict[str, Any]:
        content: list[dict[str, str]] = []
        if self.image is not None:
            content.append({"type": "image", "image": self.image})
        content.append({"type": "text", "text": self.prompt})
        return {
            "id": self.id,
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": self.target}]},
            ],
            "meta": {
                "split": self.split,
                "kind": self.kind,
                "doc_id": self.doc_id,
                "doc_type": self.doc_type,
                "part": self.part,
                "source": self.source,
                "has_errors": self.has_errors,
                "image_size": list(self.image_size) if self.image_size else None,
                "prompt_chars": len(self.prompt),
                "target_chars": len(self.target),
            },
        }


@dataclass
class _ImageJob:
    source: Path | None  # картинка страницы
    pdf: Path | None  # или страница PDF (правки /feedback)
    out: Path
    example: Example


@dataclass
class BuildResult:
    out: Path
    counts: dict[str, dict[str, int]]
    archive: Path | None
    feedback: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(dataset: Path) -> list[dict[str, Any]]:
    path = dataset / "manifest.jsonl"
    if not path.is_file():
        raise SftBuildError(f"нет {path}: сначала соберите датасет P6 (docai dataset build)")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _documents(rows: list[dict[str, Any]], split: str) -> list[dict[str, dict[str, Any]]]:
    """Документы части: {уровень: строка манифеста}, по порядку doc_id."""
    docs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["split"] == split:
            docs[row["doc_id"]].setdefault(row["level"], row)
    return [docs[doc_id] for doc_id in sorted(docs)]


def _next_kind(mix_counts: Counter, total: int, config: SftConfig) -> Kind:
    """Вид, которого сейчас больше всего не хватает до заданной доли."""
    shares = {kind: config.mix.share(kind) for kind in KINDS}
    return max(
        (k for k in KINDS if shares[k] > 0),
        key=lambda k: (shares[k] * (total + 1) - mix_counts[k], -KINDS.index(k)),
    )


def _text_examples(
    doc: dict[str, dict[str, Any]], split: str, config: SftConfig, pdf_text: PdfText
) -> list[Example]:
    row = next(iter(doc.values()))
    dataset = config.dataset
    ref = _read_json(dataset / row["json"])
    layout = _read_json(dataset / row["pdf_layout"])["fields"]
    pages = pdf_text(dataset / row["pdf"], config.text_tables)
    if any(len(text.strip()) < MIN_TEXT_CHARS_PER_PAGE for text in pages):
        return []  # такой PDF конвейер отправит в vision
    examples = []
    for part, text in _text_parts(pages, config.max_text_chars):
        page = None if part == "document" else int(part.split()[1].split("/")[0])
        target = draft_target(ref, visible_fields(layout, page))
        suffix = "doc" if page is None else f"p{page}"
        examples.append(
            Example(
                id=f"{row['doc_id']}_text_{suffix}",
                split=split,
                kind="text",
                doc_id=row["doc_id"],
                doc_type=row["doc_type"],
                part=part,
                prompt=render_prompt(
                    config.text_prompt,
                    source=TEXT_SOURCE,
                    source_title=TEXT_TITLE,
                    document_text=text,
                ),
                target=to_json(target),
                has_errors=bool(row.get("has_errors")),
            )
        )
    return examples


def _vision_examples(
    doc: dict[str, dict[str, Any]], split: str, kind: Kind, config: SftConfig
) -> list[tuple[Example, _ImageJob]]:
    row = doc.get(kind)
    if row is None:
        return []
    dataset = config.dataset
    ref = _read_json(dataset / row["json"])
    total = len(row["images"])
    out = []
    for index, (image, layout_path) in enumerate(
        zip(row["images"], row["image_layouts"], strict=True), 1
    ):
        layout = _read_json(dataset / layout_path)["fields"]
        example = Example(
            id=f"{row['doc_id']}_{kind}_p{index}",
            split=split,
            kind=kind,
            doc_id=row["doc_id"],
            doc_type=row["doc_type"],
            part=f"page {index}/{total}",
            prompt=render_prompt(config.vision_prompt, page_info=_page_info(index, total)),
            target=to_json(draft_target(ref, visible_fields(layout))),
            has_errors=bool(row.get("has_errors")),
            image=f"images/{split}/{row['doc_id']}_{kind}_p{index}.png",
        )
        job = _ImageJob(
            source=dataset / image, pdf=None, out=config.out / example.image, example=example
        )
        out.append((example, job))
    return out


def _select(
    rows: list[dict[str, Any]], split: str, n: int, config: SftConfig, pdf_text: PdfText
) -> tuple[list[Example], list[_ImageJob]]:
    docs = _documents(rows, split)
    random.Random(f"docai-sft:{config.seed}:{split}").shuffle(docs)
    examples: list[Example] = []
    jobs: list[_ImageJob] = []
    mix_counts: Counter = Counter()
    for doc in docs:
        if len(examples) >= n:
            break
        kind = _next_kind(mix_counts, len(examples), config)
        if kind == "text":
            new = _text_examples(doc, split, config, pdf_text)
            new_jobs: list[_ImageJob] = []
        else:
            pairs = _vision_examples(doc, split, kind, config)
            new = [example for example, _ in pairs]
            new_jobs = [job for _, job in pairs]
        examples.extend(new)
        jobs.extend(new_jobs)
        mix_counts[kind] += len(new)
    return examples, jobs


def _feedback_examples(report: FeedbackReport, config: SftConfig) -> list[tuple[Example, Any]]:
    """Примеры из правок (только статус unknown и только по решению владельца)."""
    out: list[tuple[Example, Any]] = []
    for job in report.jobs:
        if job.status != "unknown":
            continue
        values = dict(job.values)
        doc_type = values.get("doc_type")
        if not doc_type:
            job.status = "no_doc_type"
            continue
        ref, visible = printed_view(values)
        ref["doc_type"] = doc_type
        target = to_json(draft_target(ref, visible))
        kind = read_kind(job.path)
        base = {
            "split": "train",
            "doc_id": f"feedback_{job.job_id}",
            "doc_type": doc_type,
            "target": target,
            "source": "feedback",
        }
        if kind is InputKind.PDF:
            pages = pdf_pages_text(job.path, config.text_tables)
            if len(pages) != 1:
                job.status = "multipage"
                continue
            if len(pages[0].strip()) >= MIN_TEXT_CHARS_PER_PAGE:
                prompt = render_prompt(
                    config.text_prompt,
                    source=TEXT_SOURCE,
                    source_title=TEXT_TITLE,
                    document_text=pages[0],
                )
                example = Example(
                    id=f"feedback_{job.job_id}_text",
                    kind="text",
                    part="document",
                    prompt=prompt,
                    **base,
                )
                out.append((example, None))
                job.status = "included"
                continue
        elif kind is not InputKind.IMAGE:
            job.status = "unsupported"
            continue
        example = Example(
            id=f"feedback_{job.job_id}_image",
            kind="scan",  # уровень неизвестен; «скан» — ближайший по смыслу
            part="page 1/1",
            prompt=render_prompt(config.vision_prompt, page_info=_page_info(1, 1)),
            image=f"images/train/feedback_{job.job_id}.png",
            **base,
        )
        is_pdf = kind is InputKind.PDF
        image_job = _ImageJob(
            source=None if is_pdf else job.path,
            pdf=job.path if is_pdf else None,
            out=config.out / example.image,
            example=example,
        )
        out.append((example, image_job))
        job.status = "included"
    return out


def _render_image(job: _ImageJob, max_side: int) -> tuple[int, int]:
    """Страница → PNG того размера, который модель получит в конвейере."""
    if job.pdf is not None:
        image = pdf_pages_fit(job.pdf, max_side)[0]
    else:
        assert job.source is not None
        pages = image_files_pages([job.source])
        if len(pages) != 1:
            raise SftBuildError(f"{job.source}: ожидалась одна страница, их {len(pages)}")
        image = fit_long_side(pages[0], max_side)
    job.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(job.out, format="PNG")
    return image.size


def _write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.record(), ensure_ascii=False) + "\n")


def _counts(examples: list[Example]) -> dict[str, Any]:
    by_kind = Counter(e.kind for e in examples)
    by_type = Counter(e.doc_type for e in examples)
    target_chars = [len(e.target) for e in examples]
    prompt_chars = [len(e.prompt) for e in examples]
    return {
        "examples": len(examples),
        "documents": len({e.doc_id for e in examples}),
        "by_kind": dict(sorted(by_kind.items())),
        "by_doc_type": dict(sorted(by_type.items())),
        "multipage_parts": sum(
            e.part.startswith("page") and e.part != "page 1/1" for e in examples
        ),
        "with_generator_errors": sum(e.has_errors for e in examples),
        "from_feedback": sum(e.source == "feedback" for e in examples),
        "target_chars_mean": round(sum(target_chars) / len(examples), 1) if examples else 0,
        "target_chars_max": max(target_chars, default=0),
        "prompt_chars_max": max(prompt_chars, default=0),
    }


def _git_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


README = """# SFT-датасет {name} для дообучения Qwen3.5-2B (docai-1c, шаг P21)

Собран командой `uv run docai finetune build`. Содержимое:

- `train.jsonl`, `val.jsonl` — по строке на пример: `messages` (user: картинка и/или промпт,
  assistant: эталонный JSON-ответ), `meta` (вид примера, документ, страница);
- `images/` — страницы PNG, уменьшенные до {side} px по длинной стороне, как в конвейере;
- `dataset_info.json` — конфиг сборки и состав;
- `train_config.yaml` — параметры обучения для ноутбука Colab.

Данные синтетические (датасет P6), реальных документов нет. Части train и val взяты из
train и val P6 (разбиение по продавцам), test P6 не использован.
"""


def build_sft_dataset(
    config: SftConfig,
    *,
    force: bool = False,
    pdf_text: PdfText = pdf_pages_text,
    workers: int = 6,
    progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Собрать датасет в `config.out` (и архив `<out>.zip`, если `config.archive`)."""
    say = progress or (lambda _msg: None)
    out = config.out
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SftBuildError(f"папка {out} не пуста; --force — собрать заново")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _manifest(config.dataset)

    say("отбор train")
    train, jobs = _select(rows, "train", config.n_train, config, pdf_text)
    say("отбор val")
    val, val_jobs = _select(rows, "val", config.n_val, config, pdf_text)
    jobs += val_jobs

    feedback_summary: dict[str, Any] = {"enabled": config.feedback.enabled}
    if config.feedback.enabled:
        say("правки /feedback")
        report = collect_feedback(
            config.feedback.data_dir, [config.dataset, *config.feedback.synthetic_dirs]
        )
        if config.feedback.include_unknown:
            for example, job in _feedback_examples(report, config):
                train.append(example)
                if job is not None:
                    jobs.append(job)
        feedback_summary |= report.summary()
        feedback_summary["include_unknown"] = config.feedback.include_unknown
        feedback_summary["synthetic_matches"] = sorted(
            {job.matched for job in report.jobs if job.matched}
        )

    say(f"картинки: {len(jobs)}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = list(pool.map(lambda job: _render_image(job, config.max_image_side), jobs))
    for job, size in zip(jobs, sizes, strict=True):
        job.example.image_size = size

    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "val.jsonl", val)
    counts = {"train": _counts(train), "val": _counts(val)}
    info = {
        "name": config.name,
        "config": json.loads(config.model_dump_json()),
        "git_commit": _git_commit(),
        "counts": counts,
        "feedback": feedback_summary,
        "format": "messages (UnslothVisionDataCollator); image — путь относительно папки",
    }
    (out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not config.train_config.is_file():
        raise SftBuildError(f"нет конфига обучения {config.train_config}")
    shutil.copyfile(config.train_config, out / "train_config.yaml")
    (out / "README.md").write_text(
        README.format(name=config.name, side=config.max_image_side), encoding="utf-8"
    )

    archive = None
    if config.archive:
        say("архив")
        archive = out.with_suffix(".zip")
        _zip(out, archive)
    return BuildResult(out=out, counts=counts, archive=archive, feedback=feedback_summary)


def _zip(folder: Path, archive: Path) -> None:
    """Папка → zip с корнем `<имя папки>/`. PNG уже сжаты — хранятся без сжатия."""
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_dir():
                continue
            name = f"{folder.name}/{path.relative_to(folder).as_posix()}"
            method = zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED
            zf.write(path, name, compress_type=method)
