"""Сборка датасета одной командой: документы → JSON → PDF → картинки, разбиение, манифест."""

import json
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docai.augment.levels import Level
from docai.augment.pipeline import process_pdf
from docai.dataset.config import DatasetConfig
from docai.dataset.split import split_by_seller
from docai.render.engine import Renderer, choose_look
from docai.synth.cli import write_documents
from docai.synth.generator import SynthDocument, generate_documents

MANIFEST_NAME = "manifest.jsonl"
BUILD_INFO_NAME = "build_info.json"
LAYOUT_SUFFIX = ".layout.json"

Progress = Callable[[str, int, int], None]
Job = tuple[str, str, list[str], list[int], int]
JobRow = tuple[str, str, int, int, str, str]


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildResult:
    out: Path
    manifest: Path
    documents: int
    rows: int
    images: int


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _augment_job(job: Job) -> list[JobRow]:
    """Задача для процесса: один PDF → картинки (имя, уровень, dpi, страница, файлы)."""
    pdf, images_dir, levels, dpis, seed = job
    written = process_pdf(Path(pdf), Path(images_dir), [Level(v) for v in levels], dpis, seed)
    stem = Path(pdf).stem
    return [(stem, w.level.value, w.dpi, w.page, w.image.name, w.layout.name) for w in written]


def _render_all(documents: list[SynthDocument], pdf_dir: Path, seed: int) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with Renderer() as renderer:
        for item in documents:
            stem = Path(item.file_name).stem
            chosen, decor = choose_look(item.document.doc_type, stem, seed)
            result = renderer.render(
                item.document, variant=chosen, decor=decor, source=item.file_name
            )
            (pdf_dir / f"{stem}.pdf").write_bytes(result.pdf)
            layout = json.dumps(result.layout, ensure_ascii=False, indent=1)
            _write_text(pdf_dir / f"{stem}{LAYOUT_SUFFIX}", layout + "\n")


def build_dataset(
    config: DatasetConfig, *, force: bool = False, progress: Progress | None = None
) -> BuildResult:
    out = config.out
    if out.exists() and any(out.iterdir()):
        if not force:
            raise BuildError(f"Папка {out} не пуста. Используйте --force, чтобы собрать заново.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    notify: Progress = progress or (lambda *_: None)

    documents = generate_documents(config.n_documents, config.seed, config.error_rate)
    notify("документы", len(documents), len(documents))

    shares = config.split.model_dump()
    assignment = split_by_seller([d.document.seller.inn for d in documents], shares, config.seed)

    json_dir, pdf_dir, images_dir = out / "json", out / "pdf", out / "images"
    write_documents(documents, json_dir, config.seed)
    _render_all(documents, pdf_dir, config.seed)
    notify("PDF", len(documents), len(documents))

    levels = [level.value for level in config.levels]
    jobs: list[Job] = [
        (
            str(pdf_dir / f"{Path(d.file_name).stem}.pdf"),
            str(images_dir),
            levels,
            config.dpi,
            config.seed,
        )
        for d in documents
    ]
    # ProcessPoolExecutor.map сохраняет порядок задач, поэтому результат не зависит от workers.
    pages: dict[tuple[str, str, int], list[tuple[int, str, str]]] = {}
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        for done, rows in enumerate(pool.map(_augment_job, jobs, chunksize=4), start=1):
            for stem, level, dpi, page, image, layout in rows:
                pages.setdefault((stem, level, dpi), []).append((page, image, layout))
            if done % 100 == 0 or done == len(jobs):
                notify("картинки", done, len(jobs))

    manifest_rows: list[dict[str, Any]] = []
    images = 0
    for item in documents:
        stem = Path(item.file_name).stem
        for level in levels:
            for dpi in config.dpi:
                found = sorted(pages[(stem, level, dpi)])
                images += len(found)
                manifest_rows.append(
                    {
                        "id": f"{stem}_{level}_{dpi}dpi",
                        "doc_id": stem,
                        "doc_type": item.document.doc_type.value,
                        "split": assignment[item.document.seller.inn],
                        "seed": config.seed,
                        "level": level,
                        "dpi": dpi,
                        "pages": len(found),
                        "json": f"json/{item.file_name}",
                        "pdf": f"pdf/{stem}.pdf",
                        "pdf_layout": f"pdf/{stem}{LAYOUT_SUFFIX}",
                        "images": [f"images/{image}" for _, image, _ in found],
                        "image_layouts": [f"images/{layout}" for _, _, layout in found],
                        "seller_inn": item.document.seller.inn,
                        "has_errors": item.has_errors,
                        "error_kinds": list(item.error_kinds),
                    }
                )
    manifest = out / MANIFEST_NAME
    _write_text(manifest, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_rows))
    # Конфиг без дат и путей машины: файл тоже воспроизводим.
    info = {"config": json.loads(config.model_dump_json(exclude={"out", "workers"}))}
    _write_text(out / BUILD_INFO_NAME, json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    return BuildResult(out, manifest, len(documents), len(manifest_rows), images)
