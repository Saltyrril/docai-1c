"""Выборка документов для оценки: стратифицированная по типу, воспроизводимая по seed.

Синтетика (P6): документы берутся из манифеста датасета, сплит по продавцам уже сделан.
Число документов каждого типа пропорционально их доле в сплите (метод наибольших
остатков), конкретные документы — `random.Random(seed).sample` по отсортированному списку.
Уровни качества при `rotate` раздаются по кругу в порядке «тип → документ», так что каждый
уровень получает почти поровну документов каждого типа.

Реальные документы (P8): все размеченные из `data/real/labels`, уровень — по виду съёмки.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from docai.eval.config import EvalConfig

# Вид съёмки из разметки P8 → уровень качества.
CAPTURE_LEVEL = {"PDF (цифровой)": "clean", "скан": "scan", "фото": "photo"}


class EvalItem(BaseModel):
    """Один прогон извлечения: документ на одном уровне качества."""

    item_id: str
    doc_id: str
    doc_type: str
    level: str
    inputs: list[Path]
    truth: Path
    # Картинка первой страницы для отчёта (только синтетика; реальные картинки не обезличены).
    preview: Path | None = None
    real: bool = False
    pages: int | None = None
    # В документе есть намеренная ошибка генератора P3 (эталон = как напечатано).
    has_errors: bool = False


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Квоты, пропорциональные counts, в сумме total (метод наибольших остатков).

    При равных остатках раньше получает тип с бо́льшим исходным числом, затем по имени.
    """
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"в выборке {available} документов, а запрошено {total}")
    exact = {key: total * n / available for key, n in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    rest = total - sum(quotas.values())
    order = sorted(counts, key=lambda k: (-(exact[k] - quotas[k]), -counts[k], k))
    for key in order[:rest]:
        quotas[key] += 1
    return quotas


def item_from_row(row: dict, root: Path, level: str, clean_pdf: bool) -> EvalItem:
    """Элемент выборки из строки манифеста датасета (общий для P11 и матрицы P12)."""
    images = [root / p for p in row["images"]]
    use_pdf = level == "clean" and clean_pdf
    return EvalItem(
        item_id=row["id"],
        doc_id=row["doc_id"],
        doc_type=row["doc_type"],
        level=level,
        inputs=[root / row["pdf"]] if use_pdf else images,
        truth=root / row["json"],
        preview=images[0] if images else None,
        pages=row.get("pages"),
        has_errors=bool(row.get("has_errors")),
    )


def _read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_dataset_items(cfg: EvalConfig) -> list[EvalItem]:
    data = cfg.data
    root = data.dataset
    rows = [
        r
        for r in _read_manifest(root / "manifest.jsonl")
        if r["split"] == data.split
        and r["dpi"] == data.dpi
        and r["level"] in data.levels
        and (data.doc_types is None or r["doc_type"] in data.doc_types)
        and r["doc_id"] not in data.exclude_doc_ids
    ]
    by_key = {(r["doc_id"], r["level"]): r for r in rows}
    doc_types = {r["doc_id"]: r["doc_type"] for r in rows}
    # Документ входит в выборку, только если он есть на всех запрошенных уровнях.
    docs = sorted(d for d in doc_types if all((d, lv) in by_key for lv in data.levels))
    counts = Counter(doc_types[d] for d in docs)
    n_docs = len(docs) if data.n_docs is None else data.n_docs
    quotas = proportional_quotas(dict(counts), n_docs)

    rng = random.Random(cfg.seed)
    chosen: list[str] = []
    for doc_type in sorted(quotas):
        pool = [d for d in docs if doc_types[d] == doc_type]
        chosen += sorted(rng.sample(pool, quotas[doc_type]))

    pairs: list[tuple[str, str]] = []
    for index, doc_id in enumerate(chosen):
        if data.level_assignment == "rotate":
            pairs.append((doc_id, data.levels[index % len(data.levels)]))
        else:
            pairs += [(doc_id, level) for level in data.levels]

    items = []
    for doc_id, level in pairs:
        items.append(
            item_from_row(by_key[(doc_id, level)], root, level, cfg.extract.clean_input == "pdf")
        )
    return items


def select_real_items(cfg: EvalConfig) -> list[EvalItem]:
    # Импорт здесь: модуль разметки тянет Pillow и pypdfium2, синтетике они не нужны.
    from docai.labeling.pseudonymize import load_salt
    from docai.labeling.store import LabelStore, _page_files

    store = LabelStore(cfg.data.real_root)
    salt = load_salt(cfg.data.real_root)
    items = []
    for doc_id in store.label_ids():
        truth = store.load_label(doc_id)
        if truth is None:
            continue
        level = CAPTURE_LEVEL.get(store.load_meta(doc_id).get("capture") or "", "other")
        if level not in cfg.data.levels and level != "other":
            continue
        if cfg.data.doc_types is not None and str(truth.doc_type) not in cfg.data.doc_types:
            continue
        source = store.get_source(doc_id)
        inputs = _page_files(source.path) if source.kind == "folder" else [source.path]
        # Имя файла реального документа может называть контрагента: в кэш и отчёт — псевдоним.
        pseudo = "real-" + hashlib.sha256(f"{salt}:id:{doc_id}".encode()).hexdigest()[:10]
        items.append(
            EvalItem(
                item_id=pseudo,
                doc_id=pseudo,
                doc_type=str(truth.doc_type),
                level=level,
                inputs=inputs,
                truth=store.labels_dir / f"{doc_id}.json",
                real=True,
            )
        )
    if cfg.data.n_docs is not None and cfg.data.n_docs < len(items):
        rng = random.Random(cfg.seed)
        keep = set(rng.sample([it.item_id for it in items], cfg.data.n_docs))
        items = [it for it in items if it.item_id in keep]
    return items


def select_items(cfg: EvalConfig) -> list[EvalItem]:
    if cfg.data.source == "real":
        return select_real_items(cfg)
    return select_dataset_items(cfg)
