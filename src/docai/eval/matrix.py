"""Матрица экспериментов P12: модель × путь × качество входа, прогон партиями с кэшем.

Ячейка матрицы — пара «модель + путь» (text, vision, ocr). Уровень качества (clean, scan,
photo) — не отдельная ячейка, а разбиение документов внутри неё: документы раздаются по
уровням по кругу блоками по три в случайном порядке, поэтому в любом начале списка
уровни почти поровну. Путь text читает текстовый слой PDF, поэтому работает только на clean
и берёт все документы выборки.

Порядок документов общий для всех ячеек и стратифицирован по типу: любое начало списка —
тоже стратифицированная выборка. Прогон идёт партиями «по документам»: партия k — документы
[k·B, (k+1)·B) во всех ячейках подряд. Прервать можно в любой момент, а таблица по готовым
партиям сравнима между ячейками (в отчёте берётся общее для всех ячеек начало списка).
Ответы кэшируются по id элемента (`docai.eval.predict`), повторный запуск продолжает с места
остановки.
"""

import json
import logging
import random
import statistics
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docai.eval.config import (
    BootstrapConfig,
    DataConfig,
    EvalConfig,
    ExtractSection,
    Level,
    MetricsConfig,
    ReportConfig,
)
from docai.eval.predict import (
    ExtractFn,
    Prediction,
    fingerprint,
    load_cached,
    run_predictions,
)
from docai.eval.sample import (
    EvalItem,
    _read_manifest,
    item_from_row,
    select_dataset_items,
    select_real_items,
)
from docai.extract import ExtractConfig, Mode

log = logging.getLogger("docai.eval")

LEVELS: tuple[Level, ...] = ("clean", "scan", "photo")
# Доля модели, лежащая в VRAM, не меньше которой считаем, что модель помещается целиком.
FULL_VRAM_SHARE = 0.98


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Strict):
    id: str
    tag: str  # имя в Ollama
    paths: list[str]
    # Модель не запускается; причина попадает в таблицу.
    skip_reason: str | None = None
    # Если после `check-vram` модель с контекстом num_ctx не лежит в VRAM целиком — пропустить.
    require_full_vram: bool = False
    # Ожидаемое время одного элемента, с: только для оценки до первого прогона.
    guess_s_per_item: float = 60.0


class PathSpec(_Strict):
    mode: Mode
    levels: list[Level]
    clean_input: Literal["pdf", "image"] = "image"


class MatrixData(_Strict):
    dataset: Path = Path("data/datasets/v1")
    split: str = "test"
    n_docs: int = 300
    dpi: int = 200
    exclude_doc_ids: list[str] = Field(default_factory=list)
    # Добавить размеченные реальные документы (data/real/labels); таблица по ним отдельная.
    include_real: bool = True
    real_root: Path = Path("data/real")


class MatrixConfig(_Strict):
    name: str
    description: str = ""
    seed: int = 2026
    # Сколько документов (в порядке выборки) считать за одну партию, кратно числу уровней.
    batch_docs: int = 30
    data: MatrixData = Field(default_factory=MatrixData)
    # Базовый конфиг извлечения; модель подставляется из ячейки.
    extract_config: Path | None = Path("configs/extract.yaml")
    models: list[ModelSpec]
    paths: dict[str, PathSpec]
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    predictions_dir: Path = Path("data/experiments/eval")
    power_guard: bool = True

    @model_validator(mode="after")
    def _check(self) -> "MatrixConfig":
        if self.batch_docs % len(LEVELS):
            raise ValueError(f"batch_docs должно делиться на {len(LEVELS)}")
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError("id моделей не повторяются")
        for model in self.models:
            for path in model.paths:
                if path not in self.paths:
                    raise ValueError(f"модель {model.id}: неизвестный путь {path}")
        for name, spec in self.paths.items():
            if spec.mode is Mode.TEXT and spec.levels != ["clean"]:
                raise ValueError(f"путь {name}: text читает текстовый слой, только clean")
            if spec.mode is Mode.TEXT:
                spec.clean_input = "pdf"
        return self

    @property
    def root_dir(self) -> Path:
        return self.predictions_dir / self.name


def load_matrix_config(path: Path) -> MatrixConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return MatrixConfig.model_validate(data)


@dataclass(frozen=True)
class Cell:
    cfg: MatrixConfig
    model: ModelSpec
    path_name: str
    path: PathSpec

    @property
    def id(self) -> str:
        return f"{self.model.id}__{self.path_name}"

    @property
    def cache_dir(self) -> Path:
        return self.cfg.root_dir / self.id

    def eval_config(self) -> EvalConfig:
        c = self.cfg
        return EvalConfig(
            name=self.id,
            seed=c.seed,
            data=DataConfig(
                dataset=c.data.dataset,
                split=c.data.split,
                n_docs=c.data.n_docs,
                levels=list(self.path.levels),
                dpi=c.data.dpi,
                exclude_doc_ids=c.data.exclude_doc_ids,
                real_root=c.data.real_root,
            ),
            extract=ExtractSection(
                config=c.extract_config,
                mode=self.path.mode,
                model=self.model.tag,
                clean_input=self.path.clean_input,
            ),
            metrics=c.metrics,
            bootstrap=c.bootstrap,
            report=c.report,
            predictions_dir=c.predictions_dir,
            power_guard=c.power_guard,
        )

    def extract_config(self) -> ExtractConfig:
        return self.eval_config().extract.build()


def build_cells(cfg: MatrixConfig) -> list[Cell]:
    return [Cell(cfg, model, path, cfg.paths[path]) for model in cfg.models for path in model.paths]


# ---- проверка VRAM ----------------------------------------------------------------------


def vram_check_path(cfg: MatrixConfig) -> Path:
    return cfg.root_dir / "vram_check.json"


def load_vram_check(cfg: MatrixConfig) -> dict[str, Any]:
    path = vram_check_path(cfg)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def skip_reason(
    cfg: MatrixConfig, model: ModelSpec, check: dict[str, Any] | None = None
) -> str | None:
    """Почему модель не запускается (None — запускается)."""
    if model.skip_reason:
        return model.skip_reason
    if model.require_full_vram:
        check = load_vram_check(cfg) if check is None else check
        entry = check.get(model.id)
        if entry is None:
            return "помещаемость в VRAM не проверена: запустите docai eval-matrix check-vram"
        if not entry["fits"]:
            return entry["reason"]
    return None


def active_cells(cfg: MatrixConfig, only: Sequence[str] | None = None) -> list[Cell]:
    check = load_vram_check(cfg)
    cells = [c for c in build_cells(cfg) if skip_reason(cfg, c.model, check) is None]
    if only:
        known = {c.id for c in build_cells(cfg)}
        unknown = set(only) - known
        if unknown:
            raise ValueError(
                f"нет ячеек: {', '.join(sorted(unknown))}; есть: {', '.join(sorted(known))}"
            )
        cells = [c for c in cells if c.id in only]
    return cells


def _api(url: str, path: str, body: dict | None = None, timeout: float = 900) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(
            f"Ollama недоступна по адресу {url} ({exc}). Запустите: ollama serve"
        ) from exc


def ollama_models(url: str) -> set[str]:
    return {m["name"] for m in _api(url, "/api/tags")["models"]}


def ollama_placement(url: str, tag: str) -> dict[str, int] | None:
    """Сколько байт модели в памяти и сколько из них в VRAM (`/api/ps`)."""
    for m in _api(url, "/api/ps").get("models", []):
        if m["name"] == tag or m["model"] == tag:
            return {"size": int(m["size"]), "size_vram": int(m["size_vram"])}
    return None


def check_vram(
    cfg: MatrixConfig,
    *,
    chat_fn: Callable[[ExtractConfig, str], Any] | None = None,
    placement_fn: Callable[[str, str], dict[str, int] | None] = ollama_placement,
    peak_fn: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """Загрузить каждую модель с `require_full_vram` с контекстом из конфига и записать итог."""
    from docai.extract.ollama import build_chat_request, chat

    def default_chat(econfig: ExtractConfig, prompt: str) -> Any:
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        return chat(econfig, build_chat_request(econfig, prompt, schema))

    out: dict[str, Any] = {}
    for model in cfg.models:
        if not model.require_full_vram or model.skip_reason:
            continue
        econfig = next(c for c in build_cells(cfg) if c.model.id == model.id).extract_config()
        (chat_fn or default_chat)(econfig, 'Ответь {"ok": true}')
        place = placement_fn(econfig.ollama_url, model.tag)
        peak = peak_fn() if peak_fn else _nvidia_used_mib()
        if place is None:
            raise RuntimeError(f"{model.tag}: не вижу модель в /api/ps после запроса")
        share = place["size_vram"] / place["size"]
        fits = share >= FULL_VRAM_SHARE
        gib = place["size"] / 2**30
        out[model.id] = {
            "tag": model.tag,
            "fits": fits,
            "size_bytes": place["size"],
            "size_vram_bytes": place["size_vram"],
            "vram_share": round(share, 4),
            "num_ctx": econfig.options.num_ctx,
            "gpu_used_mib": peak,
            "reason": (
                None
                if fits
                else f"не помещается в VRAM: в памяти {gib:.2f} ГиБ при контексте "
                f"{econfig.options.num_ctx}, в VRAM {share:.0%} (нужно ≥ {FULL_VRAM_SHARE:.0%}), "
                "остальное ушло бы в CPU"
            ),
        }
    path = vram_check_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def _nvidia_used_mib() -> int | None:
    import shutil
    import subprocess

    cmd = shutil.which("nvidia-smi")
    if not cmd:
        return None
    try:
        out = subprocess.run(
            [cmd, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout  # fmt: skip
        return int(out.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


# ---- выборка и порядок -----------------------------------------------------------------


@dataclass(frozen=True)
class DocPos:
    order: int
    doc_id: str
    doc_type: str
    level: Level  # уровень при раздаче по кругу (для путей с тремя уровнями)


def plan_docs(cfg: MatrixConfig) -> list[DocPos]:
    """Общий порядок документов: стратифицирован по типу в любом начале, уровни — блоками по три."""
    probe = EvalConfig(
        name="probe",
        seed=cfg.seed,
        data=DataConfig(
            dataset=cfg.data.dataset,
            split=cfg.data.split,
            n_docs=cfg.data.n_docs,
            levels=list(LEVELS),
            level_assignment="rotate",
            exclude_doc_ids=cfg.data.exclude_doc_ids,
            dpi=cfg.data.dpi,
        ),
        extract=ExtractSection(config=None),
    )
    chosen = select_dataset_items(probe)
    rng = random.Random(f"{cfg.seed}:matrix-order")
    by_type: dict[str, list[str]] = {}
    for item in chosen:
        by_type.setdefault(item.doc_type, []).append(item.doc_id)
    keyed = []
    for doc_type in sorted(by_type):
        ids = sorted(by_type[doc_type])
        rng.shuffle(ids)
        keyed += [((rank + 0.5) / len(ids), doc_type, doc_id) for rank, doc_id in enumerate(ids)]
    keyed.sort()
    positions = []
    for start in range(0, len(keyed), len(LEVELS)):
        block = list(LEVELS)
        rng.shuffle(block)
        for offset, (_, doc_type, doc_id) in enumerate(keyed[start : start + len(LEVELS)]):
            positions.append(DocPos(start + offset, doc_id, doc_type, block[offset]))
    return positions


def real_items(cfg: MatrixConfig) -> list[EvalItem]:
    if not cfg.data.include_real or not (cfg.data.real_root / "labels").is_dir():
        return []
    probe = EvalConfig(
        name="probe",
        seed=cfg.seed,
        data=DataConfig(
            source="real", n_docs=None, levels=list(LEVELS), real_root=cfg.data.real_root
        ),
        extract=ExtractSection(config=None),
    )
    return select_real_items(probe)


def cell_items(
    cell: Cell, docs: Sequence[DocPos], real: Sequence[EvalItem]
) -> list[tuple[int, EvalItem]]:
    """Элементы ячейки с номером позиции в общем порядке (реальные идут после синтетики)."""
    cfg = cell.cfg
    root = cfg.data.dataset
    rows = {
        (r["doc_id"], r["level"]): r
        for r in _read_manifest(root / "manifest.jsonl")
        if r["split"] == cfg.data.split and r["dpi"] == cfg.data.dpi
    }
    out: list[tuple[int, EvalItem]] = []
    levels = cell.path.levels
    pdf = cell.path.clean_input == "pdf"
    for pos in docs:
        level = pos.level if len(levels) > 1 else levels[0]
        out.append((pos.order, item_from_row(rows[(pos.doc_id, level)], root, level, pdf)))
    base = len(docs)
    index = 0
    for item in real:
        if item.level in levels:
            out.append((base + index, item))
            index += 1
    return out


# ---- прогон ------------------------------------------------------------------------------


def _placement_log(cell: Cell) -> Path:
    return cell.cache_dir / "placement.jsonl"


ProgressFn = Callable[[int, int, Cell, EvalItem, Prediction, bool], None]


def run_matrix(
    cfg: MatrixConfig,
    *,
    docs: int | None = None,
    only: Sequence[str] | None = None,
    extract_fn: ExtractFn | None = None,
    placement_fn: Callable[[str, str], dict[str, int] | None] = ollama_placement,
    models_fn: Callable[[str], set[str]] = ollama_models,
    progress: ProgressFn | None = None,
    notice: Callable[[str], None] | None = None,
    on_pass: Callable[[int, int, int], None] | None = None,
) -> dict[str, int]:
    """Прогнать матрицу партиями; возвращает число свежих ответов по ячейкам."""
    cells = active_cells(cfg, only)
    if not cells:
        raise ValueError("нет ячеек для прогона")
    if extract_fn is None:
        url = cells[0].extract_config().ollama_url
        have = models_fn(url)
        missing = sorted({c.model.tag for c in cells} - have)
        if missing:
            raise RuntimeError(f"в Ollama нет моделей: {', '.join(missing)} (ollama pull <имя>)")
    plan = plan_docs(cfg)
    real = real_items(cfg)
    items = {c.id: cell_items(c, plan, real) for c in cells}
    last = max(order for rows in items.values() for order, _ in rows) + 1
    limit = last if docs is None else min(docs, last)
    batch = cfg.batch_docs
    fresh = {c.id: 0 for c in cells}
    passes = -(-limit // batch)
    for k, start in enumerate(range(0, limit, batch), start=1):
        end = min(start + batch, limit)
        if on_pass:
            on_pass(k, passes, end)
        for cell in cells:
            chunk = [it for order, it in items[cell.id] if start <= order < end]
            if not chunk:
                continue
            ran = []

            def wrapped(i, n, item, pred, cached, _cell=cell, _ran=ran):
                if not cached:
                    _ran.append(item.item_id)
                    fresh[_cell.id] += 1
                if progress:
                    progress(i, n, _cell, item, pred, cached)

            econfig = cell.extract_config()
            kwargs = {"extract_fn": extract_fn} if extract_fn is not None else {}
            run_predictions(
                chunk,
                econfig,
                cell.path.mode,
                cell.cache_dir,
                progress=wrapped,
                power_guard=cfg.power_guard,
                notice=notice,
                **kwargs,
            )
            if ran:
                _remember_placement(cell, econfig, placement_fn, extract_fn is None)
    return fresh


def _remember_placement(cell: Cell, econfig: ExtractConfig, placement_fn, enabled: bool) -> None:
    """Сколько модели лежит в VRAM — скорость зависит от разделения CPU/GPU (заметка P11)."""
    if not enabled:
        return
    try:
        place = placement_fn(econfig.ollama_url, cell.model.tag)
    except RuntimeError:
        return
    if place:
        with _placement_log(cell).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(place) + "\n")


def placement_summary(cell: Cell) -> dict[str, Any] | None:
    path = _placement_log(cell)
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return None
    shares = [r["size_vram"] / r["size"] for r in rows if r["size"]]
    return {
        "vram_share_median": round(statistics.median(shares), 3),
        "vram_share_min": round(min(shares), 3),
        "size_gib": round(rows[-1]["size"] / 2**30, 2),
    }


# ---- ход прогона и оценка времени -------------------------------------------------------


def cached_predictions(cell: Cell, items: Sequence[tuple[int, EvalItem]]) -> dict[str, Prediction]:
    econfig = cell.extract_config()
    out = {}
    for _, item in items:
        fp = fingerprint(item, econfig, cell.path.mode, None)
        cached = load_cached(cell.cache_dir / f"{item.item_id}.json", fp)
        if cached is not None:
            out[item.item_id] = cached
    return out


def progress_table(cfg: MatrixConfig) -> list[dict[str, Any]]:
    """По каждой ячейке: сколько элементов всего и готово, средняя длительность, остаток."""
    plan = plan_docs(cfg)
    real = real_items(cfg)
    check = load_vram_check(cfg)
    rows = []
    for cell in build_cells(cfg):
        reason = skip_reason(cfg, cell.model, check)
        if reason is not None:
            rows.append({"cell": cell.id, "skipped": reason})
            continue
        items = cell_items(cell, plan, real)
        done = cached_predictions(cell, items)
        walls = [p.wall_s for p in done.values() if p.wall_s]
        measured = statistics.fmean(walls) if walls else None
        per_item = measured if measured is not None else cell.model.guess_s_per_item
        total, ready = len(items), len(done)
        rows.append(
            {
                "cell": cell.id,
                "items": total,
                "done": ready,
                "seconds_per_item": round(per_item, 1),
                "measured": measured is not None,
                "remaining_hours": round((total - ready) * per_item / 3600, 2),
            }
        )
    return rows
