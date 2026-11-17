"""Прогон извлечения по выборке с кэшем «сырых» ответов модели.

Каждый ответ пишется в `<cache_dir>/<item_id>.json` сразу после получения, поэтому
прерванный прогон продолжается с места остановки. В записи хранится отпечаток всего,
от чего зависит ответ модели (конфиг извлечения без постобработки, режим, входные файлы);
запись с другим отпечатком считается устаревшей и пересчитывается. Постобработка (D41–D43)
в отпечаток не входит: она применяется при подсчёте метрик и пересчитывается по кэшу.

Исключение при извлечении (Ollama упала, таймаут) записывается как неудача с текстом
ошибки и при следующем запуске повторяется. Три исключения подряд останавливают прогон:
это почти всегда окружение (Ollama не запущена), а не документ. Исключение, которое
описывает ответ модели (Ollama прервала зациклившуюся генерацию), — не сбой окружения:
оно кэшируется как несобранный документ и не повторяется.
"""

import contextlib
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from docai.eval import power
from docai.eval.sample import EvalItem
from docai.extract import ExtractConfig, Mode, RawExtraction, extract_raw

log = logging.getLogger("docai.eval")

# Поля конфига, которые не влияют на ответ модели.
_NOT_AFFECTING = {
    "ollama_url",
    "keep_alive",
    "timeout_s",
    "vram_poll_s",
    "vram_limit_gib",
    "tesseract_cmd",
    "postprocess",
}
# Поля, добавленные после первых прогонов: при значении, равном прежнему поведению,
# в отпечаток не входят — иначе старый кэш (P11) стал бы «устаревшим» без причины.
_LEGACY_DEFAULTS = {"date_format": "iso"}
MAX_CONSECUTIVE_FAILURES = 3

ExtractFn = Callable[[Sequence[Path], Mode, ExtractConfig, Mode | None], RawExtraction]


class Prediction(BaseModel):
    item_id: str
    fingerprint: str
    raw: RawExtraction | None = None
    # Текст исключения, если извлечение упало (такая запись повторяется при следующем запуске).
    exception: str | None = None
    wall_s: float = 0.0
    # Питание в конце документа: True — сеть, False — батарея, None — неизвестно.
    on_ac: bool | None = None


def _rel(path: Path) -> str:
    """Путь для отпечатка: относительно текущей папки, с прямыми слэшами."""
    with contextlib.suppress(ValueError):
        path = path.resolve().relative_to(Path.cwd().resolve())
    return path.as_posix()


def fingerprint_config(config: ExtractConfig) -> dict:
    """Часть конфига, от которой зависит ответ модели (входит в отпечаток)."""
    dumped = config.model_dump(mode="json", exclude=_NOT_AFFECTING)
    for key, legacy in _LEGACY_DEFAULTS.items():
        if dumped.get(key) == legacy:
            dumped.pop(key)
    return dumped


def fingerprint(item: EvalItem, config: ExtractConfig, mode: Mode, cross_check: Mode | None) -> str:
    payload = {
        "config": fingerprint_config(config),
        "mode": str(mode),
        "cross_check": str(cross_check) if cross_check else None,
        "inputs": [_rel(p) for p in item.inputs],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_cached(path: Path, expected: str) -> Prediction | None:
    if not path.is_file():
        return None
    try:
        cached = Prediction.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        log.warning("кэш %s не читается — пересчитываю", path.name)
        return None
    if cached.fingerprint != expected:
        return None
    if cached.exception is not None and not is_model_failure(cached.exception):
        return None
    return cached


# Ollama сама прервала генерацию: модель зациклилась. Это ответ модели на документ, а не сбой
# окружения: такой результат кэшируется и считается несобранным документом, иначе пересборка
# отчёта прогоняла бы документ снова и могла бы молча поменять метрики.
_MODEL_FAILURES = ("prediction aborted",)


def is_model_failure(exception: str) -> bool:
    return any(marker in exception for marker in _MODEL_FAILURES)


def _default_extract(
    paths: Sequence[Path], mode: Mode, config: ExtractConfig, cross_check: Mode | None
) -> RawExtraction:
    return extract_raw(paths, mode, config, cross_check)


def run_predictions(
    items: Sequence[EvalItem],
    config: ExtractConfig,
    mode: Mode,
    cache_dir: Path,
    *,
    cross_check: Mode | None = None,
    extract_fn: ExtractFn | None = None,
    progress: Callable[[int, int, EvalItem, Prediction, bool], None] | None = None,
    power_guard: bool = False,
    notice: Callable[[str], None] | None = None,
) -> dict[str, Prediction]:
    """Ответы модели по всем элементам выборки: из кэша или новым прогоном.

    power_guard: считать только от сети; документ, во время которого был переход на батарею
    или сон, пересчитывается (модуль `power`, D47).
    """
    extract_fn = extract_fn or _default_extract
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Prediction] = {}
    failures = 0
    for index, item in enumerate(items, start=1):
        fp = fingerprint(item, config, mode, cross_check)
        path = cache_dir / f"{item.item_id}.json"
        cached = load_cached(path, fp)
        if cached is not None:
            out[item.item_id] = cached
            if progress:
                progress(index, len(items), item, cached, True)
            continue
        while True:
            if power_guard:
                power.wait_for_ac(notice)
            started = time.perf_counter()
            with power.PowerWatch() as watch:
                try:
                    raw = extract_fn(item.inputs, mode, config, cross_check)
                    pred = Prediction(item_id=item.item_id, fingerprint=fp, raw=raw)
                except Exception as exc:  # любая ошибка документа пишется и не роняет прогон
                    pred = Prediction(
                        item_id=item.item_id,
                        fingerprint=fp,
                        exception=f"{type(exc).__name__}: {exc}",
                    )
            if power_guard and not watch.clean:
                if notice:
                    notice(f"{item.item_id}: во время документа — {watch.reason}; пересчитаю")
                continue
            break
        if pred.exception is None or is_model_failure(pred.exception):
            failures = 0
        else:
            failures += 1
            log.error("%s: %s", item.item_id, pred.exception)
        pred.on_ac = power.on_ac_power()
        pred.wall_s = round(time.perf_counter() - started, 3)
        _write_atomic(path, pred.model_dump_json() + "\n")
        out[item.item_id] = pred
        if progress:
            progress(index, len(items), item, pred, False)
        if failures >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"{failures} исключения подряд, последнее: {pred.exception}. "
                "Проверьте окружение (ollama serve, модель, nvidia-smi) и запустите снова: "
                "готовые ответы возьмутся из кэша."
            )
    return out
