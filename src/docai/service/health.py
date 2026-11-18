"""Состояние окружения для `GET /v1/health`: Ollama (доступна ли, загружена ли модель) и GPU."""

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from docai.extract.config import ExtractConfig
from docai.service.models import GpuHealth, OllamaHealth

HealthProbe = Callable[[], tuple[OllamaHealth, GpuHealth]]

_TIMEOUT_S = 2.0


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
        return json.loads(response.read())


def probe_ollama(config: ExtractConfig) -> OllamaHealth:
    base = config.ollama_url.rstrip("/")
    health = OllamaHealth(reachable=False, model=config.model)
    try:
        tags = _get_json(f"{base}/api/tags")
        loaded = _get_json(f"{base}/api/ps")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        health.error = f"Ollama недоступна: {exc}"
        return health
    health.reachable = True
    health.model_available = _has_model(tags, config.model)
    health.model_loaded = _has_model(loaded, config.model)
    if not health.model_available:
        health.error = f"модель {config.model} не скачана (ollama pull)"
    return health


def _has_model(payload: dict[str, Any], name: str) -> bool:
    return any(name in (m.get("name"), m.get("model")) for m in payload.get("models") or [])


def probe_gpu() -> GpuHealth:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return GpuHealth(available=False, error="nvidia-smi не найден")
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        total, free = (int(part) for part in out.strip().splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return GpuHealth(available=False, error=str(exc))
    return GpuHealth(available=True, vram_total_mib=total, vram_free_mib=free)


def default_probe(config: ExtractConfig) -> HealthProbe:
    return lambda: (probe_ollama(config), probe_gpu())
