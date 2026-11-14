"""Пиковая VRAM: nvidia-smi в фоне опрашивает память GPU с заданным периодом.

Меряется занятая память всего GPU (как в P0), а не одного процесса: Ollama работает
отдельным процессом, а порог 3,8 ГиБ касается видеокарты целиком.
"""

import shutil
import subprocess
import threading
import time
from types import TracebackType

from pydantic import BaseModel


class VramStats(BaseModel):
    available: bool
    baseline_mib: int | None = None
    peak_mib: int | None = None
    samples: int = 0
    poll_s: float
    error: str | None = None

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak_mib is None else round(self.peak_mib / 1024, 3)


def _query(nvidia_smi: str) -> int:
    out = subprocess.run(
        [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    return int(out.strip().splitlines()[0])


class VramMonitor:
    """Контекстный менеджер: `with VramMonitor(0.5) as vram: ...; vram.stats()`."""

    def __init__(self, poll_s: float = 0.5, nvidia_smi: str | None = None) -> None:
        self.poll_s = poll_s
        self._cmd = nvidia_smi or shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: int | None = None
        self._peak: int | None = None
        self._samples = 0
        self._error: str | None = None

    def _sample(self) -> None:
        assert self._cmd is not None
        try:
            value = _query(self._cmd)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            self._error = str(exc)
            return
        self._samples += 1
        self._peak = value if self._peak is None else max(self._peak, value)
        if self._baseline is None:
            self._baseline = value

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            self._sample()
            self._stop.wait(max(0.0, self.poll_s - (time.perf_counter() - started)))

    def __enter__(self) -> "VramMonitor":
        if self._cmd is None:
            self._error = "nvidia-smi не найден"
            return self
        self._sample()  # базовая линия до начала работы
        self._thread = threading.Thread(target=self._run, name="vram-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._sample()  # последний замер после окончания

    def stats(self) -> VramStats:
        return VramStats(
            available=self._samples > 0,
            baseline_mib=self._baseline,
            peak_mib=self._peak,
            samples=self._samples,
            poll_s=self.poll_s,
            error=self._error if self._samples == 0 else None,
        )
