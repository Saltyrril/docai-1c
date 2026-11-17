"""Питание ноутбука во время прогона: замеры скорости честны только от сети.

На батарее Windows и драйвер NVIDIA урезают мощность, и документ считается в 1,5–3 раза
дольше (замерено в P11: повторы на сети в среднем вдвое быстрее). Сон посреди документа
превращает его время в часы. Поэтому прогон оценки (D47):
- перед каждым документом ждёт питания от сети;
- во время документа в фоне раз в 2 с проверяет питание и разрыв во времени между
  проверками (больше 30 с — компьютер спал); такой ответ не сохраняется, документ
  считается заново на сети.

Источник состояния — `GetSystemPowerStatus` из kernel32 (Windows), без зависимостей.
На других системах состояние неизвестно (None) и прогон не ждёт.
"""

import ctypes
import sys
import threading
import time
from collections.abc import Callable
from types import TracebackType

# Сколько ждать между проверками, пока ноутбук на батарее (с).
WAIT_POLL_S = 30.0
# Пауза ожидания; отдельное имя, чтобы тесты подменяли её, не трогая модуль time.
_sleep = time.sleep


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [  # так задаётся структура ctypes
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def on_ac_power() -> bool | None:
    """True — от сети, False — от батареи, None — неизвестно (не Windows или ошибка)."""
    if sys.platform != "win32":
        return None
    status = _SystemPowerStatus()
    try:
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    if not ok:
        return None
    return {0: False, 1: True}.get(status.ACLineStatus)


def wait_for_ac(notice: Callable[[str], None] | None = None) -> float:
    """Ждать питания от сети; вернуть, сколько секунд ждали."""
    started = time.monotonic()
    announced = False
    while on_ac_power() is False:
        if not announced and notice:
            notice("ноутбук работает от батареи — жду подключения зарядки")
            announced = True
        _sleep(WAIT_POLL_S)
    waited = time.monotonic() - started
    if announced and notice:
        notice(f"питание от сети, продолжаю (ждал {waited / 60:.0f} мин)")
    return waited


class PowerWatch:
    """`with PowerWatch() as w: ...; w.clean` — не было ли батареи и сна за время блока."""

    def __init__(
        self,
        poll_s: float = 2.0,
        sleep_gap_s: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.poll_s = poll_s
        self.sleep_gap_s = sleep_gap_s
        # Настенные часы: после сна они уходят вперёд, а пауза между замерами — нет.
        self._clock = clock
        self.battery = False
        self.slept = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last = 0.0

    def _sample(self) -> None:
        now = self._clock()
        if self._last and now - self._last > self.sleep_gap_s:
            self.slept = True
        self._last = now
        if on_ac_power() is False:
            self.battery = True

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._sample()

    def __enter__(self) -> "PowerWatch":
        self._sample()
        self._thread = threading.Thread(target=self._run, name="power-watch", daemon=True)
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
            self._thread.join(timeout=10)
        self._sample()

    @property
    def clean(self) -> bool:
        return not self.battery and not self.slept

    @property
    def reason(self) -> str:
        parts = (["работа от батареи"] if self.battery else []) + (["сон"] if self.slept else [])
        return " и ".join(parts) or "—"
