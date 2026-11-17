"""Доверительные интервалы и сравнение конфигураций (определения — D45).

Единица выборки — документ: поля и строки одного документа зависимы (одна ошибка в типе
документа портит десяток полей), поэтому бутстреп перевыбирает документы целиком
(кластерный бутстреп), а метрику каждый раз пересчитывает по сумме счётчиков выбранных
документов (micro-усреднение). Интервал — процентильный, 95%, 2000 повторов, seed из конфига.

Сравнение двух конфигураций на одних и тех же документах:
- парный бутстреп: одни и те же номера документов для A и B в каждом повторе,
  интервал для разности B − A; приближённое двустороннее p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0));
- тест Макнемара для доли документов «можно загружать без правок» (бинарный исход на
  документ): b — верно в A и неверно в B, c — наоборот; при b + c < 25 точный биномиальный
  тест, иначе χ² с поправкой Йейтса. По полям Макнемар не применяется: поля одного
  документа не независимы.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Sums = Mapping[str, np.ndarray]
MetricFn = Callable[[Sums], np.ndarray]


def ratio(num: str, den: str) -> MetricFn:
    def fn(s: Sums) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(s[num], dtype=float) / np.asarray(s[den], dtype=float)

    return fn


def f1(tp: str, n_pred: str, n_ref: str) -> MetricFn:
    def fn(s: Sums) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2 * np.asarray(s[tp], float) / (np.asarray(s[n_pred], float) + s[n_ref])

    return fn


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float | None]:
        def clean(x: float) -> float | None:
            return None if math.isnan(x) else round(x, 6)

        return {"value": clean(self.value), "low": clean(self.low), "high": clean(self.high)}


def resample_weights(n_docs: int, n_resamples: int, seed: int) -> np.ndarray:
    """Матрица B×n: сколько раз каждый документ попал в каждый повтор бутстрепа."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_docs, size=(n_resamples, n_docs))
    weights = np.zeros((n_resamples, n_docs))
    rows = np.repeat(np.arange(n_resamples), n_docs)
    np.add.at(weights, (rows, idx.ravel()), 1)
    return weights


class _Totals(dict):
    """Суммы счётчиков; счётчика, которого нет ни у одного документа, — ноль."""

    def __init__(self, data: dict, shape: tuple[int, ...]) -> None:
        super().__init__(data)
        self.shape = shape

    def __missing__(self, key: str) -> np.ndarray | float:
        return np.zeros(self.shape) if self.shape else 0.0


def _sums(matrix: np.ndarray, keys: Sequence[str], weights: np.ndarray | None) -> _Totals:
    totals = matrix.sum(axis=0) if weights is None else weights @ matrix
    return _Totals({key: totals[..., k] for k, key in enumerate(keys)}, totals.shape[:-1])


def totals(matrix: np.ndarray, keys: Sequence[str]) -> _Totals:
    """Суммы счётчиков по всем документам (точечная оценка)."""
    return _sums(matrix, keys, None)


def _percentiles(values: np.ndarray, ci: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    return float(low), float(high)


def bootstrap(
    matrix: np.ndarray,
    keys: Sequence[str],
    metrics: Mapping[str, MetricFn],
    *,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Точечная оценка и процентильный интервал каждой метрики.

    matrix — счётчики по документам (n_docs × len(keys)); метрика — функция от сумм.
    """
    if matrix.shape[0] == 0:
        nan = float("nan")
        return {name: Estimate(nan, nan, nan) for name in metrics}
    point = _sums(matrix, keys, None)
    boot = _sums(matrix, keys, resample_weights(matrix.shape[0], n_resamples, seed))
    out = {}
    for name, fn in metrics.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(fn(point))
            low, high = _percentiles(np.asarray(fn(boot), dtype=float), ci)
        out[name] = Estimate(value, low, high)
    return out


@dataclass(frozen=True, slots=True)
class PairedEstimate:
    a: float
    b: float
    diff: float  # b − a
    low: float
    high: float
    p_value: float

    def as_dict(self) -> dict[str, float | None]:
        def clean(x: float) -> float | None:
            return None if math.isnan(x) else round(x, 6)

        return {k: clean(getattr(self, k)) for k in ("a", "b", "diff", "low", "high", "p_value")}


def paired_bootstrap(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    keys: Sequence[str],
    metrics: Mapping[str, MetricFn],
    *,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, PairedEstimate]:
    """Разность метрик B − A на одних и тех же документах (строки матриц выровнены)."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("матрицы A и B должны описывать одни и те же документы")
    weights = resample_weights(matrix_a.shape[0], n_resamples, seed)
    pa, pb = _sums(matrix_a, keys, None), _sums(matrix_b, keys, None)
    ba, bb = _sums(matrix_a, keys, weights), _sums(matrix_b, keys, weights)
    out = {}
    for name, fn in metrics.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = float(fn(pa)), float(fn(pb))
            deltas = np.asarray(fn(bb), float) - np.asarray(fn(ba), float)
        low, high = _percentiles(deltas, ci)
        finite = deltas[np.isfinite(deltas)]
        if finite.size:
            p = min(1.0, 2 * min(float(np.mean(finite <= 0)), float(np.mean(finite >= 0))))
        else:
            p = float("nan")
        out[name] = PairedEstimate(a, b, b - a, low, high, p)
    return out


@dataclass(frozen=True, slots=True)
class McNemar:
    b: int  # верно в A, неверно в B
    c: int  # неверно в A, верно в B
    statistic: float | None
    p_value: float
    method: str

    def as_dict(self) -> dict[str, object]:
        return {
            "a_only": self.b,
            "b_only": self.c,
            "statistic": None if self.statistic is None else round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "method": self.method,
        }


def mcnemar(a: Sequence[bool], b: Sequence[bool], exact_below: int = 25) -> McNemar:
    """Тест Макнемара для парных бинарных исходов (True — успех)."""
    if len(a) != len(b):
        raise ValueError("исходы A и B должны быть по одним и тем же документам")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return McNemar(0, 0, None, 1.0, "нет расхождений")
    if n < exact_below:
        k = min(only_a, only_b)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        return McNemar(only_a, only_b, None, min(1.0, 2 * tail), "точный биномиальный")
    stat = (abs(only_a - only_b) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))  # P(χ²₁ ≥ stat)
    return McNemar(only_a, only_b, stat, p, "χ² с поправкой Йейтса")
