"""Задача о назначениях (венгерский алгоритм): какая строка предсказания — какая строка эталона.

Модель может пропустить или вставить строку, поэтому сравнивать строки по номеру нельзя:
одна пропущенная строка сдвинула бы все следующие, и каждая из них считалась бы ошибкой.
Строки сопоставляются так, чтобы суммарное сходство пар было наибольшим (D44).

Реализация — алгоритм Куна — Манкреса с потенциалами за O(n²·m) для прямоугольной
матрицы (Kuhn, 1955; Munkres, 1957; запись по Jonker & Volgenant). Своя, а не
scipy.optimize.linear_sum_assignment: таблицы маленькие (десятки строк), а scipy ради
одной функции — лишняя тяжёлая зависимость. Совпадение с полным перебором проверяется тестом.
"""

from collections.abc import Sequence

_INF = float("inf")


def linear_assignment(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Пары (строка, столбец) с наименьшей суммарной стоимостью.

    Матрица может быть прямоугольной: назначается min(n, m) пар, каждая строка и каждый
    столбец — не больше одного раза. Пары отсортированы по номеру строки.
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return []
    if any(len(row) != m for row in cost):
        raise ValueError("строки матрицы стоимостей разной длины")
    transposed = n > m
    matrix = (
        [list(col) for col in zip(*cost, strict=True)] if transposed else [list(r) for r in cost]
    )
    if transposed:
        n, m = m, n

    # Индексация с 1, нулевой столбец — фиктивный (как в классической записи алгоритма).
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    owner = [0] * (m + 1)  # owner[j] — строка, назначенная столбцу j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = owner[j0]
            delta = _INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                reduced = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if reduced < minv[j]:
                    minv[j] = reduced
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while True:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j] != 0]
    if transposed:
        pairs = [(col, row) for row, col in pairs]
    return sorted(pairs)
