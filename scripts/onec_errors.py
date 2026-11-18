"""Ошибки из журнала регистрации служебной базы 1С (build/onec_ib), P15.

Если обработка не скомпилировалась или упала, окно 1С показывает ошибку, а скрипт
автопроверки видит только «нет результата». Платформа пишет ту же ошибку в журнал
регистрации — файл SQLite `1Cv8Log/1Cv8.lgd`; этот скрипт печатает последние ошибки.

    uv run python scripts/onec_errors.py build/onec_ib [--minutes 30]
"""

import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Время в журнале — десятитысячные доли секунды от 01.01.0001 UTC (проверено на 8.3.8:
# запись ошибки отстаёт от местного времени файла ровно на смещение пояса).
_EPOCH = datetime(1, 1, 1, tzinfo=UTC)
_ERROR_SEVERITY = 3
_TICKS_PER_SECOND = 10_000


def ticks_to_datetime(ticks: int) -> datetime:
    return _EPOCH + timedelta(microseconds=ticks * (1_000_000 // _TICKS_PER_SECOND))


def datetime_to_ticks(moment: datetime) -> int:
    delta = moment - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * _TICKS_PER_SECOND + delta.microseconds // 100


def read_errors(infobase: Path, since: datetime) -> list[tuple[datetime, str]]:
    """Ошибки (уровень «Ошибка») не раньше `since`, от старых к новым."""
    log = infobase / "1Cv8Log" / "1Cv8.lgd"
    if not log.is_file():
        return []
    # Только чтение: журнал принадлежит платформе.
    with closing(sqlite3.connect(f"{log.resolve().as_uri()}?mode=ro", uri=True)) as db:
        rows = db.execute(
            "SELECT date, comment FROM EventLog WHERE severity = ? AND date >= ? ORDER BY rowID",
            (_ERROR_SEVERITY, datetime_to_ticks(since)),
        ).fetchall()
    return [(ticks_to_datetime(date), comment or "") for date, comment in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("infobase", type=Path, help="каталог файловой базы")
    parser.add_argument("--minutes", type=int, default=30, help="за сколько минут показать")
    args = parser.parse_args(argv)
    errors = read_errors(args.infobase, datetime.now(UTC) - timedelta(minutes=args.minutes))
    if not errors:
        print(f"В журнале {args.infobase} нет ошибок за {args.minutes} мин.")
        return 0
    print("Ошибки из журнала регистрации 1С:")
    for moment, text in errors:
        print(f"  {moment.astimezone():%H:%M:%S}  {text}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
