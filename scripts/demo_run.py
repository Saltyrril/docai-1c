"""Прогон демо-документов через HTTP API сервиса: время на документ и сводка (P17).

Вызывается из scripts/demo_e2e.ps1, можно и отдельно:
    uv run python scripts/demo_run.py --dir data/demo --port 8000 --out reports/<папка>/api_run.json

Использует только стандартную библиотеку; как и 1С, шлёт multipart/form-data и опрашивает задание.
Содержимое документов в отчёт не попадает: только вид, путь, статус, число полей и время.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".xml"}


def request(method: str, url: str, body: bytes | None = None, ctype: str = "") -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def upload(base: str, path: Path) -> dict[str, Any]:
    boundary = "docai" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    code, raw = request(
        "POST",
        f"{base}/v1/extract",
        head + path.read_bytes() + tail,
        f"multipart/form-data; boundary={boundary}",
    )
    if code != 202:
        raise RuntimeError(f"сервис ответил {code}: {raw.decode('utf-8', 'replace')[:200]}")
    return json.loads(raw)


def run_one(base: str, path: Path, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"file": path.name, "kind": path.suffix.lower().lstrip(".")}
    try:
        accepted = upload(base, path)
    except (RuntimeError, OSError) as exc:
        row.update(status="error", error=str(exc), total_s=round(time.perf_counter() - started, 2))
        return row
    job_id = accepted["id"]
    while True:
        code, raw = request("GET", f"{base}/v1/jobs/{job_id}")
        job = json.loads(raw) if code == 200 else {"status": "error", "error": raw.decode()[:200]}
        if job["status"] in {"done", "failed", "error"}:
            break
        if time.perf_counter() - started > timeout_s:
            job = {"status": "timeout", "error": f"нет результата за {timeout_s:.0f} с"}
            break
        time.sleep(1.0)
    total = time.perf_counter() - started
    fields = job.get("fields") or []
    row.update(
        status=job["status"],
        error=job.get("error"),
        total_s=round(total, 2),
        service_s=job.get("seconds"),
        doc_type=job.get("doc_type"),
        route=job.get("route"),
        document_status=job.get("document_status"),
        valid=job.get("valid"),
        fields_total=len(fields),
        fields_found=sum(1 for f in fields if f["value"] is not None),
        fields_need_review=sum(1 for f in fields if f["needs_review"]),
        issues=len(job.get("issues") or []),
        lines=len(job.get("lines") or []),
        mean_confidence=job.get("mean_confidence"),
        job_id=job_id,
    )
    return row


def print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'файл':36} {'путь':7} {'вид':11} {'статус документа':16} "
        f"{'нужно проверить':>15} {'с':>7}"
    )
    for r in rows:
        if r["status"] != "done":
            print(f"{r['file']:36} {'—':7} {'—':11} {'ОШИБКА: ' + str(r.get('error'))[:60]}")
            continue
        status = r.get("document_status") or "—"
        review = f"{r['fields_need_review']} из {r['fields_total']}"
        print(
            f"{r['file']:36} {r['route'] or '—':7} {r['doc_type'] or '—':11} "
            f"{status:16} {review:>15} {r['total_s']:>7.1f}"
        )
    done = [r for r in rows if r["status"] == "done"]
    print()
    print(f"Обработано: {len(done)} из {len(rows)}")
    if done:
        times = sorted(r["total_s"] for r in done)
        print(
            f"Время на документ: медиана {times[len(times) // 2]:.1f} с, "
            f"максимум {times[-1]:.1f} с, "
            f"сумма {sum(times):.0f} с"
        )
        by_status: dict[str, int] = {}
        for r in done:
            by_status[r["document_status"]] = by_status.get(r["document_status"], 0) + 1
        print("Статусы: " + ", ".join(f"{k} — {v}" for k, v in sorted(by_status.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=None, help="куда записать JSON с результатами")
    parser.add_argument("--timeout", type=float, default=300.0, help="секунд на один документ")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    files = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"В {args.dir} нет документов.")
        return 2
    base = f"http://127.0.0.1:{args.port}"
    print(f"Документов: {len(files)}, сервис {base}")
    rows = []
    for number, path in enumerate(files, 1):
        print(f"[{number}/{len(files)}] {path.name} ...", end=" ", flush=True)
        row = run_one(base, path, args.timeout)
        print(f"{row['status']} за {row['total_s']:.1f} с")
        rows.append(row)
    print_summary(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Результаты: {args.out}")
    return 0 if all(r["status"] == "done" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
