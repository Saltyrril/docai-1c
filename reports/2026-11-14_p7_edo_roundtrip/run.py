"""Эксперимент P7: 100 УПД → XML → проверка по XSD → обратный разбор → сравнение с исходником.

Запуск из корня репозитория:
    uv run python reports/2026-11-14_p7_edo_roundtrip/run.py
Пишет results.json и report.md рядом со скриптом. Данные синтетические, data/real не нужен.
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import yaml

from docai.edo import upd_to_xml, xml_to_upd
from docai.edo.schema import XSD_PATH, parse_xml, xsd_errors
from docai.schemas import FIELD_SPEC, DocType
from docai.schemas.documents import Upd
from docai.schemas.fields import get_field_value
from docai.synth.generator import generate_document

HERE = Path(__file__).parent
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def make_corpus() -> list[Upd]:
    docs: list[Upd] = []
    index = 0
    while len(docs) < CONFIG["documents"]:
        index += 1
        document = generate_document(CONFIG["seed"], index, CONFIG["error_rate"]).document
        if isinstance(document, Upd):
            docs.append(document)
    return docs


def main() -> None:
    corpus = make_corpus()
    started = time.perf_counter()
    xsd_ok = exact = 0
    field_checks = field_mismatches = 0
    sizes: list[int] = []
    digest = hashlib.sha256()
    failures: list[str] = []
    for document in corpus:
        data = upd_to_xml(document, validate=False)
        digest.update(data)
        sizes.append(len(data))
        errors = xsd_errors(parse_xml(data))
        xsd_ok += not errors
        if errors:
            failures.append(f"{document.number}: XSD: {errors[0]}")
            continue
        back = xml_to_upd(data)
        exact += back == document
        if back != document:
            failures.append(f"{document.number}: обратный разбор отличается")
        for spec in FIELD_SPEC:
            if not spec.applies_to(DocType.UPD):
                continue
            field_checks += 1
            before, after = get_field_value(document, spec), get_field_value(back, spec)
            if before != after:
                field_mismatches += 1
    elapsed = time.perf_counter() - started

    rates = Counter(line.vat_rate.value for d in corpus for line in d.lines if line.vat_rate)
    results = {
        "documents": len(corpus),
        "xsd_valid": xsd_ok,
        "round_trip_exact": exact,
        "field_checks": field_checks,
        "field_mismatches": field_mismatches,
        "failures": failures,
        "composition": {
            "status_1": sum(d.upd_status == 1 for d in corpus),
            "status_2": sum(d.upd_status == 2 for d in corpus),
            "seller_ip": sum(len(d.seller.inn or "") == 12 for d in corpus),
            "buyer_ip": sum(len(d.buyer.inn or "") == 12 for d in corpus),
            "without_basis": sum(d.basis is None for d in corpus),
            "lines": sum(len(d.lines) for d in corpus),
            "vat_rates": dict(sorted(rates.items())),
        },
        "xml_bytes_min_max": [min(sizes), max(sizes)],
        "xml_sha256": digest.hexdigest(),
        "xsd_sha256": hashlib.sha256(XSD_PATH.read_bytes()).hexdigest(),
        "seconds": round(elapsed, 2),
    }
    (HERE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    comp = results["composition"]
    rates_text = ", ".join(f"{k} — {v}" for k, v in comp["vat_rates"].items())
    lines = [
        "# P7: УПД из ЭДО, XML формата ФНС 5.03 туда и обратно",
        "",
        f"Серия: {len(corpus)} синтетических УПД, seed {CONFIG['seed']}, без намеренных ошибок.",
        "Путь: `Upd` → XML (windows-1251) → проверка по XSD → `Upd`. Повтор — `config.yaml`.",
        "",
        "| Проверка | Результат |",
        "|---|---|",
        f"| XML прошли XSD | {xsd_ok} из {len(corpus)} |",
        f"| Обратный разбор равен исходнику целиком | {exact} из {len(corpus)} |",
        f"| Пары «поле FIELD_SPEC × документ» совпали | {field_checks - field_mismatches} "
        f"из {field_checks} |",
        "",
        f"Состав серии: статус 1 — {comp['status_1']}, статус 2 — {comp['status_2']}; "
        f"продавец ИП — {comp['seller_ip']}, покупатель ИП — {comp['buyer_ip']}; "
        f"без основания — {comp['without_basis']}; строк всего — {comp['lines']}.",
        f"Ставки в строках: {rates_text}.",
        "",
        f"Размер XML: {min(sizes)}–{max(sizes)} байт. Время: {results['seconds']} с.",
        f"SHA-256 схемы: `{results['xsd_sha256'][:16]}…`; SHA-256 всех XML подряд: "
        f"`{results['xml_sha256'][:16]}…` (при повторе тот же: генерация детерминирована).",
        "",
        "Допущения (D30): адрес — одной строкой в `АдрИнф`; подписант вымышленный; "
        "грузоотправитель — «он же». Проверка только по XSD: правила schematron из блока "
        "appinfo схемы (например, совпадение имени файла с `ИдФайл`) не выполнялись.",
    ]
    report = chr(10).join(lines) + chr(10)
    (HERE / "report.md").write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps({k: results[k] for k in ("documents", "xsd_valid", "round_trip_exact")}))
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
