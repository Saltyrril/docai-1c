"""Отчёт оценки: reports/<дата>_<имя>/ — metrics.json, report.md, config.yaml, картинки.

Реальные документы (P8) попадают в отчёт только обезличенными: идентификаторы — псевдонимы,
значения в разнице — после `Pseudonymizer`, картинок нет (картинки не обезличиваются).
"""

import datetime as dt
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from docai.eval.calibration import reliability_svg
from docai.eval.compare import DocComparison, compare_documents
from docai.eval.config import EvalConfig
from docai.eval.metrics import GROUP_METRICS, DocRecord

MAX_DIFF_ROWS = 30


def report_dir(cfg: EvalConfig) -> Path:
    date = cfg.report.date or dt.date.today().isoformat()
    return cfg.report.out_dir / f"{date}_{cfg.name}"


def _num(x: float | None, digits: int = 1, percent: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    value = x * 100 if percent else x
    return f"{value:.{digits}f}".replace(".", ",")


def fmt(est: dict | None, percent: bool = True, digits: int | None = None) -> str:
    """«93,4% [91,0; 95,1]» — значение и 95% интервал."""
    if not est or est.get("value") is None:
        return "—"
    d = digits if digits is not None else (1 if percent else 3)
    unit = "%" if percent else ""
    value = _num(est["value"], d, percent)
    if est.get("low") is None:
        return f"{value}{unit}"
    return f"{value}{unit} [{_num(est['low'], d, percent)}; {_num(est['high'], d, percent)}]"


def fmt_p(p: float | None, *, eq: bool = False) -> str:
    """p-значение: «< 0,001» вместо «0,000»; eq — с «=» для текста («p = 0,03»)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 0.001:
        return "< 0,001"
    text = _num(p, 3, False)
    return f"= {text}" if eq else text


# Метрики, которые печатаются не процентом.
_PLAIN = {"ece": 3, "mce": 3, "brier": 3, "sec_per_page": 1}


def fmt_metric(name: str, est: dict | None) -> str:
    if name in _PLAIN:
        return fmt(est, percent=False, digits=_PLAIN[name])
    return fmt(est)


def _cell(value: Any) -> str:
    text = "∅" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= 60 else text[:57] + "…"


def diff_rows(c: DocComparison, truth, pred) -> list[tuple[str, str, str]]:
    """(поле, эталон, предсказание) для всех расхождений документа."""
    rows: list[tuple[str, str, str]] = []
    for f in c.header:
        if not f.correct:
            rows.append((f.field, _cell(f.ref), _cell(f.pred)))
    for name, value in c.empty_ref:
        if value is not None:
            rows.append((f"{name} (в эталоне пусто)", "∅", _cell(value)))
    for pair in c.pairs:
        for cell in pair.cells:
            if not cell.correct:
                label = cell.field.replace("lines[]", f"lines[{pair.ref_index}↔{pair.pred_index}]")
                rows.append((label, _cell(cell.ref), _cell(cell.pred)))
    matched_ref = {p.ref_index for p in c.pairs}
    matched_pred = {p.pred_index for p in c.pairs}
    for i, line in enumerate(truth.lines):
        if i not in matched_ref:
            rows.append((f"lines[{i}] — строка не найдена", _cell(line.name), "∅"))
    if pred is not None:
        for j, line in enumerate(pred.lines):
            if j not in matched_pred:
                rows.append((f"lines[{j}] — лишняя строка", "∅", _cell(line.name)))
    return rows


def _anonymized_diff(r: DocRecord, cfg: EvalConfig) -> list[tuple[str, str, str]] | None:
    from docai.labeling.pseudonymize import LeakError, Pseudonymizer, load_salt

    pseudo = Pseudonymizer(load_salt(cfg.data.real_root))
    try:
        truth = pseudo.document(r.truth)
        pred = pseudo.document(r.pred) if r.pred is not None else None
    except (LeakError, ValueError):
        return None
    m = cfg.metrics
    c = compare_documents(
        truth, pred, anls_threshold=m.anls_threshold, row_match_threshold=m.row_match_threshold
    )
    return diff_rows(c, truth, pred)


def _save_preview(src: Path, out: Path, side: int) -> None:
    with Image.open(src) as image:
        image = image.convert("RGB")
        image.thumbnail((side, side))
        image.save(out, "JPEG", quality=80, optimize=True)


def worst_records(records: Sequence[DocRecord], n: int) -> list[DocRecord]:
    return sorted(records, key=lambda r: (r.comparison.f1, r.item_id))[:n]


def _worst_section(records: Sequence[DocRecord], cfg: EvalConfig, out: Path) -> list[str]:
    lines = [
        "## Худшие примеры",
        "",
        f"{cfg.report.worst} документов с наименьшим F1 по всем значениям (шапка и ячейки). "
        "Обозначения: ∅ — пусто; `lines[i↔j]` — строка i эталона сопоставлена со строкой j "
        "ответа. Значения показаны после нормализации FIELD_SPEC.",
        "",
    ]
    (out / "worst").mkdir(exist_ok=True)
    for r in worst_records(records, cfg.report.worst):
        c = r.comparison
        lines += [
            f"### {r.item_id}",
            "",
            f"Тип: {r.doc_type} (ответ: {c.pred_doc_type or '—'}), уровень {r.level}, "
            f"путь {r.route or '—'}, статус {r.status or 'извлечение упало'}, F1 {_num(c.f1)}%, "
            f"строк {c.n_pred_rows} из {c.n_ref_rows}.",
            "",
        ]
        if r.exception:
            lines += [f"Исключение: `{_cell(r.exception)}`", ""]
        if r.real:
            rows = _anonymized_diff(r, cfg)
            lines += ["Реальный документ: картинки нет, значения обезличены (P8).", ""]
            if rows is None:
                rows = [(name, "скрыто", "скрыто") for name in c.required_errors]
        else:
            rows = diff_rows(c, r.truth, r.pred)
            if r.preview is not None and r.preview.is_file():
                name = f"{r.item_id}.jpg"
                _save_preview(r.preview, out / "worst" / name, cfg.report.image_side)
                lines += [f"![{r.item_id}](worst/{name})", ""]
        if rows:
            lines += ["| Поле | Эталон | Ответ |", "|---|---|---|"]
            lines += [f"| {a} | {b} | {d} |" for a, b, d in rows[:MAX_DIFF_ROWS]]
            if len(rows) > MAX_DIFF_ROWS:
                lines.append(f"| … ещё {len(rows) - MAX_DIFF_ROWS} | | |")
        else:
            lines.append("Расхождений нет.")
        lines.append("")
    return lines


def _group_table(group: dict, titles: dict[str, str]) -> list[str]:
    head = "| Значение | Документов | " + " | ".join(titles[m] for m in GROUP_METRICS) + " |"
    lines = [head, "|---|---|" + "---|" * len(GROUP_METRICS)]
    for value, cell in group["values"].items():
        metrics = cell["metrics"]
        lines.append(
            f"| {value} | {cell['documents']} | "
            + " | ".join(fmt_metric(m, metrics.get(m)) for m in GROUP_METRICS)
            + " |"
        )
    return lines


_SHORT = {
    "doc_ok": "Без правок",
    "doc_type_accuracy": "Тип верен",
    "header_accuracy": "Шапка acc",
    "text_anls": "ANLS",
    "rows_f1": "Строки F1",
    "cells_f1": "Ячейки F1",
    "silent_error": "Молча с ошибкой",
    "ece": "ECE",
    "sec_per_page": "с/стр",
}


def render_markdown(
    result: dict[str, Any], records: Sequence[DocRecord], cfg: EvalConfig, out: Path
) -> str:
    o = result["overall"]
    titles = result["metric_titles"]
    ec = result["extract_config"]
    b = cfg.bootstrap
    levels = ", ".join(cfg.data.levels)
    lines = [
        f"# Оценка: {cfg.name}",
        "",
        cfg.description.strip(),
        "",
        f"Запуск: `uv run docai eval --config configs/eval/{cfg.name}.yaml` "
        f"(копия — `config.yaml`). docai {result['docai_version']}.",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Модель | `{ec['model']}` |",
        f"| Путь извлечения | `{cfg.extract.mode}`"
        + (f", сверка `{cfg.extract.cross_check}`" if cfg.extract.cross_check else "")
        + " |",
        f"| Промпты | `{ec['text_prompt']}`, `{ec['vision_prompt']}` |",
        f"| Выборка | {result['sample']['items']} прогонов, "
        f"{result['sample']['documents']} документов, сплит `{cfg.data.split}`, "
        f"уровни {levels} ({cfg.data.level_assignment}) |",
        f"| Seed | {cfg.seed} |",
        f"| Интервалы | бутстреп по документам, {b.n_resamples} повторов, {b.ci:.0%} |",
        "",
        "## Главное",
        "",
        "В скобках — 95-процентный доверительный интервал (бутстреп по документам, D45).",
        "",
        "| Метрика | Значение |",
        "|---|---|",
    ]
    for name, est in o.items():
        lines.append(f"| {titles[name]} | {fmt_metric(name, est)} |")

    rvp = result["raw_vs_post"]
    lines += [
        "",
        "## Ответ модели до и после постобработки (P10)",
        "",
        "Одни и те же документы: парный бутстреп разности «после − до», p — приближённое "
        "двустороннее по бутстрепу; для доли документов без правок — ещё тест Макнемара.",
        "",
        "| Метрика | До | После | Разность [95% ДИ] | p |",
        "|---|---|---|---|---|",
    ]
    for name, est in rvp["metrics"].items():
        lines.append(
            f"| {titles[name]} | {_num(est['a'])}% | {_num(est['b'])}% "
            f"| {_num(est['diff'])} п.п. [{_num(est['low'])}; {_num(est['high'])}] "
            f"| {fmt_p(est['p_value'])} |"
        )
    mc = rvp["mcnemar_doc_ok"]
    lines += [
        "",
        f"Макнемар (без правок): только до — {mc['a_only']}, только после — {mc['b_only']}, "
        f"p {fmt_p(mc['p_value'], eq=True)} ({mc['method']}).",
    ]

    short = {m: _SHORT.get(m, titles[m]) for m in GROUP_METRICS}
    for group in result["groups"].values():
        lines += ["", f"## По группам: {group['title_ru'].lower()}", ""]
        lines += _group_table(group, short)
    lines += [
        "",
        "Столбцы: «Без правок» — документ без ошибок в обязательных полях; «Шапка acc» — "
        "accuracy полей шапки; ANLS — текстовые поля; «Строки F1» — найдена пара строки; "
        "«Молча с ошибкой» — статус «можно загружать» у документа с ошибкой; "
        "с/стр — секунд на страницу.",
    ]

    lines += [
        "",
        "## По полям",
        "",
        "Accuracy — верных среди непустых эталонных значений (у полей строк ячейка "
        "несопоставленной строки считается неверной); precision — верных среди непустых ответов.",
        "",
        "| Поле | Название | Эталонных значений | Accuracy | ANLS | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for name, f in result["fields"].items():
        lines.append(
            f"| `{name}` | {f['title_ru']} | {f['reference_values']} | {fmt(f.get('accuracy'))} "
            f"| {fmt(f.get('anls'))} | {fmt(f.get('precision'))} |"
        )

    cal = result["calibration"]
    auc = cal["auroc"]
    lines += [
        "",
        "## Калибровка уверенности",
        "",
        f"Полей с уверенностью: {cal['points']}. ECE {fmt_metric('ece', o['ece'])}, "
        f"MCE {fmt_metric('mce', o['mce'])}, Brier {fmt_metric('brier', o['brier'])}, "
        f"AUROC «уверенность отличает верное поле от неверного» "
        f"{fmt(auc, percent=False, digits=3)}. Порог «нужна проверка» — "
        f"{_num(cal['review_threshold'], 2, False)}.",
        "",
        "![График надёжности](reliability.svg)",
        "",
        "| Уверенность | Полей | Средняя уверенность | Доля верных |",
        "|---|---|---|---|",
    ]
    for cell in cal["bins"]:
        lines.append(
            f"| {_num(cell['low'], 1, False)}–{_num(cell['high'], 1, False)} | {cell['fields']} "
            f"| {_num(cell['mean_confidence'], 3, False)} | {_num(cell['share_correct'])}% |"
        )

    perf = result["performance"]
    lines += [
        "",
        "## Производительность",
        "",
        "Секунд на страницу (сумма времени / сумма страниц): "
        f"{fmt_metric('sec_per_page', o['sec_per_page'])}; "
        f"медиана по документам {_num(perf['seconds_per_page_median'], 1, False)}, "
        f"90-й процентиль {_num(perf['seconds_per_page_p90'], 1, False)}, "
        f"максимум {_num(perf['seconds_per_page_max'], 1, False)}. "
        f"Документ в среднем {_num(perf['seconds_per_doc_mean'], 1, False)} с, всего "
        f"{_num(perf['seconds_total'] / 3600 if perf['seconds_total'] else None, 2, False)} ч на "
        f"{perf['pages']} страниц. Пиковая VRAM (nvidia-smi, весь GPU): максимум "
        f"{perf['peak_vram_mib_max'] or '—'} МиБ, "
        f"медиана {perf['peak_vram_mib_median'] or '—'} МиБ. "
        f"Токены: {perf['prompt_tokens']} → {perf['completion_tokens']}.",
    ]

    if "comparison" in result:
        comp = result["comparison"]
        lines += ["", f"## Сравнение с `{comp.get('other')}`", ""]
        if comp["common_documents"]:
            lines += [
                f"Общих документов: {comp['common_documents']}. A — `{comp.get('other')}`, "
                f"B — `{cfg.name}`; парный бутстреп разности B − A.",
                "",
                "| Метрика | A | B | B − A [95% ДИ] | p |",
                "|---|---|---|---|---|",
            ]
            for name, est in comp["metrics"].items():
                pct = name not in _PLAIN
                unit = " п.п." if pct else ""
                d = 1 if pct else 3
                lines.append(
                    f"| {titles[name]} | {_num(est['a'], d, pct)} | {_num(est['b'], d, pct)} "
                    f"| {_num(est['diff'], d, pct)}{unit} [{_num(est['low'], d, pct)}; "
                    f"{_num(est['high'], d, pct)}] | {fmt_p(est['p_value'])} |"
                )
            mc = comp["mcnemar_doc_ok"]
            lines += [
                "",
                f"Макнемар (без правок): только A — {mc['a_only']}, только B — {mc['b_only']}, "
                f"p {fmt_p(mc['p_value'], eq=True)} ({mc['method']}).",
            ]
        else:
            lines.append("Общих документов нет.")

    notes = out / "notes.md"  # наблюдения, написанные вручную после прогона
    if notes.is_file():
        lines += ["", notes.read_text(encoding="utf-8").strip()]
    lines += ["", *_worst_section(records, cfg, out)]
    lines += [
        "## Определения",
        "",
        "Сравнение полей, сопоставление строк и «можно загружать без правок» — D44; "
        "интервалы, парное сравнение и калибровка — D45 (`docs/DECISIONS.md`). "
        "Все значения сравниваются после нормализации FIELD_SPEC.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict[str, Any], records: Sequence[DocRecord], cfg: EvalConfig) -> Path:
    out = report_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(json.loads(cfg.model_dump_json()), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    title = f"Надёжность уверенности: {cfg.name}"
    (out / "reliability.svg").write_text(
        reliability_svg(result["calibration"]["bins"], title), encoding="utf-8", newline="\n"
    )
    (out / "report.md").write_text(
        render_markdown(result, records, cfg, out), encoding="utf-8", newline="\n"
    )
    return out
