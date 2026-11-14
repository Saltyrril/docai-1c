"""Команды `docai edo export` (JSON → XML ФНС) и `docai edo parse` (XML → JSON)."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docai.edo.errors import EdoError
from docai.edo.reader import xml_to_upd
from docai.edo.writer import DEMO_SIGNER, Signer, upd_to_xml
from docai.schemas.documents import Upd, parse_document

app = typer.Typer(help="УПД из ЭДО: XML формата ФНС 5.03 ↔ наши JSON, без LLM.")
console = Console()


def _json_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json") if not p.name.startswith("_"))


def _signer(full_name: str, position: str) -> Signer:
    parts = full_name.split()
    if len(parts) not in (2, 3):
        raise typer.BadParameter("подписант: «Фамилия Имя» или «Фамилия Имя Отчество»")
    middle = parts[2] if len(parts) == 3 else None
    return Signer(parts[0], parts[1], middle, position)


@app.command()
def export(
    in_dir: Annotated[Path, typer.Option("--in", help="Папка с JSON документов.")] = Path(
        "data/synth/json"
    ),
    out: Annotated[Path, typer.Option(help="Папка для XML.")] = Path("data/synth/edo"),
    limit: Annotated[int | None, typer.Option(min=1, help="Взять только первые N УПД.")] = None,
    signer: Annotated[
        str, typer.Option(help="Подписант продавца (в JSON его нет; по умолчанию вымышленный).")
    ] = f"{DEMO_SIGNER.last_name} {DEMO_SIGNER.first_name} {DEMO_SIGNER.middle_name}",
    position: Annotated[str, typer.Option(help="Должность подписанта.")] = DEMO_SIGNER.position,
) -> None:
    """Записать каждый УПД как `<имя>.xml` (windows-1251) и проверить его по XSD.

    Имя файла берётся от JSON, чтобы пары легко находились; в XML внутри лежит
    собственный ИдФайл — перед отправкой через ЭДО файл переименовывают в него.
    """
    who = _signer(signer, position)
    out.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    for path in _json_files(in_dir):
        if limit and written >= limit:
            break
        document = parse_document(path.read_text(encoding="utf-8"))
        if not isinstance(document, Upd):
            skipped += 1
            continue
        try:
            data = upd_to_xml(document, who)
        except EdoError as exc:
            failed += 1
            console.print(f"[red]{path.name}[/red]: {exc}")
            continue
        (out / f"{path.stem}.xml").write_bytes(data)
        written += 1
    console.print(f"XML записано: {written}, не УПД пропущено: {skipped}, ошибок: {failed}. {out}")
    if failed:
        raise typer.Exit(1)


@app.command()
def parse(
    in_dir: Annotated[Path, typer.Option("--in", help="Папка с XML.")] = Path("data/synth/edo"),
    out: Annotated[Path, typer.Option(help="Папка для JSON.")] = Path("data/synth/edo_json"),
) -> None:
    """Разобрать XML в JSON нашей схемы (проверка по XSD включена)."""
    out.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for path in sorted(in_dir.glob("*.xml")):
        try:
            document = xml_to_upd(path.read_bytes())
        except EdoError as exc:
            failed += 1
            console.print(f"[red]{path.name}[/red]: {exc}")
            continue
        text = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)
        (out / f"{path.stem}.json").write_text(text + "\n", encoding="utf-8", newline="\n")
        done += 1
    console.print(f"JSON записано: {done}, ошибок: {failed}. {out}")
    if failed:
        raise typer.Exit(1)
