"""Команда `docai serve`: запустить HTTP-сервис на 127.0.0.1."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from docai.extract.config import load_config
from docai.service.settings import DEFAULT_PORT, HOST, ServiceSettings


def serve(
    port: Annotated[int, typer.Option(help="Порт; адрес всегда 127.0.0.1.")] = DEFAULT_PORT,
    config: Annotated[
        Path | None, typer.Option(help="YAML с настройками извлечения (configs/extract.yaml).")
    ] = None,
    data_dir: Annotated[
        Path, typer.Option(help="Папка для базы, загруженных файлов и страниц.")
    ] = Path("data/service"),
) -> None:
    """Запустить сервис: Swagger — http://127.0.0.1:<порт>/docs, контракт — docs/API.md."""
    import uvicorn

    from docai.service.app import create_app

    settings = ServiceSettings(data_dir=data_dir, extract=load_config(config))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"
    )
    typer.echo(f"Сервис: http://{HOST}:{port}/docs  (данные: {data_dir})")
    # Один процесс и один воркер: GPU одна, задания идут по очереди.
    uvicorn.run(create_app(settings), host=HOST, port=port, log_level="info")
