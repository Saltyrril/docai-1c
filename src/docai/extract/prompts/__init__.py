"""Промпты: файлы `<имя>_v<N>.md` в этой папке; имя файла без расширения — версия в результате."""

import re
from functools import cache
from importlib import resources

_NAME = re.compile(r"^[a-z]+_v\d+$")


@cache
def load_prompt(version: str) -> str:
    """Текст промпта по версии, например `text_v1`."""
    if not _NAME.match(version):
        raise ValueError(f"версия промпта должна быть вида «text_v1»: {version!r}")
    path = resources.files(__name__).joinpath(f"{version}.md")
    if not path.is_file():
        raise FileNotFoundError(f"нет файла промпта {version}.md в docai/extract/prompts/")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(version: str, **values: str) -> str:
    """Подставить значения в `{имя}`. Прочие фигурные скобки в промпте не трогаются."""
    text = load_prompt(version)
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text
