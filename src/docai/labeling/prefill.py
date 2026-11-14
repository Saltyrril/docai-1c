"""Предзаполнение формы моделью. Появится после шага P9 (извлечение реквизитов).

Интерфейс разметки уже вызывает `prefill_document`, поэтому в P9 достаточно
реализовать эту функцию: форма заполнится сама через `form.document_to_form`.
"""

from PIL import Image

from docai.schemas.documents import AnyDocument


class PrefillUnavailable(RuntimeError):
    """Предзаполнение недоступно; текст исключения показывается пользователю."""


def prefill_document(pages: list[Image.Image]) -> AnyDocument:
    """Распознать документ по страницам и вернуть его в нашей схеме.

    Сейчас — заглушка: модель извлечения будет подключена в P9. Поля вводятся вручную.
    """
    raise PrefillUnavailable(
        "Предзаполнение появится после шага P9 (извлечение моделью). Пока введите поля вручную."
    )
