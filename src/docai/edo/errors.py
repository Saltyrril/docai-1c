"""Ошибки пакета `docai.edo`."""


class EdoError(ValueError):
    """Документ нельзя перевести в XML формата ФНС (или обратно) без потерь."""


class EdoValidationError(EdoError):
    """XML не проходит проверку по XSD формата ФНС."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        shown = "; ".join(errors[:5])
        more = f" (и ещё {len(errors) - 5})" if len(errors) > 5 else ""
        super().__init__(f"XML не соответствует XSD: {shown}{more}")
