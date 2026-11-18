"""Настройки постобработки и уверенности. Формула и обоснование констант — D41, D42."""

from pydantic import BaseModel, ConfigDict, Field

# Группы полей для априорной уверенности (см. kind_of в confidence.py).
KINDS = ("id", "money", "date", "text", "enum", "code")


def _priors() -> dict[str, dict[str, float]]:
    """P(поле прочитано верно) по пути извлечения и группе поля, ДО проверок.

    Взяты из доли верных полей в прогоне P9 на 30 документах (по 10 на путь,
    `reports/2026-11-16_p10_validation/`), округлены и подтянуты к середине: выборка мала.
    XML разбирается парсером без LLM — там ошибок чтения нет. Пересчитать на тестовом
    сплите в P13.
    """
    return {
        "xml": dict.fromkeys(KINDS, 0.995),
        "text": {"id": 0.93, "money": 0.97, "date": 0.6, "text": 0.93, "enum": 0.95, "code": 0.8},
        "vision": {"id": 0.7, "money": 0.6, "date": 0.3, "text": 0.63, "enum": 0.85, "code": 0.45},
        "ocr": {"id": 0.5, "money": 0.2, "date": 0.4, "text": 0.2, "enum": 0.5, "code": 0.2},
    }


class PostprocessConfig(BaseModel):
    """Всё, от чего зависят исправления, уверенность и статус документа."""

    model_config = ConfigDict(extra="forbid")

    autofix: bool = True
    # Тип документа и статус УПД — по заголовку текста-источника, а не по ответу модели
    # (docai.postprocess.doctype, D50). Только для путей text и ocr: у vision текста нет.
    doc_type_by_title: bool = True
    priors: dict[str, dict[str, float]] = Field(default_factory=_priors)

    # --- Отношения правдоподобия (LR) сигналов; уверенность = σ(logit(prior) + Σ ln LR) ---
    # Проверка не пройдена: поле названо в ошибке / только участвует в сверяемом равенстве.
    lr_error_named: float = 0.01
    lr_error_involved: float = 0.2
    lr_warning: float = 0.4
    # Проверка пройдена. Основа числа: как часто неверное значение прошло бы её случайно.
    lr_inn10: float = 10.0  # одна контрольная цифра
    lr_inn12: float = 100.0  # две контрольные цифры
    lr_kpp: float = 2.0  # только формат
    lr_bik_weak: float = 3.0  # формат и префикс «04»
    lr_key: float = 10.0  # ключ счёта (зависит от БИК)
    lr_date_vat: float = 2.0  # ставка в строках действовала на эту дату
    lr_vat_rate: float = 10.0  # НДС строки = база × ставка
    lr_untaxed_rate: float = 2.0
    lr_product: float = 20.0  # количество × цена = сумма
    lr_line_sum: float = 20.0  # без НДС + НДС = с НДС
    lr_column_sum: float = 3.0  # сумма по строкам = итог (вклад в каждую строку)
    lr_total_sum: float = 20.0  # итог = сумма по строкам (вклад в итог)
    lr_total_identity: float = 3.0  # итоги согласованы между собой
    # Произведение «пройденных» LR не больше этого. Ошибки LLM приходят согласованными блоками
    # (вся строка прочитана в другом масштабе, но арифметика сходится), поэтому проверки одни
    # не поднимают уверенность выше ≈0,94 при исходной 0,6.
    validation_boost_cap: float = 10.0

    # Значение дословно в тексте (текстовый слой / OCR).
    lr_found_strong: float = 8.0
    lr_found_weak: float = 1.5  # короткие значения («2», «шт») встречаются и случайно
    lr_missing_text_layer: float = 0.05
    lr_missing_ocr: float = 0.3
    lr_missing_weak: float = 0.5
    # Другой путь извлечения того же документа.
    lr_cross_agree: float = 6.0  # ошибки одной модели коррелируют, поэтому не 50
    lr_cross_disagree: float = 0.25

    # Поле исправлено автоматически: уверенность не выше этого значения.
    corrected_cap: float = 0.85
    prob_min: float = 0.01
    prob_max: float = 0.99

    # --- Статус документа ---
    review_threshold: float = 0.8  # поле ниже — «нужна проверка»
    reject_threshold: float = 0.5  # поле ниже считается «не прочитано»
    reject_low_share: float = 0.5  # доля таких полей, с которой документ отклоняется
