"""Извлечение полей из документов: XML ФНС, текстовый слой PDF, vision-модель, OCR."""

from docai.extract.config import ExtractConfig, load_config
from docai.extract.pipeline import RawExtraction, extract, extract_raw, finish
from docai.extract.result import ExtractionResult
from docai.extract.router import Mode, Route, RoutingError, choose_route, route_inputs

__all__ = [
    "ExtractConfig",
    "ExtractionResult",
    "Mode",
    "RawExtraction",
    "Route",
    "RoutingError",
    "choose_route",
    "extract",
    "extract_raw",
    "finish",
    "load_config",
    "route_inputs",
]
