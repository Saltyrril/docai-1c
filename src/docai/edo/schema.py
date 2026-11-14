"""XSD формата ФНС и проверка XML по нему (lxml).

Схема лежит в пакете (`xsd/`) в том виде, в каком её опубликовала ФНС, — см. D30
в docs/DECISIONS.md. Разбор XML устроен так, чтобы чужой файл не мог ничего
прочитать с диска и из сети: внешние сущности и DTD отключены.
"""

from functools import cache
from pathlib import Path

from lxml import etree

from docai.edo.errors import EdoError, EdoValidationError

XSD_NAME = "ON_NSCHFDOPPR_1_997_01_05_03_05.xsd"
XSD_PATH = Path(__file__).parent / "xsd" / XSD_NAME

FORMAT_VERSION = "5.03"
KND = "1115131"  # код формы по КНД: УПД / счёт-фактура (информация продавца)
FILE_PREFIX = "ON_NSCHFDOPPR"
ENCODING = "windows-1251"


def make_parser() -> etree.XMLParser:
    """Безопасный парсер: без DTD, внешних сущностей и обращений в сеть."""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        huge_tree=False,
        remove_comments=True,
        remove_pis=True,
    )


@cache
def load_schema() -> etree.XMLSchema:
    """XSD ФНС, загруженный один раз."""
    return etree.XMLSchema(etree.parse(str(XSD_PATH), make_parser()))


def parse_xml(data: bytes) -> etree._Element:
    """Разобрать XML (кодировку берёт из заголовка файла)."""
    try:
        return etree.fromstring(data, make_parser())
    except etree.XMLSyntaxError as exc:
        raise EdoError(f"это не XML: {exc}") from exc


def xsd_errors(root: etree._Element) -> list[str]:
    """Нарушения XSD списком строк (пусто — документ корректен)."""
    schema = load_schema()
    if schema.validate(root):
        return []
    return [f"строка {e.line}: {e.message}" for e in schema.error_log]


def validate_xml(data: bytes | etree._Element) -> None:
    """Проверить XML по XSD; при нарушениях — `EdoValidationError`."""
    root = parse_xml(data) if isinstance(data, bytes) else data
    errors = xsd_errors(root)
    if errors:
        raise EdoValidationError(errors)
