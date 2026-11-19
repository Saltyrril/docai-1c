"""Тесты P16: обработка «Загрузка первичных документов (ИИ)» (onec/src/DocaiLoad).

1С в тестах не запускается. Здесь проверяется, что исходники собраны правильно и что
обработка и сервис говорят на одном языке: тело multipart, которое собирает модуль 1С,
принимается сервисом; ключи JSON, которые читает модуль, есть в моделях ответа; правки
в формате формы проверки принимаются /feedback. Проверка с настоящей 1С —
`scripts/onec_load_smoke.ps1` (onec/README.md).
"""

import importlib.util
import json
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docai.schemas.documents import parse_document
from docai.service.models import (
    CorrectionView,
    FeedbackOut,
    FieldView,
    HealthView,
    IssueView,
    JobCreated,
    JobView,
    LineView,
    OllamaHealth,
)
from docai.service.views import flat_values
from test_service import make_client, pdf_bytes, upload, wait_for  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "onec" / "src"
LOAD = SRC / "DocaiLoad"
OBJECT_MODULE = LOAD / "Ext" / "ObjectModule.bsl"
MAIN_FORM = LOAD / "Forms" / "Форма" / "Ext" / "Form.xml"
MAIN_MODULE = LOAD / "Forms" / "Форма" / "Ext" / "Form" / "Module.bsl"
REVIEW_FORM = LOAD / "Forms" / "ФормаПроверки" / "Ext" / "Form.xml"
REVIEW_MODULE = LOAD / "Forms" / "ФормаПроверки" / "Ext" / "Form" / "Module.bsl"
MODULES = [OBJECT_MODULE, MAIN_MODULE, REVIEW_MODULE]
NS = {"f": "http://v8.1c.ru/8.3/xcf/logform"}


def code(path: Path) -> str:
    """Текст модуля без комментариев."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return "\n".join(line.split("//", 1)[0] for line in lines)


# --- исходники --------------------------------------------------------------------------------


def test_processor_lists_both_forms() -> None:
    root = ET.parse(SRC / "DocaiLoad.xml").getroot()
    md = {"m": "http://v8.1c.ru/8.3/MDClasses"}
    assert root.get("version") == "2.1"
    assert root.findtext(".//m:Properties/m:Name", namespaces=md) == "DocaiLoad"
    forms = [e.text for e in root.findall(".//m:ChildObjects/m:Form", md)]
    assert forms == ["Форма", "ФормаПроверки"]
    default = root.findtext(".//m:DefaultForm", namespaces=md)
    assert default == "ExternalDataProcessor.DocaiLoad.Form.Форма"
    for name in forms:
        assert (LOAD / "Forms" / f"{name}.xml").is_file()
        assert (LOAD / "Forms" / name / "Ext" / "Form.xml").is_file()


@pytest.mark.parametrize(
    ("form", "module"), [(MAIN_FORM, MAIN_MODULE), (REVIEW_FORM, REVIEW_MODULE)]
)
def test_form_handlers_exist_in_module(form: Path, module: Path) -> None:
    root = ET.parse(form).getroot()
    handlers = {e.text for e in root.iter(f"{{{NS['f']}}}Event")}
    handlers |= {e.text for e in root.findall(".//f:Commands/f:Command/f:Action", NS)}
    text = module.read_text(encoding="utf-8-sig")
    for name in handlers:
        assert re.search(rf"^(Процедура|Функция) {name}\(", text, re.MULTILINE), name


@pytest.mark.parametrize("form", [MAIN_FORM, REVIEW_FORM])
def test_form_ids_and_data_paths_are_consistent(form: Path) -> None:
    root = ET.parse(form).getroot()
    # Идентификаторы элементов уникальны, иначе Конфигуратор не загрузит форму
    ids = [
        e.get("id")
        for e in root.iter()
        if e.get("name")
        and e.get("id")
        and e.tag.endswith(
            (
                "InputField",
                "Table",
                "UsualGroup",
                "Button",
                "PictureField",
                "LabelDecoration",
                "ContextMenu",
                "ExtendedTooltip",
                "AutoCommandBar",
                "Pages",
                "Page",
                "CheckBoxField",
            )
        )
    ]
    assert len(ids) == len(set(ids))
    # Каждый путь к данным ведёт к реквизиту формы или колонке табличного реквизита
    attrs: dict[str, set[str]] = {}
    for attr in root.findall("f:Attributes/f:Attribute", NS):
        attrs[attr.get("name")] = {c.get("name") for c in attr.findall("f:Columns/f:Column", NS)}
    for path in root.iter(f"{{{NS['f']}}}DataPath"):
        head, _, column = path.text.partition(".")
        assert head in attrs, path.text
        if column:
            assert column in attrs[head], path.text


def test_review_form_items_used_in_code_exist() -> None:
    root = ET.parse(REVIEW_FORM).getroot()
    names = {e.get("name") for e in root.iter() if e.get("name")}
    text = code(REVIEW_MODULE)
    for item in set(re.findall(r"(?<![.\w])Элементы\.([A-Za-zА-Яа-яЁё0-9_]+)", text)):
        assert item in names, item
    # Условное оформление ссылается на колонки таблиц формы по имени элемента
    for item in set(re.findall(r'ДобавитьОформление\("([^"]+)"', text)):
        assert item in names, item
    for column in re.findall(r'ДобавитьКолонку\(Колонки, "([^"]+)"', text):
        assert f"Строки{column}" in names


def test_context_directives_are_valid() -> None:
    # «&НаКлиентеНаСервере» без «БезКонтекста» платформа не считает серверной: процедура
    # с такой директивой не найдена при вызове с сервера (найдено при P16).
    allowed = {
        "&НаКлиенте",
        "&НаСервере",
        "&НаСервереБезКонтекста",
        "&НаКлиентеНаСервереБезКонтекста",
    }
    for path in [MAIN_MODULE, REVIEW_MODULE]:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("&"):
                assert line.strip() in allowed, (path.name, line)


def test_object_module_runs_only_on_server() -> None:
    text = OBJECT_MODULE.read_text(encoding="utf-8-sig")
    assert "#Если Сервер Или ТолстыйКлиентОбычноеПриложение Или ВнешнееСоединение Тогда" in text
    exported = re.findall(r"^Функция (\w+)\(.*\) Экспорт", text, re.MULTILINE)
    for name in exported:
        # Формы вызывают экспортные функции модуля объекта
        used = any(f".{name}(" in m.read_text(encoding="utf-8-sig") for m in MODULES)
        assert used, name


def test_no_variable_shadows_form_property() -> None:
    # «РазмерКартинки = ...» в модуле формы упало: имя совпало со свойством формы (P16).
    # Здесь — известные свойства формы и глобального контекста, которые легко задеть.
    reserved = ["РазмерКартинки", "Заголовок", "Документы", "Справочники", "Строка", "Файл"]
    for path in [MAIN_MODULE, REVIEW_MODULE]:
        text = code(path)
        for name in reserved:
            if name == "Заголовок":
                continue  # заголовок формы задаётся намеренно
            assert not re.search(rf"^\s*{name}\s*=", text, re.MULTILINE), (path.name, name)


# --- контракт с сервисом -------------------------------------------------------------------------


def test_service_paths_used_by_module() -> None:
    text = code(OBJECT_MODULE)
    for path in ['"/v1/extract"', '"/v1/jobs/"', '"/preview/"', '"/feedback"', '"/v1/health"']:
        assert path in text, path
    assert "multipart/form-data; boundary=" in text
    assert 'name=""file""; filename=""' in text


def test_json_keys_read_by_modules_exist_in_service_models() -> None:
    models = {
        "Данные": JobView,
        "Поле": FieldView,
        "Проблема": IssueView,
        "Исправление": CorrectionView,
        "Принято": JobCreated,
        "Итог": FeedbackOut,
        "Сервис": HealthView,
        "Ollama": OllamaHealth,
    }
    for path in MODULES:
        text = code(path)
        for var, model in models.items():
            for key in re.findall(rf"\b{var}\.([a-z_]+)\b", text):
                assert key in model.model_fields, (path.name, var, key)


def test_line_columns_match_service_lines() -> None:
    keys = re.findall(r'ДобавитьКолонку\(Колонки, "[^"]+", "([a-z_]+)"', code(REVIEW_MODULE))
    assert keys == [
        name
        for name in LineView.model_fields
        if name not in ("n", "min_confidence", "needs_review")
    ]


def test_every_header_field_has_russian_name() -> None:
    text = code(OBJECT_MODULE)
    named = set(re.findall(r'Названия\.Вставить\("([a-z_]+)"', text))
    sample = sorted((ROOT / "data" / "synth" / "json").glob("*.json"))[:200]
    if not sample:
        pytest.skip("нет синтетических документов data/synth/json")
    for path in sample:
        document = parse_document(json.loads(path.read_text(encoding="utf-8")))
        for name in flat_values(document):
            if name.startswith("lines["):
                continue
            for part in name.split("."):
                assert part in named, name


def multipart_like_1c(content: bytes, filename: str) -> tuple[bytes, str]:
    """Тело запроса так, как его собирает ОтправитьФайл в ObjectModule.bsl."""
    boundary = "docai" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + content + tail, f"multipart/form-data; boundary={boundary}"


def test_service_accepts_multipart_built_like_1c(make_client: Callable[..., TestClient]) -> None:  # noqa: F811
    client = make_client()
    body, content_type = multipart_like_1c(pdf_bytes(), "Счёт №5 от 14.11.pdf")
    response = client.post("/v1/extract", content=body, headers={"Content-Type": content_type})
    assert response.status_code == 202, response.text
    job = wait_for(client, response.json()["id"], "done")
    # Кириллица в имени (1С пишет заголовок в UTF-8) доходит до сервиса без искажений
    assert job["filename"] == "Счёт №5 от 14.11.pdf"


def test_feedback_in_review_form_format_is_accepted(make_client: Callable[..., TestClient]) -> None:  # noqa: F811
    client = make_client()
    job = wait_for(client, upload(client, pdf_bytes())["id"], "done")
    values = {f["name"]: f["value"] for f in job["fields"]}
    assert values, "заглушка должна вернуть поля"
    # Форма присылает все значения; человек поправил одно и стёр другое (null)
    name, value = next((n, v) for n, v in values.items() if v)
    values[name] = value + "1"
    empty = next(n for n in values if n != name)
    values[empty] = None
    values["lines[99].name"] = None  # удалённая человеком строка
    body = {"fields": values, "author": "Admin"}
    response = client.post(
        f"/v1/jobs/{job['id']}/feedback",
        content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total"] == len(values)
    assert result["changed"] >= 1


def test_form_sources_match_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # XML форм порождает scripts/onec_forms.py (D55): правка XML руками без генератора
    # потерялась бы при следующем запуске.
    spec = importlib.util.spec_from_file_location("onec_forms", ROOT / "scripts" / "onec_forms.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["onec_forms.py", str(tmp_path)])
    module.main()
    generated = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.xml"))
    stored = sorted(p.relative_to(SRC) for p in [SRC / "DocaiLoad.xml", *LOAD.rglob("*.xml")])
    assert generated == stored
    for rel in generated:
        assert (tmp_path / rel).read_bytes() == (SRC / rel).read_bytes(), rel


# --- сопоставление со справочниками (P16, часть 2) ------------------------------------------------

# Объекты учебной Бухгалтерии 3.0.111.25, которые использует обработка. Сверено с выгрузкой
# конфигурации (DumpConfigToFiles, 2026-11-18, D55/D58): опечатка в имени объекта видна только
# при запуске в 1С, поэтому список зафиксирован здесь.
BP_OBJECTS = {
    ("Справочник", "Организации"),
    ("Справочник", "Контрагенты"),
    ("Справочник", "ДоговорыКонтрагентов"),
    ("Справочник", "Номенклатура"),
    ("Справочник", "КлассификаторЕдиницИзмерения"),
    ("Справочник", "СтраныМира"),
    ("Перечисление", "ВидыДоговоровКонтрагентов"),
    ("Перечисление", "ЮридическоеФизическоеЛицо"),
}
BP_VALUES = {
    "ВидыДоговоровКонтрагентов": {"СПоставщиком"},
    "ЮридическоеФизическоеЛицо": {"ЮридическоеЛицо", "ФизическоеЛицо"},
    "СтраныМира": {"Россия"},
}
BP_FORMS = {"ФормаОбъекта", "ФормаВыбора", "ФормаЭлемента", "ПустаяСсылка"}


def test_configuration_objects_are_known() -> None:
    kinds = {"Справочник": "Справочник", "Справочники": "Справочник", "Справочника": "Справочник"}
    kinds |= {"Перечисление": "Перечисление", "Перечисления": "Перечисление"}
    for path in [OBJECT_MODULE, REVIEW_MODULE]:
        text = code(path)
        for kind, name, member in re.findall(
            r"\b(Справочник[иа]?|Перечислени[яе])\.(\w+)(?:\.(\w+))?", text
        ):
            assert (kinds[kind], name) in BP_OBJECTS, (path.name, kind, name)
            if member and member not in BP_FORMS:
                assert member in BP_VALUES.get(name, set()), (path.name, name, member)


def test_form_types_are_known_catalogs() -> None:
    for form in [MAIN_FORM, REVIEW_FORM]:
        text = form.read_text(encoding="utf-8-sig")
        for name in re.findall(r"cfg:CatalogRef\.(\w+)", text):
            assert ("Справочник", name) in BP_OBJECTS, (form.name, name)


def loops_with_queries(text: str) -> list[int]:
    """Номера строк, где запрос выполняется внутри цикла (стандарт 1С: без запросов в цикле)."""
    depth = 0
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.split("//", 1)[0].strip()
        if re.search(r"\bЦикл\s*$", stripped):
            depth += 1
        if depth and re.search(r"\.(Выполнить|ВыполнитьПакет)\(|Новый Запрос\b", stripped):
            found.append(number)
        if stripped.startswith("КонецЦикла"):
            depth -= 1
    return found


def test_no_queries_in_loops() -> None:
    for path in MODULES:
        assert loops_with_queries(code(path)) == [], path.name
    # сама проверка видит запрос в цикле
    sample = "Для Каждого А Из Б Цикл\n\tРезультат = Запрос.Выполнить();\nКонецЦикла;"
    assert loops_with_queries(sample) == [2]


def test_matching_reads_catalogs_in_one_batch() -> None:
    text = code(OBJECT_MODULE)
    assert "Запрос.ВыполнитьПакет()" in text
    # Пакет: организации, контрагенты, единицы, номенклатура — ровно четыре запроса
    batch = text.split("Функция ТекстЗапросаСопоставления", 1)[1].split("КонецФункции", 1)[0]
    assert batch.count("ВЫБРАТЬ") == 4
    for table in ("Организации", "Контрагенты", "КлассификаторЕдиницИзмерения", "Номенклатура"):
        assert f"Справочник.{table}" in batch


def test_review_form_waits_for_standard_write_notifications() -> None:
    # Имена оповещений — из форм элементов Бухгалтерии (ПослеЗаписи, ПриЗакрытии), D58
    text = code(REVIEW_MODULE)
    assert '"Запись_Контрагенты"' in text
    assert '"Запись_ДоговорыКонтрагентов"' in text
    assert "ДополнительныеПараметры" in text or 'Параметры.Свойство("Номенклатура")' in text
