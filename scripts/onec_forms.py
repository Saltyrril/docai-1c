r"""Генератор XML-исходников обработки DocaiLoad (P16, формат выгрузки 2.1).

Пишет описание обработки (onec/src/DocaiLoad.xml) и XML форм; модули .bsl пишутся вручную,
генератор их не трогает. Формы описаны здесь, а не правятся в Конфигураторе: учебная
платформа 8.3.18 выгружает формат 2.11 и переписала бы все файлы (D55). uuid
детерминированы, поэтому повторный запуск даёт те же файлы байт в байт.

    uv run python scripts/onec_forms.py          # перезаписать XML в onec/src
    .\scripts\onec_build.ps1 -Name DocaiLoad    # собрать build/DocaiLoad.epf
"""

import sys
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

XMLNS = {
    "app": "http://v8.1c.ru/8.2/managed-application/core",
    "cfg": "http://v8.1c.ru/8.1/data/enterprise/current-config",
    "cmi": "http://v8.1c.ru/8.2/managed-application/cmi",
    "dcscor": "http://v8.1c.ru/8.1/data-composition-system/core",
    "dcsset": "http://v8.1c.ru/8.1/data-composition-system/settings",
    "ent": "http://v8.1c.ru/8.1/data/enterprise",
    "lf": "http://v8.1c.ru/8.2/managed-application/logform",
    "style": "http://v8.1c.ru/8.1/data/ui/style",
    "sys": "http://v8.1c.ru/8.1/data/ui/fonts/system",
    "v8": "http://v8.1c.ru/8.1/data/core",
    "v8ui": "http://v8.1c.ru/8.1/data/ui",
    "web": "http://v8.1c.ru/8.1/data/ui/colors/web",
    "win": "http://v8.1c.ru/8.1/data/ui/colors/windows",
    "xen": "http://v8.1c.ru/8.3/xcf/enums",
    "xpr": "http://v8.1c.ru/8.3/xcf/predef",
    "xr": "http://v8.1c.ru/8.3/xcf/readable",
    "xs": "http://www.w3.org/2001/XMLSchema",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def namespaces(default: str, prefixes: str) -> str:
    """Атрибуты xmlns в том порядке, в каком их пишет Конфигуратор."""
    attrs = [f'xmlns="{default}"']
    attrs += [f'xmlns:{p}="{XMLNS[p]}"' for p in prefixes.split()]
    return " ".join(attrs) + ' version="2.1"'


NS_MD = namespaces(
    "http://v8.1c.ru/8.3/MDClasses",
    "app cfg cmi ent lf style sys v8 v8ui web win xen xpr xr xs xsi",
)
NS_FORM = namespaces(
    "http://v8.1c.ru/8.3/xcf/logform",
    "app cfg dcscor dcsset ent lf style sys v8 v8ui web win xr xs xsi",
)

# Постоянные uuid: повторная генерация не должна менять исходники.
NAMESPACE = uuid.UUID("5d0c7a3e-2b1f-4f0e-9c55-6a1d3e0b9f10")


def uid(key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, key))


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\r\n".join(lines)
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))


def ml(tag: str, text: str, ind: str, formatted: bool | None = None) -> list[str]:
    attr = "" if formatted is None else f' formatted="{str(formatted).lower()}"'
    return [
        f"{ind}<{tag}{attr}>",
        f"{ind}\t<v8:item>",
        f"{ind}\t\t<v8:lang>ru</v8:lang>",
        f"{ind}\t\t<v8:content>{escape(text)}</v8:content>",
        f"{ind}\t</v8:item>",
        f"{ind}</{tag}>",
    ]


# --- типы ---------------------------------------------------------------------------------


def t_str(length: int = 0) -> list[str]:
    return [
        "<v8:Type>xs:string</v8:Type>",
        "<v8:StringQualifiers>",
        f"\t<v8:Length>{length}</v8:Length>",
        "\t<v8:AllowedLength>Variable</v8:AllowedLength>",
        "</v8:StringQualifiers>",
    ]


def t_num(digits: int, frac: int, sign: str = "Any") -> list[str]:
    return [
        "<v8:Type>xs:decimal</v8:Type>",
        "<v8:NumberQualifiers>",
        f"\t<v8:Digits>{digits}</v8:Digits>",
        f"\t<v8:FractionDigits>{frac}</v8:FractionDigits>",
        f"\t<v8:AllowedSign>{sign}</v8:AllowedSign>",
        "</v8:NumberQualifiers>",
    ]


def t_date() -> list[str]:
    return [
        "<v8:Type>xs:dateTime</v8:Type>",
        "<v8:DateQualifiers>",
        "\t<v8:DateFractions>Date</v8:DateFractions>",
        "</v8:DateQualifiers>",
    ]


def t_bool() -> list[str]:
    return ["<v8:Type>xs:boolean</v8:Type>"]


def t_ref(name: str) -> list[str]:
    """Ссылка на объект конфигурации, например «CatalogRef.Контрагенты» (D58)."""
    return [f"<v8:Type>cfg:{name}</v8:Type>"]


def t_refs(*names: str) -> list[str]:
    """Составной тип из нескольких ссылок (документ поступления или счёт поставщика)."""
    return [f"<v8:Type>cfg:{name}</v8:Type>" for name in names]


def t_vt() -> list[str]:
    return ["<v8:Type>v8:ValueTable</v8:Type>"]


def type_block(t: list[str], ind: str) -> list[str]:
    return [f"{ind}<Type>", *[f"{ind}\t{x}" for x in t], f"{ind}</Type>"]


# --- форма ----------------------------------------------------------------------------------


class Form:
    def __init__(self, processor: str) -> None:
        self.processor = processor
        self.next_id = 0
        self.items: list[str] = []
        self.attrs: list[str] = []
        self.commands: list[str] = []
        self.params: list[str] = []
        self.bar: list[str] = []
        self.events: list[tuple[str, str]] = []
        self.props: list[str] = []
        self.attr_id = 1
        self.cmd_id = 0

    def nid(self) -> int:
        self.next_id += 1
        return self.next_id

    # элементы возвращают список строк с отступом ind

    def input(
        self,
        name: str,
        path: str,
        ind: str,
        *,
        ro: bool = False,
        title: str | None = None,
        title_loc: str | None = None,
        width: int | None = None,
        height: int | None = None,
        hstretch: bool | None = None,
        vstretch: bool | None = None,
        multiline: bool = False,
        wrap: bool | None = None,
        events: dict[str, str] | None = None,
        fmt: str | None = None,
        links: dict[str, str] | None = None,
        saved_hint: str | None = None,
    ) -> list[str]:
        i = self.nid()
        out = [f'{ind}<InputField name="{name}" id="{i}">', f"{ind}\t<DataPath>{path}</DataPath>"]
        if ro:
            out.append(f"{ind}\t<ReadOnly>true</ReadOnly>")
        if title is not None:
            out += ml("Title", title, ind + "\t")
        if title_loc:
            out.append(f"{ind}\t<TitleLocation>{title_loc}</TitleLocation>")
        if width is not None:
            out.append(f"{ind}\t<Width>{width}</Width>")
        if height is not None:
            out.append(f"{ind}\t<Height>{height}</Height>")
        if hstretch is not None:
            out.append(f"{ind}\t<HorizontalStretch>{str(hstretch).lower()}</HorizontalStretch>")
        if vstretch is not None:
            out.append(f"{ind}\t<VerticalStretch>{str(vstretch).lower()}</VerticalStretch>")
        if multiline:
            out.append(f"{ind}\t<MultiLine>true</MultiLine>")
        if wrap is not None:
            out.append(f"{ind}\t<Wrap>{str(wrap).lower()}</Wrap>")
        if fmt is not None:
            out += ml("Format", fmt, ind + "\t")
        if saved_hint is not None:
            out += ml("InputHint", saved_hint, ind + "\t")
        if links:
            out.append(f"{ind}\t<ChoiceParameterLinks>")
            for link_name, data_path in links.items():
                out += [
                    f"{ind}\t\t<xr:Link>",
                    f"{ind}\t\t\t<xr:Name>{link_name}</xr:Name>",
                    f'{ind}\t\t\t<xr:DataPath xsi:type="xs:string">{data_path}</xr:DataPath>',
                    f"{ind}\t\t\t<xr:ValueChange>Clear</xr:ValueChange>",
                    f"{ind}\t\t</xr:Link>",
                ]
            out.append(f"{ind}\t</ChoiceParameterLinks>")
        out.append(f'{ind}\t<ContextMenu name="{name}КонтекстноеМеню" id="{self.nid()}"/>')
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        if events:
            out.append(f"{ind}\t<Events>")
            for ev, handler in events.items():
                out.append(f'{ind}\t\t<Event name="{ev}">{handler}</Event>')
            out.append(f"{ind}\t</Events>")
        out.append(f"{ind}</InputField>")
        return out

    def checkbox(self, name: str, path: str, ind: str) -> list[str]:
        out = [f'{ind}<CheckBoxField name="{name}" id="{self.nid()}">']
        out.append(f"{ind}\t<DataPath>{path}</DataPath>")
        out.append(f"{ind}\t<EditMode>EnterOnInput</EditMode>")
        out.append(f"{ind}\t<CheckBoxType>Auto</CheckBoxType>")
        out.append(f'{ind}\t<ContextMenu name="{name}КонтекстноеМеню" id="{self.nid()}"/>')
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        out.append(f"{ind}</CheckBoxField>")
        return out

    def picture(self, name: str, path: str, ind: str, empty_text: str) -> list[str]:
        i = self.nid()
        out = [f'{ind}<PictureField name="{name}" id="{i}">', f"{ind}\t<DataPath>{path}</DataPath>"]
        out.append(f"{ind}\t<TitleLocation>None</TitleLocation>")
        out.append(f"{ind}\t<Width>60</Width>")
        out.append(f"{ind}\t<Height>25</Height>")
        out.append(f"{ind}\t<HorizontalStretch>true</HorizontalStretch>")
        out.append(f"{ind}\t<VerticalStretch>true</VerticalStretch>")
        out += ml("NonselectedPictureText", empty_text, ind + "\t")
        out.append(f"{ind}\t<PictureSize>Proportionally</PictureSize>")
        out.append(f'{ind}\t<ContextMenu name="{name}КонтекстноеМеню" id="{self.nid()}"/>')
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        out.append(f"{ind}</PictureField>")
        return out

    def label(self, name: str, text: str, ind: str) -> list[str]:
        i = self.nid()
        out = [f'{ind}<LabelDecoration name="{name}" id="{i}">']
        out.append(f"{ind}\t<HorizontalStretch>true</HorizontalStretch>")
        out += ml("Title", text, ind + "\t", formatted=False)
        out.append(f'{ind}\t<ContextMenu name="{name}КонтекстноеМеню" id="{self.nid()}"/>')
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        out.append(f"{ind}</LabelDecoration>")
        return out

    def pages(self, name: str, ind: str, pages) -> list[str]:
        out = [f'{ind}<Pages name="{name}" id="{self.nid()}">']
        out.append(
            f'{ind}	<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>'
        )
        out.append(f"{ind}	<ChildItems>")
        for pname, ptitle, children in pages:
            out.append(f'{ind}		<Page name="{pname}" id="{self.nid()}">')
            out += ml("Title", ptitle, ind + "			")
            tip = f'<ExtendedTooltip name="{pname}РасширеннаяПодсказка" id="{self.nid()}"/>'
            out.append(f"{ind}			{tip}")
            out.append(f"{ind}			<ChildItems>")
            for child in children:
                out += child(ind + "				")
            out.append(f"{ind}			</ChildItems>")
            out.append(f"{ind}		</Page>")
        out.append(f"{ind}	</ChildItems>")
        out.append(f"{ind}</Pages>")
        return out

    def button(
        self, name: str, command: str, ind: str, *, bar: bool, default: bool = False
    ) -> list[str]:
        i = self.nid()
        out = [f'{ind}<Button name="{name}" id="{i}">']
        out.append(f"{ind}\t<Type>{'CommandBarButton' if bar else 'UsualButton'}</Type>")
        if default:
            out.append(f"{ind}\t<DefaultButton>true</DefaultButton>")
        out.append(f"{ind}\t<CommandName>Form.Command.{command}</CommandName>")
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        out.append(f"{ind}</Button>")
        return out

    def group(
        self,
        name: str,
        title: str,
        ind: str,
        children,
        *,
        kind: str = "Vertical",
        collapsible: bool = False,
        collapsed: bool = False,
        show_title: bool = False,
        representation: str = "None",
        hstretch: bool | None = None,
        width: int | None = None,
    ) -> list[str]:
        i = self.nid()
        out = [f'{ind}<UsualGroup name="{name}" id="{i}">']
        out += ml("Title", title, ind + "\t")
        if width is not None:
            out.append(f"{ind}\t<Width>{width}</Width>")
        if hstretch is not None:
            out.append(f"{ind}\t<HorizontalStretch>{str(hstretch).lower()}</HorizontalStretch>")
        out.append(f"{ind}\t<Group>{kind}</Group>")
        if collapsible:
            out.append(f"{ind}\t<Behavior>Collapsible</Behavior>")
            if collapsed:
                out.append(f"{ind}\t<Collapsed>true</Collapsed>")
        out.append(f"{ind}\t<Representation>{representation}</Representation>")
        out.append(f"{ind}\t<ShowTitle>{str(show_title).lower()}</ShowTitle>")
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        out.append(f"{ind}\t<ChildItems>")
        for child in children:
            out += child(ind + "\t\t")
        out.append(f"{ind}\t</ChildItems>")
        out.append(f"{ind}</UsualGroup>")
        return out

    def table(
        self,
        name: str,
        path: str,
        title: str,
        ind: str,
        columns,
        *,
        ro: bool,
        change_rows: bool,
        height: int | None = None,
        events: dict[str, str] | None = None,
        show_title: bool = False,
        bar_buttons: list[tuple[str, str]] | None = None,
    ) -> list[str]:
        i = self.nid()
        out = [
            f'{ind}<Table name="{name}" id="{i}">',
            f"{ind}\t<Representation>List</Representation>",
        ]
        if not show_title:
            out.append(f"{ind}\t<TitleLocation>None</TitleLocation>")
        if ro:
            out.append(f"{ind}\t<ReadOnly>true</ReadOnly>")
        if not change_rows:
            out.append(f"{ind}\t<ChangeRowSet>false</ChangeRowSet>")
            out.append(f"{ind}\t<ChangeRowOrder>false</ChangeRowOrder>")
        if height is not None:
            out.append(f"{ind}\t<Height>{height}</Height>")
        out.append(f"{ind}\t<SelectionMode>SingleRow</SelectionMode>")
        out.append(f"{ind}\t<HorizontalStretch>true</HorizontalStretch>")
        out.append(f"{ind}\t<VerticalStretch>true</VerticalStretch>")
        out.append(f"{ind}\t<DataPath>{path}</DataPath>")
        out += ml("Title", title, ind + "\t")
        out.append(f'{ind}\t<ContextMenu name="{name}КонтекстноеМеню" id="{self.nid()}"/>')
        out.append(f'{ind}\t<AutoCommandBar name="{name}КоманднаяПанель" id="{self.nid()}">')
        if not change_rows:
            out.append(f"{ind}\t\t<Autofill>false</Autofill>")
        if bar_buttons:
            out.append(f"{ind}\t\t<ChildItems>")
            for button_name, command in bar_buttons:
                out += self.button(button_name, command, ind + "\t\t\t", bar=True)
            out.append(f"{ind}\t\t</ChildItems>")
        out.append(f"{ind}\t</AutoCommandBar>")
        out.append(f'{ind}\t<ExtendedTooltip name="{name}РасширеннаяПодсказка" id="{self.nid()}"/>')
        if events:
            out.append(f"{ind}\t<Events>")
            for ev, handler in events.items():
                out.append(f'{ind}\t\t<Event name="{ev}">{handler}</Event>')
            out.append(f"{ind}\t</Events>")
        out.append(f"{ind}\t<ChildItems>")
        for col in columns:
            out += col(ind + "\t\t")
        out.append(f"{ind}\t</ChildItems>")
        out.append(f"{ind}</Table>")
        return out

    # реквизиты, команды, параметры

    def attr(
        self,
        name: str,
        title: str,
        t: list[str],
        *,
        main: bool = False,
        saved: bool = False,
        columns: list[tuple[str, str, list[str]]] | None = None,
    ) -> None:
        ind = "\t\t"
        self.attr_id += 1
        out = [f'{ind}<Attribute name="{name}" id="{self.attr_id}">']
        out += ml("Title", title, ind + "\t")
        out += type_block(t, ind + "\t")
        if saved:
            out.append(f"{ind}\t<SavedData>true</SavedData>")
        if columns:
            out.append(f"{ind}\t<Columns>")
            for cid, (cname, ctitle, ctype) in enumerate(columns, start=1):
                out.append(f'{ind}\t\t<Column name="{cname}" id="{cid}">')
                out += ml("Title", ctitle, ind + "\t\t\t")
                out += type_block(ctype, ind + "\t\t\t")
                out.append(f"{ind}\t\t</Column>")
            out.append(f"{ind}\t</Columns>")
        out.append(f"{ind}</Attribute>")
        self.attrs += out

    def command(self, name: str, title: str, tooltip: str | None = None) -> None:
        ind = "\t\t"
        self.cmd_id += 1
        out = [f'{ind}<Command name="{name}" id="{self.cmd_id}">']
        out += ml("Title", title, ind + "\t")
        if tooltip:
            out += ml("ToolTip", tooltip, ind + "\t")
        out.append(f"{ind}\t<Action>{name}</Action>")
        out.append(f"{ind}</Command>")
        self.commands += out

    def param(self, name: str, t: list[str]) -> None:
        ind = "\t\t"
        self.params += [
            f'{ind}<Parameter name="{name}">',
            *type_block(t, ind + "\t"),
            f"{ind}</Parameter>",
        ]

    def render(self, title: str) -> list[str]:
        out = ['<?xml version="1.0" encoding="UTF-8"?>', f"<Form {NS_FORM}>"]
        out += ml("Title", title, "\t")
        out.append("\t<AutoTitle>false</AutoTitle>")
        out += self.props
        if self.bar:
            out.append('\t<AutoCommandBar name="ФормаКоманднаяПанель" id="-1">')
            out.append("\t\t<ChildItems>")
            out += self.bar
            out.append("\t\t</ChildItems>")
            out.append("\t</AutoCommandBar>")
        else:
            out.append('\t<AutoCommandBar name="ФормаКоманднаяПанель" id="-1"/>')
        out.append("\t<Events>")
        for ev, handler in self.events:
            out.append(f'\t\t<Event name="{ev}">{handler}</Event>')
        out.append("\t</Events>")
        out.append("\t<ChildItems>")
        out += self.items
        out.append("\t</ChildItems>")
        out.append("\t<Attributes>")
        out += [
            '\t\t<Attribute name="Объект" id="1">',
            "\t\t\t<Type>",
            f"\t\t\t\t<v8:Type>cfg:ExternalDataProcessorObject.{self.processor}</v8:Type>",
            "\t\t\t</Type>",
            "\t\t\t<MainAttribute>true</MainAttribute>",
            "\t\t</Attribute>",
        ]
        out += self.attrs
        out.append("\t</Attributes>")
        if self.commands:
            out.append("\t<Commands>")
            out += self.commands
            out.append("\t</Commands>")
        if self.params:
            out.append("\t<Parameters>")
            out += self.params
            out.append("\t</Parameters>")
        out.append("</Form>")
        return out


def form_meta(name: str, synonym: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<MetaDataObject {NS_MD}>",
        f'\t<Form uuid="{uid("form:" + name)}">',
        "\t\t<Properties>",
        f"\t\t\t<Name>{name}</Name>",
        *ml("Synonym", synonym, "\t\t\t"),
        "\t\t\t<Comment/>",
        "\t\t\t<FormType>Managed</FormType>",
        "\t\t\t<IncludeHelpInContents>false</IncludeHelpInContents>",
        "\t\t\t<UsePurposes>",
        '\t\t\t\t<v8:Value xsi:type="app:ApplicationUsePurpose">PersonalComputer</v8:Value>',
        "\t\t\t</UsePurposes>",
        "\t\t\t<ExtendedPresentation/>",
        "\t\t</Properties>",
        "\t</Form>",
        "</MetaDataObject>",
    ]


def processor_meta(name: str, synonym: str, forms: list[str], default: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<MetaDataObject {NS_MD}>",
        f'\t<ExternalDataProcessor uuid="{uid("processor:" + name)}">',
        "\t\t<InternalInfo>",
        "\t\t\t<xr:ContainedObject>",
        "\t\t\t\t<xr:ClassId>c3831ec8-d8d5-4f93-8a22-f9bfae07327f</xr:ClassId>",
        f"\t\t\t\t<xr:ObjectId>{uid('object:' + name)}</xr:ObjectId>",
        "\t\t\t</xr:ContainedObject>",
        f'\t\t\t<xr:GeneratedType name="ExternalDataProcessorObject.{name}" category="Object">',
        f"\t\t\t\t<xr:TypeId>{uid('type:' + name)}</xr:TypeId>",
        f"\t\t\t\t<xr:ValueId>{uid('value:' + name)}</xr:ValueId>",
        "\t\t\t</xr:GeneratedType>",
        "\t\t</InternalInfo>",
        "\t\t<Properties>",
        f"\t\t\t<Name>{name}</Name>",
        *ml("Synonym", synonym, "\t\t\t"),
        "\t\t\t<Comment/>",
        f"\t\t\t<DefaultForm>ExternalDataProcessor.{name}.Form.{default}</DefaultForm>",
        "\t\t\t<AuxiliaryForm/>",
        "\t\t</Properties>",
        "\t\t<ChildObjects>",
        *[f"\t\t\t<Form>{f}</Form>" for f in forms],
        "\t\t</ChildObjects>",
        "\t</ExternalDataProcessor>",
        "</MetaDataObject>",
    ]


# --- главная форма: список загрузок -------------------------------------------------------------


def main_form() -> Form:
    f = Form("DocaiLoad")
    f.events = [("OnCreateAtServer", "ПриСозданииНаСервере"), ("OnOpen", "ПриОткрытии")]
    f.bar += f.button("ФормаЗагрузитьФайлы", "ЗагрузитьФайлы", "\t\t\t", bar=True, default=True)
    f.bar += f.button("ФормаПроверить", "Проверить", "\t\t\t", bar=True)
    f.bar += f.button("ФормаОбновить", "Обновить", "\t\t\t", bar=True)
    f.bar += f.button("ФормаЖурнал", "Журнал", "\t\t\t", bar=True)

    def col(name: str, path: str, **kw):
        return lambda ind: f.input(name, path, ind, **kw)

    ind = "\t\t"
    f.items += f.input(
        "СостояниеСервиса",
        "СостояниеСервиса",
        ind,
        ro=True,
        title="Сервис",
        width=90,
        hstretch=True,
    )
    f.items += f.table(
        "Загрузки",
        "Загрузки",
        "Загруженные файлы",
        ind,
        [
            col("ЗагрузкиИмяФайла", "Загрузки.ИмяФайла", width=25),
            col("ЗагрузкиСостояниеТекст", "Загрузки.СостояниеТекст", width=22),
            col("ЗагрузкиДокумент1С", "Загрузки.Документ1С", width=25),
            col("ЗагрузкиСчетФактура", "Загрузки.СчетФактура", width=20),
            col("ЗагрузкиВидДокумента", "Загрузки.ВидДокумента", width=12),
            col("ЗагрузкиНомер", "Загрузки.Номер", width=10),
            col("ЗагрузкиДата", "Загрузки.Дата", width=10),
            col("ЗагрузкиПродавец", "Загрузки.Продавец", width=25),
            col("ЗагрузкиСумма", "Загрузки.Сумма", width=12),
            col("ЗагрузкиЗамечания", "Загрузки.Замечания", width=40),
        ],
        ro=True,
        change_rows=False,
        height=12,
        events={"Selection": "ЗагрузкиВыбор"},
    )
    f.items += f.input(
        "Прогресс", "Прогресс", ind, ro=True, title_loc="None", width=100, hstretch=True
    )
    f.items += f.group(
        "ГруппаНастройки",
        "Настройки сервиса",
        ind,
        [
            lambda i: f.input("Адрес", "Адрес", i, width=20),
            lambda i: f.input("Порт", "Порт", i, width=6),
            lambda i: f.button("ПроверитьСвязь", "ПроверитьСвязь", i, bar=False),
        ],
        kind="Horizontal",
        collapsible=True,
        collapsed=True,
        show_title=True,
        representation="NormalSeparation",
    )

    f.attr("Адрес", "Адрес сервиса", t_str(100))
    f.attr("Порт", "Порт", t_num(5, 0, "Nonnegative"))
    f.attr("СостояниеСервиса", "Сервис", t_str())
    f.attr("Прогресс", "Прогресс", t_str())
    f.attr(
        "Загрузки",
        "Загрузки",
        t_vt(),
        columns=[
            ("ИмяФайла", "Файл", t_str()),
            ("ИдЗадания", "Задание", t_str(32)),
            ("Состояние", "Код состояния", t_str(10)),
            ("СостояниеТекст", "Состояние", t_str()),
            ("ВидДокумента", "Вид", t_str()),
            ("КодВида", "Код вида", t_str(20)),
            ("СтатусДокумента", "Статус документа", t_str(10)),
            ("Номер", "Номер", t_str()),
            ("Дата", "Дата", t_date()),
            ("Продавец", "Продавец", t_str()),
            ("Сумма", "Сумма", t_num(15, 2)),
            ("Замечания", "Замечания", t_str()),
            ("Правки", "Правки", t_str()),
            ("Проверено", "Проверено", t_bool()),
            (
                "Документ1С",
                "Документ 1С",
                t_refs("DocumentRef.ПоступлениеТоваровУслуг", "DocumentRef.СчетНаОплатуПоставщика"),
            ),
            ("СчетФактура", "Счёт-фактура", t_ref("DocumentRef.СчетФактураПолученный")),
        ],
    )
    f.command(
        "ЗагрузитьФайлы",
        "Загрузить файлы...",
        "Выбрать PDF, сканы или XML и отправить на распознавание",
    )
    f.command("Проверить", "Проверить документ", "Открыть форму проверки распознанного документа")
    f.command("Обновить", "Обновить", "Запросить у сервиса состояние заданий")
    f.command("ПроверитьСвязь", "Проверить связь")
    f.command("Журнал", "Журнал", "Какие документы 1С созданы из каких файлов")
    return f


# --- форма проверки документа ---------------------------------------------------------------


LINE_COLUMNS = [
    ("Наименование", "Наименование", t_str(), 30),
    ("ЕдИзм", "Ед.", t_str(), 6),
    ("КодЕдИзм", "Код ед.", t_str(), 5),
    ("Количество", "Количество", t_num(15, 3), 9),
    ("Цена", "Цена", t_num(15, 2), 10),
    ("СуммаБезНДС", "Сумма без НДС", t_num(15, 2), 11),
    ("СтавкаНДС", "Ставка НДС", t_str(), 7),
    ("СуммаНДС", "НДС", t_num(15, 2), 10),
    ("СуммаСНДС", "Сумма с НДС", t_num(15, 2), 11),
]


def review_form() -> Form:
    f = Form("DocaiLoad")
    f.events = [
        ("OnCreateAtServer", "ПриСозданииНаСервере"),
        ("OnOpen", "ПриОткрытии"),
        ("NotificationProcessing", "ОбработкаОповещения"),
    ]
    f.bar += f.button("ФормаГотово", "Готово", "\t\t\t", bar=True, default=True)
    f.bar += f.button("ФормаСоздатьДокумент", "СоздатьДокумент", "\t\t\t", bar=True)
    f.bar += f.button("ФормаОтправитьПравки", "ОтправитьПравки", "\t\t\t", bar=True)

    def col(name: str, path: str, **kw):
        return lambda ind: f.input(name, path, ind, **kw)

    ind = "\t\t"
    f.items += f.input(
        "Сводка", "Сводка", ind, ro=True, title_loc="None", height=3, multiline=True, hstretch=True
    )
    left = lambda i: f.group(  # noqa: E731
        "ГруппаСтраница",
        "Страница документа",
        i,
        [
            lambda j: f.picture("Картинка", "АдресКартинки", j, "Картинки нет"),
            lambda j: f.group(
                "ГруппаЛистание",
                "Листание",
                j,
                [
                    lambda k: f.button("ПредыдущаяСтраница", "ПредыдущаяСтраница", k, bar=False),
                    lambda k: f.input(
                        "ТекстСтраницы", "ТекстСтраницы", k, ro=True, title_loc="None", width=14
                    ),
                    lambda k: f.button("СледующаяСтраница", "СледующаяСтраница", k, bar=False),
                ],
                kind="Horizontal",
            ),
        ],
        width=55,
    )
    fields_table = lambda i: f.table(  # noqa: E731
        "Поля",
        "Поля",
        "Реквизиты документа",
        i,
        [
            col("ПоляПредставление", "Поля.Представление", ro=True, width=18),
            col(
                "ПоляЗначение",
                "Поля.Значение",
                width=25,
                events={"OnChange": "ПоляЗначениеПриИзменении"},
            ),
            col("ПоляЗамечание", "Поля.Замечание", ro=True, width=30),
        ],
        ro=False,
        change_rows=False,
    )
    lines_table = lambda i: f.table(  # noqa: E731
        "Строки",
        "Строки",
        "Товары и услуги",
        i,
        [
            col("СтрокиН", "Строки.Н", ro=True, width=3),
            col("СтрокиНаименование", "Строки.Наименование", width=30),
            col(
                "СтрокиНоменклатура",
                "Строки.Номенклатура",
                width=25,
                events={
                    "StartChoice": "СтрокиНоменклатураНачалоВыбора",
                    "OnChange": "СтрокиНоменклатураПриИзменении",
                },
            ),
            col("СтрокиСостояниеНоменклатуры", "Строки.СостояниеНоменклатуры", ro=True, width=22),
            lambda i: f.checkbox("СтрокиЭтоУслуга", "Строки.ЭтоУслуга", i),
            col("СтрокиЕдиницаИзмерения", "Строки.ЕдиницаИзмерения", width=8),
            *[
                col(f"Строки{n}", f"Строки.{n}", width=w)
                for n, _t, _ty, w in LINE_COLUMNS
                if n != "Наименование"
            ],
            col("СтрокиЗамечание", "Строки.Замечание", ro=True, width=30),
        ],
        ro=False,
        change_rows=True,
        bar_buttons=[("СтрокиСоздатьНоменклатуру", "СоздатьНоменклатуру")],
        events={
            "OnEditEnd": "СтрокиПриОкончанииРедактирования",
            "AfterDeleteRow": "СтрокиПослеУдаления",
        },
    )
    right = lambda i: f.pages(  # noqa: E731
        "Страницы",
        i,
        [
            ("СтраницаРеквизиты", "Реквизиты", [fields_table]),
            ("СтраницаСтроки", "Товары и услуги", [lines_table]),
        ],
    )

    def match_row(name, title, field, handler, button=None, links=None):
        children = [
            lambda k: f.input(
                field, field, k, title=title, width=28, events={"OnChange": handler}, links=links
            ),
            lambda k: f.input(
                f"Состояние{field}",
                f"Состояние{field}",
                k,
                ro=True,
                title_loc="None",
                hstretch=True,
            ),
        ]
        if button:
            children.append(lambda k: f.button(button, button, k, bar=False))
        return lambda j: f.group(name, title, j, children, kind="Horizontal")

    matching = lambda i: f.group(  # noqa: E731
        "ГруппаСопоставление",
        "Справочники 1С",
        i,
        [
            match_row("ГруппаОрганизация", "Организация", "Организация", "ОрганизацияПриИзменении"),
            match_row(
                "ГруппаКонтрагент",
                "Контрагент",
                "Контрагент",
                "КонтрагентПриИзменении",
                "СоздатьКонтрагента",
            ),
            match_row(
                "ГруппаДоговор",
                "Договор",
                "Договор",
                "ДоговорПриИзменении",
                "СоздатьДоговор",
                links={"Отбор.Владелец": "Контрагент"},
            ),
        ],
        show_title=True,
        representation="NormalSeparation",
    )
    right_column = lambda i: f.group(  # noqa: E731
        "ГруппаПравая", "Реквизиты и справочники", i, [matching, right]
    )
    f.items += f.group("ГруппаОсновная", "Документ", ind, [left, right_column], kind="Horizontal")
    f.items += f.label(
        "Легенда",
        "Жёлтым — модель не уверена в значении, красным — значение не прошло проверку "
        "(причина в колонке «Замечание»), зелёным — исправлено вами. Исправьте значение "
        "прямо в таблице и нажмите «Готово».",
        ind,
    )

    f.attr("ИдЗадания", "Задание", t_str(32))
    f.attr("Адрес", "Адрес сервиса", t_str(100))
    f.attr("Порт", "Порт", t_num(5, 0, "Nonnegative"))
    f.attr("ИмяФайла", "Файл", t_str())
    f.attr("АдресКартинки", "Картинка", t_str())
    f.attr("Страница", "Страница", t_num(3, 0, "Nonnegative"))
    f.attr("ВсегоСтраниц", "Всего страниц", t_num(3, 0, "Nonnegative"))
    f.attr("ТекстСтраницы", "Страница", t_str())
    f.attr("Сводка", "Сводка", t_str())
    f.attr("Автопроверка", "Автопроверка", t_bool())
    f.attr("АвтоСоздание", "Автопроверка: создать документ", t_bool())
    f.attr("Организация", "Организация", t_ref("CatalogRef.Организации"))
    f.attr("Контрагент", "Контрагент", t_ref("CatalogRef.Контрагенты"))
    f.attr("Договор", "Договор", t_ref("CatalogRef.ДоговорыКонтрагентов"))
    for name in ("Организация", "Контрагент", "Договор"):
        f.attr(f"Состояние{name}", f"Состояние: {name}", t_str())
        f.attr(f"Уровень{name}", f"Уровень: {name}", t_num(1, 0))
    f.attr(
        "Поля",
        "Реквизиты",
        t_vt(),
        columns=[
            ("Имя", "Имя", t_str()),
            ("Представление", "Реквизит", t_str()),
            ("Значение", "Значение", t_str()),
            ("Исходное", "Исходное значение", t_str()),
            ("Уверенность", "Уверенность", t_num(3, 2)),
            ("Уровень", "Уровень", t_num(1, 0)),
            ("Замечание", "Замечание", t_str()),
            ("УровеньИсходный", "Исходный уровень", t_num(1, 0)),
            ("ЗамечаниеИсходное", "Исходное замечание", t_str()),
        ],
    )
    f.attr(
        "Строки",
        "Строки",
        t_vt(),
        columns=[
            ("Н", "N", t_num(5, 0)),
            *[(n, t, ty) for n, t, ty, _w in LINE_COLUMNS],
            *[(f"Уровень{n}", f"Уровень: {t}", t_num(1, 0)) for n, t, _ty, _w in LINE_COLUMNS],
            ("Индекс", "Индекс в документе", t_num(5, 0)),
            ("Замечание", "Замечание", t_str()),
            ("Номенклатура", "Номенклатура 1С", t_ref("CatalogRef.Номенклатура")),
            ("ЕдиницаИзмерения", "Ед. 1С", t_ref("CatalogRef.КлассификаторЕдиницИзмерения")),
            ("ЭтоУслуга", "Услуга", t_bool()),
            ("УровеньНоменклатуры", "Уровень: номенклатура", t_num(1, 0)),
            ("СостояниеНоменклатуры", "Сопоставление", t_str()),
        ],
    )
    f.attr(
        "Похожие",
        "Похожая номенклатура",
        t_vt(),
        columns=[
            ("ИдентификаторСтроки", "Строка формы", t_num(10, 0)),
            ("Номенклатура", "Номенклатура", t_ref("CatalogRef.Номенклатура")),
            ("Наименование", "Наименование", t_str()),
            ("Сходство", "Сходство", t_num(3, 2)),
        ],
    )
    f.attr(
        "Исходные",
        "Исходные значения",
        t_vt(),
        columns=[
            ("Имя", "Имя", t_str()),
            ("Значение", "Значение", t_str()),
            ("Уровень", "Уровень", t_num(1, 0)),
            ("Замечание", "Замечание", t_str()),
        ],
    )
    f.command("Готово", "Готово", "Сохранить правки, отправить их в сервис и закрыть")
    f.command(
        "ОтправитьПравки",
        "Отправить правки",
        "Отправить проверенные значения в сервис (/v1/jobs/{id}/feedback)",
    )
    f.command("СоздатьКонтрагента", "Создать...", "Новый контрагент, заполненный из документа")
    f.command(
        "СоздатьДоговор", "Создать...", "Новый договор с поставщиком, заполненный из документа"
    )
    f.command(
        "СоздатьДокумент",
        "Создать документ в 1С",
        "Создать непроведённое поступление (или счёт от поставщика) и счёт-фактуру",
    )
    f.command(
        "СоздатьНоменклатуру",
        "Создать номенклатуру",
        "Новая номенклатура для текущей строки, заполненная из документа",
    )
    f.command("ПредыдущаяСтраница", "<")
    f.command("СледующаяСтраница", ">")
    for p, t in [
        ("ИдЗадания", t_str(32)),
        ("Адрес", t_str(100)),
        ("Порт", t_num(5, 0)),
        ("ИмяФайла", t_str()),
        ("Правки", t_str()),
        ("Автопроверка", t_bool()),
        ("АвтоСоздание", t_bool()),
    ]:
        f.param(p, t)
    return f


def journal_form() -> Form:
    f = Form("DocaiLoad")
    f.events = [("OnCreateAtServer", "ПриСозданииНаСервере")]
    f.bar += f.button("ФормаОбновитьЖурнал", "ОбновитьЖурнал", "\t\t\t", bar=True, default=True)

    def col(name: str, path: str, **kw):
        return lambda ind: f.input(name, path, ind, **kw)

    ind = "\t\t"
    f.items += f.label(
        "Пояснение",
        "Документы 1С, созданные обработкой «Загрузка первичных документов (ИИ)» — по метке "
        "[docai] в комментарии. Двойной щелчок открывает документ.",
        ind,
    )
    f.items += f.table(
        "Журнал",
        "Журнал",
        "Созданные документы",
        ind,
        [
            col("ЖурналДата", "Журнал.Дата", width=10),
            col("ЖурналДокумент", "Журнал.Документ", width=35),
            col("ЖурналПроведен", "Журнал.Проведен", width=8),
            col("ЖурналКонтрагент", "Журнал.Контрагент", width=25),
            col("ЖурналСумма", "Журнал.Сумма", width=12),
            col("ЖурналФайл", "Журнал.Файл", width=40),
        ],
        ro=True,
        change_rows=False,
        events={"Selection": "ЖурналВыбор"},
    )
    f.attr(
        "Журнал",
        "Журнал",
        t_vt(),
        columns=[
            (
                "Документ",
                "Документ",
                t_refs("DocumentRef.ПоступлениеТоваровУслуг", "DocumentRef.СчетНаОплатуПоставщика"),
            ),
            ("Дата", "Дата", t_date()),
            ("Проведен", "Проведён", t_bool()),
            ("Контрагент", "Контрагент", t_ref("CatalogRef.Контрагенты")),
            ("Сумма", "Сумма", t_num(15, 2)),
            ("Файл", "Файл и задание", t_str()),
        ],
    )
    f.command("ОбновитьЖурнал", "Обновить")
    return f


def main() -> None:
    default = Path(__file__).resolve().parents[1] / "onec" / "src"
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    name = "DocaiLoad"
    forms = {
        "Форма": ("Загрузка первичных документов (ИИ)", main_form()),
        "ФормаПроверки": ("Проверка документа", review_form()),
        "ФормаЖурнала": ("Журнал загрузки документов", journal_form()),
    }
    write(
        root / f"{name}.xml",
        processor_meta(name, "Загрузка первичных документов (ИИ)", list(forms), "Форма"),
    )
    for form_name, (title, form) in forms.items():
        write(root / name / "Forms" / f"{form_name}.xml", form_meta(form_name, title))
        write(root / name / "Forms" / form_name / "Ext" / "Form.xml", form.render(title))


if __name__ == "__main__":
    main()
