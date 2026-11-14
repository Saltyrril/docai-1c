"""Обезличивание разметки для отчётов: наименования, ИНН и прочие реквизиты → псевдонимы.

Правила:
- Один и тот же исходный реквизит всегда даёт один и тот же псевдоним (в пределах соли),
  поэтому связи «продавец повторяется в нескольких документах» сохраняются.
- Псевдоним считается как SHA-256 от секретной соли и значения. Соль лежит в
  `data/real/.pseudo_salt` и никуда не уходит; без неё по псевдониму ИНН не восстановить
  перебором (пространство ИНН маленькое, без соли хэш перебрали бы за секунды).
- ИНН заменяется на ИНН той же длины с верными контрольными цифрами, БИК и счета — на
  правдоподобные числа с верным ключом, чтобы валидаторы и на обезличенных примерах
  вели себя так же, как на настоящих.
- Остаются номера, даты, суммы, количества и наименования товаров. Они обычно не
  идентифицируют сторону, но в редких случаях (уникальный товар, круглая сумма с датой)
  могут. Перед публикацией примеры нужно просмотреть глазами.
- После замены проверяется, что ни одно исходное значение не осталось в тексте результата
  (например, наименование продавца внутри названия строки). Нашлось — ошибка, файл не пишется.
"""

import hashlib
import json
import re
import secrets
from pathlib import Path

from docai.schemas.common import BankDetails, Party
from docai.schemas.documents import AnyDocument, parse_document
from docai.validation.requisites import account_key, inn_check_digits

SALT_NAME = ".pseudo_salt"
DEFAULT_OUT = "anon"

_LEGAL_FORM_RE = re.compile(
    r"^\s*(ООО|АО|ПАО|ЗАО|ОАО|НКО|АНО|ГУП|МУП|ФГУП|ИП)(?=[\s«\"“]|$)"
    r"|^\s*(Индивидуальный\s+предприниматель)\b",
    re.IGNORECASE,
)


class LeakError(RuntimeError):
    """Исходное значение осталось в обезличенном документе."""


def load_salt(root: Path) -> str:
    """Соль из `<root>/.pseudo_salt`; при первом запуске создаётся случайная."""
    path = root / SALT_NAME
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    salt = secrets.token_hex(32)
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(salt + "\n", encoding="utf-8", newline="\n")
    return salt


class Pseudonymizer:
    """Детерминированная замена реквизитов (при одной соли — одни и те же псевдонимы)."""

    def __init__(self, salt: str) -> None:
        if len(salt) < 16:
            raise ValueError("соль должна быть длиной не меньше 16 символов")
        self._salt = salt
        self.originals: set[str] = set()  # исходные значения: для проверки утечки

    def _hash(self, kind: str, value: str) -> str:
        data = f"{self._salt}\0{kind}\0{value.strip().casefold()}".encode()
        return hashlib.sha256(data).hexdigest()

    def _digits(self, kind: str, value: str, count: int) -> str:
        """`count` цифр, выведенных из хэша; первая цифра не ноль (десятичная запись числа)."""
        return str(int(self._hash(kind, value), 16))[:count]

    def _remember(self, *values: str | None) -> None:
        self.originals.update(v.strip() for v in values if v and len(v.strip()) >= 4)

    # ---- отдельные реквизиты -----------------------------------------------------------

    def name(self, value: str) -> str:
        self._remember(value)
        tag = self._hash("name", value)[:6].upper()
        match = _LEGAL_FORM_RE.match(value)
        if not match:
            return f"Контрагент-{tag}"
        form = (match.group(1) or "ИП").upper()  # «Индивидуальный предприниматель» → ИП
        return f"ИП Контрагент-{tag}" if form == "ИП" else f"{form} «Контрагент-{tag}»"

    def inn(self, value: str) -> str:
        self._remember(value)
        long = len(value) == 12
        body = self._digits("inn", value, 10 if long else 9)
        return body + inn_check_digits(body)

    def kpp(self, value: str) -> str:
        self._remember(value)
        digits = self._digits("kpp", value, 7)
        return f"{digits[:4]}01{digits[4:]}"

    def address(self, value: str) -> str:
        self._remember(value)
        return f"г. Город-{self._hash('address', value)[:4].upper()}, ул. Адресная, д. 1"

    def bank_name(self, value: str) -> str:
        self._remember(value)
        return f"Банк «Псевдоним-{self._hash('bank', value)[:4].upper()}»"

    def bik(self, value: str) -> str:
        self._remember(value)
        return "04" + self._digits("bik", value, 7)

    def account(self, value: str, bik: str) -> str:
        """Расчётный счёт: 20 цифр с верным ключом (если БИК пригоден для расчёта ключа)."""
        self._remember(value)
        raw = "40702810" + "0" + self._digits("account", value, 11)
        if len(bik) == 9 and bik.isdigit():
            return raw[:8] + account_key(raw, bik) + raw[9:]
        return raw

    def corr_account(self, value: str, bik: str) -> str:
        """Корр. счёт: 301018..., последние три цифры — как в БИК, ключ верный."""
        self._remember(value)
        tail = bik[-3:] if len(bik) == 9 and bik.isdigit() else "000"
        raw = "30101810" + "0" + self._digits("corr", value, 8) + tail
        if len(bik) == 9 and bik.isdigit():
            return raw[:8] + account_key(raw, bik, in_credit_institution=False) + raw[9:]
        return raw

    # ---- документ ----------------------------------------------------------------------

    def party(self, party: Party) -> Party:
        return Party(
            name=self.name(party.name),
            inn=self.inn(party.inn) if party.inn else None,
            kpp=self.kpp(party.kpp) if party.kpp else None,
            address=self.address(party.address) if party.address else None,
        )

    def bank(self, bank: BankDetails) -> BankDetails:
        new_bik = self.bik(bank.bik) if bank.bik else None
        return BankDetails(
            bank_name=self.bank_name(bank.bank_name) if bank.bank_name else None,
            bik=new_bik,
            corr_account=self.corr_account(bank.corr_account, new_bik or "")
            if bank.corr_account
            else None,
            account=self.account(bank.account, new_bik or "") if bank.account else None,
        )

    def document(self, doc: AnyDocument) -> AnyDocument:
        """Копия документа с замененными реквизитами; исходный не меняется."""
        data = doc.model_dump(mode="json")
        data["seller"] = self.party(doc.seller).model_dump(mode="json")
        data["buyer"] = self.party(doc.buyer).model_dump(mode="json")
        bank = getattr(doc, "seller_bank", None)
        if bank is not None:
            data["seller_bank"] = self.bank(bank).model_dump(mode="json")
        basis = getattr(doc, "basis", None)
        if basis is not None and basis.number:
            self._remember(basis.number)
            data["basis"]["number"] = "Д-" + self._hash("basis", basis.number)[:5].upper()
        result = parse_document(data)
        self.check_no_leak(result)
        return result

    def check_no_leak(self, doc: AnyDocument) -> None:
        text = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False).casefold()
        for original in self.originals:
            if original.casefold() in text:
                raise LeakError(
                    "В обезличенном документе осталось исходное значение "
                    f"(длина {len(original)} симв.). Документ не записан."
                )


def pseudonymize_dir(
    src: Path, out: Path, salt: str, *, force: bool = False
) -> tuple[list[Path], list[str]]:
    """Обезличить все эталоны из `src` в `out`. Возвращает (записанные файлы, ошибки).

    Имена файлов не сохраняются (в них могут быть названия контрагентов): файл называется
    `<тип>_<метка>.json`, метка зависит от исходного имени и соли.
    """
    written: list[Path] = []
    problems: list[str] = []
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in src.glob("*.json") if not p.name.startswith("_")):
        label = hashlib.sha256(f"{salt}\0file\0{path.stem}".encode()).hexdigest()[:6]
        try:
            doc = parse_document(path.read_text(encoding="utf-8"))
            result = Pseudonymizer(salt).document(doc)  # новый объект: утечку ищем по документу
        except (LeakError, ValueError) as exc:
            problems.append(f"{label}: {exc}")
            continue
        target = out / f"{result.doc_type.value}_{label}.json"
        if target.exists() and not force:
            problems.append(f"{target.name}: уже есть (используйте --force)")
            continue
        text = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        target.write_text(text + "\n", encoding="utf-8", newline="\n")
        written.append(target)
    return written, problems
