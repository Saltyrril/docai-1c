"""Текст договора: каждое условие — в одной из нескольких формулировок.

Одно и то же условие («оплата в течение 30 дней с даты поставки») печатается разными
словами и в разной форме чисел («30 календарных дней», «30 (тридцати) календарных дней»,
«отсрочка платежа на 30 дней»), а разделы называются по-разному. Иначе поиск по
договорам свёлся бы к совпадению шаблонной фразы.
"""

import random
from dataclasses import dataclass, replace
from decimal import Decimal

from docai.rag import facts as F
from docai.rag.document import Clause, CorpusDoc, Section
from docai.rag.terms import Contract, Kind, Penalty
from docai.rag.words import (
    date_text,
    days_gen,
    days_nom,
    decimal_text,
    months_nom,
    months_words,
    number_words,
    number_words_genitive,
    percent_text,
    rate_text,
    rubles_text,
)
from docai.synth.directory import Entity

HEADERS = {
    Kind.SUPPLY: ("ДОГОВОР ПОСТАВКИ", "ДОГОВОР ПОСТАВКИ ТОВАРА"),
    Kind.SERVICES: ("ДОГОВОР ВОЗМЕЗДНОГО ОКАЗАНИЯ УСЛУГ", "ДОГОВОР НА ОКАЗАНИЕ УСЛУГ"),
    Kind.LEASE: ("ДОГОВОР АРЕНДЫ НЕЖИЛОГО ПОМЕЩЕНИЯ", "ДОГОВОР АРЕНДЫ"),
    Kind.WORKS: ("ДОГОВОР ПОДРЯДА", "ДОГОВОР ПОДРЯДА НА ВЫПОЛНЕНИЕ РАБОТ"),
}
# «к договору поставки № …»
KIND_DAT = {
    Kind.SUPPLY: "договору поставки",
    Kind.SERVICES: "договору возмездного оказания услуг",
    Kind.LEASE: "договору аренды нежилого помещения",
    Kind.WORKS: "договору подряда",
}


@dataclass(slots=True)
class Style:
    """Оформление одного документа: числа прописью, формат дат, сумм и процентов."""

    rng: random.Random
    words: bool
    date_style: int
    money_style: int
    pct_style: int
    rate_style: int

    @classmethod
    def random(cls, rng: random.Random) -> "Style":
        return cls(
            rng,
            rng.random() < 0.5,
            rng.randrange(3),
            rng.randrange(3),
            rng.randrange(3),
            rng.randrange(3),
        )

    def pick(self, *options: str) -> str:
        return self.rng.choice(options)

    def d(self, n: int, kind: str | None = "calendar") -> str:
        """«в течение 10 (десяти) рабочих дней» — родительный падеж."""
        return days_gen(n, kind, words=self.words)

    def d_acc(self, n: int, kind: str | None = "calendar") -> str:
        """«за 30 (тридцать) календарных дней» — винительный падеж (= именительный)."""
        text = days_nom(n, kind)
        if self.words:
            number, rest = text.split(" ", 1)
            return f"{number} ({number_words(n)}) {rest}"
        return text

    def date(self, value) -> str:
        return date_text(value, self.date_style)

    def money(self, amount: Decimal) -> str:
        return rubles_text(amount, self.money_style)

    def pct(self, p: int) -> str:
        return percent_text(p, self.pct_style)

    def rate(self, rate: Decimal) -> str:
        return rate_text(rate, self.rate_style)

    def months_gen(self, n: int) -> str:
        return f"{n} ({number_words_genitive(n)}) месяцев" if self.words else f"{n} месяцев"

    def months_acc(self, n: int) -> str:
        return months_words(n) if self.words else months_nom(n)


def party_intro(entity: Entity, role: str, s: Style) -> str:
    """Сторона в преамбуле: «ООО «Вектор», именуемое в дальнейшем «Поставщик», ...»."""
    if entity.is_individual:
        fio = entity.name.removeprefix("ИП ").strip()
        female = fio.split()[-1].endswith(("вна", "чна"))
        named = "именуемая" if female else "именуемый"
        acting = "действующая" if female else "действующий"
        return s.pick(
            f"индивидуальный предприниматель {fio}, {named} в дальнейшем «{role}», "
            f"{acting} на основании записи в ЕГРИП,",
            f"индивидуальный предприниматель {fio} (далее — «{role}»)",
        )
    return s.pick(
        f"{entity.name}, именуемое в дальнейшем «{role}», в лице генерального директора, "
        f"действующего на основании Устава,",
        f"{entity.name} (далее — «{role}») в лице генерального директора, действующего "
        f"на основании Устава,",
        f"{entity.name}, в лице директора, действующего на основании Устава, далее "
        f"именуемое «{role}»",
    )


def tidy_commas(text: str) -> str:
    """Убрать двойные запятые после оборота «действующего на основании Устава,»."""
    return text.replace(",,", ",").replace(", :", ":")


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


# --- Пункты с фактами --------------------------------------------------------------------


def clause_subject(c: Contract, s: Style) -> str:
    our, them = c.roles
    if c.kind is Kind.SUPPLY:
        return s.pick(
            f"{them.nom} обязуется передать в собственность {our.dat}, а {our.nom} — принять "
            f"и оплатить {c.subject} (далее — «Товар») в порядке и на условиях Договора.",
            f"По настоящему Договору {them.nom} поставляет, а {our.nom} принимает и оплачивает "
            f"{c.subject} (далее — Товар).",
            f"{them.nom} обязуется в течение срока действия Договора поставлять {our.dat} "
            f"партиями {c.subject}, а {our.nom} обязуется принимать и оплачивать Товар.",
        )
    if c.kind is Kind.SERVICES:
        return s.pick(
            f"{them.nom} обязуется по заданию {our.gen} оказывать услуги: {c.subject} "
            f"(далее — «Услуги»), а {our.nom} обязуется оплачивать Услуги.",
            f"Предметом Договора является оказание {them.ins} услуг ({c.subject}) "
            f"для {our.gen} на условиях, указанных ниже (далее — Услуги).",
            f"{our.nom} поручает, а {them.nom} принимает на себя обязательство оказывать "
            f"следующие услуги: {c.subject} (далее — «Услуги»).",
        )
    if c.kind is Kind.LEASE:
        return s.pick(
            f"{them.nom} передаёт, а {our.nom} принимает во временное владение и пользование "
            f"(аренду) нежилое помещение (далее — «Помещение») для размещения офиса.",
            f"{them.nom} предоставляет {our.dat} за плату во временное пользование нежилое "
            f"помещение под офис (далее — Помещение).",
        )
    return s.pick(
        f"{them.nom} обязуется по заданию {our.gen} выполнить работы: {c.subject} "
        f"(далее — «Работы») на объекте по адресу: {c.works_address}, а {our.nom} "
        f"обязуется принять результат Работ и оплатить его.",
        f"{our.nom} поручает, а {them.nom} принимает на себя выполнение работ ({c.subject}, "
        f"далее — Работы) на объекте: {c.works_address}.",
        f"Предмет Договора — {c.subject} (далее — «Работы») на объекте {our.gen} по адресу: "
        f"{c.works_address}.",
    )


def clause_premises(c: Contract, s: Style) -> str:
    assert c.area is not None
    area = decimal_text(c.area)
    return s.pick(
        f"Помещение общей площадью {area} кв. м расположено по адресу: {c.premises_address}.",
        f"Площадь Помещения — {area} м²; адрес: {c.premises_address}.",
        f"Арендуемое Помещение находится по адресу: {c.premises_address}; его общая площадь "
        f"составляет {area} кв. м.",
    )


def clause_price(c: Contract, s: Style) -> str:
    assert c.price is not None
    our, them = c.roles
    money = s.money(c.price)
    if c.kind is Kind.SUPPLY:
        return s.pick(
            f"Общая стоимость Товара, поставляемого по Договору, не может превышать {money}, "
            f"включая НДС.",
            f"Цена Договора (предельная сумма поставок) составляет {money} с учётом НДС.",
            f"Сумма всех поставок по настоящему Договору не должна превышать {money}, "
            f"в том числе НДС.",
        )
    if c.kind is Kind.SERVICES:
        return s.pick(
            f"Стоимость Услуг составляет {money} в месяц, в том числе НДС по ставке, "
            f"установленной Налоговым кодексом РФ.",
            f"Ежемесячная стоимость Услуг — {money}, включая НДС.",
            f"За оказание Услуг {our.nom} выплачивает {them.dat} абонентскую плату в размере "
            f"{money} за каждый календарный месяц, включая НДС.",
        )
    if c.kind is Kind.LEASE:
        return s.pick(
            f"Арендная плата составляет {money} в месяц.",
            f"Размер ежемесячной арендной платы — {money}.",
            f"{our.nom} уплачивает арендную плату в сумме {money} за каждый месяц аренды.",
        )
    return s.pick(
        f"Стоимость Работ по Договору составляет {money}, включая НДС, и является твёрдой.",
        f"Цена Работ — {money} с учётом НДС; цена твёрдая и может быть изменена только "
        f"соглашением Сторон.",
        f"Общая стоимость Работ определена сметой (Приложение № 2) и составляет {money}, "
        f"в том числе НДС.",
    )


def _trigger(c: Contract, s: Style) -> str:
    our, them = c.roles
    return {
        "delivery": lambda: s.pick(
            "с даты поставки Товара",
            f"с момента передачи Товара {our.dat}",
            "со дня поставки соответствующей партии Товара",
        ),
        "upd": lambda: s.pick(
            "с даты подписания УПД (товарной накладной)",
            "с даты подписания Сторонами универсального передаточного документа "
            "(товарной накладной)",
            "со дня подписания товарной накладной (УПД)",
        ),
        "invoice": lambda: s.pick(
            "с даты получения счёта",
            f"с даты получения {our.ins} счёта {them.gen}",
            "со дня получения счёта на оплату",
        ),
        "act": lambda: s.pick(
            "с даты подписания акта оказанных услуг",
            "с даты подписания Сторонами акта сдачи-приёмки Услуг",
            "со дня подписания акта об оказании Услуг",
        ),
        "ks": lambda: s.pick(
            "с даты подписания акта о приёмке выполненных работ (КС-2) и справки о стоимости "
            "работ (КС-3)",
            "с даты подписания Сторонами форм КС-2 и КС-3",
            "со дня подписания акта КС-2 и справки КС-3",
        ),
    }[c.payment.trigger]()


def clause_payment(c: Contract, s: Style) -> str:
    our, them = c.roles
    p = c.payment
    if p.scheme == "monthly":
        due = p.due_day
        if c.kind is Kind.LEASE and p.due_month == "current":
            return s.pick(
                f"Арендная плата вносится ежемесячно, не позднее {due} числа текущего "
                f"(оплачиваемого) месяца.",
                f"{our.nom} перечисляет арендную плату авансом за каждый месяц не позднее "
                f"{due}-го числа этого месяца.",
                f"Арендная плата за текущий месяц оплачивается не позднее {due} числа "
                f"такого месяца.",
            )
        if c.kind is Kind.LEASE:
            return s.pick(
                f"Арендная плата вносится ежемесячно не позднее {due} числа месяца, следующего "
                f"за расчётным.",
                f"Плата за месяц аренды перечисляется не позднее {due}-го числа следующего месяца.",
                f"{our.nom} оплачивает каждый истекший месяц аренды не позднее {due} числа "
                f"месяца, следующего за ним.",
            )
        return s.pick(
            f"{our.nom} оплачивает Услуги ежемесячно, не позднее {due} числа месяца, следующего "
            f"за отчётным, на основании акта и счёта.",
            f"Оплата Услуг производится не позднее {due}-го числа месяца, следующего за месяцем "
            f"оказания Услуг.",
            f"Расчёты производятся ежемесячно: плата за истекший месяц перечисляется не позднее "
            f"{due} числа следующего месяца.",
        )

    obj, obj_gen, part_gen = {
        Kind.SUPPLY: ("Товар", "Товара", "партии Товара"),
        Kind.SERVICES: ("Услуги", "Услуг", "Услуг"),
        Kind.WORKS: ("выполненные Работы", "Работ", "Работ"),
    }[c.kind]
    pd = s.d(p.prepay_days, "banking") if p.prepay_percent else ""
    if p.prepay_percent == 100:
        options = [
            f"Оплата {obj_gen} производится на условиях 100% предварительной оплаты в течение "
            f"{pd} с даты получения счёта.",
            f"{our.nom} оплачивает {obj} авансом в полном объёме (100%) в течение {pd} с даты "
            f"получения счёта.",
        ]
        if c.kind is Kind.SUPPLY:
            options.append(
                f"Отгрузка Товара производится после стопроцентной предоплаты, которую "
                f"{our.nom} вносит в течение {pd} со дня получения счёта."
            )
        return s.pick(*options)

    assert p.days is not None
    days = s.d(p.days, p.day_kind)
    trig = _trigger(c, s)
    if p.prepay_percent:
        pre, rest = s.pct(p.prepay_percent), s.pct(100 - p.prepay_percent)
        if c.kind is Kind.WORKS:
            return s.pick(
                f"{our.nom} выплачивает {them.dat} аванс в размере {pre} от стоимости Работ "
                f"в течение {pd} с даты получения счёта. Окончательный расчёт производится "
                f"в течение {days} {trig}.",
                f"Аванс составляет {pre} цены Работ и перечисляется в течение {pd} со дня "
                f"получения счёта; оставшиеся {rest} {our.nom} оплачивает не позднее {days} "
                f"{trig}.",
            )
        return s.pick(
            f"{our.nom} перечисляет предоплату в размере {pre} от стоимости {part_gen} в течение "
            f"{pd} с даты получения счёта; оставшиеся {rest} оплачиваются в течение {days} "
            f"{trig}.",
            f"Оплата производится в два этапа: аванс {pre} — в течение {pd} с даты получения "
            f"счёта, окончательный расчёт ({rest}) — не позднее {days} {trig}.",
            f"{our.nom} вносит предварительную оплату в размере {pre} в течение {pd} со дня "
            f"получения счёта. Остаток стоимости оплачивается в течение {days} {trig}.",
        )
    return s.pick(
        f"{our.nom} оплачивает {obj} в течение {days} {trig}.",
        f"Оплата производится путём перечисления денежных средств на расчётный счёт {them.gen} "
        f"в течение {days} {trig}.",
        f"Расчёты за {obj} осуществляются в безналичном порядке в срок не позднее {days} {trig}.",
        f"{our.nom} обязуется произвести оплату {obj_gen} не позднее {days} {trig}.",
        f"{them.nom} предоставляет {our.dat} отсрочку платежа на {s.d_acc(p.days, p.day_kind)} "
        f"{trig}.",
    )


def clause_deposit(c: Contract, s: Style) -> str:
    assert c.deposit is not None
    our, them = c.roles
    money = s.money(c.deposit)
    return s.pick(
        f"В течение 5 (пяти) банковских дней с даты подписания Договора {our.nom} вносит "
        f"обеспечительный платёж в размере {money}, который засчитывается в счёт арендной "
        f"платы за последний месяц аренды.",
        f"{our.nom} перечисляет {them.dat} обеспечительный платёж — {money}. Платёж "
        f"возвращается при прекращении Договора за вычетом задолженности {our.gen}.",
        f"Обеспечительный платёж по Договору составляет {money} и вносится одновременно "
        f"с первой арендной платой.",
    )


def clause_indexation(c: Contract, s: Style) -> str:
    assert c.indexation is not None
    our, them = c.roles
    pct = s.pct(c.indexation)
    return s.pick(
        f"Размер арендной платы ежегодно индексируется на {pct}, начиная с первого числа месяца, "
        f"следующего за годовщиной подписания Договора.",
        f"Арендная плата увеличивается ежегодно на {pct}.",
        f"{them.nom} ежегодно индексирует арендную плату на {pct}, уведомив {our.acc} "
        f"об этом не позднее чем за 30 дней.",
    )


def clause_utilities(c: Contract, s: Style) -> str:
    our, them = c.roles
    if c.utilities_included:
        return s.pick(
            "Арендная плата включает стоимость коммунальных услуг (отопление, водоснабжение, "
            "электроэнергия).",
            "Коммунальные платежи и электроэнергия входят в арендную плату и отдельно "
            "не оплачиваются.",
            f"В арендную плату включены расходы {them.gen} на коммунальные услуги.",
        )
    return s.pick(
        f"Коммунальные услуги и электроэнергия в арендную плату не входят и оплачиваются "
        f"{our.ins} отдельно по показаниям приборов учёта.",
        f"Стоимость коммунальных услуг возмещается {our.ins} сверх арендной платы по счетам "
        f"{them.gen} на основании показаний счётчиков.",
        "Расходы на электроэнергию, отопление и воду не включены в арендную плату "
        "и компенсируются отдельно по приборам учёта.",
    )


def clause_retention(c: Contract, s: Style) -> str:
    assert c.retention_percent is not None
    our, them = c.roles
    pct = s.pct(c.retention_percent)
    return s.pick(
        f"{our.nom} удерживает {pct} от стоимости выполненных Работ в качестве гарантийного "
        f"удержания; удержанная сумма выплачивается {them.dat} после окончания гарантийного "
        f"срока.",
        f"Гарантийное удержание составляет {pct} от стоимости каждого акта КС-2 и возвращается "
        f"по истечении гарантийного срока.",
        f"Из каждого платежа {our.nom} удерживает {pct} стоимости Работ в обеспечение "
        f"гарантийных обязательств {them.gen}; эти средства перечисляются после окончания "
        f"гарантийного срока.",
    )


def clause_delivery(c: Contract, s: Style) -> str:
    assert c.delivery_days is not None
    our, them = c.roles
    if c.delivery_by_supplier:
        how = s.pick(
            f"силами и за счёт {them.gen} на склад {our.gen}",
            f"транспортом {them.gen} до склада {our.gen}",
        )
    else:
        how = s.pick(
            f"самовывозом со склада {them.gen}",
            f"путём выборки Товара {our.ins} со склада {them.gen} (самовывоз)",
        )
    kind = c.delivery_day_kind
    return s.pick(
        f"Поставка каждой партии Товара осуществляется в течение {s.d(c.delivery_days, kind)} "
        f"с даты получения заявки {our.gen} {how}.",
        f"{them.nom} поставляет Товар {how} не позднее {s.d(c.delivery_days, kind)} со дня "
        f"получения заявки.",
        f"Срок поставки партии Товара составляет {s.d_acc(c.delivery_days, kind)} с момента "
        f"получения заявки. Товар передаётся {how}.",
    )


def clause_works_period(c: Contract, s: Style) -> str:
    assert c.works_start and c.works_end
    _, them = c.roles
    start, end = s.date(c.works_start), s.date(c.works_end)
    return s.pick(
        f"Срок выполнения Работ: начало — {start}, окончание — {end}.",
        f"{them.nom} выполняет Работы в период с {start} по {end}.",
        f"Работы должны быть начаты не позднее {start} и завершены не позднее {end}.",
    )


def clause_warranty(c: Contract, s: Style) -> str:
    assert c.warranty_months
    _, them = c.roles
    n = c.warranty_months
    return s.pick(
        f"Гарантийный срок на результат Работ составляет {s.months_acc(n)} с даты подписания "
        f"акта о приёмке.",
        f"{them.nom} гарантирует качество Работ в течение {s.months_gen(n)} с даты подписания "
        f"Сторонами акта КС-2.",
        f"Гарантия на выполненные Работы — {months_nom(n)} с даты их приёмки {c.roles[0].ins}.",
    )


def clause_acceptance(c: Contract, s: Style) -> str:
    our, them = c.roles
    d = s.d(c.acceptance_days, "working")
    if c.kind is Kind.SUPPLY:
        return s.pick(
            f"Приёмка Товара по количеству и комплектности производится при его получении. "
            f"Претензии по качеству Товара предъявляются {them.dat} в течение {d} с даты приёмки.",
            f"Товар принимается по количеству в момент передачи, по качеству — в течение {d} "
            f"со дня приёмки; о выявленных недостатках {our.nom} письменно уведомляет {them.acc}.",
            f"При обнаружении несоответствия Товара по качеству {our.nom} вправе заявить "
            f"{them.dat} претензию в течение {d} с даты приёмки Товара. Расхождения по "
            f"количеству оформляются актом по форме ТОРГ-2 в момент приёмки.",
        )
    if c.kind is Kind.SERVICES:
        return s.pick(
            f"В течение {d} с даты получения акта оказанных услуг {our.nom} подписывает акт или "
            f"направляет {them.dat} мотивированный отказ. Если в этот срок отказ не получен, "
            f"Услуги считаются принятыми.",
            f"{our.nom} рассматривает и подписывает акт сдачи-приёмки Услуг в течение {d} со дня "
            f"его получения либо в тот же срок направляет мотивированные возражения; при их "
            f"отсутствии Услуги считаются оказанными надлежащим образом.",
            f"Приёмка Услуг оформляется актом. Срок рассмотрения акта {our.ins} — "
            f"{s.d_acc(c.acceptance_days, 'working')}; молчание {our.gen} по истечении этого "
            f"срока означает приёмку Услуг.",
        )
    if c.kind is Kind.WORKS:
        return s.pick(
            f"{our.nom} в течение {d} с даты получения акта о приёмке выполненных работ (КС-2) "
            f"и справки о стоимости (КС-3) подписывает их или направляет {them.dat} "
            f"мотивированный отказ.",
            f"Сдача-приёмка Работ оформляется актом КС-2 и справкой КС-3, которые {our.nom} "
            f"обязан подписать в течение {d} со дня получения либо заявить мотивированный отказ.",
        )
    return s.pick(
        f"Помещение передаётся {our.dat} по акту приёма-передачи в течение {d} с даты "
        f"подписания Договора.",
        f"{them.nom} обязан передать Помещение по акту приёма-передачи не позднее {d} со дня "
        f"подписания Договора.",
        f"Передача Помещения оформляется актом приёма-передачи, который Стороны подписывают "
        f"в течение {d} с даты заключения Договора.",
    )


def _cap(penalty: Penalty, s: Style) -> str:
    if not penalty.cap_percent:
        return ""
    pct = s.pct(penalty.cap_percent)
    return s.pick(
        f", но не более {pct} от суммы задолженности",
        f"; общий размер неустойки не может превышать {pct} от неоплаченной суммы",
        f" (не более {pct} от просроченной суммы)",
    )


def clause_penalty_us(c: Contract, s: Style) -> str:
    our, them = c.roles
    penalty = c.penalty_us
    if penalty is None:
        return s.pick(
            "За неисполнение или ненадлежащее исполнение обязательств по Договору Стороны "
            "несут ответственность в соответствии с законодательством Российской Федерации.",
            "Ответственность Сторон за нарушение обязательств по Договору определяется "
            "действующим законодательством РФ; договорная неустойка не устанавливается.",
        )
    base = s.pick("от неоплаченной в срок суммы", "от суммы задолженности",
                  "от суммы просроченного платежа")  # fmt: skip
    if penalty.kind == "key_rate":
        rate = s.pick(
            "1/300 (одной трёхсотой) ключевой ставки Банка России, действующей на день "
            "уплаты пени,",
            "1/300 ключевой ставки ЦБ РФ",
        )
    else:
        rate = s.rate(penalty.rate)
    cap = _cap(penalty, s)
    return s.pick(
        f"За нарушение сроков оплаты {them.nom} вправе потребовать от {our.gen} уплаты неустойки "
        f"(пени) в размере {rate} {base} за каждый день просрочки{cap}.",
        f"В случае просрочки платежа {our.nom} уплачивает {them.dat} пени в размере {rate} "
        f"{base} за каждый день просрочки{cap}.",
        f"При задержке оплаты {our.nom} по требованию {them.gen} выплачивает неустойку — "
        f"{rate} {base} за каждый календарный день просрочки{cap}.",
    )


def clause_penalty_them(c: Contract, s: Style) -> str:
    assert c.penalty_them is not None
    our, them = c.roles
    rate = s.rate(c.penalty_them.rate)
    cap = (
        _cap(c.penalty_them, s)
        .replace("задолженности", "обязательства")
        .replace("неоплаченной суммы", "стоимости обязательства")
        .replace("просроченной суммы", "стоимости обязательства")
    )
    if c.kind is Kind.SUPPLY:
        return s.pick(
            f"За просрочку поставки {them.nom} уплачивает {our.dat} неустойку в размере {rate} "
            f"от стоимости не поставленного в срок Товара за каждый день просрочки{cap}.",
            f"Если {them.nom} нарушает срок поставки, {our.nom} вправе взыскать пени — {rate} "
            f"от стоимости непоставленного Товара за каждый день просрочки{cap}.",
            f"В случае просрочки поставки {them.nom} выплачивает неустойку в размере {rate} "
            f"стоимости несвоевременно поставленного Товара за каждый день просрочки{cap}.",
        )
    if c.kind is Kind.SERVICES:
        return s.pick(
            f"За нарушение сроков оказания Услуг {them.nom} уплачивает {our.dat} неустойку "
            f"в размере {rate} от стоимости Услуг, оказанных с нарушением срока, за каждый день "
            f"просрочки{cap}.",
            f"Если Услуги оказаны с опозданием, {our.nom} вправе потребовать пени — {rate} "
            f"от стоимости таких Услуг за каждый день просрочки{cap}.",
        )
    return s.pick(
        f"За нарушение срока окончания Работ {them.nom} уплачивает {our.dat} неустойку "
        f"в размере {rate} от стоимости Работ за каждый день просрочки{cap}.",
        f"При просрочке завершения Работ {our.nom} вправе взыскать с {them.gen} пени — {rate} "
        f"от цены Работ за каждый день просрочки{cap}.",
    )


def clause_validity(c: Contract, s: Style) -> str:
    v = c.validity
    if v.kind == "until_date":
        end = s.date(v.end)
        return s.pick(
            f"Договор вступает в силу с даты подписания и действует до {end}.",
            f"Срок действия Договора — с момента подписания по {end} включительно.",
            f"Настоящий Договор действует по {end}, а в части расчётов — до полного исполнения "
            f"Сторонами обязательств.",
        )
    if v.kind == "months":
        assert v.months is not None
        return s.pick(
            f"Договор заключён сроком на {s.months_acc(v.months)} и вступает в силу с даты "
            f"его подписания.",
            f"Срок действия Договора составляет {s.months_acc(v.months)} с даты подписания.",
            f"Договор действует в течение {s.months_gen(v.months)} с момента его подписания "
            f"Сторонами.",
        )
    return s.pick(
        "Договор вступает в силу с момента подписания и действует до полного исполнения "
        "Сторонами своих обязательств.",
        "Настоящий Договор действует с даты его подписания до момента исполнения Сторонами "
        "всех обязательств по нему.",
    )


def clause_prolongation(c: Contract, s: Style) -> str:
    pr = c.prolongation
    assert pr is not None
    if not pr.auto:
        return s.pick(
            "По истечении срока действия Договор прекращается. Продление срока оформляется "
            "дополнительным соглашением Сторон.",
            "Автоматическое продление (пролонгация) Договора не предусмотрено; продлить Договор "
            "можно только дополнительным соглашением.",
            "Договор не подлежит автоматической пролонгации. Для продолжения отношений Стороны "
            "заключают дополнительное соглашение.",
        )
    assert pr.notice_days is not None
    period = "на тот же срок" if pr.period == "same" else "на каждый следующий календарный год"
    n = pr.notice_days
    return s.pick(
        f"Если ни одна из Сторон не позднее чем за {s.d(n)} до окончания срока действия Договора "
        f"не заявит письменно о его прекращении, Договор считается продлённым {period} на тех "
        f"же условиях.",
        f"Договор автоматически пролонгируется {period}, если за {s.d_acc(n)} до истечения срока "
        f"его действия ни одна из Сторон не уведомит другую об отказе от продления.",
        f"При отсутствии письменного заявления одной из Сторон о прекращении Договора, "
        f"направленного не менее чем за {s.d(n)} до окончания срока, действие Договора "
        f"продлевается {period}. Количество пролонгаций не ограничено.",
    )


def clause_termination(c: Contract, s: Style) -> str:
    assert c.termination_days is not None
    n = c.termination_days
    return s.pick(
        f"Каждая из Сторон вправе в одностороннем порядке отказаться от исполнения Договора, "
        f"письменно уведомив другую Сторону не менее чем за {s.d(n)} до предполагаемой даты "
        f"расторжения.",
        f"Договор может быть расторгнут по инициативе любой из Сторон с письменным "
        f"предупреждением другой Стороны за {s.d_acc(n)}.",
        f"Односторонний отказ любой Стороны от Договора допускается при условии направления "
        f"уведомления не позднее чем за {s.d(n)} до даты прекращения Договора.",
    )


def clause_claims(c: Contract, s: Style) -> str:
    n = c.claim_days
    return s.pick(
        f"Споры разрешаются в претензионном порядке. Срок ответа на претензию — {s.d_acc(n)} "
        f"с даты её получения.",
        f"До обращения в суд Сторона направляет другой Стороне претензию; претензия "
        f"рассматривается в течение {s.d(n)} со дня получения.",
        f"Соблюдение претензионного порядка обязательно. Сторона, получившая претензию, обязана "
        f"ответить на неё в течение {s.d(n)} с момента получения.",
    )


def clause_court(c: Contract, s: Style) -> str:
    prep = c.court.replace("Арбитражный суд", "Арбитражном суде")
    ins = c.court.replace("Арбитражный суд", "Арбитражным судом")
    return s.pick(
        f"Если спор не урегулирован в претензионном порядке, он передаётся на рассмотрение "
        f"в {c.court}.",
        f"Споры, не урегулированные путём переговоров, подлежат рассмотрению в {prep}.",
        f"Все неурегулированные споры по Договору разрешаются {ins}.",
    )


def clause_force_majeure(c: Contract, s: Style) -> str:
    assert c.force_majeure_days is not None
    n = c.force_majeure_days
    return s.pick(
        f"Сторона, для которой создалась невозможность исполнения обязательств вследствие "
        f"обстоятельств непреодолимой силы, обязана уведомить другую Сторону в течение {s.d(n)} "
        f"с момента их наступления.",
        f"О наступлении форс-мажорных обстоятельств Сторона письменно извещает другую Сторону "
        f"не позднее {s.d(n)} со дня их возникновения.",
        f"При наступлении обстоятельств непреодолимой силы Сторона должна известить другую "
        f"Сторону в срок не более {s.d(n)}; несвоевременное извещение лишает её права "
        f"ссылаться на эти обстоятельства.",
    )


def requisites_block(entity: Entity, role: str, bank_name: str, bik: str, corr: str,
                     account: str) -> str:  # fmt: skip
    kpp = f", КПП {entity.kpp}" if entity.kpp else ""
    return (
        f"**{role}:** {entity.name}\n"
        f"ИНН {entity.inn}{kpp}\n"
        f"Адрес: {entity.address}\n"
        f"р/с {account} в {bank_name}\n"
        f"БИК {bik}, к/с {corr}"
    )


def clause_requisites_them(c: Contract, s: Style) -> str:
    _, them = c.roles
    b = c.them_bank
    return requisites_block(c.them, them.nom, b.name, b.bik, b.corr_account, c.them_account)


# Факт → функция печати пункта (нужна и доп. соглашениям: «изложить в редакции»).
CLAUSE_RENDERERS = {
    F.SUBJECT: clause_subject,
    F.PREMISES: clause_premises,
    F.PRICE: clause_price,
    F.PAYMENT: clause_payment,
    F.DEPOSIT: clause_deposit,
    F.INDEXATION: clause_indexation,
    F.UTILITIES: clause_utilities,
    F.RETENTION: clause_retention,
    F.DELIVERY: clause_delivery,
    F.WORKS_PERIOD: clause_works_period,
    F.WARRANTY: clause_warranty,
    F.ACCEPTANCE: clause_acceptance,
    F.PENALTY_US: clause_penalty_us,
    F.PENALTY_THEM: clause_penalty_them,
    F.VALIDITY: clause_validity,
    F.PROLONGATION: clause_prolongation,
    F.TERMINATION: clause_termination,
    F.CLAIMS: clause_claims,
    F.COURT: clause_court,
    F.FORCE_MAJEURE: clause_force_majeure,
    F.REQUISITES: clause_requisites_them,
}


def fact_clause(c: Contract, fact: str, s: Style) -> Clause:
    return Clause(CLAUSE_RENDERERS[fact](c, s), (fact,))


# --- Пункты без фактов (наполнение, как в настоящих договорах) ---------------------------


def _filler(s: Style, options: list[str], k: int) -> list[Clause]:
    k = min(k, len(options))
    return [Clause(text) for text in s.rng.sample(options, k)]


def render_contract(c: Contract, s: Style) -> CorpusDoc:
    """Договор целиком: разделы, пункты, реквизиты сторон."""
    our, them = c.roles
    sections: list[Section] = []

    subject = [fact_clause(c, F.SUBJECT, s)]
    if c.kind is Kind.LEASE:
        subject.append(fact_clause(c, F.PREMISES, s))
    subject += _filler(
        s,
        {
            Kind.SUPPLY: [
                "Наименование, количество, ассортимент и цена Товара согласуются Сторонами "
                "в спецификациях (заявках), являющихся неотъемлемой частью Договора.",
                f"Ассортимент и количество каждой партии определяются заявкой {our.gen}.",
            ],
            Kind.SERVICES: [
                "Перечень, объём и сроки оказания Услуг согласуются Сторонами в заявках.",
                f"{them.nom} оказывает Услуги лично или с привлечением третьих лиц, оставаясь "
                f"ответственным перед {our.ins} за их действия.",
            ],
            Kind.LEASE: [
                f"Помещение принадлежит {them.dat} на праве собственности.",
                "Помещение предоставляется для использования под офис; изменение назначения "
                "не допускается.",
            ],
            Kind.WORKS: [
                "Объём и содержание Работ определяются техническим заданием (Приложение № 1).",
                f"Работы выполняются иждивением {them.gen}: из его материалов, его силами "
                f"и средствами.",
            ],
        }[c.kind],
        s.rng.randint(1, 2),
    )
    sections.append(Section(s.pick("Предмет договора", "Предмет Договора", "Общие положения"),
                            subject))  # fmt: skip

    price: list[Clause] = []
    if c.price is not None:
        price.append(fact_clause(c, F.PRICE, s))
    elif c.kind is Kind.SUPPLY:
        price.append(
            Clause(
                f"Цена Товара указывается в спецификациях и счетах {them.gen} и включает НДС "
                f"по ставке, установленной законодательством РФ."
            )
        )
    else:
        price.append(
            Clause(
                f"Стоимость Услуг определяется по тарифам {them.gen}, действующим на дату "
                f"заявки, и указывается в актах оказанных услуг."
            )
        )
    price.append(fact_clause(c, F.PAYMENT, s))
    if c.kind is Kind.LEASE:
        price.append(fact_clause(c, F.UTILITIES, s))
        if c.deposit is not None:
            price.append(fact_clause(c, F.DEPOSIT, s))
        if c.indexation is not None:
            price.append(fact_clause(c, F.INDEXATION, s))
    if c.kind is Kind.WORKS and c.retention_percent is not None:
        price.append(fact_clause(c, F.RETENTION, s))
    price += _filler(
        s,
        [
            f"Датой оплаты считается дата списания денежных средств с расчётного счёта {our.gen}.",
            "Все расчёты по Договору производятся в рублях в безналичном порядке.",
            "Стороны ежеквартально проводят сверку взаиморасчётов с подписанием акта сверки.",
        ],
        1,
    )
    sections.append(
        Section(
            s.pick(
                "Цена и порядок расчётов",
                "Стоимость и порядок оплаты",
                "Цена Договора и расчёты",
                "Порядок расчётов",
            ),
            price,
        )
    )

    if c.kind is Kind.SUPPLY:
        sections.append(
            Section(
                s.pick("Порядок и сроки поставки", "Условия поставки", "Поставка Товара"),
                [
                    fact_clause(c, F.DELIVERY, s),
                    Clause(
                        f"Право собственности на Товар и риск его случайной гибели переходят "
                        f"к {our.dat} с момента подписания УПД (товарной накладной)."
                    ),
                ],
            )
        )
    elif c.kind is Kind.WORKS:
        sections.append(
            Section(
                s.pick("Сроки выполнения работ", "Сроки и порядок выполнения работ"),
                [
                    fact_clause(c, F.WORKS_PERIOD, s),
                    Clause(f"{them.nom} вправе выполнить Работы досрочно с согласия {our.gen}."),
                ],
            )
        )
    elif c.kind is Kind.SERVICES:
        sections.append(
            Section(
                "Порядок оказания услуг",
                _filler(
                    s,
                    [
                        "Услуги оказываются в рабочие дни с 9:00 до 18:00 по местному времени.",
                        f"{them.nom} ежемесячно представляет {our.dat} отчёт об оказанных "
                        f"Услугах вместе с актом.",
                        f"{our.nom} обеспечивает {them.dat} доступ к оборудованию и помещениям, "
                        f"необходимый для оказания Услуг.",
                    ],
                    2,
                ),
            )
        )
    else:
        sections.append(
            Section(
                "Права и обязанности сторон",
                _filler(
                    s,
                    [
                        f"{our.nom} вправе производить перепланировку Помещения только "
                        f"с письменного согласия {them.gen}.",
                        f"Текущий ремонт Помещения производит {our.nom} за свой счёт, "
                        f"капитальный — {them.nom}.",
                        f"{our.nom} не вправе сдавать Помещение в субаренду без согласия "
                        f"{them.gen}.",
                    ],
                    2,
                ),
            )
        )

    acceptance_title = {
        Kind.SUPPLY: ("Приёмка Товара", "Порядок приёмки"),
        Kind.SERVICES: ("Сдача и приёмка услуг", "Порядок приёмки услуг"),
        Kind.WORKS: ("Сдача и приёмка работ", "Порядок сдачи и приёмки работ"),
        Kind.LEASE: ("Передача помещения", "Порядок передачи помещения"),
    }[c.kind]
    sections.append(Section(s.pick(*acceptance_title), [fact_clause(c, F.ACCEPTANCE, s)]))
    if c.kind is Kind.WORKS:
        sections.append(
            Section(
                s.pick("Гарантии качества", "Гарантийные обязательства"),
                [
                    fact_clause(c, F.WARRANTY, s),
                    Clause(
                        f"Недостатки, выявленные в гарантийный срок, {them.nom} устраняет "
                        f"за свой счёт в согласованный Сторонами срок."
                    ),
                ],
            )
        )

    responsibility = [fact_clause(c, F.PENALTY_US, s)]
    if c.penalty_them is not None:
        responsibility.append(fact_clause(c, F.PENALTY_THEM, s))
    if c.penalty_us is not None:
        responsibility += _filler(
            s,
            [
                "Уплата неустойки не освобождает Стороны от исполнения обязательств по Договору.",
                "Неустойка начисляется только на основании письменного требования Стороны.",
            ],
            1,
        )
    resp_section = Section(
        s.pick("Ответственность сторон", "Ответственность Сторон", "Санкции"), responsibility
    )

    validity = [fact_clause(c, F.VALIDITY, s)]
    if c.prolongation is not None:
        validity.append(fact_clause(c, F.PROLONGATION, s))
    if c.termination_days is not None:
        validity.append(fact_clause(c, F.TERMINATION, s))
    validity_section = Section(
        s.pick(
            "Срок действия договора",
            "Срок действия и порядок расторжения",
            "Срок действия Договора",
        ),
        validity,
    )
    if s.rng.random() < 0.3:
        sections += [validity_section, resp_section]
    else:
        sections += [resp_section, validity_section]

    if c.force_majeure_days is not None:
        sections.append(
            Section(
                s.pick("Обстоятельства непреодолимой силы", "Форс-мажор"),
                [
                    fact_clause(c, F.FORCE_MAJEURE, s),
                    Clause(
                        "Наступление обстоятельств непреодолимой силы подтверждается документом "
                        "уполномоченного органа или Торгово-промышленной палаты."
                    ),
                ],
            )
        )
    sections.append(
        Section(
            s.pick(
                "Разрешение споров",
                "Порядок разрешения споров",
                "Претензионный порядок и подсудность",
            ),
            [fact_clause(c, F.CLAIMS, s), fact_clause(c, F.COURT, s)],
        )
    )
    if s.rng.random() < 0.3:
        sections.append(
            Section(
                "Конфиденциальность",
                [
                    Clause(
                        "Стороны обязуются не разглашать условия Договора и сведения, полученные "
                        "при его исполнении, без письменного согласия другой Стороны."
                    )
                ],
            )
        )
    sections.append(
        Section(
            s.pick("Заключительные положения", "Прочие условия"),
            _filler(
                s,
                [
                    "Все изменения и дополнения к Договору действительны, если они совершены "
                    "в письменной форме и подписаны уполномоченными представителями Сторон.",
                    "Договор составлен в двух экземплярах, имеющих равную юридическую силу, "
                    "по одному для каждой из Сторон.",
                    "Стороны допускают обмен документами по Договору через систему "
                    "электронного документооборота с использованием квалифицированной "
                    "электронной подписи.",
                    "Во всём, что не предусмотрено Договором, Стороны руководствуются "
                    "законодательством Российской Федерации.",
                    "Сторона, изменившая адрес или банковские реквизиты, обязана уведомить "
                    "об этом другую Сторону в течение 5 рабочих дней.",
                ],
                s.rng.randint(2, 3),
            ),
        )
    )
    our_bank = c.our.bank
    sections.append(
        Section(
            s.pick("Адреса и реквизиты сторон", "Реквизиты и подписи Сторон"),
            [
                Clause(clause_requisites_them(c, s), (F.REQUISITES,), label="req-1"),
                Clause(
                    requisites_block(
                        c.our,
                        our.nom,
                        our_bank.name,
                        our_bank.bik,
                        our_bank.corr_account,
                        c.our.account,
                    ),
                    label="req-2",
                ),
            ],
            numbered=False,
        )
    )

    intro_their = party_intro(c.them, them.nom, s)
    intro_our = party_intro(c.our, our.nom, s)
    preamble = s.pick(
        f"{capitalize(intro_our)}, с одной стороны, и {intro_their}, с другой стороны, совместно "
        f"именуемые «Стороны», заключили настоящий Договор (далее — «Договор») о нижеследующем:",
        f"{capitalize(intro_their)} и {intro_our}, далее вместе — «Стороны», заключили настоящий "
        f"договор (далее — Договор) о следующем:",
        f"{capitalize(intro_our)} и {intro_their} заключили настоящий Договор на следующих "
        f"условиях:",
    )
    preamble = tidy_commas(preamble)
    header = [
        f"{s.pick(*HEADERS[c.kind])} № {c.number}",
        f"г. {c.city}, {s.date(c.date)}",
    ]
    doc = CorpusDoc(
        doc_id=c.contract_id,
        doc_kind="contract",
        header=header,
        preamble=preamble,
        sections=sections,
        meta=contract_meta(c),
    )
    doc.render()
    return doc


def contract_meta(c: Contract) -> dict[str, object]:
    return {
        "contract_kind": c.kind.value,
        "number": c.number,
        "date": c.date.isoformat(),
        "counterparty": c.them.name,
        "counterparty_inn": c.them.inn,
        "city": c.city,
    }


def with_style(s: Style, rng: random.Random) -> Style:
    """Тот же стиль с другим генератором (для текста доп. соглашения)."""
    return replace(s, rng=rng)
