"""Numbers, ranges, units and names as written in Chinese process patents.

Input is normalized text (normalize_text), so decimals are "0.5", ranges use "-" or
"~", and pH is spelled "pH".
"""

import re
from dataclasses import dataclass

NUMBER = r"\d+(?:\.\d+)?"
# "45-65", "50℃-80℃" (unit repeated inside the range), "0.5~0.9", "5至7"
QUANTITY = rf"(?P<low>{NUMBER})\s*(?:℃|%)?\s*(?:[-~至到]\s*(?P<high>{NUMBER}))?"

UNITS: dict[str, str] = {
    "份": "份",
    "重量份": "份",
    "质量份": "份",
    "℃": "℃",
    "°C": "℃",
    "转/分钟": "rpm",
    "转/分": "rpm",
    "r/min": "rpm",
    "rpm": "rpm",
    "分钟": "min",
    "min": "min",
    "小时": "h",
    "h": "h",
    "秒": "s",
    "%": "%",
    "g": "g",
    "kg": "kg",
    "mL": "mL",
    "ml": "mL",
    "L": "L",
    "μm": "μm",
    "nm": "nm",
}

# Names that a document writes in more than one way. Keys are what was written.
NAME_ALIASES: dict[str, str] = {
    "去离子": "去离子水",   # CN116694108A claim 2 / [0008] omit "水"
    "纯水": "去离子水",
    "TiO2": "二氧化钛",
    "钛白": "二氧化钛",
}


@dataclass(frozen=True)
class Quantity:
    low: float
    high: float

    @property
    def is_range(self) -> bool:
        return self.low != self.high

    @property
    def value(self) -> float | None:
        return None if self.is_range else self.low


def parse_quantity(low: str, high: str | None) -> Quantity:
    a = float(low)
    b = float(high) if high is not None else a
    return Quantity(min(a, b), max(a, b))


def canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return UNITS.get(unit.strip(), unit.strip())


def canonical_name(name: str) -> tuple[str, str | None]:
    """(canonical name, raw name if it was changed)."""
    name = name.strip()
    alias = NAME_ALIASES.get(name)
    return (alias, name) if alias else (name, None)


def parse_number(text: str) -> float | None:
    """A cell such as "1.21" or "1. 21" -> 1.21; anything else -> None."""
    compact = re.sub(r"\s+", "", text)
    return float(compact) if re.fullmatch(NUMBER, compact) else None


def normalize_chemical_name(name: str) -> str:
    """"2，6-二叔丁基甲酚" -> "2,6-二叔丁基甲酚" (locants use an ASCII comma)."""
    return re.sub(r"(?<=\d)，(?=\d)", ",", name.strip())
