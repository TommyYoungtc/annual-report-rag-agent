from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ChangeResult:
    old_value: Decimal
    new_value: Decimal
    absolute_change: Decimal
    percentage_change: Decimal | None


def calculate_change(old_value: Decimal, new_value: Decimal) -> ChangeResult:
    absolute = new_value - old_value
    percentage = None
    if old_value != 0:
        percentage = absolute / abs(old_value) * Decimal(100)
    return ChangeResult(old_value, new_value, absolute, percentage)


def format_decimal(value: Decimal, *, places: int | None = None) -> str:
    if places is not None:
        value = value.quantize(Decimal(1).scaleb(-places))
    text = format(value, "f")
    if "." in text:
        integer, fraction = text.split(".", 1)
        return f"{int(integer):,}.{fraction}"
    return f"{int(text):,}"
