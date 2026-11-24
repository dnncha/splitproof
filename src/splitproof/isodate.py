"""The one date reader splitproof uses.

`date.fromisoformat` accepts basic and week dates on Python 3.11 and later and refuses
them on 3.10, so the same file split differently by interpreter. Every date passes
through `calendar_day`, which accepts exactly one spelling on every Python.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Optional

ISO_DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)


def calendar_day(text: str) -> Optional[_dt.date]:
    if not ISO_DATE.fullmatch(text):
        return None
    try:
        return _dt.date(int(text[0:4]), int(text[5:7]), int(text[8:10]))
    except ValueError:
        return None
