from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Literal,
    TypeVar,
    overload,
)

import numpy as np

from pandas._typing import npt

from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.offsets import BaseOffset, BusinessMixin

if TYPE_CHECKING:
    from pandas.core.indexes.datetimes import DatetimeIndex
_BaseOffsetT = TypeVar("_BaseOffsetT", bound=BaseOffset)
_DatetimeT = TypeVar("_DatetimeT", bound=datetime)
_TimedeltaT = TypeVar("_TimedeltaT", bound=timedelta)

_relativedelta_kwds: set[str]
prefix_mapping: dict[str, type]

class BusinessDay(BusinessMixin): ...

class BusinessHour(BusinessMixin):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        start: str | Collection[str] = ...,
        end: str | Collection[str] = ...,
        offset: timedelta = ...,
    ) -> None: ...

class CustomBusinessDay(BusinessDay):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekmask: str = ...,
        holidays: list | None = ...,
        calendar: np.busdaycalendar | None = ...,
        offset: timedelta = ...,
    ) -> None: ...

class CustomBusinessHour(BusinessHour):
    def __init__(
        self,
        n: int = ...,
        normalize: bool = ...,
        weekmask: str = ...,
        holidays: list | None = ...,
        calendar: np.busdaycalendar | None = ...,
        start: str = ...,
        end: str = ...,
        offset: timedelta = ...,
    ) -> None: ...

class CustomBusinessIntradayOffset(BusinessHour):
    def __init__(
        self,
        n: int = ...,
        step: timedelta = ...,
        normalize: bool = ...,
        weekmask: str = ...,
        holidays: list | None = ...,
        calendar: np.busdaycalendar | None = ...,
        start: str = ...,
        end: str = ...,
        offset: timedelta = ...,
    ) -> None: ...
    step: timedelta




