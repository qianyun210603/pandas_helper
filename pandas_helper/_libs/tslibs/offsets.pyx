import operator
import re
import time
import warnings

import cython

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDateTime_IMPORT,
    PyDelta_Check,
    date,
    datetime,
    time as dt_time,
    timedelta,
)

PyDateTime_IMPORT

from dateutil.easter import easter
from dateutil.relativedelta import relativedelta
import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    ndarray,
)

cnp.import_array()

# TODO: formalize having _libs.properties "above" tslibs in the dependency structure

from pandas._libs.properties import cache_readonly

from pandas._libs.tslibs cimport util
from pandas._libs.tslibs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.ccalendar import (
    MONTH_ALIASES,
    MONTH_TO_CAL_NUM,
    int_to_weekday,
    weekday_to_int,
)

from pandas._libs.tslibs.ccalendar cimport (
    DAY_NANOS,
    dayofweek,
    get_days_in_month,
    get_firstbday,
    get_lastbday,
)
from pandas._libs.tslibs.conversion cimport (
    convert_datetime_to_tsobject,
    localize_pydatetime,
)
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    dt64_to_dtstruct,
    dtstruct_to_dt64,
    npy_datetimestruct,
    pydate_to_dtstruct,
)
from pandas._libs.tslibs.tzconversion cimport tz_convert_from_utc_single

from pandas._libs.tslibs.dtypes cimport PeriodDtypeCode
from pandas._libs.tslibs.timedeltas cimport (
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from pandas._libs.tslibs.timedeltas import Timedelta

from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs.tslibs.timestamps import Timestamp

from pandas._libs.tslibs.offsets import delta_to_tick
from pandas._libs.tslibs.offsets cimport BaseOffset

# ---------------------------------------------------------------------
# Misc Helpers

cdef bint is_offset_object(object obj):
    return isinstance(obj, BaseOffset)


cdef datetime _as_datetime(datetime obj):
    if isinstance(obj, _Timestamp):
        return obj.to_pydatetime()
    return obj


cdef bint _is_normalized(datetime dt):
    if dt.hour != 0 or dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        # Regardless of whether dt is datetime vs Timestamp
        return False
    if isinstance(dt, _Timestamp):
        return dt.nanosecond == 0
    return True


def apply_wrapper_core(func, self, other) -> ndarray:
    result = func(self, other)
    result = np.asarray(result)

    if self.normalize:
        # TODO: Avoid circular/runtime import
        from .vectorized import normalize_i8_timestamps
        result = normalize_i8_timestamps(result.view("i8"), None)

    return result


def apply_index_wraps(func):
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods
    def wrapper(self, other):
        # other is a DatetimeArray
        result = apply_wrapper_core(func, self, other)
        result = type(other)(result)
        warnings.warn("'Offset.apply_index(other)' is deprecated. "
                      "Use 'offset + other' instead.", FutureWarning)
        return result

    return wrapper


def apply_array_wraps(func):
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods
    def wrapper(self, other) -> np.ndarray:
        # other is a DatetimeArray
        result = apply_wrapper_core(func, self, other)
        return result

    # do @functools.wraps(func) manually since it doesn't work on cdef funcs
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def apply_wraps(func):
    # Note: normally we would use `@functools.wraps(func)`, but this does
    # not play nicely with cython class methods

    def wrapper(self, other):

        if other is NaT:
            return NaT
        elif (
            isinstance(other, BaseOffset)
            or PyDelta_Check(other)
            or util.is_timedelta64_object(other)
        ):
            # timedelta path
            return func(self, other)
        elif is_datetime64_object(other) or PyDate_Check(other):
            # PyDate_Check includes date, datetime
            other = Timestamp(other)
        else:
            # This will end up returning NotImplemented back in __add__
            raise ApplyTypeError

        tz = other.tzinfo
        nano = other.nanosecond

        if self._adjust_dst:
            other = other.tz_localize(None)

        result = func(self, other)

        result = Timestamp(result)
        if self._adjust_dst:
            result = result.tz_localize(tz)

        if self.normalize:
            result = result.normalize()

        # If the offset object does not have a nanoseconds component,
        # the result's nanosecond component may be lost.
        if not self.normalize and nano != 0 and not hasattr(self, "nanoseconds"):
            if result.nanosecond != nano:
                if result.tz is not None:
                    # convert to UTC
                    value = result.tz_localize(None).value
                else:
                    value = result.value
                result = Timestamp(value + nano)

        if tz is not None and result.tzinfo is None:
            result = result.tz_localize(tz)

        return result

    # do @functools.wraps(func) manually since it doesn't work on cdef funcs
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


cdef _wrap_timedelta_result(result):
    """
    Tick operations dispatch to their Timedelta counterparts.  Wrap the result
    of these operations in a Tick if possible.

    Parameters
    ----------
    result : object

    Returns
    -------
    object
    """
    if PyDelta_Check(result):
        # convert Timedelta back to a Tick
        return delta_to_tick(result)

    return result

# ---------------------------------------------------------------------
# Business Helpers


cdef _get_calendar(weekmask, holidays, calendar):
    """
    Generate busdaycalendar
    """
    if isinstance(calendar, np.busdaycalendar):
        if not holidays:
            holidays = tuple(calendar.holidays)
        elif not isinstance(holidays, tuple):
            holidays = tuple(holidays)
        else:
            # trust that calendar.holidays and holidays are
            # consistent
            pass
        return calendar, holidays

    if holidays is None:
        holidays = []
    try:
        holidays = holidays + calendar.holidays().tolist()
    except AttributeError:
        pass
    holidays = [_to_dt64D(dt) for dt in holidays]
    holidays = tuple(sorted(holidays))

    kwargs = {'weekmask': weekmask}
    if holidays:
        kwargs['holidays'] = holidays

    busdaycalendar = np.busdaycalendar(**kwargs)
    return busdaycalendar, holidays


cdef _to_dt64D(dt):
    # Currently
    # > np.datetime64(dt.datetime(2013,5,1),dtype='datetime64[D]')
    # numpy.datetime64('2013-05-01T02:00:00.000000+0200')
    # Thus astype is needed to cast datetime to datetime64[D]
    if getattr(dt, 'tzinfo', None) is not None:
        # Get the nanosecond timestamp,
        #  equiv `Timestamp(dt).value` or `dt.timestamp() * 10**9`
        nanos = getattr(dt, "nanosecond", 0)
        i8 = convert_datetime_to_tsobject(dt, tz=None, nanos=nanos).value
        dt = tz_convert_from_utc_single(i8, dt.tzinfo)
        dt = np.int64(dt).astype('datetime64[ns]')
    else:
        dt = np.datetime64(dt)
    if dt.dtype.name != "datetime64[D]":
        dt = dt.astype("datetime64[D]")
    return dt


# ---------------------------------------------------------------------
# Validation


cdef _validate_business_time(t_input):
    if isinstance(t_input, str):
        try:
            t = time.strptime(t_input, '%H:%M')
            return dt_time(hour=t.tm_hour, minute=t.tm_min)
        except ValueError:
            raise ValueError("time data must match '%H:%M' format")
    elif isinstance(t_input, dt_time):
        if t_input.second != 0 or t_input.microsecond != 0:
            raise ValueError(
                "time data must be specified only with hour and minute")
        return t_input
    else:
        raise ValueError("time data must be string or datetime.time")


# ---------------------------------------------------------------------
# Constructor Helpers

_relativedelta_kwds = {"years", "months", "weeks", "days", "year", "month",
                       "day", "weekday", "hour", "minute", "second",
                       "microsecond", "nanosecond", "nanoseconds", "hours",
                       "minutes", "seconds", "microseconds"}


cdef _determine_offset(kwds):
    # timedelta is used for sub-daily plural offsets and all singular
    # offsets relativedelta is used for plural offsets of daily length or
    # more nanosecond(s) are handled by apply_wraps
    kwds_no_nanos = dict(
        (k, v) for k, v in kwds.items()
        if k not in ('nanosecond', 'nanoseconds')
    )
    # TODO: Are nanosecond and nanoseconds allowed somewhere?

    _kwds_use_relativedelta = ('years', 'months', 'weeks', 'days',
                               'year', 'month', 'week', 'day', 'weekday',
                               'hour', 'minute', 'second', 'microsecond')

    use_relativedelta = False
    if len(kwds_no_nanos) > 0:
        if any(k in _kwds_use_relativedelta for k in kwds_no_nanos):
            offset = relativedelta(**kwds_no_nanos)
            use_relativedelta = True
        else:
            # sub-daily offset - use timedelta (tz-aware)
            offset = timedelta(**kwds_no_nanos)
    elif any(nano in kwds for nano in ('nanosecond', 'nanoseconds')):
        offset = timedelta(days=0)
    else:
        # GH 45643/45890: (historically) defaults to 1 day for non-nano
        # since datetime.timedelta doesn't handle nanoseconds
        offset = timedelta(days=1)
    return offset, use_relativedelta


# ---------------------------------------------------------------------
# Mixins & Singletons


class ApplyTypeError(TypeError):
    # sentinel class for catching the apply error to return NotImplemented
    pass

# ---------------------------------------------------------------------
# Base Classes

cdef class SingleConstructorOffset(BaseOffset):
    @classmethod
    def _from_name(cls, suffix=None):
        # default _from_name calls cls with no args
        if suffix:
            raise ValueError(f"Bad freq suffix {suffix}")
        return cls()

    def __reduce__(self):
        # This __reduce__ implementation is for all BaseOffset subclasses
        #  except for RelativeDeltaOffset
        # np.busdaycalendar objects do not pickle nicely, but we can reconstruct
        #  from attributes that do get pickled.
        tup = tuple(
            getattr(self, attr) if attr != "calendar" else None
            for attr in self._attributes
        )
        return type(self), tup

# --------------------------------------------------------------------


cdef class BusinessMixin(SingleConstructorOffset):
    """
    Mixin to business types to provide related functions.
    """

    cdef readonly:
        timedelta _offset
        # Only Custom subclasses use weekmask, holiday, calendar
        object weekmask, holidays, calendar

    def __init__(self, n=1, normalize=False, offset=timedelta(0)):
        BaseOffset.__init__(self, n, normalize)
        self._offset = offset

    cpdef _init_custom(self, weekmask, holidays, calendar):
        """
        Additional __init__ for Custom subclasses.
        """
        calendar, holidays = _get_calendar(
            weekmask=weekmask, holidays=holidays, calendar=calendar
        )
        # Custom offset instances are identified by the
        # following two attributes. See DateOffset._params()
        # holidays, weekmask
        self.weekmask = weekmask
        self.holidays = holidays
        self.calendar = calendar

    @property
    def offset(self):
        """
        Alias for self._offset.
        """
        # Alias for backward compat
        return self._offset

    def _repr_attrs(self) -> str:
        if self.offset:
            attrs = [f"offset={repr(self.offset)}"]
        else:
            attrs = []
        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    cpdef __setstate__(self, state):
        # We need to use a cdef/cpdef method to set the readonly _offset attribute
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            # Older (<0.22.0) versions have offset attribute instead of _offset
            self._offset = state.pop("offset")

        if self._prefix.startswith("C"):
            # i.e. this is a Custom class
            weekmask = state.pop("weekmask")
            holidays = state.pop("holidays")
            calendar, holidays = _get_calendar(weekmask=weekmask,
                                               holidays=holidays,
                                               calendar=None)
            self.weekmask = weekmask
            self.calendar = calendar
            self.holidays = holidays

        BaseOffset.__setstate__(self, state)


cdef class BusinessDay(BusinessMixin):
    """
    DateOffset subclass representing possibly n business days.
    """
    _period_dtype_code = PeriodDtypeCode.B
    _prefix = "B"
    _attributes = tuple(["n", "normalize", "offset"])

    cpdef __setstate__(self, state):
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        if "_offset" in state:
            self._offset = state.pop("_offset")
        elif "offset" in state:
            self._offset = state.pop("offset")
        self._cache = state.pop("_cache", {})

    def _offset_str(self) -> str:
        def get_str(td):
            off_str = ""
            if td.days > 0:
                off_str += str(td.days) + "D"
            if td.seconds > 0:
                s = td.seconds
                hrs = int(s / 3600)
                if hrs != 0:
                    off_str += str(hrs) + "H"
                    s -= hrs * 3600
                mts = int(s / 60)
                if mts != 0:
                    off_str += str(mts) + "Min"
                    s -= mts * 60
                if s != 0:
                    off_str += str(s) + "s"
            if td.microseconds > 0:
                off_str += str(td.microseconds) + "us"
            return off_str

        if PyDelta_Check(self.offset):
            zero = timedelta(0, 0, 0)
            if self.offset >= zero:
                off_str = "+" + get_str(self.offset)
            else:
                off_str = "-" + get_str(-self.offset)
            return off_str
        else:
            return "+" + repr(self.offset)

    @apply_wraps
    def _apply(self, other):
        if PyDateTime_Check(other):
            n = self.n
            wday = other.weekday()

            # avoid slowness below by operating on weeks first
            weeks = n // 5
            if n <= 0 and wday > 4:
                # roll forward
                n += 1

            n -= 5 * weeks

            # n is always >= 0 at this point
            if n == 0 and wday > 4:
                # roll back
                days = 4 - wday
            elif wday > 4:
                # roll forward
                days = (7 - wday) + (n - 1)
            elif wday + n <= 4:
                # shift by n days without leaving the current week
                days = n
            else:
                # shift by n days plus 2 to get past the weekend
                days = n + 2

            result = other + timedelta(days=7 * weeks + days)
            if self.offset:
                result = result + self.offset
            return result

        elif is_any_td_scalar(other):
            td = Timedelta(self.offset) + other
            return BusinessDay(
                self.n, offset=td.to_pytimedelta(), normalize=self.normalize
            )
        else:
            raise ApplyTypeError(
                "Only know how to combine business day with datetime or timedelta."
            )

    @apply_index_wraps
    def apply_index(self, dtindex):
        return self._apply_array(dtindex)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        i8other = dtarr.view("i8")
        res = _shift_bdays(i8other, self.n)
        if self.offset:
            res = res.view("M8[ns]") + Timedelta(self.offset)
            res = res.view("i8")
        return res

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return dt.weekday() < 5


cdef class BusinessHour(BusinessMixin):
    """
    DateOffset subclass representing possibly n business hours.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    start : str, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, default: "17:00"
        End time of your custom business hour in 24h format.
    """

    _prefix = "BH"
    _anchor = 0
    _attributes = tuple(["n", "normalize", "start", "end", "offset"])
    _adjust_dst = False

    cdef readonly:
        tuple start, end

    def __init__(
            self, n=1, normalize=False, start="09:00", end="17:00", offset=timedelta(0)
    ):
        BusinessMixin.__init__(self, n, normalize, offset)

        # must be validated here to equality check
        if np.ndim(start) == 0:
            # i.e. not is_list_like
            start = [start]
        if not len(start):
            raise ValueError("Must include at least 1 start time")

        if np.ndim(end) == 0:
            # i.e. not is_list_like
            end = [end]
        if not len(end):
            raise ValueError("Must include at least 1 end time")

        start = np.array([_validate_business_time(x) for x in start])
        end = np.array([_validate_business_time(x) for x in end])

        # Validation of input
        if len(start) != len(end):
            raise ValueError("number of starting time and ending time must be the same")
        num_openings = len(start)

        # sort starting and ending time by starting time
        index = np.argsort(start)

        # convert to tuple so that start and end are hashable
        start = tuple(start[index])
        end = tuple(end[index])

        total_secs = 0
        for i in range(num_openings):
            total_secs += self._get_business_hours_by_sec(start[i], end[i])
            total_secs += self._get_business_hours_by_sec(
                end[i], start[(i + 1) % num_openings]
            )
        if total_secs != 24 * 60 * 60:
            raise ValueError(
                "invalid starting and ending time(s): "
                "opening hours should not touch or overlap with "
                "one another"
            )

        self.start = start
        self.end = end

    cpdef __setstate__(self, state):
        start = state.pop("start")
        start = (start,) if np.ndim(start) == 0 else tuple(start)
        end = state.pop("end")
        end = (end,) if np.ndim(end) == 0 else tuple(end)
        self.start = start
        self.end = end

        state.pop("kwds", {})
        state.pop("next_bday", None)
        BusinessMixin.__setstate__(self, state)

    def _repr_attrs(self) -> str:
        out = super()._repr_attrs()
        hours = ",".join(
            f'{st.strftime("%H:%M")}-{en.strftime("%H:%M")}'
            for st, en in zip(self.start, self.end)
        )
        attrs = [f"{self._prefix}={hours}"]
        out += ": " + ", ".join(attrs)
        return out

    def _get_business_hours_by_sec(self, start, end):
        """
        Return business hours in a day by seconds.
        """
        # create dummy datetime to calculate business hours in a day
        dtstart = datetime(2014, 4, 1, start.hour, start.minute)
        day = 1 if start < end else 2
        until = datetime(2014, 4, day, end.hour, end.minute)
        return int((until - dtstart).total_seconds())

    def _get_closing_time(self, dt: datetime) -> datetime:
        """
        Get the closing time of a business hour interval by its opening time.

        Parameters
        ----------
        dt : datetime
            Opening time of a business hour interval.

        Returns
        -------
        result : datetime
            Corresponding closing time.
        """
        for i, st in enumerate(self.start):
            if st.hour == dt.hour and st.minute == dt.minute:
                return dt + timedelta(
                    seconds=self._get_business_hours_by_sec(st, self.end[i])
                )
        assert False

    @cache_readonly
    def next_bday(self):
        """
        Used for moving to next business day.
        """
        if self.n >= 0:
            nb_offset = 1
        else:
            nb_offset = -1
        if self._prefix.startswith("C"):
            # CustomBusinessHour
            return CustomBusinessDay(
                n=nb_offset,
                weekmask=self.weekmask,
                holidays=self.holidays,
                calendar=self.calendar,
            )
        else:
            return BusinessDay(n=nb_offset)

    def _next_opening_time(self, other, sign=1):
        """
        If self.n and sign have the same sign, return the earliest opening time
        later than or equal to current time.
        Otherwise the latest opening time earlier than or equal to current
        time.

        Opening time always locates on BusinessDay.
        However, closing time may not if business hour extends over midnight.

        Parameters
        ----------
        other : datetime
            Current time.
        sign : int, default 1.
            Either 1 or -1. Going forward in time if it has the same sign as
            self.n. Going backward in time otherwise.

        Returns
        -------
        result : datetime
            Next opening time.
        """
        earliest_start = self.start[0]
        latest_start = self.start[-1]

        if not self.next_bday.is_on_offset(other):
            # today is not business day
            other = other + sign * self.next_bday
            if self.n * sign >= 0:
                hour, minute = earliest_start.hour, earliest_start.minute
            else:
                hour, minute = latest_start.hour, latest_start.minute
        else:
            if self.n * sign >= 0:
                if latest_start < other.time():
                    # current time is after latest starting time in today
                    other = other + sign * self.next_bday
                    hour, minute = earliest_start.hour, earliest_start.minute
                else:
                    # find earliest starting time no earlier than current time
                    for st in self.start:
                        if other.time() <= st:
                            hour, minute = st.hour, st.minute
                            break
            else:
                if other.time() < earliest_start:
                    # current time is before earliest starting time in today
                    other = other + sign * self.next_bday
                    hour, minute = latest_start.hour, latest_start.minute
                else:
                    # find latest starting time no later than current time
                    for st in reversed(self.start):
                        if other.time() >= st:
                            hour, minute = st.hour, st.minute
                            break

        return datetime(other.year, other.month, other.day, hour, minute)

    def _prev_opening_time(self, other: datetime) -> datetime:
        """
        If n is positive, return the latest opening time earlier than or equal
        to current time.
        Otherwise the earliest opening time later than or equal to current
        time.

        Parameters
        ----------
        other : datetime
            Current time.

        Returns
        -------
        result : datetime
            Previous opening time.
        """
        return self._next_opening_time(other, sign=-1)

    @apply_wraps
    def rollback(self, dt: datetime) -> datetime:
        """
        Roll provided date backward to next offset only if not on offset.
        """
        if not self.is_on_offset(dt):
            if self.n >= 0:
                dt = self._prev_opening_time(dt)
            else:
                dt = self._next_opening_time(dt)
            return self._get_closing_time(dt)
        return dt

    @apply_wraps
    def rollforward(self, dt: datetime) -> datetime:
        """
        Roll provided date forward to next offset only if not on offset.
        """
        if not self.is_on_offset(dt):
            if self.n >= 0:
                return self._next_opening_time(dt)
            else:
                return self._prev_opening_time(dt)
        return dt

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # used for detecting edge condition
        nanosecond = getattr(other, "nanosecond", 0)
        # reset timezone and nanosecond
        # other may be a Timestamp, thus not use replace
        other = datetime(
            other.year,
            other.month,
            other.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        n = self.n

        # adjust other to reduce number of cases to handle
        if n >= 0:
            if other.time() in self.end or not self._is_on_offset(other):
                other = self._next_opening_time(other)
        else:
            if other.time() in self.start:
                # adjustment to move to previous business day
                other = other - timedelta(seconds=1)
            if not self._is_on_offset(other):
                other = self._next_opening_time(other)
                other = self._get_closing_time(other)

        # get total business hours by sec in one business day
        businesshours = sum(
            self._get_business_hours_by_sec(st, en)
            for st, en in zip(self.start, self.end)
        )

        bd, r = divmod(abs(n * 60), businesshours // 60)
        if n < 0:
            bd, r = -bd, -r

        # adjust by business days first
        if bd != 0:
            if self._prefix.startswith("C"):
                # GH#30593 this is a Custom offset
                skip_bd = CustomBusinessDay(
                    n=bd,
                    weekmask=self.weekmask,
                    holidays=self.holidays,
                    calendar=self.calendar,
                )
            else:
                skip_bd = BusinessDay(n=bd)
            # midnight business hour may not on BusinessDay
            if not self.next_bday.is_on_offset(other):
                prev_open = self._prev_opening_time(other)
                remain = other - prev_open
                other = prev_open + skip_bd + remain
            else:
                other = other + skip_bd

        # remaining business hours to adjust
        bhour_remain = timedelta(minutes=r)

        if n >= 0:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = (
                    self._get_closing_time(self._prev_opening_time(other)) - other
                )
                if bhour_remain < bhour:
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._next_opening_time(other + bhour)
        else:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = self._next_opening_time(other) - other
                if (
                    bhour_remain > bhour
                    or bhour_remain == bhour
                    and nanosecond != 0
                ):
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._get_closing_time(
                        self._next_opening_time(
                            other + bhour - timedelta(seconds=1)
                        )
                    )

        return other

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False

        if dt.tzinfo is not None:
            dt = datetime(
                dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
            )
        # Valid BH can be on the different BusinessDay during midnight
        # Distinguish by the time spent from previous opening time
        return self._is_on_offset(dt)

    def _is_on_offset(self, dt: datetime) -> bool:
        """
        Slight speedups using calculated values.
        """
        # if self.normalize and not _is_normalized(dt):
        #     return False
        # Valid BH can be on the different BusinessDay during midnight
        # Distinguish by the time spent from previous opening time
        if self.n >= 0:
            op = self._prev_opening_time(dt)
        else:
            op = self._next_opening_time(dt)
        span = (dt - op).total_seconds()
        businesshours = 0
        for i, st in enumerate(self.start):
            if op.hour == st.hour and op.minute == st.minute:
                businesshours = self._get_business_hours_by_sec(st, self.end[i])
        if span <= businesshours:
            return True
        else:
            return False


# ----------------------------------------------------------------------
# Custom Offset classes


cdef class CustomBusinessDay(BusinessDay):
    """
    DateOffset subclass representing custom business days excluding holidays.

    Parameters
    ----------
    n : int, default 1
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    holidays : list
        List/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``.
    calendar : pd.HolidayCalendar or np.busdaycalendar
    offset : timedelta, default timedelta(0)
    """

    _prefix = "C"
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        offset=timedelta(0),
    ):
        BusinessDay.__init__(self, n, normalize, offset)
        self._init_custom(weekmask, holidays, calendar)

    cpdef __setstate__(self, state):
        self.holidays = state.pop("holidays")
        self.weekmask = state.pop("weekmask")
        BusinessDay.__setstate__(self, state)

    @apply_wraps
    def _apply(self, other):
        if self.n <= 0:
            roll = "forward"
        else:
            roll = "backward"

        if PyDateTime_Check(other):
            date_in = other
            np_dt = np.datetime64(date_in.date())

            np_incr_dt = np.busday_offset(
                np_dt, self.n, roll=roll, busdaycal=self.calendar
            )

            dt_date = np_incr_dt.astype(datetime)
            result = datetime.combine(dt_date, date_in.time())

            if self.offset:
                result = result + self.offset
            return result

        elif is_any_td_scalar(other):
            td = Timedelta(self.offset) + other
            return BusinessDay(self.n, offset=td.to_pytimedelta(), normalize=self.normalize)
        else:
            raise ApplyTypeError(
                "Only know how to combine trading day with "
                "datetime, datetime64 or timedelta."
            )

    def apply_index(self, dtindex):
        raise NotImplementedError

    def _apply_array(self, dtarr):
        raise NotImplementedError

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        day64 = _to_dt64D(dt)
        return np.is_busday(day64, busdaycal=self.calendar)


cdef class CustomBusinessHour(BusinessHour):
    """
    DateOffset subclass representing possibly n custom business days.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    start : str, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, default: "17:00"
        End time of your custom business hour in 24h format.
    """

    _prefix = "CBH"
    _anchor = 0
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "start", "end", "offset"]
    )

    def __init__(
        self,
        n=1,
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        start="09:00",
        end="17:00",
        offset=timedelta(0),
    ):
        BusinessHour.__init__(self, n, normalize, start=start, end=end, offset=offset)
        self._init_custom(weekmask, holidays, calendar)

cdef class CustomBusinessIntradayOffset(BusinessHour):
    """
    DateOffset subclass representing possibly arbitary intraday offset within business hours.

    Parameters
    ----------
    n : int, default 1
        The number of months represented.
    step : timedelta
        intraday offset step
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``.
    start : str, default "09:00"
        Start time of your custom business hour in 24h format.
    end : str, default: "17:00"
        End time of your custom business hour in 24h format.
    """

    _prefix = "CBI"
    _anchor = 0
    _attributes = tuple(
        ["n", "step", "normalize", "weekmask", "holidays", "calendar", "start", "end", "offset"]
    )

    cdef readonly:
        timedelta step

    def __init__(
        self,
        n=1,
        step=timedelta(hours=1),
        normalize=False,
        weekmask="Mon Tue Wed Thu Fri",
        holidays=None,
        calendar=None,
        start="09:00",
        end="17:00",
        offset=timedelta(0),
    ):
        self.step = step
        BusinessHour.__init__(self, n, normalize, start=start, end=end, offset=offset)
        self._init_custom(weekmask, holidays, calendar)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        # used for detecting edge condition
        nanosecond = getattr(other, "nanosecond", 0)
        # reset timezone and nanosecond
        # other may be a Timestamp, thus not use replace
        other = datetime(
            other.year,
            other.month,
            other.day,
            other.hour,
            other.minute,
            other.second,
            other.microsecond,
        )
        n = self.n

        # adjust other to reduce number of cases to handle
        if n >= 0:
            if other.time() in self.end or not self._is_on_offset(other):
                other = self._next_opening_time(other)
        else:
            if other.time() in self.start:
                # adjustment to move to previous business day
                other = other - timedelta(microseconds=1)
            if not self._is_on_offset(other):
                other = self._next_opening_time(other)
                other = self._get_closing_time(other)

        # get total business hours by sec in one business day
        businesshours = sum(
            self._get_business_hours_by_sec(st, en)
            for st, en in zip(self.start, self.end)
        )

        bd, r = divmod(abs(n * int(self.step.total_seconds())), businesshours)
        if n < 0:
            bd, r = -bd, -r

        # adjust by business days first
        if bd != 0:
            if self._prefix.startswith("C"):
                # GH#30593 this is a Custom offset
                skip_bd = CustomBusinessDay(
                    n=bd,
                    weekmask=self.weekmask,
                    holidays=self.holidays,
                    calendar=self.calendar,
                )
            else:
                skip_bd = BusinessDay(n=bd)
            # midnight business hour may not on BusinessDay
            if not self.next_bday.is_on_offset(other):
                prev_open = self._prev_opening_time(other)
                remain = other - prev_open
                other = prev_open + skip_bd + remain
            else:
                other = other + skip_bd

        # remaining business hours to adjust
        bhour_remain = timedelta(seconds=r)

        if n >= 0:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = (
                    self._get_closing_time(self._prev_opening_time(other)) - other
                )
                if bhour_remain < bhour:
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._next_opening_time(other + bhour)
        else:
            while bhour_remain != timedelta(0):
                # business hour left in this business time interval
                bhour = self._next_opening_time(other) - other
                if (
                    bhour_remain > bhour
                    or bhour_remain == bhour
                    and nanosecond != 0
                ):
                    # finish adjusting if possible
                    other += bhour_remain
                    bhour_remain = timedelta(0)
                else:
                    # go to next business time interval
                    bhour_remain -= bhour
                    other = self._get_closing_time(
                        self._next_opening_time(
                            other + bhour - timedelta(seconds=1)
                        )
                    )

        return other

# ----------------------------------------------------------------------
# to_offset helpers

prefix_mapping = {
    offset._prefix: offset
    for offset in [
        BusinessDay,  # 'B'
        BusinessHour,  # 'BH'
        CustomBusinessDay,  # 'C'
        CustomBusinessHour,  # 'CBH'
    ]
}

# hack to handle WOM-1MON
opattern = re.compile(
    r"([+\-]?\d*|[+\-]?\d*\.\d*)\s*([A-Za-z]+([\-][\dA-Za-z\-]+)?)"
)

_lite_rule_alias = {
    "W": "W-SUN",
    "Q": "Q-DEC",

    "Min": "T",
    "min": "T",
    "ms": "L",
    "us": "U",
    "ns": "N",
}

_dont_uppercase = {"MS", "ms"}

INVALID_FREQ_ERR_MSG = "Invalid frequency: {0}"

# TODO: still needed?
# cache of previously seen offsets
_offset_map = {}


# TODO: better name?
def _get_offset(name: str) -> BaseOffset:
    """
    Return DateOffset object associated with rule name.

    Examples
    --------
    _get_offset('EOM') --> BMonthEnd(1)
    """
    if name not in _dont_uppercase:
        name = name.upper()
        name = _lite_rule_alias.get(name, name)
        name = _lite_rule_alias.get(name.lower(), name)
    else:
        name = _lite_rule_alias.get(name, name)

    if name not in _offset_map:
        try:
            split = name.split("-")
            klass = prefix_mapping[split[0]]
            # handles case where there's no suffix (and will TypeError if too
            # many '-')
            offset = klass._from_name(*split[1:])
        except (ValueError, TypeError, KeyError) as err:
            # bad prefix or suffix
            raise ValueError(INVALID_FREQ_ERR_MSG.format(name)) from err
        # cache
        _offset_map[name] = offset

    return _offset_map[name]


cpdef to_offset(freq):
    """
    Return DateOffset object from string or tuple representation
    or datetime.timedelta object.

    Parameters
    ----------
    freq : str, datetime.timedelta, BaseOffset or None

    Returns
    -------
    DateOffset or None

    Raises
    ------
    ValueError
        If freq is an invalid frequency

    See Also
    --------
    BaseOffset : Standard kind of date increment used for a date range.

    Examples
    --------
    >>> to_offset("5min")
    <5 * Minutes>

    >>> to_offset("1D1H")
    <25 * Hours>

    >>> to_offset("2W")
    <2 * Weeks: weekday=6>

    >>> to_offset("2B")
    <2 * BusinessDays>

    >>> to_offset(pd.Timedelta(days=1))
    <Day>

    >>> to_offset(Hour())
    <Hour>
    """
    if freq is None:
        return None

    if isinstance(freq, BaseOffset):
        return freq

    if isinstance(freq, tuple):
        raise TypeError(
            f"to_offset does not support tuples {freq}, pass as a string instead"
        )

    elif PyDelta_Check(freq):
        return delta_to_tick(freq)

    elif isinstance(freq, str):
        delta = None
        stride_sign = None

        try:
            split = opattern.split(freq)
            if split[-1] != "" and not split[-1].isspace():
                # the last element must be blank
                raise ValueError("last element must be blank")

            tups = zip(split[0::4], split[1::4], split[2::4])
            for n, (sep, stride, name) in enumerate(tups):
                if sep != "" and not sep.isspace():
                    raise ValueError("separator must be spaces")
                prefix = _lite_rule_alias.get(name) or name
                if stride_sign is None:
                    stride_sign = -1 if stride.startswith("-") else 1
                if not stride:
                    stride = 1

                if prefix in {"D", "H", "T", "S", "L", "U", "N"}:
                    # For these prefixes, we have something like "3H" or
                    #  "2.5T", so we can construct a Timedelta with the
                    #  matching unit and get our offset from delta_to_tick
                    td = Timedelta(1, unit=prefix)
                    off = delta_to_tick(td)
                    offset = off * float(stride)
                    if n != 0:
                        # If n==0, then stride_sign is already incorporated
                        #  into the offset
                        offset *= stride_sign
                else:
                    stride = int(stride)
                    offset = _get_offset(name)
                    offset = offset * int(np.fabs(stride) * stride_sign)

                if delta is None:
                    delta = offset
                else:
                    delta = delta + offset
        except (ValueError, TypeError) as err:
            raise ValueError(INVALID_FREQ_ERR_MSG.format(freq)) from err
    else:
        delta = None

    if delta is None:
        raise ValueError(INVALID_FREQ_ERR_MSG.format(freq))

    return delta


# ----------------------------------------------------------------------
# RelativeDelta Arithmetic

def shift_day(other: datetime, days: int) -> datetime:
    """
    Increment the datetime `other` by the given number of days, retaining
    the time-portion of the datetime.  For tz-naive datetimes this is
    equivalent to adding a timedelta.  For tz-aware datetimes it is similar to
    dateutil's relativedelta.__add__, but handles pytz tzinfo objects.

    Parameters
    ----------
    other : datetime or Timestamp
    days : int

    Returns
    -------
    shifted: datetime or Timestamp
    """
    if other.tzinfo is None:
        return other + timedelta(days=days)

    tz = other.tzinfo
    naive = other.replace(tzinfo=None)
    shifted = naive + timedelta(days=days)
    return localize_pydatetime(shifted, tz)


cdef inline int year_add_months(npy_datetimestruct dts, int months) nogil:
    """
    New year number after shifting npy_datetimestruct number of months.
    """
    return dts.year + (dts.month + months - 1) // 12


cdef inline int month_add_months(npy_datetimestruct dts, int months) nogil:
    """
    New month number after shifting npy_datetimestruct
    number of months.
    """
    cdef:
        int new_month = (dts.month + months) % 12
    return 12 if new_month == 0 else new_month


@cython.wraparound(False)
@cython.boundscheck(False)
cdef shift_quarters(
    const int64_t[:] dtindex,
    int quarters,
    int q1start_month,
    object day_opt,
    int modby=3,
):
    """
    Given an int64 array representing nanosecond timestamps, shift all elements
    by the specified number of quarters using DateOffset semantics.

    Parameters
    ----------
    dtindex : int64_t[:] timestamps for input dates
    quarters : int number of quarters to shift
    q1start_month : int month in which Q1 begins by convention
    day_opt : {'start', 'end', 'business_start', 'business_end'}
    modby : int (3 for quarters, 12 for years)

    Returns
    -------
    out : ndarray[int64_t]
    """
    cdef:
        Py_ssize_t count = len(dtindex)
        int64_t[:] out = np.empty(count, dtype="int64")

    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    _shift_quarters(dtindex, out, count, quarters, q1start_month, day_opt, modby)
    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
def shift_months(const int64_t[:] dtindex, int months, object day_opt=None):
    """
    Given an int64-based datetime index, shift all elements
    specified number of months using DateOffset semantics

    day_opt: {None, 'start', 'end', 'business_start', 'business_end'}
       * None: day of month
       * 'start' 1st day of month
       * 'end' last day of month
    """
    cdef:
        Py_ssize_t i
        npy_datetimestruct dts
        int count = len(dtindex)
        int64_t[:] out = np.empty(count, dtype="int64")

    if day_opt is None:
        with nogil:
            for i in range(count):
                if dtindex[i] == NPY_NAT:
                    out[i] = NPY_NAT
                    continue

                dt64_to_dtstruct(dtindex[i], &dts)
                dts.year = year_add_months(dts, months)
                dts.month = month_add_months(dts, months)

                dts.day = min(dts.day, get_days_in_month(dts.year, dts.month))
                out[i] = dtstruct_to_dt64(&dts)
    elif day_opt in ["start", "end", "business_start", "business_end"]:
        _shift_months(dtindex, out, count, months, day_opt)
    else:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    return np.asarray(out)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _shift_months(const int64_t[:] dtindex,
                               int64_t[:] out,
                               Py_ssize_t count,
                               int months,
                               str day_opt) nogil:
    """
    See shift_months.__doc__
    """
    cdef:
        Py_ssize_t i
        int months_to_roll
        npy_datetimestruct dts

    for i in range(count):
        if dtindex[i] == NPY_NAT:
            out[i] = NPY_NAT
            continue

        dt64_to_dtstruct(dtindex[i], &dts)
        months_to_roll = months

        months_to_roll = _roll_qtrday(&dts, months_to_roll, 0, day_opt)

        dts.year = year_add_months(dts, months_to_roll)
        dts.month = month_add_months(dts, months_to_roll)
        dts.day = get_day_of_month(&dts, day_opt)

        out[i] = dtstruct_to_dt64(&dts)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void _shift_quarters(const int64_t[:] dtindex,
                                 int64_t[:] out,
                                 Py_ssize_t count,
                                 int quarters,
                                 int q1start_month,
                                 str day_opt,
                                 int modby) nogil:
    """
    See shift_quarters.__doc__
    """
    cdef:
        Py_ssize_t i
        int months_since, n
        npy_datetimestruct dts

    for i in range(count):
        if dtindex[i] == NPY_NAT:
            out[i] = NPY_NAT
            continue

        dt64_to_dtstruct(dtindex[i], &dts)
        n = quarters

        months_since = (dts.month - q1start_month) % modby
        n = _roll_qtrday(&dts, n, months_since, day_opt)

        dts.year = year_add_months(dts, modby * n - months_since)
        dts.month = month_add_months(dts, modby * n - months_since)
        dts.day = get_day_of_month(&dts, day_opt)

        out[i] = dtstruct_to_dt64(&dts)


cdef ndarray[int64_t] _shift_bdays(const int64_t[:] i8other, int periods):
    """
    Implementation of BusinessDay.apply_offset.

    Parameters
    ----------
    i8other : const int64_t[:]
    periods : int

    Returns
    -------
    ndarray[int64_t]
    """
    cdef:
        Py_ssize_t i, n = len(i8other)
        int64_t[:] result = np.empty(n, dtype="i8")
        int64_t val, res
        int wday, nadj, days
        npy_datetimestruct dts

    for i in range(n):
        val = i8other[i]
        if val == NPY_NAT:
            result[i] = NPY_NAT
        else:
            # The rest of this is effectively a copy of BusinessDay.apply
            nadj = periods
            weeks = nadj // 5
            dt64_to_dtstruct(val, &dts)
            wday = dayofweek(dts.year, dts.month, dts.day)

            if nadj <= 0 and wday > 4:
                # roll forward
                nadj += 1

            nadj -= 5 * weeks

            # nadj is always >= 0 at this point
            if nadj == 0 and wday > 4:
                # roll back
                days = 4 - wday
            elif wday > 4:
                # roll forward
                days = (7 - wday) + (nadj - 1)
            elif wday + nadj <= 4:
                # shift by n days without leaving the current week
                days = nadj
            else:
                # shift by nadj days plus 2 to get past the weekend
                days = nadj + 2

            res = val + (7 * weeks + days) * DAY_NANOS
            result[i] = res

    return result.base


def shift_month(stamp: datetime, months: int, day_opt: object = None) -> datetime:
    """
    Given a datetime (or Timestamp) `stamp`, an integer `months` and an
    option `day_opt`, return a new datetimelike that many months later,
    with day determined by `day_opt` using relativedelta semantics.

    Scalar analogue of shift_months

    Parameters
    ----------
    stamp : datetime or Timestamp
    months : int
    day_opt : None, 'start', 'end', 'business_start', 'business_end', or int
        None: returned datetimelike has the same day as the input, or the
              last day of the month if the new month is too short
        'start': returned datetimelike has day=1
        'end': returned datetimelike has day on the last day of the month
        'business_start': returned datetimelike has day on the first
            business day of the month
        'business_end': returned datetimelike has day on the last
            business day of the month
        int: returned datetimelike has day equal to day_opt

    Returns
    -------
    shifted : datetime or Timestamp (same as input `stamp`)
    """
    cdef:
        int year, month, day
        int days_in_month, dy

    dy = (stamp.month + months) // 12
    month = (stamp.month + months) % 12

    if month == 0:
        month = 12
        dy -= 1
    year = stamp.year + dy

    if day_opt is None:
        days_in_month = get_days_in_month(year, month)
        day = min(stamp.day, days_in_month)
    elif day_opt == "start":
        day = 1
    elif day_opt == "end":
        day = get_days_in_month(year, month)
    elif day_opt == "business_start":
        # first business day of month
        day = get_firstbday(year, month)
    elif day_opt == "business_end":
        # last business day of month
        day = get_lastbday(year, month)
    elif is_integer_object(day_opt):
        days_in_month = get_days_in_month(year, month)
        day = min(day_opt, days_in_month)
    else:
        raise ValueError(day_opt)
    return stamp.replace(year=year, month=month, day=day)


cdef inline int get_day_of_month(npy_datetimestruct* dts, str day_opt) nogil:
    """
    Find the day in `other`'s month that satisfies a DateOffset's is_on_offset
    policy, as described by the `day_opt` argument.

    Parameters
    ----------
    dts : npy_datetimestruct*
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        'start': returns 1
        'end': returns last day of the month
        'business_start': returns the first business day of the month
        'business_end': returns the last business day of the month

    Returns
    -------
    day_of_month : int

    Examples
    -------
    >>> other = datetime(2017, 11, 14)
    >>> get_day_of_month(other, 'start')
    1
    >>> get_day_of_month(other, 'end')
    30

    Notes
    -----
    Caller is responsible for ensuring one of the four accepted day_opt values
    is passed.
    """

    if day_opt == "start":
        return 1
    elif day_opt == "end":
        return get_days_in_month(dts.year, dts.month)
    elif day_opt == "business_start":
        # first business day of month
        return get_firstbday(dts.year, dts.month)
    else:
        # i.e. day_opt == "business_end":
        # last business day of month
        return get_lastbday(dts.year, dts.month)


cpdef int roll_convention(int other, int n, int compare) nogil:
    """
    Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : int, generally the day component of a datetime
    n : number of periods to increment, before adjusting for rolling
    compare : int, generally the day component of a datetime, in the same
              month as the datetime form which `other` was taken.

    Returns
    -------
    n : int number of periods to increment
    """
    if n > 0 and other < compare:
        n -= 1
    elif n <= 0 and other > compare:
        # as if rolled forward already
        n += 1
    return n


def roll_qtrday(other: datetime, n: int, month: int,
                day_opt: str, modby: int) -> int:
    """
    Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : datetime or Timestamp
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_opt : {'start', 'end', 'business_start', 'business_end'}
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    get_day_of_month : Find the day in a month provided an offset.
    """
    cdef:
        int months_since
        npy_datetimestruct dts

    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError(day_opt)

    pydate_to_dtstruct(other, &dts)

    if modby == 12:
        # We care about the month-of-year, not month-of-quarter, so skip mod
        months_since = other.month - month
    else:
        months_since = other.month % modby - month % modby

    return _roll_qtrday(&dts, n, months_since, day_opt)


cdef inline int _roll_qtrday(npy_datetimestruct* dts,
                             int n,
                             int months_since,
                             str day_opt) nogil except? -1:
    """
    See roll_qtrday.__doc__
    """

    if n > 0:
        if months_since < 0 or (months_since == 0 and
                                dts.day < get_day_of_month(dts, day_opt)):
            # pretend to roll back if on same month but
            # before compare_day
            n -= 1
    else:
        if months_since > 0 or (months_since == 0 and
                                dts.day > get_day_of_month(dts, day_opt)):
            # make sure to roll forward, so negate
            n += 1
    return n
