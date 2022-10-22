import re
import time
import warnings

from pandas.util._exceptions import find_stack_level

cimport cython
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    date,
    datetime,
    import_datetime,
    time as dt_time,
    timedelta,
)

import_datetime()

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

from pandas._libs.tslibs.ccalendar cimport (
    dayofweek,
    get_days_in_month,
    get_firstbday,
    get_lastbday,
)
from pandas._libs.tslibs.conversion cimport localize_pydatetime
from pandas._libs.tslibs.dtypes cimport periods_per_day
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_unit_from_dtype,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydate_to_dtstruct,
)

from pandas._libs.tslibs.dtypes cimport PeriodDtypeCode
from pandas._libs.tslibs.timedeltas cimport (
    _Timedelta,
    delta_to_nanoseconds,
    is_any_td_scalar,
)

from pandas._libs.tslibs.timedeltas import Timedelta

from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs.tslibs.timestamps import Timestamp

# ---------------------------------------------------------------------
# Misc Helpers

cdef bint is_offset_object(object obj):
    return isinstance(obj, BaseOffset)


cdef bint is_tick_object(object obj):
    return isinstance(obj, Tick)


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
        reso = get_unit_from_dtype(other.dtype)
        result = normalize_i8_timestamps(result.view("i8"), None, reso=reso)

    return result


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
        naive = dt.astimezone(None)
        dt = np.datetime64(naive, "D")
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
                       "microsecond", "millisecond", "nanosecond",
                       "nanoseconds", "hours", "minutes", "seconds",
                       "milliseconds", "microseconds"}


cdef _determine_offset(kwds):
    # timedelta is used for sub-daily plural offsets and all singular
    # offsets, relativedelta is used for plural offsets of daily length or
    # more, nanosecond(s) are handled by apply_wraps
    kwds_no_nanos = dict(
        (k, v) for k, v in kwds.items()
        if k not in ('nanosecond', 'nanoseconds')
    )
    # TODO: Are nanosecond and nanoseconds allowed somewhere?

    _kwds_use_relativedelta = ('years', 'months', 'weeks', 'days',
                               'year', 'month', 'week', 'day', 'weekday',
                               'hour', 'minute', 'second', 'microsecond',
                               'millisecond')

    use_relativedelta = False
    if len(kwds_no_nanos) > 0:
        if any(k in _kwds_use_relativedelta for k in kwds_no_nanos):
            if "millisecond" in kwds_no_nanos:
                raise NotImplementedError(
                    "Using DateOffset to replace `millisecond` component in "
                    "datetime object is not supported. Use "
                    "`microsecond=timestamp.microsecond % 1000 + ms * 1000` "
                    "instead."
                )
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

cdef class BaseOffset:
    """
    Base class for DateOffset methods that are not overridden by subclasses.
    """
    # ensure that reversed-ops with numpy scalars return NotImplemented
    __array_priority__ = 1000

    _day_opt = None
    _attributes = tuple(["n", "normalize"])
    _use_relativedelta = False
    _adjust_dst = True
    _deprecations = frozenset(["isAnchored", "onOffset"])

    # cdef readonly:
    #    int64_t n
    #    bint normalize
    #    dict _cache

    def __init__(self, n=1, normalize=False):
        n = self._validate_n(n)
        self.n = n
        """
        Number of multiples of the frequency.

        Examples
        --------
        >>> pd.offsets.Hour(5).n
        5
        """
        self.normalize = normalize
        """
        Return boolean whether the frequency can align with midnight.

        Examples
        --------
        >>> pd.offsets.Hour(5).normalize
        False
        """
        self._cache = {}

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            try:
                # GH#23524 if to_offset fails, we are dealing with an
                #  incomparable type so == is False and != is True
                other = to_offset(other)
            except ValueError:
                # e.g. "infer"
                return False
        try:
            return self._params == other._params
        except AttributeError:
            # other is not a DateOffset object
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self) -> int:
        return hash(self._params)

    @cache_readonly
    def _params(self):
        """
        Returns a tuple containing all of the attributes needed to evaluate
        equality between two DateOffset objects.
        """
        d = getattr(self, "__dict__", {})
        all_paras = d.copy()
        all_paras["n"] = self.n
        all_paras["normalize"] = self.normalize
        for attr in self._attributes:
            if hasattr(self, attr) and attr not in d:
                # cython attributes are not in __dict__
                all_paras[attr] = getattr(self, attr)

        if 'holidays' in all_paras and not all_paras['holidays']:
            all_paras.pop('holidays')
        exclude = ['kwds', 'name', 'calendar']
        attrs = [(k, v) for k, v in all_paras.items()
                 if (k not in exclude) and (k[0] != '_')]
        attrs = sorted(set(attrs))
        params = tuple([str(type(self))] + attrs)
        return params

    @property
    def kwds(self) -> dict:
        """
        Return a dict of extra parameters for the offset.

        Examples
        --------
        >>> pd.DateOffset(5).kwds
        {}

        >>> pd.offsets.FY5253Quarter().kwds
        {'weekday': 0,
         'startingMonth': 1,
         'qtr_with_extra_week': 1,
         'variation': 'nearest'}
        """
        # for backwards-compatibility
        kwds = {name: getattr(self, name, None) for name in self._attributes
                if name not in ["n", "normalize"]}
        return {name: kwds[name] for name in kwds if kwds[name] is not None}

    @property
    def base(self):
        """
        Returns a copy of the calling offset object with n=1 and all other
        attributes equal.
        """
        return type(self)(n=1, normalize=self.normalize, **self.kwds)

    def __add__(self, other):
        if not isinstance(self, BaseOffset):
            # cython semantics; this is __radd__
            # TODO(cython3): remove this, this moved to __radd__
            return other.__add__(self)

        elif util.is_array(other) and other.dtype == object:
            return np.array([self + x for x in other])

        try:
            return self._apply(other)
        except ApplyTypeError:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if PyDateTime_Check(other):
            raise TypeError('Cannot subtract datetime from offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n, normalize=self.normalize,
                              **self.kwds)
        elif not isinstance(self, BaseOffset):
            # TODO(cython3): remove, this moved to __rsub__
            # cython semantics, this is __rsub__
            return (-other).__add__(self)
        else:
            # e.g. PeriodIndex
            return NotImplemented

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __call__(self, other):
        warnings.warn(
            "DateOffset.__call__ is deprecated and will be removed in a future "
            "version.  Use `offset + other` instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._apply(other)

    def apply(self, other):
        # GH#44522
        warnings.warn(
            f"{type(self).__name__}.apply is deprecated and will be removed "
            "in a future version. Use `offset + other` instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._apply(other)

    def __mul__(self, other):
        if util.is_array(other):
            return np.array([self * x for x in other])
        elif is_integer_object(other):
            return type(self)(n=other * self.n, normalize=self.normalize,
                              **self.kwds)
        elif not isinstance(self, BaseOffset):
            # TODO(cython3): remove this, this moved to __rmul__
            # cython semantics, this is __rmul__
            return other.__mul__(self)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        # Note: we are deferring directly to __mul__ instead of __rmul__, as
        # that allows us to use methods that can go in a `cdef class`
        return self * -1

    def copy(self):
        # Note: we are deferring directly to __mul__ instead of __rmul__, as
        # that allows us to use methods that can go in a `cdef class`
        """
        Return a copy of the frequency.

        Examples
        --------
        >>> freq = pd.DateOffset(1)
        >>> freq_copy = freq.copy()
        >>> freq is freq_copy
        False
        """
        return self * 1

    # ------------------------------------------------------------------
    # Name and Rendering Methods

    def __repr__(self) -> str:
        # _output_name used by B(Year|Quarter)(End|Begin) to
        #  expand "B" -> "Business"
        class_name = getattr(self, "_output_name", type(self).__name__)

        if abs(self.n) != 1:
            plural = "s"
        else:
            plural = ""

        n_str = ""
        if self.n != 1:
            n_str = f"{self.n} * "

        out = f"<{n_str}{class_name}{plural}{self._repr_attrs()}>"
        return out

    def _repr_attrs(self) -> str:
        exclude = {"n", "inc", "normalize"}
        attrs = []
        for attr in sorted(self._attributes):
            # _attributes instead of __dict__ because cython attrs are not in __dict__
            if attr.startswith("_") or attr == "kwds" or not hasattr(self, attr):
                # DateOffset may not have some of these attributes
                continue
            elif attr not in exclude:
                value = getattr(self, attr)
                attrs.append(f"{attr}={value}")

        out = ""
        if attrs:
            out += ": " + ", ".join(attrs)
        return out

    @property
    def name(self) -> str:
        """
        Return a string representing the base frequency.

        Examples
        --------
        >>> pd.offsets.Hour().name
        'H'

        >>> pd.offsets.Hour(5).name
        'H'
        """
        return self.rule_code

    @property
    def _prefix(self) -> str:
        raise NotImplementedError("Prefix not defined")

    @property
    def rule_code(self) -> str:
        return self._prefix

    @cache_readonly
    def freqstr(self) -> str:
        """
        Return a string representing the frequency.

        Examples
        --------
        >>> pd.DateOffset(5).freqstr
        '<5 * DateOffsets>'

        >>> pd.offsets.BusinessHour(2).freqstr
        '2BH'

        >>> pd.offsets.Nano().freqstr
        'N'

        >>> pd.offsets.Nano(-3).freqstr
        '-3N'
        """
        try:
            code = self.rule_code
        except NotImplementedError:
            return str(repr(self))

        if self.n != 1:
            fstr = f"{self.n}{code}"
        else:
            fstr = code

        try:
            if self._offset:
                fstr += self._offset_str()
        except AttributeError:
            # TODO: standardize `_offset` vs `offset` naming convention
            pass

        return fstr

    def _offset_str(self) -> str:
        return ""

    # ------------------------------------------------------------------

    def apply_index(self, dtindex):
        """
        Vectorized apply of DateOffset to DatetimeIndex.

        .. deprecated:: 1.1.0

           Use ``offset + dtindex`` instead.

        Parameters
        ----------
        index : DatetimeIndex

        Returns
        -------
        DatetimeIndex

        Raises
        ------
        NotImplementedError
            When the specific offset subclass does not have a vectorized
            implementation.
        """
        warnings.warn("'Offset.apply_index(other)' is deprecated. "
                      "Use 'offset + other' instead.", FutureWarning)

        res = self._apply_array(dtindex)
        return type(dtindex)(res)

    def _apply(self, other):
        raise NotImplementedError("implemented by subclasses")

    @apply_array_wraps
    def _apply_array(self, dtarr):
        raise NotImplementedError(
            f"DateOffset subclass {type(self).__name__} "
            "does not have a vectorized implementation"
        )

    def rollback(self, dt) -> datetime:
        """
        Roll provided date backward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
        dt = Timestamp(dt)
        if not self.is_on_offset(dt):
            dt = dt - type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def rollforward(self, dt) -> datetime:
        """
        Roll provided date forward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
        dt = Timestamp(dt)
        if not self.is_on_offset(dt):
            dt = dt + type(self)(1, normalize=self.normalize, **self.kwds)
        return dt

    def _get_offset_day(self, other: datetime) -> int:
        # subclass must implement `_day_opt`; calling from the base class
        # will implicitly assume day_opt = "business_end", see get_day_of_month.
        cdef:
            npy_datetimestruct dts
        pydate_to_dtstruct(other, &dts)
        return get_day_of_month(&dts, self._day_opt)

    def is_on_offset(self, dt: datetime) -> bool:
        """
        Return boolean whether a timestamp intersects with this frequency.

        Parameters
        ----------
        dt : datetime.datetime
            Timestamp to check intersections with frequency.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Day(1)
        >>> freq.is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 6)
        >>> ts.day_name()
        'Saturday'
        >>> freq = pd.offsets.BusinessDay(1)
        >>> freq.is_on_offset(ts)
        False
        """
        if self.normalize and not _is_normalized(dt):
            return False

        # Default (slow) method for determining if some date is a member of the
        # date range generated by this offset. Subclasses may have this
        # re-implemented in a nicer way.
        a = dt
        b = (dt + self) - self
        return a == b

    # ------------------------------------------------------------------

    # Staticmethod so we can call from Tick.__init__, will be unnecessary
    #  once BaseOffset is a cdef class and is inherited by Tick
    @staticmethod
    def _validate_n(n) -> int:
        """
        Require that `n` be an integer.

        Parameters
        ----------
        n : int

        Returns
        -------
        nint : int

        Raises
        ------
        TypeError if `int(n)` raises
        ValueError if n != int(n)
        """
        if util.is_timedelta64_object(n):
            raise TypeError(f'`n` argument must be an integer, got {type(n)}')
        try:
            nint = int(n)
        except (ValueError, TypeError):
            raise TypeError(f'`n` argument must be an integer, got {type(n)}')
        if n != nint:
            raise ValueError(f'`n` argument must be an integer, got {n}')
        return nint

    def __setstate__(self, state):
        """
        Reconstruct an instance from a pickled state
        """
        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})
        # At this point we expect state to be empty

    def __getstate__(self):
        """
        Return a pickleable state
        """
        state = {}
        state["n"] = self.n
        state["normalize"] = self.normalize

        # we don't want to actually pickle the calendar object
        # as its a np.busyday; we recreate on deserialization
        state.pop("calendar", None)
        if "kwds" in state:
            state["kwds"].pop("calendar", None)

        return state

    @property
    def nanos(self):
        raise ValueError(f"{self} is a non-fixed frequency")

    def onOffset(self, dt) -> bool:
        warnings.warn(
            "onOffset is a deprecated, use is_on_offset instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.is_on_offset(dt)

    def isAnchored(self) -> bool:
        warnings.warn(
            "isAnchored is a deprecated, use is_anchored instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.is_anchored()

    def is_anchored(self) -> bool:
        # TODO: Does this make sense for the general case?  It would help
        # if there were a canonical docstring for what is_anchored means.
        """
        Return boolean whether the frequency is a unit frequency (n=1).

        Examples
        --------
        >>> pd.DateOffset().is_anchored()
        True
        >>> pd.DateOffset(2).is_anchored()
        False
        """
        return self.n == 1

    # ------------------------------------------------------------------

    def is_month_start(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_start(ts)
        True
        """
        return ts._get_start_end_field("is_month_start", self)

    def is_month_end(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the month end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_end(ts)
        False
        """
        return ts._get_start_end_field("is_month_end", self)

    def is_quarter_start(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the quarter start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_start(ts)
        True
        """
        return ts._get_start_end_field("is_quarter_start", self)

    def is_quarter_end(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the quarter end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_end(ts)
        False
        """
        return ts._get_start_end_field("is_quarter_end", self)

    def is_year_start(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the year start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_start(ts)
        True
        """
        return ts._get_start_end_field("is_year_start", self)

    def is_year_end(self, _Timestamp ts):
        """
        Return boolean whether a timestamp occurs on the year end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_end(ts)
        False
        """
        return ts._get_start_end_field("is_year_end", self)


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


# ---------------------------------------------------------------------
# Tick Offsets

cdef class Tick(SingleConstructorOffset):
    _adjust_dst = False
    _prefix = "undefined"
    _td64_unit = "undefined"
    _attributes = tuple(["n", "normalize"])

    def __init__(self, n=1, normalize=False):
        n = self._validate_n(n)
        self.n = n
        self.normalize = False
        self._cache = {}
        if normalize:
            # GH#21427
            raise ValueError(
                "Tick offset with `normalize=True` are not allowed."
            )

    # Note: Without making this cpdef, we get AttributeError when calling
    #  from __mul__
    cpdef Tick _next_higher_resolution(Tick self):
        if type(self) is Day:
            return Hour(self.n * 24)
        if type(self) is Hour:
            return Minute(self.n * 60)
        if type(self) is Minute:
            return Second(self.n * 60)
        if type(self) is Second:
            return Milli(self.n * 1000)
        if type(self) is Milli:
            return Micro(self.n * 1000)
        if type(self) is Micro:
            return Nano(self.n * 1000)
        raise ValueError("Could not convert to integer offset at any resolution")

    # --------------------------------------------------------------------

    def _repr_attrs(self) -> str:
        # Since cdef classes have no __dict__, we need to override
        return ""

    @property
    def delta(self):
        return self.n * Timedelta(self._nanos_inc)

    @property
    def nanos(self) -> int64_t:
        """
        Return an integer of the total number of nanoseconds.

        Raises
        ------
        ValueError
            If the frequency is non-fixed.

        Examples
        --------
        >>> pd.offsets.Hour(5).nanos
        18000000000000
        """
        return self.n * self._nanos_inc

    def is_on_offset(self, dt: datetime) -> bool:
        return True

    def is_anchored(self) -> bool:
        return False

    # This is identical to BaseOffset.__hash__, but has to be redefined here
    # for Python 3, because we've redefined __eq__.
    def __hash__(self) -> int:
        return hash(self._params)

    # --------------------------------------------------------------------
    # Comparison and Arithmetic Methods

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                # GH#23524 if to_offset fails, we are dealing with an
                #  incomparable type so == is False and != is True
                other = to_offset(other)
            except ValueError:
                # e.g. "infer"
                return False
        return self.delta == other

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        return self.delta.__le__(other)

    def __lt__(self, other):
        return self.delta.__lt__(other)

    def __ge__(self, other):
        return self.delta.__ge__(other)

    def __gt__(self, other):
        return self.delta.__gt__(other)

    def __mul__(self, other):
        if not isinstance(self, Tick):
            # TODO(cython3), remove this, this moved to __rmul__
            # cython semantics, this is __rmul__
            return other.__mul__(self)
        if is_float_object(other):
            n = other * self.n
            # If the new `n` is an integer, we can represent it using the
            #  same Tick subclass as self, otherwise we need to move up
            #  to a higher-resolution subclass
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return BaseOffset.__mul__(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(self, Tick):
            # cython semantics mean the args are sometimes swapped
            result = other.delta.__rtruediv__(self)
        else:
            result = self.delta.__truediv__(other)
        return _wrap_timedelta_result(result)

    def __rtruediv__(self, other):
        result = self.delta.__rtruediv__(other)
        return _wrap_timedelta_result(result)

    def __add__(self, other):
        if not isinstance(self, Tick):
            # cython semantics; this is __radd__
            # TODO(cython3): remove this, this moved to __radd__
            return other.__add__(self)

        if isinstance(other, Tick):
            if type(self) == type(other):
                return type(self)(self.n + other.n)
            else:
                return delta_to_tick(self.delta + other.delta)
        try:
            return self._apply(other)
        except ApplyTypeError:
            # Includes pd.Period
            return NotImplemented
        except OverflowError as err:
            raise OverflowError(
                f"the add operation between {self} and {other} will overflow"
            ) from err

    def __radd__(self, other):
        return self.__add__(other)

    def _apply(self, other):
        # Timestamp can handle tz and nano sec, thus no need to use apply_wraps
        if isinstance(other, _Timestamp):
            # GH#15126
            return other + self.delta
        elif other is NaT:
            return NaT
        elif is_datetime64_object(other) or PyDate_Check(other):
            # PyDate_Check includes date, datetime
            return Timestamp(other) + self

        if util.is_timedelta64_object(other) or PyDelta_Check(other):
            return other + self.delta
        elif isinstance(other, type(self)):
            # TODO(2.0): remove once apply deprecation is enforced.
            #  This is reached in tests that specifically call apply,
            #  but should not be reached "naturally" because __add__ should
            #  catch this case first.
            return type(self)(self.n + other.n)

        raise ApplyTypeError(f"Unhandled type: {type(other).__name__}")

    # --------------------------------------------------------------------
    # Pickle Methods

    def __setstate__(self, state):
        self.n = state["n"]
        self.normalize = False


cdef class Day(Tick):
    _nanos_inc = 24 * 3600 * 1_000_000_000
    _prefix = "D"
    _td64_unit = "D"
    _period_dtype_code = PeriodDtypeCode.D
    _creso = NPY_DATETIMEUNIT.NPY_FR_D


cdef class Hour(Tick):
    _nanos_inc = 3600 * 1_000_000_000
    _prefix = "H"
    _td64_unit = "h"
    _period_dtype_code = PeriodDtypeCode.H
    _creso = NPY_DATETIMEUNIT.NPY_FR_h


cdef class Minute(Tick):
    _nanos_inc = 60 * 1_000_000_000
    _prefix = "T"
    _td64_unit = "m"
    _period_dtype_code = PeriodDtypeCode.T
    _creso = NPY_DATETIMEUNIT.NPY_FR_m


cdef class Second(Tick):
    _nanos_inc = 1_000_000_000
    _prefix = "S"
    _td64_unit = "s"
    _period_dtype_code = PeriodDtypeCode.S
    _creso = NPY_DATETIMEUNIT.NPY_FR_s


cdef class Milli(Tick):
    _nanos_inc = 1_000_000
    _prefix = "L"
    _td64_unit = "ms"
    _period_dtype_code = PeriodDtypeCode.L
    _creso = NPY_DATETIMEUNIT.NPY_FR_ms


cdef class Micro(Tick):
    _nanos_inc = 1000
    _prefix = "U"
    _td64_unit = "us"
    _period_dtype_code = PeriodDtypeCode.U
    _creso = NPY_DATETIMEUNIT.NPY_FR_us


cdef class Nano(Tick):
    _nanos_inc = 1
    _prefix = "N"
    _td64_unit = "ns"
    _period_dtype_code = PeriodDtypeCode.N
    _creso = NPY_DATETIMEUNIT.NPY_FR_ns


def delta_to_tick(delta: timedelta) -> Tick:
    if delta.microseconds == 0 and getattr(delta, "nanoseconds", 0) == 0:
        # nanoseconds only for pd.Timedelta
        if delta.seconds == 0:
            return Day(delta.days)
        else:
            seconds = delta.days * 86400 + delta.seconds
            if seconds % 3600 == 0:
                return Hour(seconds / 3600)
            elif seconds % 60 == 0:
                return Minute(seconds / 60)
            else:
                return Second(seconds)
    else:
        nanos = delta_to_nanoseconds(delta)
        if nanos % 1_000_000 == 0:
            return Milli(nanos // 1_000_000)
        elif nanos % 1000 == 0:
            return Micro(nanos // 1000)
        else:  # pragma: no cover
            return Nano(nanos)


# --------------------------------------------------------------------

cdef class RelativeDeltaOffset(BaseOffset):
    """
    DateOffset subclass backed by a dateutil relativedelta object.
    """
    _attributes = tuple(["n", "normalize"] + list(_relativedelta_kwds))
    _adjust_dst = False

    def __init__(self, n=1, normalize=False, **kwds):
        BaseOffset.__init__(self, n, normalize)

        off, use_rd = _determine_offset(kwds)
        object.__setattr__(self, "_offset", off)
        object.__setattr__(self, "_use_relativedelta", use_rd)
        for key in kwds:
            val = kwds[key]
            object.__setattr__(self, key, val)

    def __getstate__(self):
        """
        Return a pickleable state
        """
        # RelativeDeltaOffset (technically DateOffset) is the only non-cdef
        #  class, so the only one with __dict__
        state = self.__dict__.copy()
        state["n"] = self.n
        state["normalize"] = self.normalize
        return state

    def __setstate__(self, state):
        """
        Reconstruct an instance from a pickled state
        """

        if "offset" in state:
            # Older (<0.22.0) versions have offset attribute instead of _offset
            if "_offset" in state:  # pragma: no cover
                raise AssertionError("Unexpected key `_offset`")
            state["_offset"] = state.pop("offset")
            state["kwds"]["offset"] = state["_offset"]

        self.n = state.pop("n")
        self.normalize = state.pop("normalize")
        self._cache = state.pop("_cache", {})

        self.__dict__.update(state)

    @apply_wraps
    def _apply(self, other: datetime) -> datetime:
        if self._use_relativedelta:
            other = _as_datetime(other)

        if len(self.kwds) > 0:
            tzinfo = getattr(other, "tzinfo", None)
            if tzinfo is not None and self._use_relativedelta:
                # perform calculation in UTC
                other = other.replace(tzinfo=None)

            if hasattr(self, "nanoseconds"):
                td_nano = Timedelta(nanoseconds=self.nanoseconds)
            else:
                td_nano = Timedelta(0)

            if self.n > 0:
                for i in range(self.n):
                    other = other + self._offset + td_nano
            else:
                for i in range(-self.n):
                    other = other - self._offset - td_nano

            if tzinfo is not None and self._use_relativedelta:
                # bring tz back from UTC calculation
                other = localize_pydatetime(other, tzinfo)

            return Timestamp(other)
        else:
            return other + timedelta(self.n)

    @apply_array_wraps
    def _apply_array(self, dtarr):
        reso = get_unit_from_dtype(dtarr.dtype)
        dt64other = np.asarray(dtarr)
        kwds = self.kwds
        relativedelta_fast = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
        }
        # relativedelta/_offset path only valid for base DateOffset
        if self._use_relativedelta and set(kwds).issubset(relativedelta_fast):

            months = (kwds.get("years", 0) * 12 + kwds.get("months", 0)) * self.n
            if months:
                shifted = shift_months(dt64other.view("i8"), months, reso=reso)
                dt64other = shifted.view(dtarr.dtype)

            weeks = kwds.get("weeks", 0) * self.n
            if weeks:
                delta = Timedelta(days=7 * weeks)
                td = (<_Timedelta>delta)._as_creso(reso)
                dt64other = dt64other + td

            timedelta_kwds = {
                k: v
                for k, v in kwds.items()
                if k in ["days", "hours", "minutes", "seconds", "microseconds"]
            }
            if timedelta_kwds:
                delta = Timedelta(**timedelta_kwds)
                td = (<_Timedelta>delta)._as_creso(reso)
                dt64other = dt64other + (self.n * td)
            return dt64other
        elif not self._use_relativedelta and hasattr(self, "_offset"):
            # timedelta
            num_nano = getattr(self, "nanoseconds", 0)
            if num_nano != 0:
                rem_nano = Timedelta(nanoseconds=num_nano)
                delta = Timedelta((self._offset + rem_nano) * self.n)
            else:
                delta = Timedelta(self._offset * self.n)
            td = (<_Timedelta>delta)._as_creso(reso)
            return dt64other + td
        else:
            # relativedelta with other keywords
            kwd = set(kwds) - relativedelta_fast
            raise NotImplementedError(
                "DateOffset with relativedelta "
                f"keyword(s) {kwd} not able to be "
                "applied vectorized"
            )

    def is_on_offset(self, dt: datetime) -> bool:
        if self.normalize and not _is_normalized(dt):
            return False
        return True


class OffsetMeta(type):
    """
    Metaclass that allows us to pretend that all BaseOffset subclasses
    inherit from DateOffset (which is needed for backward-compatibility).
    """

    @classmethod
    def __instancecheck__(cls, obj) -> bool:
        return isinstance(obj, BaseOffset)

    @classmethod
    def __subclasscheck__(cls, obj) -> bool:
        return issubclass(obj, BaseOffset)


# TODO: figure out a way to use a metaclass with a cdef class
class DateOffset(RelativeDeltaOffset, metaclass=OffsetMeta):
    """
    Standard kind of date increment used for a date range.

    Works exactly like the keyword argument form of relativedelta.
    Note that the positional argument form of relativedelata is not
    supported. Use of the keyword n is discouraged-- you would be better
    off specifying n in the keywords you use, but regardless it is
    there for you. n is needed for DateOffset subclasses.

    DateOffset works as follows.  Each offset specify a set of dates
    that conform to the DateOffset.  For example, Bday defines this
    set to be the set of dates that are weekdays (M-F).  To test if a
    date is in the set of a DateOffset dateOffset we can use the
    is_on_offset method: dateOffset.is_on_offset(date).

    If a date is not on a valid date, the rollback and rollforward
    methods can be used to roll the date to the nearest valid date
    before/after the date.

    DateOffsets can be created to move dates forward a given number of
    valid dates.  For example, Bday(2) can be added to a date to move
    it two business days forward.  If the date does not start on a
    valid date, first it is moved to a valid date.  Thus pseudo code
    is:

    def __add__(date):
      date = rollback(date) # does nothing if date is valid
      return date + <n number of periods>

    When a date offset is created for a negative number of periods,
    the date is first rolled forward.  The pseudo code is:

    def __add__(date):
      date = rollforward(date) # does nothing is date is valid
      return date + <n number of periods>

    Zero presents a problem.  Should it roll forward or back?  We
    arbitrarily have it rollforward:

    date + BDay(0) == BDay.rollforward(date)

    Since 0 is a bit weird, we suggest avoiding its use.

    Besides, adding a DateOffsets specified by the singular form of the date
    component can be used to replace certain component of the timestamp.

    Parameters
    ----------
    n : int, default 1
        The number of time periods the offset represents.
        If specified without a temporal pattern, defaults to n days.
    normalize : bool, default False
        Whether to round the result of a DateOffset addition down to the
        previous midnight.
    **kwds
        Temporal parameter that add to or replace the offset value.

        Parameters that **add** to the offset (like Timedelta):

        - years
        - months
        - weeks
        - days
        - hours
        - minutes
        - seconds
        - milliseconds
        - microseconds
        - nanoseconds

        Parameters that **replace** the offset value:

        - year
        - month
        - day
        - weekday
        - hour
        - minute
        - second
        - microsecond
        - nanosecond.

    See Also
    --------
    dateutil.relativedelta.relativedelta : The relativedelta type is designed
        to be applied to an existing datetime an can replace specific components of
        that datetime, or represents an interval of time.

    Examples
    --------
    >>> from pandas.tseries.offsets import DateOffset
    >>> ts = pd.Timestamp('2017-01-01 09:10:11')
    >>> ts + DateOffset(months=3)
    Timestamp('2017-04-01 09:10:11')

    >>> ts = pd.Timestamp('2017-01-01 09:10:11')
    >>> ts + DateOffset(months=2)
    Timestamp('2017-03-01 09:10:11')
    >>> ts + DateOffset(day=31)
    Timestamp('2017-01-31 09:10:11')

    >>> ts + pd.DateOffset(hour=8)
    Timestamp('2017-01-01 08:10:11')
    """
    def __setattr__(self, name, value):
        raise AttributeError("DateOffset objects are immutable.")

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

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 8, 5)
    >>> ts + pd.offsets.BusinessDay()
    Timestamp('2022-08-08 00:00:00')
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

    @apply_array_wraps
    def _apply_array(self, dtarr):
        i8other = dtarr.view("i8")
        reso = get_unit_from_dtype(dtarr.dtype)
        res = _shift_bdays(i8other, self.n, reso=reso)
        if self.offset:
            res = res.view(dtarr.dtype) + Timedelta(self.offset)
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

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.BusinessHour()
    Timestamp('2022-08-08 09:00:00')
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
        # Use python string formatting to be faster than strftime
        hours = ",".join(
            f'{st.hour:02d}:{st.minute:02d}-{en.hour:02d}:{en.minute:02d}'
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
    calendar : np.busdaycalendar
    offset : timedelta, default timedelta(0)

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 8, 5)
    >>> ts + pd.offsets.CustomBusinessDay(1)
    Timestamp('2022-08-08 00:00:00')
    """

    _prefix = "C"
    _attributes = tuple(
        ["n", "normalize", "weekmask", "holidays", "calendar", "offset"]
    )

    _apply_array = BaseOffset._apply_array

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
            return BDay(self.n, offset=td.to_pytimedelta(), normalize=self.normalize)
        else:
            raise ApplyTypeError(
                "Only know how to combine trading day with "
                "datetime, datetime64 or timedelta."
            )

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

    Examples
    --------
    >>> ts = pd.Timestamp(2022, 8, 5, 16)
    >>> ts + pd.offsets.CustomBusinessHour()
    Timestamp('2022-08-08 09:00:00')
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



BDay = BusinessDay
CDay = CustomBusinessDay

# ----------------------------------------------------------------------
# to_offset helpers

prefix_mapping = {
    offset._prefix: offset
    for offset in [
        BusinessDay,  # 'B'
        BusinessHour,  # 'BH'
        CustomBusinessDay,  # 'C'
        CustomBusinessHour,  # 'CBH'
        Nano,  # 'N'
        Second,  # 'S'
        Minute,  # 'T'
        Micro,  # 'U'
        Milli,  # 'L'
        Hour,  # 'H'
        Day,  # 'D'
    ]
}

# hack to handle WOM-1MON
opattern = re.compile(
    r"([+\-]?\d*|[+\-]?\d*\.\d*)\s*([A-Za-z]+([\-][\dA-Za-z\-]+)?)"
)

_lite_rule_alias = {
    "W": "W-SUN",
    "Q": "Q-DEC",

    "A": "A-DEC",      # YearEnd(month=12),
    "Y": "A-DEC",
    "AS": "AS-JAN",    # YearBegin(month=1),
    "YS": "AS-JAN",
    "BA": "BA-DEC",    # BYearEnd(month=12),
    "BY": "BA-DEC",
    "BAS": "BAS-JAN",  # BYearBegin(month=1),
    "BYS": "BAS-JAN",

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
    Return DateOffset object from string or datetime.timedelta object.

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

cdef datetime _shift_day(datetime other, int days):
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
cdef ndarray shift_quarters(
    ndarray dtindex,
    int quarters,
    int q1start_month,
    str day_opt,
    int modby=3,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
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
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    out : ndarray[int64_t]
    """
    if day_opt not in ["start", "end", "business_start", "business_end"]:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    cdef:
        Py_ssize_t count = dtindex.size
        ndarray out = cnp.PyArray_EMPTY(dtindex.ndim, dtindex.shape, cnp.NPY_INT64, 0)
        Py_ssize_t i
        int64_t val, res_val
        int months_since, n
        npy_datetimestruct dts
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, dtindex)

    with nogil:
        for i in range(count):
            # Analogous to: val = dtindex[i]
            val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

            if val == NPY_NAT:
                res_val = NPY_NAT
            else:
                pandas_datetime_to_datetimestruct(val, reso, &dts)
                n = quarters

                months_since = (dts.month - q1start_month) % modby
                n = _roll_qtrday(&dts, n, months_since, day_opt)

                dts.year = year_add_months(dts, modby * n - months_since)
                dts.month = month_add_months(dts, modby * n - months_since)
                dts.day = get_day_of_month(&dts, day_opt)

                res_val = npy_datetimestruct_to_datetime(reso, &dts)

            # Analogous to: out[i] = res_val
            (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

            cnp.PyArray_MultiIter_NEXT(mi)

    return out


@cython.wraparound(False)
@cython.boundscheck(False)
def shift_months(
    ndarray dtindex,  # int64_t, arbitrary ndim
    int months,
    str day_opt=None,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
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
        int count = dtindex.size
        ndarray out = cnp.PyArray_EMPTY(dtindex.ndim, dtindex.shape, cnp.NPY_INT64, 0)
        int months_to_roll
        int64_t val, res_val

        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(out, dtindex)

    if day_opt not in [None, "start", "end", "business_start", "business_end"]:
        raise ValueError("day must be None, 'start', 'end', "
                         "'business_start', or 'business_end'")

    if day_opt is None:
        # TODO: can we combine this with the non-None case?
        with nogil:
            for i in range(count):
                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    dts.year = year_add_months(dts, months)
                    dts.month = month_add_months(dts, months)

                    dts.day = min(dts.day, get_days_in_month(dts.year, dts.month))
                    res_val = npy_datetimestruct_to_datetime(reso, &dts)

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

    else:
        with nogil:
            for i in range(count):

                # Analogous to: val = i8other[i]
                val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

                if val == NPY_NAT:
                    res_val = NPY_NAT
                else:
                    pandas_datetime_to_datetimestruct(val, reso, &dts)
                    months_to_roll = months

                    months_to_roll = _roll_qtrday(&dts, months_to_roll, 0, day_opt)

                    dts.year = year_add_months(dts, months_to_roll)
                    dts.month = month_add_months(dts, months_to_roll)
                    dts.day = get_day_of_month(&dts, day_opt)

                    res_val = npy_datetimestruct_to_datetime(reso, &dts)

                # Analogous to: out[i] = res_val
                (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

                cnp.PyArray_MultiIter_NEXT(mi)

    return out


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ndarray _shift_bdays(
    ndarray i8other,
    int periods,
    NPY_DATETIMEUNIT reso=NPY_DATETIMEUNIT.NPY_FR_ns,
):
    """
    Implementation of BusinessDay.apply_offset.

    Parameters
    ----------
    i8other : const int64_t[:]
    periods : int
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    ndarray[int64_t]
    """
    cdef:
        Py_ssize_t i, n = i8other.size
        ndarray result = cnp.PyArray_EMPTY(i8other.ndim, i8other.shape, cnp.NPY_INT64, 0)
        int64_t val, res_val
        int wday, nadj, days
        npy_datetimestruct dts
        int64_t DAY_PERIODS = periods_per_day(reso)
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(result, i8other)

    for i in range(n):
        # Analogous to: val = i8other[i]
        val = (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        if val == NPY_NAT:
            res_val = NPY_NAT
        else:
            # The rest of this is effectively a copy of BusinessDay.apply
            nadj = periods
            weeks = nadj // 5
            pandas_datetime_to_datetimestruct(val, reso, &dts)
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

            res_val = val + (7 * weeks + days) * DAY_PERIODS

        # Analogous to: out[i] = res_val
        (<int64_t*>cnp.PyArray_MultiIter_DATA(mi, 0))[0] = res_val

        cnp.PyArray_MultiIter_NEXT(mi)

    return result


def shift_month(stamp: datetime, months: int, day_opt: object = None) -> datetime:
    """
    Given a datetime (or Timestamp) `stamp`, an integer `months` and an
    option `day_opt`, return a new datetimelike that many months later,
    with day determined by `day_opt` using relativedelta semantics.

    Scalar analogue of shift_months.

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
