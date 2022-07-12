from __future__ import annotations
from typing import Literal

import pandas as pd
from pandas.core.generic import NDFrame
from pandas.core.groupby.grouper import _check_deprecated_resample_kwargs
from pandas.core.resample import TimeGrouper as _TimeGrouper
from pandas.core.resample import Resampler, DatetimeIndexResampler, PeriodIndexResampler, TimedeltaIndexResampler
from pandas.core.indexes.datetimes import DatetimeIndex, date_range
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas._libs import lib
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    Timedelta,
    Timestamp,
)
from pandas.tseries.offsets import Tick
from pandas_helper.tseries.offsets import BusinessHour
from pandas._typing import TimestampConvertibleTypes, TimedeltaConvertibleTypes

bool_t = bool

class TimeGrouper1(_TimeGrouper):

    def __new__(cls, *args, **kwargs):
        if kwargs.get("freq") is not None:
            _check_deprecated_resample_kwargs(kwargs, origin=cls)
            cls = TimeGrouper1
        return super().__new__(cls)

    def __init__(
        self,
        freq="Min",
        closed: Literal["left", "right"] | None = None,
        label: str | None = None,
        how="mean",
        axis=0,
        fill_method=None,
        limit=None,
        loffset=None,
        kind: str | None = None,
        convention: str | None = None,
        base: int | None = None,
        origin: str | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        **kwargs,
    ):
        if isinstance(freq, Tick):
            raise TypeError(f"Please use resample in offical pandas for {str(freq)}")
        super().__init__(
            freq, closed, label, how, axis, fill_method, limit, loffset, kind, convention, base, origin, offset, **kwargs
        )


    def _get_resampler(self, obj, kind=None):
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : input object
        kind : string, optional
            'period','timestamp','timedelta' are valid

        Returns
        -------
        a Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
        self._set_grouper(obj)

        ax = self.ax
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(obj, groupby=self, kind=kind, axis=self.axis)
        elif isinstance(ax, PeriodIndex) or kind == "period":
            return PeriodIndexResampler(obj, groupby=self, kind=kind, axis=self.axis)
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(obj, groupby=self, axis=self.axis)

        raise TypeError(
            "Only valid with DatetimeIndex, "
            "TimedeltaIndex or PeriodIndex, "
            f"but got an instance of '{type(ax).__name__}'"
        )

    def _get_time_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if len(ax) == 0:
            binner = labels = DatetimeIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels

        first, last = _get_timestamp_range_edges(
            ax.min(),
            ax.max(),
            self.freq,
            closed=self.closed,
        )
        # GH #12037
        # use first/last directly instead of call replace() on them
        # because replace() will swallow the nanosecond part
        # thus last bin maybe slightly before the end if the end contains
        # nanosecond part and lead to `Values falls after last bin` error
        # GH 25758: If DST lands at midnight (e.g. 'America/Havana'), user feedback
        # has noted that ambiguous=True provides the most sensible result
        binner = labels = date_range(
            freq=self.freq,
            start=first,
            end=last,
            tz=ax.tz,
            name=ax.name,
            ambiguous=True,
            nonexistent="shift_forward",
        )

        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)

        # general version, knowing nothing about relative frequencies
        bins = lib.generate_bins_dt64(
            ax_values, bin_edges, self.closed, hasnans=ax.hasnans
        )

        if self.closed == "right":
            labels = binner
            if self.label == "right":
                labels = labels[1:]
        elif self.label == "right":
            labels = labels[1:]

        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)

        # if we end up with more labels than bins
        # adjust the labels
        # GH4076
        if len(bins) < len(labels):
            labels = labels[: len(bins)]

        return binner, bins, labels


def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    _="start_day",
    __: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    """
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    """
    if isinstance(freq, Tick):
        raise TypeError(f"Please use resample in offical pandas for {str(freq)}")
    elif isinstance(freq, BusinessHour):
        first_tmp = first
        first = freq.rollback(first.normalize())
        new_first = first + freq
        while new_first < first_tmp:
            first = new_first
            new_first = first + freq
        if first.time() in freq.end:
            first = freq.rollforward(first + Timedelta(microseconds=1))

        last = Timestamp(last+freq)

    else:

        first = first.normalize()
        last = last.normalize()

        if closed == "left":
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp(first - freq)

        last = Timestamp(last + freq)

    return first, last


def get_resampler(
    obj, kind=None, **kwds
) -> DatetimeIndexResampler | PeriodIndexResampler | TimedeltaIndexResampler:
    """
    Create a TimeGrouper and return our resampler.
    """
    tg = TimeGrouper1(**kwds)
    return tg._get_resampler(obj, kind=kind)


def resample(
    df: NDFrame,
    rule,
    axis=0,
    closed: str | None = None,
    label: str | None = None,
    convention: str = "start",
    kind: str | None = None,
    loffset=None,
    base: int | None = None,
    on=None,
    level=None,
    origin: str | TimestampConvertibleTypes = "start_day",
    offset: TimedeltaConvertibleTypes | None = None,
) -> Resampler:
    """
    Resample time-series data.

    Convenience method for frequency conversion and resampling of time series.
    The object must have a datetime-like index (`DatetimeIndex`, `PeriodIndex`,
    or `TimedeltaIndex`), or the caller must pass the label of a datetime-like
    series/index to the ``on``/``level`` keyword parameter.

    Parameters
    ----------
    rule : DateOffset, Timedelta or str
        The offset string or object representing target conversion.
    axis : {{0 or 'index', 1 or 'columns'}}, default 0
        Which axis to use for up- or down-sampling. For `Series` this parameter
        is unused and defaults to 0. Must be
        `DatetimeIndex`, `TimedeltaIndex` or `PeriodIndex`.
    closed : {{'right', 'left'}}, default None
        Which side of bin interval is closed. The default is 'left'
        for all frequency offsets except for 'M', 'A', 'Q', 'BM',
        'BA', 'BQ', and 'W' which all have a default of 'right'.
    label : {{'right', 'left'}}, default None
        Which bin edge label to label bucket with. The default is 'left'
        for all frequency offsets except for 'M', 'A', 'Q', 'BM',
        'BA', 'BQ', and 'W' which all have a default of 'right'.
    convention : {{'start', 'end', 's', 'e'}}, default 'start'
        For `PeriodIndex` only, controls whether to use the start or
        end of `rule`.
    kind : {{'timestamp', 'period'}}, optional, default None
        Pass 'timestamp' to convert the resulting index to a
        `DateTimeIndex` or 'period' to convert it to a `PeriodIndex`.
        By default the input representation is retained.
    loffset : timedelta, default None
        Adjust the resampled time labels.

        .. deprecated:: 1.1.0
            You should add the loffset to the `df.index` after the resample.
            See below.

    base : int, default 0
        For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '5min' frequency, base could
        range from 0 through 4. Defaults to 0.

        .. deprecated:: 1.1.0
            The new arguments that you should use are 'offset' or 'origin'.

    on : str, optional
        For a DataFrame, column to use instead of index for resampling.
        Column must be datetime-like.
    level : str or int, optional
        For a MultiIndex, level (name or number) to use for
        resampling. `level` must be datetime-like.
    origin : Timestamp or str, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.
        If string, must be one of the following:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries

        .. versionadded:: 1.1.0

        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day

        .. versionadded:: 1.3.0

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

        .. versionadded:: 1.1.0

    Returns
    -------
    pandas.core.Resampler
        :class:`~pandas.core.Resampler` object.
    """

    axis = df._get_axis_number(axis)
    return get_resampler(
        df,
        freq=rule,
        label=label,
        closed=closed,
        axis=axis,
        kind=kind,
        loffset=loffset,
        convention=convention,
        base=base,
        key=on,
        level=level,
        origin=origin,
        offset=offset,
    )