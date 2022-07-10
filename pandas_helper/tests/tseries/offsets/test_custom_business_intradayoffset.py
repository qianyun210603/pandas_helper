"""
Tests for offsets.CustomBusinessHour
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from dateutil.tz.tz import tzlocal
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas._libs.tslibs.offsets import Nano
from pandas_helper._libs.tslibs.offsets import (
    BusinessHour,
    CustomBusinessIntradayOffset,
)

import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
    Base,
    assert_offset_equal,
)

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.compat import IS64

class TestCustomBusinessIntradayOffset(Base):
    _offset: type[CustomBusinessIntradayOffset] = CustomBusinessIntradayOffset
    holidays = ["2014-06-27", datetime(2014, 6, 30), np.datetime64("2014-07-02")]

    def setup_method(self):
        # 2014 Calendar to check custom holidays
        #   Sun Mon Tue Wed Thu Fri Sat
        #  6/22  23  24  25  26  27  28
        #    29  30 7/1   2   3   4   5
        #     6   7   8   9  10  11  12
        self.d = datetime(2014, 7, 1, 10, 00)
        self.offset1 = CustomBusinessIntradayOffset(weekmask="Tue Wed Thu Fri")

        self.offset2 = CustomBusinessIntradayOffset(holidays=self.holidays)

    def test_constructor_errors(self):
        from datetime import time as dt_time

        msg = "time data must be specified only with hour and minute"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessIntradayOffset(start=dt_time(11, 0, 5))
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessIntradayOffset(start="AAA")
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessIntradayOffset(start="14:00:05")

    def test_different_normalize_equals(self):
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = self._offset()
        offset2 = self._offset(normalize=True)
        assert offset != offset2

    def test_repr(self):
        assert repr(self.offset1) == '<CustomBusinessIntradayOffset: CBI=09:00-17:00>'
        assert repr(self.offset2) == '<CustomBusinessIntradayOffset: CBI=09:00-17:00>'

    def test_with_offset(self):
        expected = Timestamp("2014-07-01 13:00")

        assert self.d + CustomBusinessIntradayOffset() * 3 == expected
        assert self.d + CustomBusinessIntradayOffset(n=3) == expected

    def test_eq(self):
        for offset in [self.offset1, self.offset2]:
            assert offset == offset

        assert CustomBusinessIntradayOffset() != CustomBusinessIntradayOffset(-1)
        assert CustomBusinessIntradayOffset(start="09:00") == CustomBusinessIntradayOffset()
        assert CustomBusinessIntradayOffset(start="09:00") != CustomBusinessIntradayOffset(start="09:01")
        assert CustomBusinessIntradayOffset(start="09:00", end="17:00") != CustomBusinessIntradayOffset(
            start="17:00", end="09:01"
        )

        assert CustomBusinessIntradayOffset(weekmask="Tue Wed Thu Fri") != CustomBusinessIntradayOffset(
            weekmask="Mon Tue Wed Thu Fri"
        )
        assert CustomBusinessIntradayOffset(holidays=["2014-06-27"]) != CustomBusinessIntradayOffset(
            holidays=["2014-06-28"]
        )

    def test_sub(self):
        # override the Base.test_sub implementation because self.offsets2 is
        # defined differently in this class than the test expects
        pass

    def test_hash(self):
        assert hash(self.offset1) == hash(self.offset1)
        assert hash(self.offset2) == hash(self.offset2)
        assert hash(self.offset1) != hash(self.offset2)

    def test_call(self):
        with tm.assert_produces_warning(FutureWarning):
            # GH#34171 DateOffset.__call__ is deprecated
            assert self.offset1(self.d) == datetime(2014, 7, 1, 11)
            assert self.offset2(self.d) == datetime(2014, 7, 1, 11)

    def testRollback1(self):
        assert self.offset1.rollback(self.d) == self.d
        assert self.offset2.rollback(self.d) == self.d

        d = datetime(2014, 7, 1, 0)

        # 2014/07/01 is Tuesday, 06/30 is Monday(holiday)
        assert self.offset1.rollback(d) == datetime(2014, 6, 27, 17)

        # 2014/6/30 and 2014/6/27 are holidays
        assert self.offset2.rollback(d) == datetime(2014, 6, 26, 17)

    def testRollback2(self):
        assert self._offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(self):
        assert self.offset1.rollforward(self.d) == self.d
        assert self.offset2.rollforward(self.d) == self.d

        d = datetime(2014, 7, 1, 0)
        assert self.offset1.rollforward(d) == datetime(2014, 7, 1, 9)
        assert self.offset2.rollforward(d) == datetime(2014, 7, 1, 9)

    def testRollforward2(self):
        assert self._offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    def test_roll_date_object(self):
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    normalize_cases = [
        (
            CustomBusinessIntradayOffset(normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        ),
        (
            CustomBusinessIntradayOffset(-1, normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 26),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        ),
        (
            CustomBusinessIntradayOffset(
                1, normalize=True, start="17:00", end="04:00", holidays=holidays
            ),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 3),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        ),
    ]

    @pytest.mark.parametrize("norm_cases", normalize_cases)
    def test_normalize(self, norm_cases):
        offset, cases = norm_cases
        for dt, expected in cases.items():
            assert offset._apply(dt) == expected

    def test_is_on_offset(self):
        tests = [
            (
                CustomBusinessIntradayOffset(start="10:00", end="15:00", holidays=self.holidays),
                {
                    datetime(2014, 7, 1, 9): False,
                    datetime(2014, 7, 1, 10): True,
                    datetime(2014, 7, 1, 15): True,
                    datetime(2014, 7, 1, 15, 1): False,
                    datetime(2014, 7, 5, 12): False,
                    datetime(2014, 7, 6, 12): False,
                },
            )
        ]

        for offset, cases in tests:
            for dt, expected in cases.items():
                assert offset.is_on_offset(dt) == expected

    apply_cases = [
        (
            CustomBusinessIntradayOffset(holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 3, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 10),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            CustomBusinessIntradayOffset(4, holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 3, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
    ]

    @pytest.mark.parametrize("apply_case", apply_cases)
    def test_apply(self, apply_case):
        offset, cases = apply_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    nano_cases = [
        (
            CustomBusinessIntradayOffset(holidays=holidays),
            {
                Timestamp("2014-07-01 15:00")
                + Nano(5): Timestamp("2014-07-01 16:00")
                + Nano(5),
                Timestamp("2014-07-01 16:00")
                + Nano(5): Timestamp("2014-07-03 09:00")
                + Nano(5),
                Timestamp("2014-07-01 16:00")
                - Nano(5): Timestamp("2014-07-01 17:00")
                - Nano(5),
            },
        ),
        (
            CustomBusinessIntradayOffset(-1, holidays=holidays),
            {
                Timestamp("2014-07-01 15:00")
                + Nano(5): Timestamp("2014-07-01 14:00")
                + Nano(5),
                Timestamp("2014-07-01 10:00")
                + Nano(5): Timestamp("2014-07-01 09:00")
                + Nano(5),
                Timestamp("2014-07-01 10:00")
                - Nano(5): Timestamp("2014-06-26 17:00")
                - Nano(5),
            },
        ),
    ]

    @pytest.mark.parametrize("nano_case", nano_cases)
    def test_apply_nanoseconds(self, nano_case):
        offset, cases = nano_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_us_federal_holiday_with_datetime(self):
        # GH 16867
        bhour_us = CustomBusinessIntradayOffset(calendar=USFederalHolidayCalendar())
        t0 = datetime(2014, 1, 17, 15)
        result = t0 + bhour_us * 8
        expected = Timestamp("2014-01-21 15:00:00")
        assert result == expected

    def test_apply_out_of_range(self, request, tz_naive_fixture):
        tz = tz_naive_fixture
        if self._offset is None:
            return

        # try to create an out-of-bounds result timestamp; if we can't create
        # the offset skip
        try:
            offset = self._get_offset(self._offset, value=100000)

            result = Timestamp("20080101") + offset
            assert isinstance(result, datetime)
            assert result.tzinfo is None

            # Check tz is preserved
            t = Timestamp("20080101", tz=tz)
            result = t + offset
            assert isinstance(result, datetime)

            if isinstance(tz, tzlocal) and not IS64:
                # If we hit OutOfBoundsDatetime on non-64 bit machines
                # we'll drop out of the try clause before the next test
                request.node.add_marker(
                    pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
                )
            assert t.tzinfo == result.tzinfo

        except OutOfBoundsDatetime:
            pass
        except (ValueError, KeyError):
            # we are creating an invalid offset
            # so ignore
            pass


@pytest.mark.parametrize(
    "weekmask, expected_time, mult",
    [
        ["Mon Tue Wed Thu Fri Sat", "2018-11-10 09:00:00", 10],
        ["Tue Wed Thu Fri Sat", "2018-11-13 08:00:00", 18],
    ],
)
def test_custom_businesshour_weekmask_and_holidays(weekmask, expected_time, mult):
    # GH 23542
    holidays = ["2018-11-09"]
    bh = CustomBusinessIntradayOffset(
        start="08:00", end="17:00", weekmask=weekmask, holidays=holidays
    )
    result = Timestamp("2018-11-08 08:00") + mult * bh
    expected = Timestamp(expected_time)
    assert result == expected
