"""Time functions for working with exchange data."""

import datetime
from typing import Union, cast
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

DateTimeLike = Union[datetime.date, datetime.datetime, pd.Timestamp]

tz_chicago = ZoneInfo("America/Chicago")
us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def as_ct(
    utc: Union[pd.DatetimeIndex, pd.Series],
) -> Union[pd.DatetimeIndex, pd.Series]:
    """Convert indexes and series to Chicago time."""
    if isinstance(utc, pd.Series):
        result = utc.dt.tz_convert(tz_chicago)
        return cast("pd.Series", result)
    return utc.tz_convert(tz_chicago)


def get_cme_next_session_end(dt: DateTimeLike) -> pd.Timestamp:
    """Offset datetime-like input to 16:00 Chicago.

    :param dt: A datetime-like object with a `date()` method.
    :return:

    >>> get_cme_next_session_end(datetime.date(2025, 10, 9))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_next_session_end(pd.Timestamp("2025-10-09", tz="America/Chicago"))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_next_session_end(pd.Timestamp("2025-10-09 23:00-05:00"))
    Timestamp('2025-10-10 16:00:00-0500', tz='America/Chicago')
    """
    if type(dt) is datetime.date:
        dt = pd.Timestamp.combine(
            dt,
            datetime.time(0, 0),
        ).tz_localize(tz_chicago)
    assert isinstance(dt, datetime.datetime)
    session_switch_chicago = pd.Timestamp.combine(
        dt.date(),
        datetime.time(16, 0),
    ).tz_localize(tz_chicago)
    if dt > session_switch_chicago:
        next_end = session_switch_chicago + us_business_day
    else:
        next_end = session_switch_chicago
    return next_end


def get_cme_session_end(dt: DateTimeLike) -> pd.Timestamp:
    """Set datetime-like input to 16:00 Chicago.

    :param dt: A datetime-like object with a `date()` method.
    :return:

    >>> get_cme_session_end(datetime.date(2025, 10, 9))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_session_end(pd.Timestamp("2025-10-09"))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_session_end(pd.Timestamp("2025-10-09", tz="America/Chicago"))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_session_end(pd.Timestamp("2025-10-09 23:00-05:00"))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_session_end(pd.Timestamp("2025-10-09 23:00"))
    Timestamp('2025-10-09 16:00:00-0500', tz='America/Chicago')
    >>> get_cme_session_end(pd.Timestamp("2025-10-10 01:00"))
    Timestamp('2025-10-10 16:00:00-0500', tz='America/Chicago')
    """
    if type(dt) is datetime.date:
        date = dt
    else:
        assert isinstance(dt, datetime.datetime)
        date = dt.date()
    session_switch_chicago = pd.Timestamp.combine(
        date,
        datetime.time(16, 0),
    ).tz_localize(tz_chicago)
    return session_switch_chicago
