"""
The goal of these tests is to provide methods to produce constant maturity
futures prices. Producing such values in real cases would require a data
source like a databento historical client, but for ease and speed of testing,
we remove the dependence on the data source. There is an example integration
test `test_real_data` that is skipped by default.

The key components to build are `get_roll_spec` and `constant_maturity_splice`,
which do not require the data source directly, only its outputs `instrument_defs`
and `raw_data`.

In this assignment, you need to build these components in your fork of `finm37000`
package so that the following tests pass as part of running `python -m pytest`.
You may change the tests if you find that my suggested API does not suit you,
but the basic logic of the intended use above should be easily adapted to your
new API. For example, the `databento.Historical.time_series.get_range()`
method produces many fields that I
do not use, and you may prefer to use different columns than I use in the
fake test data, or you may prefer to work directly with `databento.DBNStore`
rather than the `pandas.DataFrame`.

There is some awkward code below to allow static type checking across Python
versions (look for the `# type: ignore` comments or `sys.version_info`).
You only need to support a single version with your tests, and you only
need to get `pytest` to pass. You do not need to address type checking.

You may want to test your implementation against real data in `test_real_data`,
but it is not required.
"""

import datetime
import sys
from io import StringIO

import databento as db
import pandas as pd
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from finm37000 import (
    constant_maturity_splice,
    get_databento_api_key,
    get_roll_spec,
    temp_env,
    us_business_day,
)


def test_get_roll_spec() -> None:
    """Confirm that get_roll_spec finds the right pairs.

    Pairs' expirations should straddle the date plus the number of
    maturity days.

    Include some red herring spreads in the instrument definitions
    as might be found in normal use cases. Also include contracts
    that become live during the requested range (e.g. SR3Q5 and
    SR3V5 as indicated by the ts_recv value).

    Databento gives datetimes in UTC, so that is what we are
    testing against.

    THIS TEST IS ONLY ACCURATE TO THE DATE. That is, the expiration
    time can be ignored for this exercise. Getting it right
    with the time as well is a little trickier.
    """
    t = "ts_recv"
    first_date = pd.to_datetime("2025-01-01").tz_localize("UTC")
    csv_data = """instrument_id,raw_symbol,expiration,instrument_class,ts_recv
    1,SR3V4,2025-01-14 22:00:00+00:00,F,2025-01-01
    2,SR3X4,2025-02-18 22:00:00+00:00,F,2025-01-01
    3,SR3Z4,2025-03-18 21:00:00+00:00,F,2025-01-01
    4,SR3F5,2025-04-15 21:00:00+00:00,F,2025-01-01
    5,SR3F5-SR3H5,2025-04-15 21:00:00+00:00,S,2025-01-01
    6,SR3G5,2025-05-20 21:00:00+00:00,F,2025-01-01
    7,SR3H5,2025-06-17 21:00:00+00:00,F,2025-01-01
    8,SR3J5,2025-07-15 21:00:00+00:00,F,2025-01-01
    9,SR3K5,2025-08-19 21:00:00+00:00,F,2025-01-01
    10,SR3M5-SR3N5,2025-09-16 21:00:00+00:00,S,2025-01-01
    11,SR3M5,2025-09-16 21:00:00+00:00,F,2025-01-01
    12,SR3N5,2025-10-14 21:00:00+00:00,F,2025-01-01
    13,SR3Q5,2025-11-18 22:00:00+00:00,F,2025-01-30
    14,SR3U5,2025-12-16 22:00:00+00:00,F,2025-01-01
    15,SR3V5,2026-01-20 22:00:00+00:00,F,2025-03-31
    20,SR3X5,2026-02-17 22:00:00+00:00,F,2025-04-30
    25,SR3Z5,2026-03-17 22:00:00+00:00,F,2025-01-01
    """

    instrument_df = pd.read_csv(
        StringIO(csv_data), parse_dates=["expiration", "ts_recv"]
    ).assign(ts_recv=lambda df: df["ts_recv"].dt.tz_localize("UTC"))

    instrument_df[t] = pd.to_datetime(instrument_df[t])
    end_march = pd.to_datetime("2025-03-31").date()
    actual = get_roll_spec(
        "SR3.cm.182", instrument_df, start=first_date.date(), end=end_march
    )
    expected = [
        {"d0": "2025-01-01", "d1": "2025-01-15", "p": "7", "n": "8"},
        {"d0": "2025-01-15", "d1": "2025-02-19", "p": "8", "n": "9"},
        {"d0": "2025-02-19", "d1": "2025-03-19", "p": "9", "n": "11"},
        {"d0": "2025-03-19", "d1": "2025-03-31", "p": "11", "n": "12"},
    ]
    assert len(actual) == len(expected)
    for i, spec in enumerate(actual):
        assert spec == expected[i], f"Spec {i} mismatch: {spec} != {expected[i]}"

    actual = get_roll_spec(
        "SR3.cm.273", instrument_df, start=first_date.date(), end=end_march
    )
    expected = [
        {"d0": "2025-01-01", "d1": "2025-01-15", "p": "11", "n": "12"},
        {"d0": "2025-01-15", "d1": "2025-01-30", "p": "12", "n": "14"},
        {"d0": "2025-01-30", "d1": "2025-02-19", "p": "12", "n": "13"},
        {"d0": "2025-02-19", "d1": "2025-03-19", "p": "13", "n": "14"},
        {"d0": "2025-03-19", "d1": "2025-03-31", "p": "14", "n": "25"},
    ]
    assert len(actual) == len(expected)
    for i, spec in enumerate(actual):
        assert spec == expected[i], f"Spec {i} mismatch: {spec} != {expected[i]}"


def test_constant_maturity_splice() -> None:
    symbol = "SR3.cm.182"
    maturity_days = pd.Timedelta(days=182)
    roll_spec = [
        {"d0": "2025-01-01", "d1": "2025-01-14", "p": "7", "n": "8"},
        {"d0": "2025-01-14", "d1": "2025-02-18", "p": "8", "n": "9"},
        {"d0": "2025-02-18", "d1": "2025-03-19", "p": "9", "n": "11"},
        {"d0": "2025-03-19", "d1": "2025-03-31", "p": "11", "n": "12"},
    ]
    expirations = pd.Series(
        {
            7: "2025-06-17 21:00:00+00:00",
            8: "2025-07-15 21:00:00+00:00",
            9: "2025-08-19 21:00:00+00:00",
            11: "2025-09-16 21:00:00+00:00",
            12: "2025-10-14 21:00:00+00:00",
        }
    ).map(pd.to_datetime)

    time = pd.date_range(start="2025-01-01", end="2025-03-30", tz="UTC")

    segments = [
        (slice(0, 13), 7, 8),
        (slice(13, 48), 8, 9),
        (slice(48, 77), 9, 11),
        (slice(77, 89), 11, 12),
    ]

    pre_factors = {}
    if sys.version_info <= (3, 9):
        for idx, pre, nxt in segments:
            f = (expirations[nxt] - (time[idx] + maturity_days)) / (  # type: ignore
                expirations[nxt] - expirations[pre]
            )
            pre_factors[(pre, nxt)] = f
    else:
        for idx, pre, nxt in segments:
            f = (expirations[nxt] - (time[idx] + maturity_days)) / (
                expirations[nxt] - expirations[pre]
            )
            pre_factors[(pre, nxt)] = f

    # Test fake constant prices matching instrument id.
    expected_segments = []
    for r in roll_spec:
        d0, d1 = r["d0"], r["d1"]
        pre, nxt = int(r["p"]), int(r["n"])
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        f = pre_factors[(pre, nxt)]

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre,
                "pre_id": pre,
                "pre_expiration": expirations[pre],
                "next_price": nxt,
                "next_id": nxt,
                "next_expiration": expirations[nxt],
                "pre_weight": f,
                symbol: f * pre + (1 - f) * nxt,  # expected weighted price
            }
        )
        expected_segments.append(seg)

    expected = pd.concat(expected_segments, ignore_index=True)
    instrument_ids = expirations.index
    dfs = {
        k: pd.DataFrame(
            {
                "instrument_id": k,
                "datetime": expected["datetime"],
                "price": k,
                "expiration": expirations[k],
            }
        )
        for k in instrument_ids
    }
    all_data = pd.concat(dfs)
    actual = constant_maturity_splice(
        symbol,
        roll_spec,
        all_data,
        date_col="datetime",
        price_col="price",
    )
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.skip(reason="Integration testing against real data client.")
def test_real_data() -> None:
    with temp_env(DATABENTO_API_KEY=get_databento_api_key()):
        client = db.Historical()
    if sys.version_info <= (3, 9):
        start_of_this_year = (datetime.date(2025, 1, 1) + us_business_day).date()  # type: ignore
    else:
        start_of_this_year = (datetime.date(2025, 1, 1) + us_business_day).date()
    yesterday = (start_of_this_year + 2 * us_business_day).date()

    product = "SR3"
    num_contracts = 4
    constant_maturity = tuple(
        f"{product}.cm.{91 * (i + 1)}" for i in range(num_contracts)
    )
    instrument_defs = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="definition",
        symbols=f"{product}.FUT",
        stype_in="parent",
        start=start_of_this_year,
        end=yesterday,
    ).to_df()
    quarters = ["H", "M", "U", "Z"]
    quarterlies = instrument_defs[
        instrument_defs["raw_symbol"].str.slice(3, 4).isin(quarters)
    ]
    cm_specs = {
        symbol: get_roll_spec(
            symbol,
            quarterlies.reset_index(),
            start=start_of_this_year,
            end=yesterday,
        )
        for symbol in constant_maturity
    }
    cm_instruments = {
        int(spec[key])
        for spec_list in cm_specs.values()
        for spec in spec_list
        for key in (
            "p",
            "n",
        )
    }
    ohlcv = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1d",
        symbols=cm_instruments,
        stype_in="instrument_id",
        start=start_of_this_year,
        end=yesterday,
    ).to_df()
    expirations = quarterlies[["instrument_id", "expiration"]].drop_duplicates()
    ohlcv_with_exp = ohlcv.reset_index().merge(
        expirations, on="instrument_id", how="left"
    )
    ohlcv_with_exp = ohlcv_with_exp.set_index("ts_event")
    cm_pieces = {
        symbol: constant_maturity_splice(
            symbol,
            roll_spec,
            ohlcv_with_exp.reset_index(),
            date_col="ts_event",
            price_col="close",
        )
        for symbol, roll_spec in cm_specs.items()
    }
    assert len(cm_pieces) == num_contracts
    cm_long = [
        pd.wide_to_long(
            df, stubnames=["SR3.cm"], i="ts_event", j="days_to_maturity", sep="."
        )
        for df in cm_pieces.values()
    ]
    ohlcv_cm = pd.concat(cm_long)
    expected_num_dates = 12
    assert len(ohlcv_cm) == expected_num_dates
