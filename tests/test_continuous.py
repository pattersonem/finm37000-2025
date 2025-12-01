from io import StringIO

import numpy as np
import pandas as pd
import pytest

from finm37000 import additive_splice, multiplicative_splice, tz_chicago


@pytest.mark.parametrize(
    "time",
    [
        {
            651434: pd.date_range("2025-09-01", "2025-10-13"),
            432669: pd.date_range("2025-09-05", "2025-10-05"),
        },
        {
            651434: pd.date_range("2025-09-01", "2025-10-13", tz="UTC"),
            432669: pd.date_range("2025-09-05", "2025-10-05", tz="UTC"),
        },
        {
            651434: pd.date_range("2025-09-01", "2025-10-13", tz=tz_chicago),
            432669: pd.date_range("2025-09-05", "2025-10-05", tz=tz_chicago),
        },
    ],
)
def test_additive_splice(time: dict[int, pd.Series]) -> None:
    # db.Historical().symbology.resolve(
    #     dataset=cme,
    #     symbols=continuous_by_oi,
    #     stype_in="continuous",
    #     stype_out="instrument_id",
    #     start_date="2025-09-12",
    #     end_date="2025-10-10",
    # )["result"]["CL.n.0"]
    roll_spec = [
        {"d0": "2025-09-12", "d1": "2025-09-17", "s": "651434"},
        {"d0": "2025-09-17", "d1": "2025-09-28", "s": "432669"},
        {"d0": "2025-09-28", "d1": "2025-10-10", "s": "651434"},
    ]
    df = {}
    prices = [
        np.cumsum(range(43)),
        np.cumsum(range(31)),
    ]
    df[0] = pd.DataFrame(
        {
            "instrument_id": 651434,
            "datetime": time[651434],
            "price": prices[0],
            "alt_price": np.arange(len(prices[0])),
        },
    )
    df[1] = pd.DataFrame(
        {
            "instrument_id": 432669,
            "datetime": time[432669],
            "price": prices[1],
            "alt_price": np.arange(len(prices[1])),
        },
    )

    unadjusted_splice = np.concat(
        [prices[0][11:16], prices[1][12:23], prices[0][27:39]],
    )
    unadjusted_alt = np.concat(
        [np.arange(11, 16), np.arange(12, 23), np.arange(27, 39)],
    )
    adjustment = [
        prices[0][15] - prices[1][11],
        prices[1][22] - prices[0][26],
    ]
    adjusted_price = np.concat(
        [
            unadjusted_splice[0:5],
            adjustment[0] + unadjusted_splice[5:16],
            adjustment[0] + adjustment[1] + unadjusted_splice[16:28],
        ],
    ).astype(float)
    adjusted_alt = np.concat(
        [
            unadjusted_alt[0:5],
            adjustment[0] + unadjusted_alt[5:16],
            adjustment[0] + adjustment[1] + unadjusted_alt[16:28],
        ],
    ).astype(float)
    cumulative_adjustment = np.concat(
        [
            np.repeat(0.0, 5),
            np.repeat(adjustment[0], 11),
            np.repeat(adjustment[0] + adjustment[1], 12),
        ],
    )
    tz = time[651434].tz
    expected = pd.DataFrame(
        {
            "instrument_id": 5 * [651434] + 11 * [432669] + 12 * [651434],
            "datetime": pd.date_range("2025-09-12", "2025-10-09", tz=tz),
            "price": adjusted_price,
            "alt_price": unadjusted_alt,
            "additive_adjustment": cumulative_adjustment,
        },
    )
    all_data = pd.concat(df)
    actual = additive_splice(
        roll_spec,
        all_data,
        date_col="datetime",
        adjust_by="price",
    )
    pd.testing.assert_frame_equal(actual, expected)
    actual = additive_splice(
        roll_spec,
        all_data,
        date_col="datetime",
        adjust_by="price",
        adjustment_cols=["price", "alt_price"],
    )
    expected["alt_price"] = adjusted_alt
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "time",
    [
        {
            651434: pd.date_range("2025-09-01", "2025-10-13"),
            432669: pd.date_range("2025-09-05", "2025-10-05"),
        },
        {
            651434: pd.date_range("2025-09-01", "2025-10-13", tz="UTC"),
            432669: pd.date_range("2025-09-05", "2025-10-05", tz="UTC"),
        },
        {
            651434: pd.date_range("2025-09-01", "2025-10-13", tz=tz_chicago),
            432669: pd.date_range("2025-09-05", "2025-10-05", tz=tz_chicago),
        },
    ],
)
def test_multiplicative_splice(time: dict[int, pd.Series]) -> None:
    # db.Historical().symbology.resolve(
    #     dataset=cme,
    #     symbols=continuous_by_oi,
    #     stype_in="continuous",
    #     stype_out="instrument_id",
    #     start_date="2025-09-12",
    #     end_date="2025-10-10",
    # )["result"]["CL.n.0"]
    roll_spec = [
        {"d0": "2025-09-12", "d1": "2025-09-17", "s": "651434"},
        {"d0": "2025-09-17", "d1": "2025-09-28", "s": "432669"},
        {"d0": "2025-09-28", "d1": "2025-10-10", "s": "651434"},
    ]
    df = {}
    prices = [
        np.cumsum(range(43)),
        np.cumsum(range(31)),
    ]
    df[0] = pd.DataFrame(
        {
            "instrument_id": 651434,
            "datetime": time[651434],
            "price": prices[0],
            "alt_price": np.arange(len(prices[0])),
        },
    )
    df[1] = pd.DataFrame(
        {
            "instrument_id": 432669,
            "datetime": time[432669],
            "price": prices[1],
            "alt_price": np.arange(len(prices[1])),
        },
    )

    unadjusted_splice = np.concat(
        [prices[0][11:16], prices[1][12:23], prices[0][27:39]],
    )
    unadjusted_alt = np.concat(
        [np.arange(11, 16), np.arange(12, 23), np.arange(27, 39)],
    )
    adjustment = [
        prices[0][15] / prices[1][11],
        prices[1][22] / prices[0][26],
    ]
    adjusted_price = np.concat(
        [
            unadjusted_splice[0:5],
            adjustment[0] * unadjusted_splice[5:16],
            adjustment[0] * adjustment[1] * unadjusted_splice[16:28],
        ],
    )
    adjusted_alt = np.concat(
        [
            unadjusted_alt[0:5],
            adjustment[0] * unadjusted_alt[5:16],
            adjustment[0] * adjustment[1] * unadjusted_alt[16:28],
        ],
    )
    cumulative_adjustment = np.concat(
        [
            np.repeat(1.0, 5),
            np.repeat(adjustment[0], 11),
            np.repeat(adjustment[0] * adjustment[1], 12),
        ],
    )
    tz = time[651434].tz
    expected = pd.DataFrame(
        {
            "instrument_id": 5 * [651434] + 11 * [432669] + 12 * [651434],
            "datetime": pd.date_range("2025-09-12", "2025-10-09", tz=tz),
            "price": adjusted_price,
            "alt_price": unadjusted_alt,
            "multiplicative_adjustment": cumulative_adjustment,
        },
    )
    all_data = pd.concat(df)
    actual = multiplicative_splice(
        roll_spec,
        all_data,
        date_col="datetime",
        adjust_by="price",
    )
    pd.testing.assert_frame_equal(actual, expected)
    actual = multiplicative_splice(
        roll_spec,
        all_data,
        date_col="datetime",
        adjust_by="price",
        adjustment_cols=["price", "alt_price"],
    )
    expected["alt_price"] = adjusted_alt
    pd.testing.assert_frame_equal(actual, expected)


def test_simple_additive_splice_example() -> None:
    df_csv = """datetime,instrument_id,close
2024-02-27,1,100
2024-02-28,1,102
2024-02-27,2,101
2024-02-28,2,105
2024-03-02,2,107
2024-03-03,2,103
"""
    df = pd.read_csv(StringIO(df_csv), parse_dates=["datetime"])
    roll_spec = [
        {"d0": "2024-02-27", "d1": "2024-03-02", "s": "1"},
        {"d0": "2024-03-02", "d1": "2024-03-04", "s": "2"},
    ]
    expected = pd.read_csv(
        StringIO(
            (
                "datetime,instrument_id,close,additive_adjustment\n"
                "2024-02-27,1,100.0,0.0\n"
                "2024-02-28,1,102.0,0.0\n"
                "2024-03-02,2,104.0,-3.0\n"
                "2024-03-03,2,100.0,-3.0\n"
            )
        ),
        parse_dates=["datetime"],
    )
    actual = additive_splice(
        roll_spec,
        df,
        date_col="datetime",
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_simple_multiplicative_splice_example() -> None:
    df_csv = """datetime,instrument_id,close
2024-02-27,1,94
2024-02-28,1,96
2024-02-27,2,96
2024-02-28,2,100
2024-03-02,2,102
2024-03-03,2,98
"""
    df = pd.read_csv(StringIO(df_csv), parse_dates=["datetime"])
    roll_spec = [
        {"d0": "2024-02-27", "d1": "2024-03-02", "s": "1"},
        {"d0": "2024-03-02", "d1": "2024-03-04", "s": "2"},
    ]

    expected = pd.read_csv(
        StringIO(
            (
                "datetime,instrument_id,close,multiplicative_adjustment\n"
                "2024-02-27,1,94.0,1.0\n"
                "2024-02-28,1,96.0,1.0\n"
                "2024-03-02,2,97.92,0.96\n"
                "2024-03-03,2,94.08,0.96\n"
            )
        ),
        parse_dates=["datetime"],
    )
    actual = multiplicative_splice(
        roll_spec,
        df,
        date_col="datetime",
    )
    pd.testing.assert_frame_equal(actual, expected)
