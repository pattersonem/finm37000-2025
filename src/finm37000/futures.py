"""Extract futures data from databento objects."""

import datetime

import databento as db
import pandas as pd

favorite_def_cols = [
    "instrument_id",
    "raw_symbol",
    "expiration",
    "unit_of_measure",
    "unit_of_measure_qty",
    "min_price_increment",
    "currency",
    "group",
    "exchange",
    "security_type",
    "trading_reference_price",
]


def get_official_stats(raw_stats: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
    """Filter official daily statistics with instrument expiration.

    Args:
        raw_stats: raw daily statistics including columns `instrument_id`,
                   `raw_symbol`, `ts_ref`, `stat_type`, `stat_flags`, `price`,
                    and `quantity` as returned by `databento` clients for
                    futures `statistics` schemas.
        def_df: instrument definitions including columns `instrument_id`,
            `expiration`.

    Returns:
        pd.DataFrame indexed on `Trade date` and `Symbol` with columns
        `Settlement price`, `Cleared volume`, `Open interest`, and `expiration`.

    """
    def_df = def_df[["instrument_id", "expiration", "raw_symbol"]]
    stats_df = raw_stats.merge(def_df, on="instrument_id")
    stats_df = stats_df.rename(columns={"raw_symbol": "Symbol"})
    stats_df["Trade date"] = stats_df["ts_ref"].dt.date
    final_actual_flag = 3
    # CME MDP3 tag 715 SettlPriceType flag: bit 0 = 1 (final) bit 1 = 1 (actual)
    # https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457414586/Settlement+Prices#SettlementPrices-SettlementatTradingTick/SettlementatClearingTick
    # https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457226917/MDP+3.0+-+Settlement+Price
    stats_df["Settlement price"] = stats_df[
        (stats_df["stat_type"] == db.StatType.SETTLEMENT_PRICE)
        & (stats_df["stat_flags"] == final_actual_flag)
    ]["price"]
    stats_df["Cleared volume"] = stats_df[
        stats_df["stat_type"] == db.StatType.CLEARED_VOLUME
    ]["quantity"]
    stats_df["Open interest"] = stats_df[
        stats_df["stat_type"] == db.StatType.OPEN_INTEREST
    ]["quantity"]
    stats_df = (
        stats_df.groupby(["Trade date", "Symbol"])
        .agg("last")
        .sort_values(["Trade date", "expiration"])
    )
    return stats_df[
        ["Settlement price", "Cleared volume", "Open interest", "expiration"]
    ]


def filter_legs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the futures legs from the data.

    :param df: `pd.DataFrame` with an "instrument_class" and "expiration" column.
    :return: Rows of `df` matching `db.InstrumentClass.FUTURE` indexed and sorted by
        "expiration".
    """
    df = df[df["instrument_class"] == db.InstrumentClass.FUTURE]
    df = df.set_index("expiration").sort_index()
    return df


def get_all_legs_on(
    client: db.Historical,
    date: datetime.date,
    parent: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve all futures legs on a given date.

    :param client: Databento client to make data requests.
    :param date: Date on which to get the futures legs.
    :param parent: Futures parent product symbol
    :return: A pair of `pd.DataFrame`, the statistics and the definitions.
    """
    all_defs = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="definition",
        symbols=parent,
        stype_in="parent",
        start=date,
    )
    leg_defs = filter_legs(all_defs.to_df())
    legs = leg_defs["raw_symbol"].unique()
    raw_stats = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="statistics",
        symbols=legs,
        start=date,
    )
    stats = get_official_stats(raw_stats.to_df(), leg_defs.reset_index())
    return stats, leg_defs

# in src/finm37000/futures.py

import datetime as _dt
from typing import List, Dict

import pandas as pd


def _parse_cm_maturity(symbol: str) -> int:
    """
    Parse a constant-maturity symbol like 'SR3.cm.182' -> 182 (days).
    """
    try:
        return int(symbol.split(".")[-1])
    except Exception as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    start: _dt.date,
    end: _dt.date,
    expiration_col: str = "expiration",
    ts_recv_col: str = "ts_recv",
) -> List[Dict[str, str]]:
    """
    Building a roll specification for a constant-maturity future.

    Parameters
    ----------
    symbol
        Constant-maturity symbol like 'SR3.cm.182'.
    instrument_defs
        DataFrame with at least:
        ['instrument_id', 'instrument_class', expiration_col, ts_recv_col].
        Expirations and ts_recv should be datetime-like.
    start, end
        Date range for which to build the spec. `end` is treated as *exclusive*.
    expiration_col, ts_recv_col
        Column names for expiration and listing time.

    Returns
    -------
    List of dictionaries with keys: 'd0', 'd1', 'p', 'n'.
    Dates are ISO strings; 'p' and 'n' are instrument_id strings.
    """
    maturity_days = _parse_cm_maturity(symbol)

    df = instrument_defs.copy()

    df = df[df["instrument_class"] == "F"].copy()

    df[expiration_col] = pd.to_datetime(df[expiration_col])
    df[ts_recv_col] = pd.to_datetime(df[ts_recv_col])
    df["ts_recv_date"] = df[ts_recv_col].dt.date

    specs: List[Dict[str, str]] = []

    current_pair = None
    seg_start: _dt.date | None = None

    cur = start
    while cur < end:
        # Instruments that are live by this date
        avail = df[df["ts_recv_date"] <= cur]
        if avail.empty or avail["instrument_id"].nunique() < 2:
            raise RuntimeError(f"Not enough live instruments on {cur!r}")

        # Sorting by expiration
        avail_sorted = avail.sort_values(expiration_col)
        exp_pairs = list(
            zip(
                avail_sorted["instrument_id"].astype(int).tolist(),
                avail_sorted[expiration_col].tolist(),
            )
        )

        target = (
            pd.Timestamp(cur).tz_localize("UTC")
            + pd.Timedelta(days=maturity_days)
        )

        pre_id = None
        nxt_id = None

        for inst_id, exp in exp_pairs:
            if exp <= target:
                pre_id = inst_id
            elif exp > target and pre_id is not None:
                nxt_id = inst_id
                break

        # Handling edge cases where target is outside expiration range
        if pre_id is None:
            # target before all expirations -> take first two
            pre_id = exp_pairs[0][0]
            nxt_id = exp_pairs[1][0]
        elif nxt_id is None:
            # target after all expirations -> take last two
            pre_id = exp_pairs[-2][0]
            nxt_id = exp_pairs[-1][0]

        pair = (pre_id, nxt_id)

        if current_pair is None:
            current_pair = pair
            seg_start = cur
        elif pair != current_pair:
            assert seg_start is not None
            specs.append(
                {
                    "d0": seg_start.isoformat(),
                    "d1": cur.isoformat(),
                    "p": str(current_pair[0]),
                    "n": str(current_pair[1]),
                }
            )
            seg_start = cur
            current_pair = pair

        cur += _dt.timedelta(days=1)

    
    assert current_pair is not None
    assert seg_start is not None
    specs.append(
        {
            "d0": seg_start.isoformat(),
            "d1": end.isoformat(),
            "p": str(current_pair[0]),
            "n": str(current_pair[1]),
        }
    )

    return specs
