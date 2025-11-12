"""Extract futures data from databento objects."""

import datetime
import numpy as np # modify
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



# modify
def _parse_cm_days(symbol: str) -> int:
    """
    Parse a constant-maturity symbol like 'SR3.cm.182' -> 182 (days).
    Falls back to raising ValueError if not found.
    """
    import re
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Could not parse constant-maturity days from symbol: {symbol}")
    return int(m.group(1))


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    *,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    """
    Build a roll specification for a constant-maturity series.

    For each calendar day t in [start, end), choose the pair (pre, next) of
    futures expirations whose expiration dates straddle t + CM_days:
        expiration[pre] <= (t + CM_days) < expiration[next]

    Only instruments with instrument_class == 'F' (futures legs) are considered.
    Contracts are only eligible on a date t if their ts_recv <= t (i.e., they
    have become "live" in the definitions by that date).

    Adjacent days with the same (pre, next) are coalesced into segments with
    dicts of the form:
        {'d0': 'YYYY-MM-DD', 'd1': 'YYYY-MM-DD', 'p': '<pre_id>', 'n': '<next_id>'}
    where d1 is exclusive (left-inclusive, right-exclusive).
    """
    # Ensure expected columns exist
    cols = {c.lower(): c for c in instrument_df.columns}
    exp_col = cols.get("expiration")
    cls_col = cols.get("instrument_class")
    recv_col = cols.get("ts_recv")
    id_col  = cols.get("instrument_id")
    if not all([exp_col, cls_col, recv_col, id_col]):
        raise ValueError("instrument_df must have columns: instrument_id, expiration, instrument_class, ts_recv")

    cm_days = _parse_cm_days(symbol)
    # Filter to futures legs only
    df = instrument_df.copy()
    df = df[df[cls_col].astype(str).str.upper().str.startswith("F")]
    # Normalize types/timezones
    df[exp_col] = pd.to_datetime(df[exp_col], utc=True)
    df[recv_col] = pd.to_datetime(df[recv_col], utc=True)

    # Daily loop over [start, end)
    days = pd.date_range(start=start, end=end, inclusive="left", tz="UTC")
    segments: list[dict[str, str]] = []

    last_pair: tuple[str, str] | None = None
    seg_start: pd.Timestamp | None = None

    for t in days:
        # Instruments visible as of this date
        visible = df[df[recv_col] <= t]
        if visible.empty:
            continue
        # Sorted by expiration ascending
        visible = visible.sort_values(exp_col)
        target = t + pd.Timedelta(days=cm_days)

        # find pre, next so that exp_pre <= target < exp_next
        exps = visible[exp_col].to_numpy()
        ids  = visible[id_col].to_numpy()

        # locate index of first expiration strictly greater than target
        idx_next = int(np.searchsorted(exps, target, side="right"))
        # idx_next must be at least 1 to have a pre
        idx_next = max(1, min(idx_next, len(exps)-1))
        idx_pre = idx_next - 1

        pre_id = str(ids[idx_pre])
        nxt_id = str(ids[idx_next])

        pair = (pre_id, nxt_id)
        if last_pair is None:
            last_pair = pair
            seg_start = t
        elif pair != last_pair:
            # close previous segment
            segments.append(
                {
                    "d0": seg_start.date().isoformat(),
                    "d1": t.date().isoformat(),
                    "p": last_pair[0],
                    "n": last_pair[1],
                }
            )
            last_pair = pair
            seg_start = t

    # close tail segment (if any days iterated)
    if last_pair is not None and seg_start is not None:
        segments.append(
            {
                "d0": seg_start.date().isoformat(),
                "d1": end.isoformat(),  # exclusive
                "p": last_pair[0],
                "n": last_pair[1],
            }
        )

    return segments
