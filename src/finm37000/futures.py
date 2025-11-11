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

def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    maturity_days = int(symbol.split(".")[-1])

    df = instrument_df.copy()
    df = df[df["instrument_class"].astype(str).str.upper().str.startswith("F")].copy()
    df = df.sort_values("expiration").reset_index(drop=True)

    df["expiration"] = pd.to_datetime(df["expiration"])
    if df["expiration"].dt.tz is None:
        df["expiration"] = df["expiration"].dt.tz_localize("UTC")
    else:
        df["expiration"] = df["expiration"].dt.tz_convert("UTC")

    df["ts_recv"] = pd.to_datetime(df["ts_recv"])
    if df["ts_recv"].dt.tz is None:
        df["ts_recv"] = df["ts_recv"].dt.tz_localize("UTC")
    else:
        df["ts_recv"] = df["ts_recv"].dt.tz_convert("UTC")

    results: list[dict[str, str]] = []
    current = start

    while current < end:
        target = pd.Timestamp(current, tz="UTC") + pd.Timedelta(days=maturity_days)

        live = df[df["ts_recv"] <= pd.Timestamp(current, tz="UTC")]
        if live.empty:
            break

        pre = live[live["expiration"] <= target].tail(1)
        nxt = live[live["expiration"] > target].head(1)
        if pre.empty or nxt.empty:
            break

        pre_id = str(pre["instrument_id"].iloc[0])
        nxt_row = nxt.iloc[0]
        nxt_id = str(nxt_row["instrument_id"])

        cut_ts = nxt_row["expiration"] - pd.Timedelta(days=maturity_days)
        d1_exp = cut_ts.tz_convert("UTC").date() + datetime.timedelta(days=1)

        d1_live = end
        upcoming = df[df["ts_recv"] > pd.Timestamp(current, tz="UTC")]
        if not upcoming.empty:
            next_live_ts = upcoming["ts_recv"].min()
            next_live_date = next_live_ts.date()
            if next_live_date < min(d1_exp, end):
                target2 = pd.Timestamp(next_live_date, tz="UTC") + pd.Timedelta(days=maturity_days)
                live2 = df[df["ts_recv"] <= pd.Timestamp(next_live_date, tz="UTC")]
                pre2 = live2[live2["expiration"] <= target2].tail(1)
                nxt2 = live2[live2["expiration"] > target2].head(1)
                if not pre2.empty and not nxt2.empty:
                    pre2_id = str(pre2["instrument_id"].iloc[0])
                    nxt2_id = str(nxt2["instrument_id"].iloc[0])
                    if (pre2_id, nxt2_id) != (pre_id, nxt_id):
                        d1_live = next_live_date

        d1 = min(d1_exp, d1_live, end)
        if d1 <= current:
            d1 = min(current + datetime.timedelta(days=1), end)
            if d1 <= current:
                break

        results.append({"d0": str(current), "d1": str(d1), "p": pre_id, "n": nxt_id})
        current = d1

    return results