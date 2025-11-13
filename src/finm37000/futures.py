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


"""Helpers to build constant-maturity time series."""

def _parse_constant_maturity_symbol(symbol: str):
    parts = symbol.split(".")
    if len(parts) < 3 or parts[-2] != "cm":
        raise ValueError(f"Unrecognized constant-maturity symbol: {symbol}")

    product = ".".join(parts[:-2]) or parts[0]

    try:
        maturity = pd.Timedelta(days=int(parts[-1]))
    except ValueError as exc:
        raise ValueError(f"Invalid maturity in symbol {symbol}") from exc

    return product, maturity


def _as_date(value):
    return value if isinstance(value, datetime.date) else pd.Timestamp(value).date()


def _localize(ts, tz):
    """Localize or convert timestamps into the provided timezone."""
    ts = pd.Timestamp(ts)
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


# ----------------------------------------------------------------------
# Roll specification generator
# ----------------------------------------------------------------------

def get_roll_spec(symbol, instrument_defs, start, end):
    product, maturity = _parse_constant_maturity_symbol(symbol)
    start_date, end_date = _as_date(start), _as_date(end)

    if start_date >= end_date:
        raise ValueError("start must be before end")

    df = instrument_defs.copy()
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df["ts_recv_date"] = df["ts_recv"].dt.date

    futures = (
        df[(df["instrument_class"] == "F") &
           df["raw_symbol"].astype(str).str.startswith(product)]
        .sort_values("expiration")
        .reset_index(drop=True)
    )

    if futures.empty:
        raise ValueError(f"No futures found for {product}")

    tz = futures["expiration"].dt.tz
    day, segment_start = start_date, start_date
    current_pair, out = None, []

    while day < end_date:
        live = futures[futures["ts_recv_date"] <= day]
        if live.empty:
            raise ValueError(f"No instruments live on {day.isoformat()}")

        target = _localize(day, tz) + maturity

        pre = live[live["expiration"] <= target]
        pre_row = pre.iloc[-1] if not pre.empty else live.iloc[0]

        nxt = live[live["expiration"] >= target]
        next_row = nxt.iloc[0] if not nxt.empty else live.iloc[-1]

        pair = (int(pre_row["instrument_id"]), int(next_row["instrument_id"]))

        if current_pair is None:
            current_pair = pair
        elif pair != current_pair:
            out.append({
                "d0": segment_start.isoformat(),
                "d1": day.isoformat(),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            })
            current_pair, segment_start = pair, day

        day += datetime.timedelta(days=1)

    out.append({
        "d0": segment_start.isoformat(),
        "d1": end_date.isoformat(),
        "p": str(current_pair[0]),
        "n": str(current_pair[1]),
    })

    return out


# ----------------------------------------------------------------------
# Splicing routine
# ----------------------------------------------------------------------

def constant_maturity_splice(symbol, roll_spec, raw_data,
                             date_col="datetime", price_col="close"):

    if not roll_spec:
        return pd.DataFrame(columns=[
            date_col, "pre_price", "pre_id", "pre_expiration",
            "next_price", "next_id", "next_expiration",
            "pre_weight", symbol
        ])

    _, maturity = _parse_constant_maturity_symbol(symbol)

    df = raw_data.copy()
    if {"instrument_id", "expiration"} - set(df.columns):
        raise KeyError("raw_data must include instrument_id and expiration columns")

    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])
    tz = df[date_col].dt.tz
    groups = df.sort_values(date_col).groupby("instrument_id")

    pieces = []
    for spec in roll_spec:
        d0, d1 = _localize(spec["d0"], tz), _localize(spec["d1"], tz)
        pre_id, next_id = int(spec["p"]), int(spec["n"])

        try:
            pre_grp = groups.get_group(pre_id)
            next_grp = groups.get_group(next_id)
        except KeyError as exc:
            raise KeyError(f"Missing data for instrument {exc.args[0]}") from exc

        pre = pre_grp[(pre_grp[date_col] >= d0) & (pre_grp[date_col] < d1)]
        nxt = next_grp[(next_grp[date_col] >= d0) & (next_grp[date_col] < d1)]

        pre = pre[[date_col, "instrument_id", price_col, "expiration"]].rename(
            columns={
                "instrument_id": "pre_id",
                price_col: "pre_price",
                "expiration": "pre_expiration",
            }
        )
        nxt = nxt[[date_col, "instrument_id", price_col, "expiration"]].rename(
            columns={
                "instrument_id": "next_id",
                price_col: "next_price",
                "expiration": "next_expiration",
            }
        )

        merged = pre.merge(nxt, on=date_col, how="inner")
        if merged.empty:
            continue

        denom = merged["next_expiration"] - merged["pre_expiration"]
        if (denom == pd.Timedelta(0)).any():
            raise ValueError("Zero day spread between expirations")

        maturity_point = merged[date_col] + maturity
        merged["pre_weight"] = (merged["next_expiration"] - maturity_point) / denom
        merged[symbol] = (
            merged["pre_weight"] * merged["pre_price"]
            + (1 - merged["pre_weight"]) * merged["next_price"]
        )

        pieces.append(merged[
            [date_col, "pre_price", "pre_id", "pre_expiration",
             "next_price", "next_id", "next_expiration",
             "pre_weight", symbol]
        ])

    if not pieces:
        raise ValueError("No data matched the supplied roll specification")

    return pd.concat(pieces, ignore_index=True)