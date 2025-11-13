from __future__ import annotations

import datetime as _dt
from typing import List, Dict

import pandas as pd


def _maturity_from_symbol(symbol: str) -> pd.Timedelta:
    """
    Parse constant-maturity days from a symbol like 'SR3.cm.182'
    -> pd.Timedelta(days=182)
    """
    # Split on '.cm.' so we don't care what the product prefix is
    try:
        days_str = symbol.split(".cm.")[1]
    except IndexError:
        raise ValueError(f"Cannot parse constant-maturity days from symbol {symbol!r}")
    days = int(days_str)
    return pd.Timedelta(days=days)


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    start: _dt.date,
    end: _dt.date,
) -> List[Dict[str, str]]:
    """
    Build a roll specification for a constant-maturity future.

    Parameters
    ----------
    symbol : str
        Constant-maturity symbol like 'SR3.cm.182'.
    instrument_defs : DataFrame
        Must contain columns:
        - 'instrument_id' (int)
        - 'expiration' (datetime64[ns, UTC])
        - 'instrument_class' (str, 'F' for futures)
        - 'ts_recv' (datetime64[ns, UTC]) — first time the contract is live
    start, end : date
        Date range for which to build the spec. 'end' is the
        exclusive right bound in the returned segments.

    Returns
    -------
    List[Dict[str, str]]
        Each dict has keys:
        - 'd0' : start date of segment (YYYY-MM-DD)
        - 'd1' : end date (exclusive) of segment (YYYY-MM-DD)
        - 'p'  : previous contract instrument_id (str)
        - 'n'  : next contract instrument_id (str)
    """
    maturity = _maturity_from_symbol(symbol)

    df = instrument_defs.copy()

    # Only actual futures, ignore spreads etc.
    if "instrument_class" in df.columns:
        df = df[df["instrument_class"] == "F"].copy()

    # Ensure datetime columns and also keep date-only versions for gating logic
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    # Work at date granularity for eligibility and for the math in the tests
    df["exp_date"] = df["expiration"].dt.date
    df["live_date"] = df["ts_recv"].dt.date

    # Make sure instrument_id is simple int for comparison
    df["instrument_id"] = df["instrument_id"].astype(int)

    # For safety, sort by expiration
    df = df.sort_values("exp_date").reset_index(drop=True)

    current_pair = None  # type: ignore[assignment]
    current_start: _dt.date | None = None
    specs: List[Dict[str, str]] = []

    # iterate over each calendar day in [start, end)
    day = start
    one_day = _dt.timedelta(days=1)

    while day < end:
        # contracts that are live on this day
        live = df[df["live_date"] <= day]
        if live.empty:
            # No live contracts, skip day (unlikely in our tests)
            day += one_day
            continue

        target_date = (pd.Timestamp(day) + maturity).date()

        # among live contracts, find expirations before/after the target
        before = live[live["exp_date"] < target_date]
        after = live[live["exp_date"] >= target_date]

        if not after.empty and not before.empty:
            # Typical case: straddle target_date
            nxt_row = after.sort_values("exp_date").iloc[0]
            pre_row = before.sort_values("exp_date").iloc[-1]
        elif after.empty:
            # target beyond last expiration: use the last two expiries
            ordered = live.sort_values("exp_date")
            nxt_row = ordered.iloc[-1]
            if len(ordered) >= 2:
                pre_row = ordered.iloc[-2]
            else:
                pre_row = ordered.iloc[-1]
        else:  # before.empty
            # target before first expiration: use first two expiries
            ordered = live.sort_values("exp_date")
            nxt_row = ordered.iloc[0]
            if len(ordered) >= 2:
                pre_row = ordered.iloc[1]
            else:
                pre_row = ordered.iloc[0]

        pre_id = int(pre_row["instrument_id"])
        nxt_id = int(nxt_row["instrument_id"])
        pair = (pre_id, nxt_id)

        if current_pair is None:
            # first day we see a valid pair
            current_pair = pair
            current_start = day
        elif pair != current_pair:
            # pair changed: close previous segment at 'day'
            assert current_start is not None
            specs.append(
                {
                    "d0": current_start.isoformat(),
                    "d1": day.isoformat(),  # day is first date of new pair → exclusive
                    "p": str(current_pair[0]),
                    "n": str(current_pair[1]),
                }
            )
            current_pair = pair
            current_start = day

        day += one_day

    # close final segment running up to 'end'
    if current_pair is not None and current_start is not None:
        specs.append(
            {
                "d0": current_start.isoformat(),
                "d1": end.isoformat(),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            }
        )

    return specs


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Construct a constant-maturity price series from underlying futures.

    Parameters
    ----------
    symbol : str
        Constant-maturity symbol, e.g. 'SR3.cm.182'.
    roll_spec : list of dict
        Output from get_roll_spec. Each dict has keys:
        'd0', 'd1', 'p', 'n' (dates as YYYY-MM-DD strings, ids as strings).
    raw_data : DataFrame
        Must contain:
        - 'instrument_id' (int)
        - date_col (datetime64, may be tz-aware)
        - price_col (float)
        - 'expiration' (datetime64[ns, UTC])
    date_col : str
        Name of the datetime column.
    price_col : str
        Name of the price column to splice.

    Returns
    -------
    DataFrame
        Columns:
        - date_col
        - pre_price, pre_id, pre_expiration
        - next_price, next_id, next_expiration
        - pre_weight
        - <symbol> (weighted price)
    """
    maturity = _maturity_from_symbol(symbol)

    df = raw_data.copy().reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["instrument_id"] = df["instrument_id"].astype(int)

    # ensure sorted for deterministic behaviour
    df = df.sort_values([date_col, "instrument_id"])

    segments: list[pd.DataFrame] = []

    for spec in roll_spec:
        d0_str = spec["d0"]
        d1_str = spec["d1"]
        pre_id = int(spec["p"])
        nxt_id = int(spec["n"])

        d0_date = _dt.date.fromisoformat(d0_str)
        d1_date = _dt.date.fromisoformat(d1_str)

        # mask for [d0, d1) at date granularity
        dates = df[date_col].dt.date
        mask_range = (dates >= d0_date) & (dates < d1_date)

        df_range = df[mask_range]

        pre_df = df_range[df_range["instrument_id"] == pre_id].copy()
        nxt_df = df_range[df_range["instrument_id"] == nxt_id].copy()

        # keep only the relevant columns, rename for clarity
        pre_df = pre_df[[date_col, price_col, "instrument_id", "expiration"]].rename(
            columns={
                price_col: "pre_price",
                "instrument_id": "pre_id",
                "expiration": "pre_expiration",
            }
        )
        nxt_df = nxt_df[[date_col, price_col, "instrument_id", "expiration"]].rename(
            columns={
                price_col: "next_price",
                "instrument_id": "next_id",
                "expiration": "next_expiration",
            }
        )

        # inner join on the datetime column to align prices
        seg = pre_df.merge(nxt_df, on=date_col, how="inner")

        # compute pre_weight as in the test
        # f = (next_exp - (t + maturity)) / (next_exp - pre_exp)
        t = seg[date_col]
        pre_exp = seg["pre_expiration"]
        nxt_exp = seg["next_expiration"]

        # (Timedelta / Timedelta) → float
        f = (nxt_exp - (t + maturity)) / (nxt_exp - pre_exp)
        seg["pre_weight"] = f

        # constant-maturity price
        seg[symbol] = seg["pre_weight"] * seg["pre_price"] + (1.0 - seg["pre_weight"]) * seg[
            "next_price"
        ]

        segments.append(seg)

    result = pd.concat(segments, ignore_index=True)

    # For tests, column order matters; align to what the test builds:
    # ['datetime', 'pre_price', 'pre_id', 'pre_expiration',
    #  'next_price', 'next_id', 'next_expiration', 'pre_weight', symbol]
    # but keep it generic using date_col
    cols = [
        date_col,
        "pre_price",
        "pre_id",
        "pre_expiration",
        "next_price",
        "next_id",
        "next_expiration",
        "pre_weight",
        symbol,
    ]
    # Only re-order if all columns exist
    result = result[cols]

    return result