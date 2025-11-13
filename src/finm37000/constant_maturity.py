from __future__ import annotations

import datetime as dt
from typing import Iterable, Mapping, Any, List, Dict

import pandas as pd


def _maturity_days_from_symbol(symbol: str) -> int:
    # "SR3.cm.182" -> 182
    try:
        return int(symbol.split(".")[-1])
    except Exception as exc:
        raise ValueError(f"Cannot parse maturity days from symbol {symbol!r}") from exc


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
) -> List[Dict[str, str]]:
    """
    Build the roll specification for a constant-maturity future.

    Returns a list of dicts with keys:
      - 'd0': start date (YYYY-MM-DD, inclusive)
      - 'd1': end date   (YYYY-MM-DD, exclusive)
      - 'p': instrument_id of the 'previous' contract (as string)
      - 'n': instrument_id of the 'next' contract (as string)
    """
    maturity_days = _maturity_days_from_symbol(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = instrument_df.copy()

    # Only outright futures (ignore spreads etc.)
    df = df[df["instrument_class"] == "F"].copy()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    # Work with pure dates for comparisons (ignore time and tz)
    df["expiration_date"] = df["expiration"].dt.date
    df["ts_recv_date"] = df["ts_recv"].dt.date

    # We'll iterate over [start, end) as daily dates
    last_date = end - dt.timedelta(days=1)
    dates = pd.date_range(start=start, end=last_date, freq="D")

    df = df.sort_values("expiration")

    specs: List[Dict[str, str]] = []
    cur_p: str | None = None
    cur_n: str | None = None
    block_start: pd.Timestamp | None = None

    for d in dates:
        d_date = d.date()

        # Only contracts that are live by this date
        available = df[df["ts_recv_date"] <= d_date]
        if available.empty:
            continue

        target_date = (d + maturity_delta).date()

        # *** key fix: how we "straddle" the target_date ***
        # pre expires strictly before target_date
        # next expires on or after target_date
        pre_mask = available["expiration_date"] < target_date
        nxt_mask = available["expiration_date"] >= target_date

        pre = available[pre_mask]
        nxt = available[nxt_mask]

        if pre.empty or nxt.empty:
            continue

        pre_row = pre.sort_values("expiration").iloc[-1]
        nxt_row = nxt.sort_values("expiration").iloc[0]

        p_id = str(pre_row["instrument_id"])
        n_id = str(nxt_row["instrument_id"])
        pair = (p_id, n_id)

        if cur_p is None:
            # first pair we see
            cur_p, cur_n = pair
            block_start = d
        elif pair != (cur_p, cur_n):
            # pair changed: close previous block at this date d
            assert block_start is not None
            specs.append(
                {
                    "d0": block_start.strftime("%Y-%m-%d"),
                    "d1": d.strftime("%Y-%m-%d"),  # exclusive upper bound
                    "p": cur_p,
                    "n": cur_n,
                }
            )
            cur_p, cur_n = pair
            block_start = d

    # close the final block up to `end` (exclusive)
    if cur_p is not None and block_start is not None:
        specs.append(
            {
                "d0": block_start.strftime("%Y-%m-%d"),
                "d1": end.strftime("%Y-%m-%d"),
                "p": cur_p,
                "n": cur_n,
            }
        )

    return specs


def constant_maturity_splice(
    symbol: str,
    roll_spec: Iterable[Mapping[str, Any]],
    all_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Use roll_spec plus underlying futures data to build a constant-maturity
    price series.

    Returns a DataFrame with columns:
      - 'datetime'
      - 'pre_price', 'pre_id', 'pre_expiration'
      - 'next_price', 'next_id', 'next_expiration'
      - 'pre_weight'
      - <symbol> (e.g. 'SR3.cm.182') with the weighted price
    """
    maturity_days = _maturity_days_from_symbol(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = all_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Expiration for each instrument_id
    expirations = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    # Price series per instrument_id, indexed by datetime
    price_by_id: dict[int, pd.Series] = {
        int(k): g.set_index(date_col)[price_col].sort_index()
        for k, g in df.groupby("instrument_id")
    }

    segments: list[pd.DataFrame] = []

    for r in roll_spec:
        d0 = str(r["d0"])
        d1 = str(r["d1"])
        pre_id = int(r["p"])
        next_id = int(r["n"])

        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        pre_series = price_by_id[pre_id].reindex(t)
        next_series = price_by_id[next_id].reindex(t)

        pre_exp = expirations[pre_id]
        next_exp = expirations[next_id]

        # weight on the "pre" contract, vectorised over t
        f = (next_exp - (t + maturity_delta)) / (next_exp - pre_exp)
        f = pd.Series(f, index=t)

        cm_price = f * pre_series + (1.0 - f) * next_series

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_series.values,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": next_series.values,
                "next_id": next_id,
                "next_expiration": next_exp,
                "pre_weight": f.values,
                symbol: cm_price.values,
            }
        )
        segments.append(seg)

    result = pd.concat(segments, ignore_index=True)
    return result
