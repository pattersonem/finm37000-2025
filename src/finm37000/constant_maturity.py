# src/finm37000/constant_maturity.py
from __future__ import annotations

from typing import List, Dict
import pandas as pd
import numpy as np


# -------------------- helpers --------------------

def _parse_maturity_days(symbol: str) -> int:
    """
    Parse constant-maturity days from a symbol like 'SR3.cm.182' -> 182.
    """
    parts = symbol.split(".")
    for i, p in enumerate(parts):
        if p.lower() == "cm" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Could not parse constant maturity from symbol: {symbol}")


def _to_utc(s: pd.Series) -> pd.Series:
    """
    Ensure a datetime series is timezone-aware, converted to UTC.
    """
    dt = pd.to_datetime(s)
    # If tz-naive, localize to UTC; else convert to UTC
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert("UTC")


# -------------------- public API --------------------

def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: pd.Timestamp | pd.Timestamp.date,
    end: pd.Timestamp | pd.Timestamp.date,
) -> List[Dict[str, str]]:
    """
    Build roll segments for a constant-maturity series.

    For each calendar date d in [start, end], compute target_dt = d + maturity_days (UTC midnight).
    Choose futures whose expirations straddle target_dt:
      - 'p' (previous/near): max expiration <= target_dt
      - 'n' (next/far):     min expiration  > target_dt
    Compress consecutive dates with the same (p, n) into segments:
        {'d0': 'YYYY-MM-DD', 'd1': 'YYYY-MM-DD', 'p': '<id>', 'n': '<id>'}
    Boundaries are date-accurate and treated as [d0, d1) (right-exclusive).
    """
    maturity_days = _parse_maturity_days(symbol)

    df = instrument_defs.copy()

    # Only outright futures; ignore spreads
    if "instrument_class" in df.columns:
        df = df[df["instrument_class"] == "F"].copy()

    if "instrument_id" not in df.columns or "expiration" not in df.columns:
        raise ValueError("instrument_defs must contain 'instrument_id' and 'expiration'")

    # Normalize expiration and ts_recv to UTC
    df["expiration"] = _to_utc(df["expiration"])
    if "ts_recv" in df.columns:
        df["ts_recv"] = _to_utc(df["ts_recv"])
    else:
        # If missing, assume available from a very early date
        df["ts_recv"] = pd.Timestamp("1900-01-01", tz="UTC")

    base = df[["instrument_id", "expiration", "ts_recv"]].copy()

    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    pairs: list[tuple[str, str]] = []
    valid_dates: list[pd.Timestamp] = []

    for d in dates:
        d_utc = pd.Timestamp(d.date()).tz_localize("UTC")

        # Only instruments live by this day
        available = base[base["ts_recv"].dt.date <= d_utc.date()]
        if available.empty:
            continue

        target_dt = d_utc + pd.Timedelta(days=maturity_days)

        pre = available[available["expiration"] <= target_dt]
        nxt = available[available["expiration"] > target_dt]
        if pre.empty or nxt.empty:
            # Cannot bracket -> skip this date
            continue

        pre_row = pre.sort_values("expiration").iloc[-1]
        nxt_row = nxt.sort_values("expiration").iloc[0]

        p_id = str(int(pre_row["instrument_id"]))
        n_id = str(int(nxt_row["instrument_id"]))

        pairs.append((p_id, n_id))
        valid_dates.append(d_utc)

    # Compress consecutive runs of identical (p, n)
    segments: List[Dict[str, str]] = []
    if not valid_dates:
        return segments

    end_date = pd.to_datetime(end).date()
    start_idx = 0

    for i in range(1, len(valid_dates) + 1):
        if i == len(valid_dates) or pairs[i] != pairs[i - 1]:
            d0_date = valid_dates[start_idx].date()
            last_day = valid_dates[i - 1].date()
            next_day = (pd.Timestamp(last_day) + pd.Timedelta(days=1)).date()
            # Clamp right boundary to 'end' (exclusive)
            d1_date = min(next_day, end_date)

            # Skip zero-length segments (required by tests)
            if d0_date < d1_date:
                segments.append(
                    {
                        "d0": d0_date.isoformat(),
                        "d1": d1_date.isoformat(),
                        "p": pairs[i - 1][0],
                        "n": pairs[i - 1][1],
                    }
                )
            start_idx = i

    return segments


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict],
    raw_data: pd.DataFrame,
    *,
    date_col: str,
    price_col: str,
    id_col: str = "instrument_id",
    exp_col: str = "expiration",
) -> pd.DataFrame:
    """
    Using roll_spec and per-contract prices, compute the constant-maturity series by
    blending the 'pre' and 'next' contracts each day with weight:

        pre_weight(t) = (exp_next - (t + T)) / (exp_next - exp_pre)

    Returns a DataFrame with columns:
      - 'datetime'
      - 'pre_price', 'pre_id', 'pre_expiration'
      - 'next_price', 'next_id', 'next_expiration'
      - 'pre_weight'
      - a column named exactly as `symbol` with the blended value
    """
    maturity_days = _parse_maturity_days(symbol)
    T = pd.Timedelta(days=maturity_days)

    df = raw_data.copy()

    # Normalize to UTC
    df[date_col] = _to_utc(df[date_col])
    df[exp_col] = _to_utc(df[exp_col])

    # Expiration per instrument_id (constant over time)
    exp_map = (
        df[[id_col, exp_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)[exp_col]
    )

    # Price lookup by (datetime, instrument_id)
    price_lookup = (
        df[[date_col, id_col, price_col]]
        .drop_duplicates()
        .set_index([date_col, id_col])[price_col]
    )

    pieces: list[pd.DataFrame] = []

    for seg in roll_spec:
        d0 = pd.to_datetime(seg["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(seg["d1"]).tz_localize("UTC")

        # Left-inclusive, right-exclusive
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left", freq="D")
        if t.empty:
            continue

        pre_id = int(seg["p"])
        nxt_id = int(seg["n"])

        exp_pre = exp_map.loc[pre_id]
        exp_nxt = exp_map.loc[nxt_id]
        denom = (exp_nxt - exp_pre)

        # Vectorized pre_weight over t
        pre_weight = ((exp_nxt - (t + T)) / denom).astype(float).to_numpy()

        # Vectorized price pulls
        idx_pre = pd.MultiIndex.from_product([t, [pre_id]])
        idx_nxt = pd.MultiIndex.from_product([t, [nxt_id]])
        pre_price = price_lookup.reindex(idx_pre).to_numpy()
        next_price = price_lookup.reindex(idx_nxt).to_numpy()

        seg_df = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_price,
                "pre_id": pre_id,
                "pre_expiration": exp_pre,
                "next_price": next_price,
                "next_id": nxt_id,
                "next_expiration": exp_nxt,
                "pre_weight": pre_weight,
            }
        )
        seg_df[symbol] = seg_df["pre_weight"] * seg_df["pre_price"] + (
            1.0 - seg_df["pre_weight"]
        ) * seg_df["next_price"]

        pieces.append(seg_df)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "datetime",
                "pre_price",
                "pre_id",
                "pre_expiration",
                "next_price",
                "next_id",
                "next_expiration",
                "pre_weight",
                symbol,
            ]
        )

    return pd.concat(pieces, ignore_index=True)
