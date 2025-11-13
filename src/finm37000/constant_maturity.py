from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@dataclass(frozen=True)
class _Cols:
    date: str = "datetime"
    price: str = "price"
    inst: str = "instrument_id"
    exp: str = "expiration"


def _coerce_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Ensure given columns are tz-aware UTC datetimes.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if not is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], utc=True)
        else:
            if out[c].dt.tz is None:
                out[c] = out[c].dt.tz_localize("UTC")
            else:
                out[c] = out[c].dt.tz_convert("UTC")
    return out


def _parse_symbol_days(symbol: str) -> pd.Timedelta:
    """
    Parse symbols like 'SR3.cm.182' → 182 days as a Timedelta.
    """
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Unrecognized constant-maturity symbol: {symbol}")
    return pd.to_timedelta(int(m.group(1)), unit="D")


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Build a constant-maturity time series by linearly interpolating in
    days-to-expiry space between a 'pre' and 'next' contract for each date
    segment specified in roll_spec.

    Parameters
    ----------
    symbol
        Constant-maturity symbol like 'SR3.cm.182'.
    roll_spec
        List of dicts with keys:
          - 'd0', 'd1': segment start/end dates in 'YYYY-MM-DD'
                        interpreted as [d0, d1) in calendar days (UTC),
          - 'p': instrument_id of 'pre' contract (string),
          - 'n': instrument_id of 'next' contract (string).
    raw_data
        DataFrame with at least:
          - instrument_id (int),
          - {date_col} (tz-aware datetime),
          - {price_col} (float or int),
          - expiration (tz-aware datetime).

    Returns
    -------
    DataFrame with columns:
      {date_col}, 'pre_price', 'pre_id', 'pre_expiration',
      'next_price', 'next_id', 'next_expiration',
      'pre_weight', and one column named `symbol` containing the
      constant-maturity price series.
    """
    if raw_data.empty or not roll_spec:
        return pd.DataFrame(
            columns=[
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
        )

    C = _Cols(date=date_col, price=price_col)

    required = {C.inst, C.exp, C.date, C.price}
    missing = required - set(raw_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw_data.copy()
    df = _coerce_datetime(df, [C.date, C.exp])

    inst_info = df.drop_duplicates(subset=[C.inst]).set_index(C.inst)
    exp_map = inst_info[C.exp].to_dict()

    maturity = _parse_symbol_days(symbol)

    pieces: list[pd.DataFrame] = []

    for seg in roll_spec:
        # Segment dates are date strings in UTC; interpret as [d0, d1)
        d0 = pd.to_datetime(seg["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(seg["d1"]).tz_localize("UTC")
        pre_id = int(seg["p"])
        nxt_id = int(seg["n"])

        # Filter raw_data to this date window and these two instruments
        mask_time = (df[C.date] >= d0) & (df[C.date] < d1)
        mask_inst = df[C.inst].isin([pre_id, nxt_id])
        sub = df.loc[mask_time & mask_inst, [C.date, C.inst, C.price]]

        if sub.empty:
            continue

        # Pivot to wide format: index = datetime, columns = instrument_id
        prices_wide = (
            sub.pivot(index=C.date, columns=C.inst, values=C.price)
              .sort_index()
        )

        prices_wide = prices_wide.ffill()

        # Ensure both contracts are present
        if pre_id not in prices_wide.columns or nxt_id not in prices_wide.columns:
            continue

        times = prices_wide.index

        pre_exp = pd.to_datetime(exp_map[pre_id])
        nxt_exp = pd.to_datetime(exp_map[nxt_id])

        denom = (nxt_exp - pre_exp)
        if denom == pd.Timedelta(0):
            pre_weight = pd.Series(1.0, index=times)
        else:
            pre_weight = (nxt_exp - (times + maturity)) / denom

        pre_price = prices_wide[pre_id]
        next_price = prices_wide[nxt_id]

        cm_price = (
            pre_weight.to_numpy(dtype=float) * pre_price.to_numpy(dtype=float)
            + (1.0 - pre_weight.to_numpy(dtype=float)) * next_price.to_numpy(dtype=float)
        )

        seg_df = pd.DataFrame(
            {
                C.date: times,
                "pre_price": pre_price.to_numpy(),
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": next_price.to_numpy(),
                "next_id": nxt_id,
                "next_expiration": nxt_exp,
                "pre_weight": pre_weight.to_numpy(dtype=float),
                symbol: cm_price,
            }
        )
        pieces.append(seg_df.reset_index(drop=True))

    if not pieces:
        return pd.DataFrame(
            columns=[
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
        )

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(by=[C.date]).reset_index(drop=True)
    return out
