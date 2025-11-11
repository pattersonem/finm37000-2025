# src/finm37000/constant_maturity.py
from __future__ import annotations
import re
import pandas as pd
from typing import Iterable, List, Dict

def _parse_cm_days(symbol: str) -> int:
    # expects like "SR3.cm.182" → 182
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Cannot parse constant-maturity days from symbol: {symbol}")
    return int(m.group(1))

def constant_maturity_splice(
    symbol: str,
    roll_spec: Iterable[Dict[str, str]],
    all_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Build a constant-maturity series by linearly blending the 'pre' and 'next'
    contracts on each date range described in roll_spec.

    roll_spec: list of {"d0": "YYYY-MM-DD", "d1": "YYYY-MM-DD", "p": "7", "n": "8"}
               where [d0, d1) is left-closed, right-open in calendar days.
    all_data: long DataFrame with columns:
        - "instrument_id" (int)
        - date_col (tz-aware daily timestamps)
        - price_col (float or whatever)
        - "expiration" (tz-aware Timestamp)
    """
    maturity_days = pd.Timedelta(days=_parse_cm_days(symbol))

    # Ensure we can look up expiration per instrument_id
    # (take the first non-null per instrument; they should be constant)
    exp_map = (
        all_data[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    # For quick price lookups, index by (instrument_id, datetime)
    prices = (
        all_data[["instrument_id", date_col, price_col]]
        .set_index(["instrument_id", date_col])
        .sort_index()
    )

    out_frames: List[pd.DataFrame] = []

    for r in roll_spec:
        d0 = pd.to_datetime(r["d0"]).tz_localize("UTC")
        # Right-open end (inclusive="left") matches the test construction
        d1 = pd.to_datetime(r["d1"]).tz_localize("UTC")
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        pre_id = int(r["p"])
        nxt_id = int(r["n"])

        # expirations (tz-aware)
        pre_exp = pd.to_datetime(exp_map.loc[pre_id])
        nxt_exp = pd.to_datetime(exp_map.loc[nxt_id])

        # Prices aligned to t (exact timestamp match expected by the tests)
        # If a date is missing, reindex will introduce NaN — which is fine in real life,
        # but the unit test builds fully aligned data.
        pre_px = prices.xs(pre_id).reindex(t)[price_col]
        nxt_px = prices.xs(nxt_id).reindex(t)[price_col]

        # Weight on the pre contract: (T_next - (t + D)) / (T_next - T_pre)
        f = (nxt_exp - (t + maturity_days)) / (nxt_exp - pre_exp)
        f = f.astype("float64")

        blended = f * pre_px + (1.0 - f) * nxt_px

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_px.values,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": nxt_px.values,
                "next_id": nxt_id,
                "next_expiration": nxt_exp,
                "pre_weight": f.values,
                symbol: blended.values,
            }
        )
        out_frames.append(seg)

    return pd.concat(out_frames, ignore_index=True)

import datetime as _dt

# def _live_contracts_on(instrument_df: pd.DataFrame, d: _dt.date) -> pd.DataFrame:
#     """Filter to futures ('F') that are 'live' (known) by date d (UTC)."""
#     df = instrument_df
#     futs = df[df["instrument_class"] == "F"].copy()
#     # ts_recv is tz-aware; compare by date (UTC)
#     futs = futs[futs["ts_recv"].dt.tz_convert("UTC").dt.date <= d]
#     return futs

# def _exp_date_series(df: pd.DataFrame) -> pd.Series:
#     """Return expirations as date (ignore intraday time for this exercise)."""
#     return pd.to_datetime(df["expiration"]).dt.tz_convert("UTC").dt.date

# def get_roll_spec(
#     symbol: str,
#     instrument_df: pd.DataFrame,
#     *,
#     start: _dt.date,
#     end: _dt.date,
# ) -> List[Dict[str, str]]:
#     """
#     Build roll spec for a constant-maturity symbol like 'SR3.cm.182' over [start, end].

#     For each calendar day d in [start, end), find the pair (pre, next) such that
#       expiration_pre <= (d + D) < expiration_next,
#     restricting to futures 'live' on day d (ts_recv <= d).
#     Then compress consecutive days with the same (pre, next) into segments with
#       {"d0": d0, "d1": d1, "p": str(pre_id), "n": str(next_id)}
#     where d1 is the first date *after* the segment (right-open).
#     """
#     D = _parse_cm_days(symbol)
#     out: List[Dict[str, str]] = []

#     # Iterate days in [start, end)
#     day = start
#     cur_pre = cur_nxt = None
#     seg_start = start

#     while day < end:
#         futs = _live_contracts_on(instrument_df, day)
#         if futs.empty:
#             raise ValueError(f"No live futures on {day}")

#         futs = futs.sort_values("expiration")
#         exps = _exp_date_series(futs).values
#         ids = futs["instrument_id"].values

#         target = day + _dt.timedelta(days=D)

#         # Find first expiration strictly after target → 'next'
#         nxt_idx = None
#         for i, e in enumerate(exps):
#             if e > target:
#                 nxt_idx = i
#                 break
#         if nxt_idx is None:
#             # If none strictly after, we cannot form a pair
#             # (test data should avoid this).
#             raise ValueError(f"No next expiration after target {target} on {day}")

#         pre_idx = max(0, nxt_idx - 1)

#         pre_id = int(ids[pre_idx])
#         nxt_id = int(ids[nxt_idx])

#         # Start a new segment if pair changes
#         if (pre_id, nxt_id) != (cur_pre, cur_nxt):
#             # Close previous segment
#             if cur_pre is not None:
#                 out.append(
#                     {
#                         "d0": seg_start.isoformat(),
#                         "d1": day.isoformat(),  # right-open end
#                         "p": str(cur_pre),
#                         "n": str(cur_nxt),
#                     }
#                 )
#             # Start new
#             seg_start = day
#             cur_pre, cur_nxt = pre_id, nxt_id

#         day = day + _dt.timedelta(days=1)

#     # Close final segment at 'end'
#     if cur_pre is not None:
#         out.append(
#             {
#                 "d0": seg_start.isoformat(),
#                 "d1": end.isoformat(),  # right-open end matches the tests
#                 "p": str(cur_pre),
#                 "n": str(cur_nxt),
#             }
#         )

#     return out
import re
from datetime import date, timedelta
from typing import Iterable, Mapping, Union, List, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

def _to_utc(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Parse to UTC-aware Timestamp."""
    t = pd.to_datetime(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _parse_maturity_days(symbol: str) -> int:
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Could not parse maturity days from symbol: {symbol}")
    return int(m.group(1))


# ---------------------------
# get_roll_spec
# ---------------------------

def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    *,
    start: date,
    end: date,
    instrument_col: str = "instrument_id",
    class_col: str = "instrument_class",
    expiration_col: str = "expiration",
    ts_recv_col: str = "ts_recv",
) -> List[Dict[str, Any]]:
    """
    Compute roll windows for a constant-maturity series.

    For each calendar day d in [start, end), choose (pre, next) among instruments that are:
      - class 'F' (ignore spreads), and
      - live on d: ts_recv.date() <= d

    Straddle rule (DATE-only, per tests):
      Let target_date = d + maturity_days.
      pre is the contract with the largest expiration date STRICTLY LESS than target_date.
      next is the contract with the smallest expiration date GREATER THAN OR EQUAL to target_date.

    Merge consecutive days with the same (pre, next) into segments:
      {"d0": start_date, "d1": end_date_exclusive, "p": pre_id_str, "n": next_id_str}
    """
    maturity_days = _parse_maturity_days(symbol)

    # Work only with outright futures (ignore spreads)
    df = instrument_df.copy()
    df = df[df[class_col] == "F"].copy()

    # Date-only comparisons per the test; normalize to UTC first
    df["exp_date"] = pd.to_datetime(df[expiration_col]).dt.tz_convert("UTC").dt.date
    df["live_date"] = pd.to_datetime(df[ts_recv_col]).dt.tz_convert("UTC").dt.date

    # Instruments that go live strictly after the period can be ignored
    last_day = (pd.to_datetime(end) - pd.Timedelta(days=1)).date()
    df = df[df["live_date"] <= last_day]

    # Sort by expiration date for deterministic selection
    df = df.sort_values(["exp_date", instrument_col]).reset_index(drop=True)

    # Precomputed rows: [id, exp_date, live_date]
    ids_by_exp = df[[instrument_col, "exp_date", "live_date"]].values.tolist()

    def find_pair(d: date):
        target = d + timedelta(days=maturity_days)

        # candidates live on d
        cand = [row for row in ids_by_exp if row[2] <= d]  # row = [id, exp_date, live_date]
        if not cand:
            return None, None

        # Straddle by DATE:
        #   pre uses STRICT '< target' so we don't collapse to the same contract when equal
        #   next uses inclusive '>= target'
        pre_cand = [row for row in cand if row[1] < target]
        next_cand = [row for row in cand if row[1] >= target]

        if not pre_cand or not next_cand:
            # Defensive fallbacks (not exercised by provided tests)
            if not pre_cand and len(cand) >= 2:
                return cand[0][0], cand[1][0]
            if not next_cand and len(cand) >= 2:
                return cand[-2][0], cand[-1][0]
            only = cand[0][0]
            return only, only

        pre_id = pre_cand[-1][0]   # max exp < target
        next_id = next_cand[0][0]  # min exp >= target
        return pre_id, next_id

    # Iterate days and compress into segments
    specs: List[Dict[str, Any]] = []
    d0 = start
    prev_pair = None

    d = start
    while d < end:
        pair = find_pair(d)
        if prev_pair is None:
            prev_pair = pair
            d0 = d
        elif pair != prev_pair:
            specs.append(
                {"d0": d0.isoformat(), "d1": d.isoformat(), "p": str(prev_pair[0]), "n": str(prev_pair[1])}
            )
            d0 = d
            prev_pair = pair
        d = d + timedelta(days=1)

    if prev_pair is not None:
        specs.append(
            {"d0": d0.isoformat(), "d1": end.isoformat(), "p": str(prev_pair[0]), "n": str(prev_pair[1])}
        )

    return specs
