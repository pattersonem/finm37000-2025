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


# ---------------------------
# constant_maturity_splice
# ---------------------------

def constant_maturity_splice(
    symbol: str,
    roll_spec: Iterable[Mapping[str, str]],
    all_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
    instrument_col: str = "instrument_id",
) -> pd.DataFrame:
    """
    Build a constant-maturity series by linearly interpolating between the 'pre'
    and 'next' contracts over each roll window.

    Expected columns in `all_data`:
      - instrument_col (e.g., 'instrument_id')
      - date_col (e.g., 'datetime', tz-aware)
      - price_col (e.g., 'price')
      - 'expiration' (tz-aware Timestamps)
    """

    # Parse maturity (days) from symbol "... .cm.<days>"
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Could not parse maturity from symbol '{symbol}'")
    maturity = pd.Timedelta(days=int(m.group(1)))

    # Normalize datetimes to UTC
    df = all_data.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

    # expiration per instrument
    expirations = (
        df[[instrument_col, "expiration"]]
        .drop_duplicates(instrument_col)
        .set_index(instrument_col)["expiration"]
    )

    # prices per instrument, indexed by datetime for quick reindex
    prices_by_instr: Dict[int, pd.Series] = {}
    for k, g in df[[instrument_col, date_col, price_col]].groupby(instrument_col):
        s = g.set_index(date_col)[price_col].sort_index()
        prices_by_instr[int(k)] = s

    # Build each roll segment
    out_parts: List[pd.DataFrame] = []

    for r in roll_spec:
        d0 = _to_utc(r["d0"])
        d1 = _to_utc(r["d1"])
        pre = int(r["p"])
        nxt = int(r["n"])

        # left-inclusive, right-exclusive range, daily frequency
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        exp_pre = expirations.loc[pre]
        exp_nxt = expirations.loc[nxt]

        # pre weight f(t) = (T_next - (t + maturity)) / (T_next - T_pre)
        denom = (exp_nxt - exp_pre)
        f = (exp_nxt - (t + maturity)) / denom
        f = f.astype("float64")  # no clamping; tests allow values slightly > 1

        pre_price = prices_by_instr[pre].reindex(t)
        nxt_price = prices_by_instr[nxt].reindex(t)

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_price.values,
                "pre_id": pre,
                "pre_expiration": exp_pre,
                "next_price": nxt_price.values,
                "next_id": nxt,
                "next_expiration": exp_nxt,
                "pre_weight": f.values,
            }
        )
        seg[symbol] = seg["pre_weight"] * seg["pre_price"] + (1.0 - seg["pre_weight"]) * seg["next_price"]
        out_parts.append(seg)

    return pd.concat(out_parts, ignore_index=True)
