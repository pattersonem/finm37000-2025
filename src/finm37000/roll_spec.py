# --- get_roll_spec ------------------------------------------------------------
from __future__ import annotations
import datetime as _dt
from typing import List, Dict, Iterable, Tuple
import pandas as pd


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: _dt.date,
    end: _dt.date,
    ts_col: str = "ts_recv",
    exp_col: str = "expiration",
    cls_col: str = "instrument_class",
    id_col: str = "instrument_id",
) -> List[Dict[str, str]]:
    """
    Build roll windows [d0, d1) with the pair of contracts (p, n) whose expirations
    straddle the target maturity date (d + maturity_days), using only instruments that
    were 'live' by that date (ts_recv <= d). Spreads (instrument_class=='S') are ignored.

    Returns a list of dicts with keys: "d0", "d1", "p", "n" (all strings).
    """
    # Parse maturity in days from "...cm.<days>"
    try:
        maturity_days = int(symbol.split(".")[-1])
    except Exception as e:
        raise ValueError(f"Symbol must end with days to maturity, got {symbol!r}") from e
    md = pd.Timedelta(days=maturity_days)

    df = instrument_defs.copy()

    # Normalize datetimes and reduce to futures only
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df[exp_col] = pd.to_datetime(df[exp_col], utc=True)
    if cls_col in df.columns:
        df = df[df[cls_col] == "F"].copy()  # ignore spreads 'S'

    # Work at DATE granularity (tests say “only accurate to the date”)
    df["exp_date"] = df[exp_col].dt.date
    df["ts_date"] = df[ts_col].dt.date

    # Unique instruments with their first ts_recv and expiration date
    inst = (
        df[[id_col, "exp_date", "ts_date"]]
        .drop_duplicates(subset=[id_col])
        .sort_values("exp_date")
        .reset_index(drop=True)
    )

    # Build a daily calendar [start, end)
    days = pd.date_range(start=start, end=end, inclusive="left").date

    # Helper to choose the bracketing contracts for a given day
    def pick_pair(d: _dt.date) -> Tuple[int, int] | None:
        target = (pd.Timestamp(d, tz="UTC") + md).date()
        # Only instruments that are 'live' by d
        pool = inst[inst["ts_date"] <= d]
        if pool.empty:
            return None
        before = pool[pool["exp_date"] < target]
        after  = pool[pool["exp_date"] >= target]
        if before.empty or after.empty:
            return None
        pre_id  = int(before.iloc[-1][id_col])   # latest exp before target
        next_id = int(after.iloc[0][id_col])     # earliest exp on/after target
        # Guard against identical (shouldn’t happen with strictly increasing expiries)
        if pre_id == next_id:
            return None
        return pre_id, next_id

    # Compute pair per day and compress into windows
    specs: List[Dict[str, str]] = []
    cur_pair: Tuple[int, int] | None = None
    window_start: _dt.date | None = None

    for d in days:
        pair = pick_pair(d)
        if pair != cur_pair:
            # Close previous window
            if cur_pair is not None and window_start is not None:
                specs.append(
                    {
                        "d0": window_start.isoformat(),
                        "d1": d.isoformat(),
                        "p": str(cur_pair[0]),
                        "n": str(cur_pair[1]),
                    }
                )
            # Start new window
            cur_pair = pair
            window_start = d

    # Close trailing window at 'end' if a pair was active
    if cur_pair is not None and window_start is not None and window_start < end:
        specs.append(
            {
                "d0": window_start.isoformat(),
                "d1": end.isoformat(),
                "p": str(cur_pair[0]),
                "n": str(cur_pair[1]),
            }
        )

    # Remove any windows where pick_pair returned None (no valid bracketing)
    specs = [s for s in specs if s["p"] != "None" and s["n"] != "None"]
    return specs