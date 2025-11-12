# constant_maturity.py

import pandas as pd
from datetime import timedelta


def parse_maturity_days(symbol: str) -> int:
    """
    Parse something like 'SR3.cm.182' → 182
    """
    return int(symbol.split(".")[-1])

def get_roll_spec(symbol, instrument_defs, start, end):
    maturity_days = int(symbol.split(".")[-1])

    # Keep futures only
    fut = instrument_defs[instrument_defs["instrument_class"] == "F"].copy()

    # Normalize
    fut["expiration"] = pd.to_datetime(fut["expiration"], utc=True)
    fut["ts_recv"] = pd.to_datetime(fut["ts_recv"], utc=True)

    # FULL FIX: convert once to Python date objects
    fut["ts_recv_date"] = fut["ts_recv"].dt.date
    fut["exp_date"] = fut["expiration"].dt.date

    # Work day by day
    dates = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="D").date

    segments = []
    last_pair = None
    seg_start = None

    for dt in dates:
        target = dt + pd.Timedelta(days=maturity_days).to_pytimedelta()

        # Use Python dates ONLY
        active = fut[fut["ts_recv_date"] <= dt]
        after = active[active["exp_date"] >= target]

        if after.empty:
            continue

        # Next = earliest exp_date >= target
        next_row = after.sort_values("exp_date").iloc[0]
        nxt = str(int(next_row["instrument_id"]))

        # Pre = the previous contract in expiration ordering
        ordered = active.sort_values("exp_date")
        idx = ordered.index.get_loc(next_row.name)
        if idx == 0:
            continue
        pre_row = ordered.iloc[idx - 1]
        pre = str(int(pre_row["instrument_id"]))

        pair = (pre, nxt)

        if pair != last_pair:
            # Close previous
            if last_pair is not None:
                segments.append({
                    "d0": seg_start.strftime("%Y-%m-%d"),
                    "d1": dt.strftime("%Y-%m-%d"),
                    "p": last_pair[0],
                    "n": last_pair[1],
                })
            seg_start = pd.to_datetime(dt)
            last_pair = pair

    # Close final segment
    if last_pair is not None:
        segments.append({
            "d0": seg_start.strftime("%Y-%m-%d"),
            "d1": end.strftime("%Y-%m-%d"),
            "p": last_pair[0],
            "n": last_pair[1],
        })

    return segments
