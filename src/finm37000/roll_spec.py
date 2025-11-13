from __future__ import annotations

from typing import List, Dict
import re
import pandas as pd


def _parse_symbol_days(symbol: str) -> pd.Timedelta:
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Unrecognized constant-maturity symbol: {symbol}")
    return pd.to_timedelta(int(m.group(1)), unit="D")


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: pd.Timestamp | pd.Date = None,
    end: pd.Timestamp | pd.Date = None,
) -> List[Dict[str, str]]:
    """
    Return segments with fields:
      {'d0': 'YYYY-MM-DD', 'd1': 'YYYY-MM-DD', 'p': str(pre_id), 'n': str(next_id)}
    such that on each US date d in [d0, d1) the pair (pre_id, next_id)
    are the bracketing expirations around (d + maturity_days).
    """
    # Normalize
    df = instrument_defs.copy()
    if "instrument_class" in df:
        df = df[df["instrument_class"] == "F"]

    # Ensure tz-aware datetimes for expiration and ts_recv
    for c in ["expiration", "ts_recv"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC") if df[c].dt.tz is None else pd.to_datetime(df[c])

    maturity = _parse_symbol_days(symbol)

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    # Pre-sort by expiration
    df = df.sort_values("expiration")
    expir_map = df.set_index("instrument_id")["expiration"].to_dict()
    recv_map = df.set_index("instrument_id")["ts_recv"].to_dict()

    # For each date: filter to instruments live by that date; pick bracketing pair
    segments: list[dict[str, str]] = []
    prev_pair: tuple[int, int] | None = None
    seg_start: pd.Date | None = None

    iter_dates = pd.date_range(start=start_date, end=end_date, inclusive="left", tz="UTC").date

    for d in iter_dates:
        # instruments live by date d
        live = [iid for iid, rcv in recv_map.items() if pd.to_datetime(rcv).date() <= d]
        if not live:
            continue
        # target timestamp for maturity calculation: d + maturity
        target = (pd.Timestamp(d).tz_localize("UTC") + maturity)

        # among live instruments, find expirations bracketing target
        live_exps = sorted(((iid, expir_map[iid]) for iid in live), key=lambda x: x[1])
        # find first with exp >= target
        idx_next = None
        for i, (_, exp) in enumerate(live_exps):
            if exp >= target:
                idx_next = i
                break

        if idx_next is None:
            # all expirations < target: pre = last, next = last
            pre_id = live_exps[-1][0]
            nxt_id = live_exps[-1][0]
        elif idx_next == 0:
            # target is before the earliest: pre = earliest, next = earliest
            pre_id = live_exps[0][0]
            nxt_id = live_exps[0][0]
        else:
            pre_id = live_exps[idx_next - 1][0]
            nxt_id = live_exps[idx_next][0]

        pair = (pre_id, nxt_id)
        if prev_pair is None:
            prev_pair = pair
            seg_start = d
        elif pair != prev_pair:
            # close previous segment at this date
            segments.append(
                {
                    "d0": pd.Timestamp(seg_start).date().isoformat(),
                    "d1": pd.Timestamp(d).date().isoformat(),
                    "p": str(prev_pair[0]),
                    "n": str(prev_pair[1]),
                }
            )
            prev_pair = pair
            seg_start = d

    # close last segment at end_date
    if prev_pair is not None and seg_start is not None:
        segments.append(
            {
                "d0": pd.Timestamp(seg_start).date().isoformat(),
                "d1": pd.Timestamp(end_date).date().isoformat(),
                "p": str(prev_pair[0]),
                "n": str(prev_pair[1]),
            }
        )

    return segments
