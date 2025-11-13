import pandas as pd
from datetime import timedelta
from typing import List, Dict, Any


# ----------------------------------------------------------------------
# Parse maturity days from symbol, e.g. "SR3.cm.182" -> 182
# ----------------------------------------------------------------------
def _parse_maturity_days(symbol: str) -> int:
    """
    Given a symbol like 'SR3.cm.182', return maturity days (e.g. 182).
    """
    try:
        return int(symbol.split(".")[-1])
    except Exception:
        raise ValueError(f"Cannot parse maturity days from symbol: {symbol}")


# ----------------------------------------------------------------------
# get_roll_spec: build roll schedule to match test behavior EXACTLY
# ----------------------------------------------------------------------
def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start,
    end,
) -> List[Dict[str, Any]]:
    """
    Build roll specification for constant maturity futures.

    This implementation is intentionally "dumb but correct":
    - Work day by day from `start` (inclusive) to `end` (exclusive).
    - For each date d, compute target date T = d + maturity_days.
    - Among FUTURES (instrument_class == "F") that are LIVE on d
      (ts_recv <= d), find:
        * nxt: earliest expiration >= T
        * pre: latest expiration < T
      If both exist, that (pre, nxt) pair applies to day d.
    - Compress consecutive days with the same (pre, nxt) pair into
      segments [d0, d1).
    - Return segments as list of dicts with keys: d0, d1, p, n.

    This logic reproduces the expected specs in tests/test_constant_maturity.py.
    """

    maturity_days = _parse_maturity_days(symbol)

    # Work only with futures, sorted by expiration
    fut = instrument_df[instrument_df["instrument_class"] == "F"].copy()
    fut = fut.sort_values("expiration").reset_index(drop=True)

    # Precompute plain dates for expiration and ts_recv
    fut["exp_date"] = fut["expiration"].dt.date
    fut["ts_date"] = fut["ts_recv"].dt.date

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    # For each calendar date, determine which (pre, nxt) pair applies
    day_pairs: List[tuple] = []
    d = start_date
    while d < end_date:
        # Contracts that are live on date d
        live = fut[fut["ts_date"] <= d]

        target = d + timedelta(days=maturity_days)

        # Next contract: first live expiration >= target
        nxt_candidates = live[live["exp_date"] >= target]
        if nxt_candidates.empty:
            pair = None
        else:
            nxt = nxt_candidates.iloc[0]

            # Pre contract: latest live expiration < target
            pre_candidates = live[live["exp_date"] < target]
            if pre_candidates.empty:
                pair = None
            else:
                pre = pre_candidates.iloc[-1]
                pair = (int(pre.instrument_id), int(nxt.instrument_id))  # type: ignore[attr-defined]

        day_pairs.append((d, pair))
        d += timedelta(days=1)

    # Compress consecutive days with same non-None pair into [d0, d1)
    segments: List[tuple] = []
    cur_pair = None
    seg_start = None

    for d, p in day_pairs:
        if p is None:
            # Close any existing segment
            if cur_pair is not None:
                segments.append((seg_start, d, cur_pair))
                cur_pair = None
                seg_start = None
            continue

        if p != cur_pair:
            # Close previous segment
            if cur_pair is not None:
                segments.append((seg_start, d, cur_pair))
            # Start new segment
            cur_pair = p
            seg_start = d

    # Close the last segment if still open
    if cur_pair is not None:
        segments.append((seg_start, end_date, cur_pair))

    # Convert segments to test-expected dict format
    result: List[Dict[str, Any]] = []
    for d0, d1, (pre_id, nxt_id) in segments:
        result.append(
            {
                "d0": d0.isoformat(),
                "d1": d1.isoformat(),
                "p": str(pre_id),
                "n": str(nxt_id),
            }
        )

    return result


# ----------------------------------------------------------------------
# constant_maturity_splice: match test_constant_maturity_splice exactly
# ----------------------------------------------------------------------
def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, Any]],
    all_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Build a spliced constant-maturity price series.

    This is tuned to match tests/test_constant_maturity.py::test_constant_maturity_splice:

    - For each roll segment r in roll_spec:
        d0, d1 = r["d0"], r["d1"]
        pre, nxt = instrument ids
      we build a time index:
        t = date_range(start=d0, end=d1, inclusive="left", tz="UTC")
      pick prices and expirations for `pre` and `nxt`, then compute:

        f = (next_exp - (t + maturity_days)) / (next_exp - pre_exp)

      and the spliced price:

        symbol_col = f * pre_price + (1 - f) * next_price

    - Column order and names are made identical to the expected DataFrame:
        ["datetime",
         "pre_price", "pre_id", "pre_expiration",
         "next_price", "next_id", "next_expiration",
         "pre_weight",
         symbol]
    """

    maturity_days = _parse_maturity_days(symbol)
    maturity_offset = pd.Timedelta(days=maturity_days)

    df = all_data.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df = df.sort_values([date_col, "instrument_id"])

    segments: List[pd.DataFrame] = []

    for r in roll_spec:
        d0 = r["d0"]
        d1 = r["d1"]
        pre = int(r["p"])
        nxt = int(r["n"])

        # Time grid for this segment [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        # Pull pre & next series; reindex to t for safety
        df_pre = df[df["instrument_id"] == pre].set_index(date_col).reindex(t)
        df_nxt = df[df["instrument_id"] == nxt].set_index(date_col).reindex(t)

        pre_exp = df_pre["expiration"].iloc[0]
        nxt_exp = df_nxt["expiration"].iloc[0]

        # pre-weight f, identical formula to the test
        f = (nxt_exp - (t + maturity_offset)) / (nxt_exp - pre_exp)

        pre_prices = df_pre[price_col].to_numpy()
        nxt_prices = df_nxt[price_col].to_numpy()
        f_vals = f.to_numpy()

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_prices,
                "pre_id": pre,
                "pre_expiration": pre_exp,
                "next_price": nxt_prices,
                "next_id": nxt,
                "next_expiration": nxt_exp,
                "pre_weight": f_vals,
                symbol: f_vals * pre_prices + (1.0 - f_vals) * nxt_prices,
            }
        )

        segments.append(seg)

    return pd.concat(segments, ignore_index=True)

