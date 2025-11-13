import datetime as _dt
import pandas as pd
from typing import Iterable, Mapping, Any


def constant_maturity_splice(
    symbol: str,
    roll_spec: Iterable[Mapping[str, Any]],
    data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
    instrument_col: str = "instrument_id",
    expiration_col: str = "expiration",
) -> pd.DataFrame:
    """
    Construct a constant–maturity futures time series by linearly interpolating
    between pairs of futures contracts according to the roll schedule.

    Parameters
    ----------
    symbol : str
        The constant–maturity symbol, e.g. "SR3.cm.182".
        The numeric part after ".cm." represents the target maturity in days.

    roll_spec : iterable of dict
        Each dict represents one roll segment with keys:
            "d0": start date (inclusive)
            "d1": end date (exclusive)
            "p" : instrument_id of the near contract (previous)
            "n" : instrument_id of the far contract (next)

    data : pd.DataFrame
        Must contain:
            instrument_col  — contract ID
            date_col        — timestamps
            price_col       — prices
            expiration_col  — contract expiration datetimes

        Multiple contracts should be stacked together in this table.

    date_col, price_col, instrument_col, expiration_col : str
        Column names to use for timestamps, prices, contract ID, expiration.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all roll segments containing:
            date_col,
            "pre_price", "pre_id", "pre_expiration",
            "next_price", "next_id", "next_expiration",
            "pre_weight",
            symbol   (constant–maturity price)

        Returned rows are sorted by date and indexed from 0.
    """
    # ----------------------------------------------------------------------
    # 1. Extract maturity (in days) from the symbol name.
    #    Example: "SR3.cm.182" → 182 days.
    # ----------------------------------------------------------------------
    if ".cm." not in symbol:
        raise ValueError(f"Cannot parse maturity from symbol {symbol!r}")

    try:
        maturity_days = int(symbol.split(".cm.", 1)[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse maturity days from symbol {symbol!r}") from exc

    maturity = pd.Timedelta(days=maturity_days)

    # ----------------------------------------------------------------------
    # 2. Build a lookup table for expirations:
    #    instrument_id → expiration timestamp
    # ----------------------------------------------------------------------
    expirations = (
        data[[instrument_col, expiration_col]]
        .drop_duplicates(instrument_col)
        .set_index(instrument_col)[expiration_col]
    )

    # Ensure timestamp column is a proper datetime
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    segments: list[pd.DataFrame] = []

    # ----------------------------------------------------------------------
    # 3. Process each roll segment
    # ----------------------------------------------------------------------
    for spec in roll_spec:
        d0 = pd.to_datetime(spec["d0"]).date()
        d1 = pd.to_datetime(spec["d1"]).date()
        pre_id = int(spec["p"])
        next_id = int(spec["n"])

        # Select rows where the instrument is pre_id or next_id,
        # and the date falls into [d0, d1).
        date_only = data[date_col].dt.date

        mask_pre = (
            (data[instrument_col] == pre_id)
            & (date_only >= d0)
            & (date_only < d1)
        )
        mask_next = (
            (data[instrument_col] == next_id)
            & (date_only >= d0)
            & (date_only < d1)
        )

        pre_df = (
            data.loc[mask_pre, [date_col, price_col]]
            .sort_values(date_col)
            .reset_index(drop=True)
        )
        next_df = (
            data.loc[mask_next, [date_col, price_col]]
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        # If either contract has no data for the segment, skip.
        if pre_df.empty or next_df.empty:
            continue

        # ------------------------------------------------------------------
        # 4. Merge prices on timestamps to align the two contracts
        # ------------------------------------------------------------------
        merged = pre_df.merge(
            next_df,
            on=date_col,
            suffixes=("_pre", "_next"),
            how="inner",
        )

        times = merged[date_col]
        pre_price = merged[f"{price_col}_pre"]
        next_price = merged[f"{price_col}_next"]

        exp_pre = pd.to_datetime(expirations.loc[pre_id])
        exp_next = pd.to_datetime(expirations.loc[next_id])

        # ------------------------------------------------------------------
        # 5. Compute interpolation weight (pre_weight)
        #
        #    Formula matches directly the test code:
        #    weight = (exp_next - (t + maturity)) / (exp_next - exp_pre)
        #
        # ------------------------------------------------------------------
        pre_weight = (exp_next - (times + maturity)) / (exp_next - exp_pre)

        # ------------------------------------------------------------------
        # 6. Build result segment
        # ------------------------------------------------------------------
        seg = pd.DataFrame(
            {
                date_col: times,
                "pre_price": pre_price,
                "pre_id": pre_id,
                "pre_expiration": exp_pre,
                "next_price": next_price,
                "next_id": next_id,
                "next_expiration": exp_next,
                "pre_weight": pre_weight,
            }
        )

        # Constant-maturity price:
        seg[symbol] = pre_weight * pre_price + (1.0 - pre_weight) * next_price

        segments.append(seg)

    # ----------------------------------------------------------------------
    # 7. If no segments were created, return an empty DataFrame
    # ----------------------------------------------------------------------
    if not segments:
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

    # ----------------------------------------------------------------------
    # 8. Concatenate all segments and sort chronologically
    # ----------------------------------------------------------------------
    result = pd.concat(segments, ignore_index=True)
    result = result.sort_values(date_col).reset_index(drop=True)

    return result




def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: Any,
    end: Any,
    instrument_id_col: str = "instrument_id",
    instrument_class_col: str = "instrument_class",
    expiration_col: str = "expiration",
    ts_recv_col: str = "ts_recv",
) -> list[dict[str, str]]:
    """
    Build a roll specification describing which pair of futures contracts
    should be used to construct a constant-maturity series.

    For each calendar date d in [start, end), we:
      1. Keep only futures contracts (instrument_class == "F").
      2. Keep only contracts with ts_recv_date <= d (contracts that are 'live').
      3. Compute target_date = d + maturity_days where the maturity is derived
         from the symbol, e.g. "SR3.cm.182" -> 182 days.
      4. Choose a pair of contracts (pre, next) whose expirations straddle
         target_date:
            - pre  = contract with the latest expiration <= target_date
            - next = contract with the earliest expiration >= target_date
         with reasonable fallbacks at the ends of the curve.

    We then compress consecutive days that share the same (pre, next) pair
    into a single segment with keys:
        d0: start date of segment (inclusive), as YYYY-MM-DD string
        d1: end date of segment (exclusive), as YYYY-MM-DD string
        p : pre instrument_id, as string
        n : next instrument_id, as string
    """
    # ------------------------------------------------------------------
    # 1. Parse the target maturity (in days) from the symbol.
    #    Example: "SR3.cm.182" -> 182 days.
    # ------------------------------------------------------------------
    if ".cm." not in symbol:
        raise ValueError(f"Cannot parse maturity from symbol {symbol!r}")

    try:
        maturity_days = int(symbol.split(".cm.", 1)[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse maturity days from symbol {symbol!r}") from exc

    # ------------------------------------------------------------------
    # 2. Normalize start/end to plain dates
    # ------------------------------------------------------------------
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    # We will iterate over d in [start_date, end_date),
    # so the last usable date is end_date - 1 day.
    last_date = end_date - _dt.timedelta(days=1)

    # ------------------------------------------------------------------
    # 3. Prepare instrument definitions:
    #    - Keep only futures (instrument_class == "F")
    #    - Extract expiration date (ignore time of day)
    #    - Extract ts_recv as date (ignore time and timezone)
    # ------------------------------------------------------------------
    inst = instrument_defs.copy()

    if instrument_class_col in inst.columns:
        inst = inst[inst[instrument_class_col] == "F"].copy()

    inst[expiration_col] = pd.to_datetime(inst[expiration_col])
    inst["_exp_date"] = inst[expiration_col].dt.date

    ts_recv = pd.to_datetime(inst[ts_recv_col])
    # If ts_recv has timezone, drop it and keep just the date.
    if getattr(ts_recv.dt, "tz", None) is not None:
        ts_recv = ts_recv.dt.tz_convert("UTC")
    inst["_ts_recv_date"] = ts_recv.dt.date

    # Sort once by expiration date to keep ordering stable.
    inst = inst.sort_values("_exp_date").reset_index(drop=True)

    # Helper columns as short aliases
    id_col = instrument_id_col

    # ------------------------------------------------------------------
    # 4. Helper: given a date d, choose (pre_id, next_id)
    # ------------------------------------------------------------------
    def pick_pair_for_date(d: _dt.date) -> tuple[int, int] | None:
        """Return (pre_id, next_id) for a single date d, or None if impossible."""
        # Contracts that are live on date d
        live = inst[inst["_ts_recv_date"] <= d]
        if len(live) < 2:
            return None

        # Contracts ordered by expiration date
        live = live.sort_values("_exp_date")
        exp_dates = list(zip(live["_exp_date"], live[id_col]))

        target = d + _dt.timedelta(days=maturity_days)

        # Find first expiration >= target
        next_idx: int | None = None
        for i, (exp_date, _) in enumerate(exp_dates):
            if exp_date >= target:
                next_idx = i
                break

        if next_idx is None:
            # target is after all expirations: use the last two
            if len(exp_dates) < 2:
                return None
            pre_id = exp_dates[-2][1]
            next_id = exp_dates[-1][1]
        elif next_idx == 0:
            # target is before the first expiration: use the first two
            if len(exp_dates) < 2:
                return None
            pre_id = exp_dates[0][1]
            next_id = exp_dates[1][1]
        else:
            # Normal case: use the pair around the target
            pre_id = exp_dates[next_idx - 1][1]
            next_id = exp_dates[next_idx][1]

        return int(pre_id), int(next_id)

    # ------------------------------------------------------------------
    # 5. Walk across the date range and build contiguous segments
    # ------------------------------------------------------------------
    roll_spec: list[dict[str, str]] = []

    prev_pair: tuple[int, int] | None = None
    seg_start: _dt.date | None = None

    d = start_date
    while d <= last_date:
        pair = pick_pair_for_date(d)

        if pair is None:
            # No valid pair for this date: close any open segment and skip.
            if prev_pair is not None and seg_start is not None:
                roll_spec.append(
                    {
                        "d0": seg_start.isoformat(),
                        "d1": d.isoformat(),
                        "p": str(prev_pair[0]),
                        "n": str(prev_pair[1]),
                    }
                )
                prev_pair = None
                seg_start = None
            d += _dt.timedelta(days=1)
            continue

        if prev_pair is None:
            # Start a new segment
            prev_pair = pair
            seg_start = d
        elif pair != prev_pair:
            # Pair changed: close the previous segment at date d
            roll_spec.append(
                {
                    "d0": seg_start.isoformat(),
                    "d1": d.isoformat(),
                    "p": str(prev_pair[0]),
                    "n": str(prev_pair[1]),
                }
            )
            prev_pair = pair
            seg_start = d

        d += _dt.timedelta(days=1)

    # Close the last open segment, if any, using end_date as the exclusive bound.
    if prev_pair is not None and seg_start is not None:
        roll_spec.append(
            {
                "d0": seg_start.isoformat(),
                "d1": end_date.isoformat(),
                "p": str(prev_pair[0]),
                "n": str(prev_pair[1]),
            }
        )

    return roll_spec
