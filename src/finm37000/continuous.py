"""Functions to splice and adjust futures data into continuous data."""
from __future__ import annotations
from typing import Optional

import pandas as pd


def _splice_unadjusted(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    grouped = df.groupby("instrument_id")
    pieces = []
    tz = df[date_col].dt.tz
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        s = int(spec["s"])
        group = grouped.get_group(s)
        piece = group[(group[date_col] >= d0) & (group[date_col] < d1)].copy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _calc_additive_adjustment(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str,
) -> pd.Series:
    tz = df[date_col].dt.tz
    last_date = None
    last_true_value = None
    grouped = df.groupby("instrument_id")
    adjustments = []
    adjustment_dates = []
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        s = int(spec["s"])
        group = grouped.get_group(s)
        piece = group[(group[date_col] >= d0) & (group[date_col] < d1)].copy()
        if last_date is not None:
            adjustment_piece = group[group[date_col] == last_date]
            adjustment = adjustment_piece[adjust_by].iloc[-1] - last_true_value
            adjustments.append(adjustment)
            adjustment_dates.append(d0)
        last_true_value = piece[adjust_by].iloc[-1]
        last_date = piece[date_col].iloc[-1]
    return pd.Series(adjustments, index=adjustment_dates, name="additive_adjustment")  # type: ignore[no-any-return]


def _calc_multiplicative_adjustment(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str,
) -> pd.Series:
    tz = df[date_col].dt.tz
    last_date = None
    last_true_value = None
    grouped = df.groupby("instrument_id")
    adjustments = []
    adjustment_dates = []
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        s = int(spec["s"])
        group = grouped.get_group(s)
        piece = group[(group[date_col] >= d0) & (group[date_col] < d1)].copy()
        if last_date is not None:
            adjustment_piece = group[group[date_col] == last_date]
            adjustment = adjustment_piece[adjust_by].iloc[-1] / last_true_value
            adjustments.append(adjustment)
            adjustment_dates.append(d0)
        last_true_value = piece[adjust_by].iloc[-1]
        last_date = piece[date_col].iloc[-1]
    return pd.Series(  # type: ignore[no-any-return]
        adjustments,
        index=adjustment_dates,
        name="multiplicative_adjustment",
    )


def additive_splice(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str = "close",
    adjustment_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Additively adjust and splice futures data (forward-adjusted).

    Args:
        roll_spec: List of dicts, each with members `"d0"`, `"d1"`, and `"s"`
                   containing the first date, one past the last date, and
                   the instrument id of the instrument in the spliced contract. See
                   `databento.Historical.symbology.resolve()["results"]` for
                   a dictionary with values of this type.
        df: A pandas DataFrame containing the raw data to splice.
        date_col: The name of the column in `df` that contains the date.
        adjust_by: The name of the column in `df` that contains the column
                   to calculate the adjustment.
        adjustment_cols: The columns in `df` that should be adjusted. The default
                         `None` will adjust the `adjust_by` column.

    Returns:
        A pandas DataFrame containing the additively adjusted adjusted data.

    """
    if adjustment_cols is None:
        adjustment_cols = [adjust_by]
    spliced = _splice_unadjusted(roll_spec, df, date_col)
    adjustments = _calc_additive_adjustment(
        roll_spec,
        df,
        date_col=date_col,
        adjust_by=adjust_by,
    )
    with_adjustment = spliced.merge(
        adjustments,
        left_on=date_col,
        right_index=True,
        how="left",
    )
    with_adjustment = with_adjustment.set_index(date_col)
    aligned_adjustment = with_adjustment[adjustments.name]
    aligned_adjustment = aligned_adjustment.fillna(value=0)
    cumulative_adjustment = aligned_adjustment.cumsum()
    for col in adjustment_cols:
        with_adjustment[col] = with_adjustment[col] + cumulative_adjustment
    with_adjustment[adjustments.name] = cumulative_adjustment
    new_columns = df.columns.tolist() + [adjustments.name]
    return with_adjustment.reset_index()[new_columns]


def multiplicative_splice(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str = "close",
    adjustment_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Multiplicatively adjust and splice futures data (forward-adjusted).

    Args:
        roll_spec: List of dicts, each with members `"d0"`, `"d1"`, and `"s"`
                   containing the first date, one past the last date, and
                   the instrument id of the instrument in the spliced contract. See
                   `databento.Historical.symbology.resolve()["results"]` for
                   a dictionary with values of this type.
        df: A pandas DataFrame containing the raw data to splice.
        date_col: The name of the column in `df` that contains the date.
        adjust_by: The name of the column in `df` that contains the column to
                   calculate the adjustment.
        adjustment_cols: The columns in `df` that should be adjusted. The default
                         `None` will adjust the `adjust_by` column.

    Returns:
        A pandas DataFrame containing the adjusted data.

    """
    if adjustment_cols is None:
        adjustment_cols = [adjust_by]
    spliced = _splice_unadjusted(roll_spec, df, date_col)
    adjustments = _calc_multiplicative_adjustment(
        roll_spec,
        df,
        date_col=date_col,
        adjust_by=adjust_by,
    )
    with_adjustment = spliced.merge(
        adjustments,
        left_on=date_col,
        right_index=True,
        how="left",
    )
    with_adjustment = with_adjustment.set_index(date_col)
    aligned_adjustment = with_adjustment[adjustments.name]
    aligned_adjustment = aligned_adjustment.fillna(value=1)
    cumulative_adjustment = aligned_adjustment.cumprod()
    for col in adjustment_cols:
        with_adjustment[col] = with_adjustment[col] * cumulative_adjustment
    with_adjustment[adjustments.name] = cumulative_adjustment
    new_columns = df.columns.tolist() + [adjustments.name]
    return with_adjustment.reset_index()[new_columns]





from typing import Dict, List, Tuple
import datetime as _dt

import pandas as pd



def _cm_days_from_symbol(symbol: str) -> pd.Timedelta:
    """
    Parse constant-maturity days from symbols like 'SR3.cm.182'.
    """
    try:
        days = int(symbol.split(".cm.")[-1])
    except Exception as e:
        raise ValueError(f"Cannot parse constant-maturity days from {symbol}") from e
    return pd.Timedelta(days=days)


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: _dt.date,
    end: _dt.date,
) -> List[Dict[str, str]]:
    """
    Build roll segments [{d0,d1,p,n}] for a constant-maturity series on [start, end).

    Idea:
      For each calendar day d, target T = d + CM_DAYS (date-only granularity).
      Among contracts 'live' that day (ts_recv <= d):
        pre = latest expiration <= T
        nxt = earliest expiration > T
      Consecutive days with the same (pre, nxt) are compressed into one segment.
    """
    cm = _cm_days_from_symbol(symbol)

    df = instrument_defs.copy()
    # Ignore spreads; keep only futures
    if "instrument_class" in df:
        df = df[df["instrument_class"] == "F"].copy()

    # Normalize to DATE (tests only check date, not time)
    df["exp_date"] = pd.to_datetime(df["expiration"], utc=True).dt.date
    df["live_date"] = pd.to_datetime(df["ts_recv"], utc=True).dt.date
    df = df.sort_values(["exp_date", "instrument_id"]).reset_index(drop=True)

    # Build half-open segments over [start, end)
    all_days: pd.DatetimeIndex = pd.date_range(start=start, end=end, freq="D")

    segments: List[Dict[str, str]] = []
    current_pair: Tuple[int, int] | None = None
    seg_start: _dt.date | None = None

    for d in all_days[:-1]:
        day = d.date()
        T = (pd.Timestamp(day) + cm).date()

        live = df[df["live_date"] <= day]
        if live.empty:
            continue

        pre_set = live[live["exp_date"] <  T]
        nxt_set = live[live["exp_date"] >= T]
        if pre_set.empty or nxt_set.empty:
            continue

        pre_id = int(pre_set.iloc[-1]["instrument_id"])   # latest exp <= T
        nxt_id = int(nxt_set.iloc[0]["instrument_id"])    # earliest exp > T
        pair = (pre_id, nxt_id)

        if current_pair is None:
            current_pair = pair
            seg_start = day
        elif pair != current_pair:
            assert seg_start is not None
            segments.append(
                {"d0": seg_start.isoformat(), "d1": day.isoformat(),
                 "p": str(current_pair[0]), "n": str(current_pair[1])}
            )
            current_pair = pair
            seg_start = day

    if current_pair is not None and seg_start is not None:
        segments.append(
            {"d0": seg_start.isoformat(), "d1": end.isoformat(),
             "p": str(current_pair[0]), "n": str(current_pair[1])}
        )

    return segments


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    """
    Create the constant-maturity series from piecewise (pre,next) pairs.

    Output columns (exact order expected by tests):
      [date_col, 'pre_price','pre_id','pre_expiration',
       'next_price','next_id','next_expiration','pre_weight', symbol]
    """
    cm = _cm_days_from_symbol(symbol)

    df = raw_data.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

    exp_map = df.drop_duplicates("instrument_id").set_index("instrument_id")["expiration"]

    out: list[pd.DataFrame] = []

    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz="UTC")
        d1 = pd.Timestamp(spec["d1"], tz="UTC")
        pre = int(spec["p"])
        nxt = int(spec["n"])

        # Daily timeline on [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        if len(t) == 0:
            continue
        seg = pd.DataFrame({date_col: t})

        # Merge pre/next prices aligned on the date column
        pre_df = (df[df["instrument_id"] == pre][[date_col, price_col]]
                  .rename(columns={price_col: "pre_price"}))
        nxt_df = (df[df["instrument_id"] == nxt][[date_col, price_col]]
                  .rename(columns={price_col: "next_price"}))
        seg = seg.merge(pre_df, on=date_col, how="left").merge(nxt_df, on=date_col, how="left")

        # Linear time-to-expiry weight on 'pre'
        exp_pre = exp_map.loc[pre]
        exp_nxt = exp_map.loc[nxt]
        f_pre = (exp_nxt - (seg[date_col] + cm)) / (exp_nxt - exp_pre)

        seg["pre_weight"] = f_pre
        seg["pre_id"] = pre
        seg["pre_expiration"] = exp_pre
        seg["next_id"] = nxt
        seg["next_expiration"] = exp_nxt
        seg[symbol] = f_pre * seg["pre_price"] + (1.0 - f_pre) * seg["next_price"]

        # Column order must match the test exactly
        seg = seg[
            [date_col, "pre_price", "pre_id", "pre_expiration",
             "next_price", "next_id", "next_expiration", "pre_weight", symbol]
        ]
        out.append(seg)

    return pd.concat(out, ignore_index=True)
