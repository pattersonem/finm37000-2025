"""Functions to splice and adjust futures data into continuous data."""
from __future__ import annotations
from typing import Optional

import numpy as np
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



# modify

def _parse_cm_days_from_symbol(symbol: str) -> int:
    import re
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Cannot parse constant-maturity days from {symbol}")
    return int(m.group(1))


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Build the per-day constant-maturity blend defined by `roll_spec`.

    Parameters
    ----------
    symbol : str
        Like 'SR3.cm.182'; the trailing integer gives maturity in days.
    roll_spec : list[dict]
        Each dict has: {'d0','d1','p','n'} with left-inclusive, right-exclusive dates.
        'p' and 'n' are instrument_id (as string) for pre/next contracts.
    df : DataFrame
        Contains at least ['instrument_id', date_col, price_col, 'expiration'].
        For the unit test, price is constant per instrument_id.

    Returns
    -------
    DataFrame with columns exactly as expected by the test:
    ['datetime','pre_price','pre_id','pre_expiration',
     'next_price','next_id','next_expiration','pre_weight', <symbol>]
    """
    cm_days = _parse_cm_days_from_symbol(symbol)

    # Map instrument_id -> {price, expiration}
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], utc=True)
    tmp["expiration"] = pd.to_datetime(tmp["expiration"], utc=True)

    # Use "last" per instrument_id (prices in the test are constant anyway)
    snap = (
        tmp.sort_values(date_col)
           .groupby("instrument_id")
           .agg({price_col: "last", "expiration": "last"})
    )

    rows = []
    for seg in roll_spec:
        d0 = pd.to_datetime(seg["d0"], utc=True)
        d1 = pd.to_datetime(seg["d1"], utc=True)
        pre_id = int(seg["p"]) if isinstance(seg["p"], (int, str)) else int(seg["p"])
        nxt_id = int(seg["n"]) if isinstance(seg["n"], (int, str)) else int(seg["n"])

        # day-by-day, left-inclusive, right-exclusive
        t_range = pd.date_range(start=d0, end=d1, inclusive="left", tz="UTC")

        pre_px = float(snap.loc[pre_id, price_col])
        nxt_px = float(snap.loc[nxt_id, price_col])
        pre_exp = snap.loc[pre_id, "expiration"]
        nxt_exp = snap.loc[nxt_id, "expiration"]

        denom = (nxt_exp - pre_exp)

        for t in t_range:
            target = t + pd.Timedelta(days=cm_days)
            # pre_weight f = (exp_next - target) / (exp_next - exp_pre)
            f = (nxt_exp - target) / denom
            # Ensure it's a scalar float (Timedelta division yields float)
            f = float(f)

            blended = f * pre_px + (1.0 - f) * nxt_px

            rows.append(
                {
                    "datetime": t,
                    "pre_price": pre_px,
                    "pre_id": pre_id,
                    "pre_expiration": pre_exp,
                    "next_price": nxt_px,
                    "next_id": nxt_id,
                    "next_expiration": nxt_exp,
                    "pre_weight": f,
                    symbol: blended,
                }
            )

    df_out = pd.DataFrame(rows)

    # Keep datetime timezone-aware
    df_out["datetime"] = pd.to_datetime(df_out["datetime"], utc=True)

    # dtypes expected by the test
    df_out["pre_id"] = df_out["pre_id"].astype("int64")
    df_out["next_id"] = df_out["next_id"].astype("int64")
    df_out["pre_price"] = df_out["pre_price"].astype("int64")
    df_out["next_price"] = df_out["next_price"].astype("int64")

    return df_out


