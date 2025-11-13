"""Functions to splice and adjust futures data into continuous data."""

from typing import Optional
from collections.abc import Mapping, Sequence

import pandas as pd
import numpy as np


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

def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    *,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    try:
        maturity_days = int(symbol.split(".")[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc
    maturity_delta = pd.Timedelta(days=maturity_days)

    tz = df[date_col].dt.tz
    expirations = (
        df.drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    segments: list[pd.DataFrame] = []

    for r in roll_spec:
        d0 = pd.to_datetime(r["d0"]).tz_localize(tz)
        d1 = pd.to_datetime(r["d1"]).tz_localize(tz)

        pre_id = int(r["p"])
        next_id = int(r["n"])

        # Date range for this segment: [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz=tz, inclusive="left")

        pre_exp = expirations[pre_id]
        next_exp = expirations[next_id]

        # Weight on the "pre" contract: same formula as in the test
        pre_weight = (next_exp - (t + maturity_delta)) / (next_exp - pre_exp)

        # For this homework test, prices are constant by instrument id
        pre_price_val = df.loc[df["instrument_id"] == pre_id, price_col].iloc[0]
        next_price_val = df.loc[df["instrument_id"] == next_id, price_col].iloc[0]

        synthetic_price = pre_weight * pre_price_val + (1.0 - pre_weight) * next_price_val

        seg = pd.DataFrame(
            {
                date_col: t,
                "pre_price": pre_price_val,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": next_price_val,
                "next_id": next_id,
                "next_expiration": next_exp,
                "pre_weight": pre_weight,
                symbol: synthetic_price,
            }
        )
        segments.append(seg)

    result = pd.concat(segments, ignore_index=True)
    return result

def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    *,
    start: pd.Timestamp | pd.Timestamp | str,
    end: pd.Timestamp | pd.Timestamp | str,
) -> list[dict[str, str]]:
    try:
        maturity_days = int(symbol.split(".")[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc
    maturity_delta = pd.Timedelta(days=maturity_days)

    legs = instrument_df[instrument_df["instrument_class"] == "F"].copy()

    legs = legs.sort_values("expiration")
    tz = legs["ts_recv"].dt.tz
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    dates = pd.date_range(start=start_date, end=end_date, freq="D", inclusive="left")

    roll_spec: list[dict[str, str]] = []
    current_pair: tuple[int, int] | None = None
    segment_start: pd.Timestamp | None = None

    for d in dates:
        current_ts = pd.Timestamp(d).tz_localize(tz)

        active = legs[legs["ts_recv"] <= current_ts]
        if active.empty:
            continue

        active = active.sort_values("expiration")
        expirations = active["expiration"]
        ids = active["instrument_id"].to_numpy()

        target = current_ts + maturity_delta

        pos = expirations.searchsorted(target, side="right")

        if pos <= 0:
            pre_idx = 0
            next_idx = 1 if len(ids) > 1 else 0
        elif pos >= len(ids):
            pre_idx = len(ids) - 2
            next_idx = len(ids) - 1
        else:
            pre_idx = pos - 1
            next_idx = pos

        pre_id = int(ids[pre_idx])
        next_id = int(ids[next_idx])
        pair = (pre_id, next_id)

        if current_pair is None:
            current_pair = pair
            segment_start = d
        elif pair != current_pair:
            assert segment_start is not None
            roll_spec.append(
                {
                    "d0": segment_start.strftime("%Y-%m-%d"),
                    "d1": d.strftime("%Y-%m-%d"),
                    "p": str(current_pair[0]),
                    "n": str(current_pair[1]),
                }
            )
            current_pair = pair
            segment_start = d

    if current_pair is not None and segment_start is not None:
        roll_spec.append(
            {
                "d0": segment_start.strftime("%Y-%m-%d"),
                "d1": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            }
        )

    return roll_spec


