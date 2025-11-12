"""Functions to splice and adjust futures data into continuous data."""

import datetime as _dt
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


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: _dt.date | str | pd.Timestamp,
    end: _dt.date | str | pd.Timestamp,
) -> list[dict[str, str]]:
    """Build roll segments for a constant-maturity symbol.

    For each UTC date d in [start, end), choose two live outright futures whose
    expirations straddle d + maturity_days and coalesce consecutive days with
    the same pair into one segment.

    Args:
        symbol: Constant-maturity symbol of the form ``'<prod>.cm.<days>'``.
        instrument_defs: DataFrame with at least columns
            ``['instrument_id','instrument_class','expiration','ts_recv']``.
            Timestamps may be naive or tz-aware; they are coerced to UTC.
        start: Start date (inclusive) for segmenting.
        end: End date (exclusive) for segmenting.

    Returns:
        List of dicts with keys:
        ``{'d0': 'YYYY-MM-DD', 'd1': 'YYYY-MM-DD', 'p': '<pre_id>', 'n': '<next_id>'}``
    """
    # Parse maturity days from '<product>.cm.<days>'
    try:
        maturity_days = int(str(symbol).split(".")[-1])
    except Exception as exc:
        msg = "Could not parse maturity days from symbol '{sym}'".format(sym=symbol)
        raise ValueError(msg) from exc

    # Normalize inputs
    df = instrument_defs.copy()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Keep outright futures only and coerce to UTC
    df = df[df["instrument_class"] == "F"].copy()
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    if start_date >= end_date:
        return []

    df = df.sort_values(["expiration", "instrument_id"]).reset_index(drop=True)
    maturity = pd.Timedelta(days=maturity_days)
    segments: list[dict[str, str]] = []
    current_pair: tuple[int, int] | None = None
    seg_start: pd.Timestamp | None = None
    for d in pd.date_range(start=start_date, end=end_date, inclusive="left", tz="UTC"):
        # Only contracts live by this date
        d_date = d.date()
        live = df[df["ts_recv"].dt.date <= d_date]
        if live.empty:
            continue

        target = d + maturity
        pre = live[live["expiration"] <= target].tail(1)
        nxt = live[live["expiration"] > target].head(1)
        if pre.empty or nxt.empty:
            continue

        pre_id = int(pre["instrument_id"].iloc[0])
        nxt_id = int(nxt["instrument_id"].iloc[0])
        pair = (pre_id, nxt_id)
        if current_pair is None:
            current_pair = pair
            seg_start = d
        elif pair != current_pair:
            assert seg_start is not None
            segments.append(
                {
                    "d0": seg_start.date().isoformat(),
                    "d1": d.date().isoformat(),
                    "p": str(current_pair[0]),
                    "n": str(current_pair[1]),
                }
            )
            current_pair = pair
            seg_start = d

    # Close final segment at end
    if current_pair is not None and seg_start is not None:
        segments.append(
            {
                "d0": seg_start.date().isoformat(),
                "d1": pd.to_datetime(end_date).date().isoformat(),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            }
        )

    return segments
