"""Functions to splice and adjust futures data into continuous data."""

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

def _parse_cm_maturity(symbol: str) -> int:
    """
    Parse a constant-maturity symbol like 'SR3.cm.182'
    """
    try:
        return int(symbol.split(".")[-1])
    except Exception as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol}")



def constant_maturity_splice(
    symbol: str,
    roll_spec,
    data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "close",
    expiration_col: str = "expiration",
) -> pd.DataFrame:
    """
    Build a constant-maturity price series by linearly interpolating between two
    futures along a roll schedule.

    Returns
    -------
    pd.DataFrame
        Columns:
        [date_col, 'pre_price', 'pre_id', 'pre_expiration',
         'next_price', 'next_id', 'next_expiration',
         'pre_weight', symbol]
    """
    maturity_days = _parse_cm_maturity(symbol)
    maturity = pd.Timedelta(days=maturity_days)

    df = data.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize("UTC")

    df[expiration_col] = pd.to_datetime(df[expiration_col])
    if df[expiration_col].dt.tz is None:
        df[expiration_col] = df[expiration_col].dt.tz_localize("UTC")

    expirations = (
        df[["instrument_id", expiration_col]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")[expiration_col]
    )

    segments: list[pd.DataFrame] = []

    for r in roll_spec:
        pre_id = int(r["p"])
        nxt_id = int(r["n"])
        d0 = r["d0"]
        d1 = r["d1"]

        # Date grid for this segment: [d0, d1) in UTC
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        if t.empty:
            continue

        t_start, t_end = t[0], t[-1]

        # Get price series for the two contracts over this segment
        pre_mask = (
            (df["instrument_id"] == pre_id)
            & (df[date_col] >= t_start)
            & (df[date_col] <= t_end)
        )
        nxt_mask = (
            (df["instrument_id"] == nxt_id)
            & (df[date_col] >= t_start)
            & (df[date_col] <= t_end)
        )

        pre_series = (
            df.loc[pre_mask]
            .set_index(date_col)[price_col]
            .reindex(t)
        )
        nxt_series = (
            df.loc[nxt_mask]
            .set_index(date_col)[price_col]
            .reindex(t)
        )

        pre_exp = expirations.loc[pre_id]
        nxt_exp = expirations.loc[nxt_id]

        # Weight on the pre- contract
        f = (nxt_exp - (t + maturity)) / (nxt_exp - pre_exp)

        pre_vals = pre_series.to_numpy()
        nxt_vals = nxt_series.to_numpy()
        f_vals = f.to_numpy()

        seg = pd.DataFrame(
            {
                date_col: t,
                "pre_price": pre_vals,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": nxt_vals,
                "next_id": nxt_id,
                "next_expiration": nxt_exp,
                "pre_weight": f_vals,
                symbol: f_vals * pre_vals + (1.0 - f_vals) * nxt_vals,
            }
        )
        segments.append(seg)

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

    return pd.concat(segments, ignore_index=True)
