"""Functions to splice and adjust futures data into continuous data."""

from typing import Optional
import datetime

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
            adjustment = last_true_value - adjustment_piece[adjust_by].iloc[-1]
            adjustments.append(adjustment)
            adjustment_dates.append(d0)
        last_true_value = piece[adjust_by].iloc[-1]
        last_date = piece[date_col].iloc[-1]
    return pd.Series(adjustments, index=adjustment_dates, name="additive_adjustment")


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
            adjustment = last_true_value / adjustment_piece[adjust_by].iloc[-1]
            adjustments.append(adjustment)
            adjustment_dates.append(d0)
        last_true_value = piece[adjust_by].iloc[-1]
        last_date = piece[date_col].iloc[-1]
    return pd.Series(
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
    with_adjustment[adjustments.name] = cumulative_adjustment  # type: ignore
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
    with_adjustment[adjustments.name] = cumulative_adjustment  # type: ignore
    new_columns = df.columns.tolist() + [adjustments.name]
    return with_adjustment.reset_index()[new_columns]


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict],
    df: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Construct a constant maturity futures series using weighted interpolation.
    """
    maturity_days = int(symbol.split(".")[-1])
    maturity_td = pd.Timedelta(days=maturity_days)

    tz = df[date_col].dt.tz
    exp_map = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    pieces: list[pd.DataFrame] = []

    for spec in roll_spec:
        pre_id = int(spec["p"])
        next_id = int(spec["n"])

        d0 = pd.Timestamp(spec["d0"]).tz_localize(tz)
        d1 = pd.Timestamp(spec["d1"]).tz_localize(tz)

        t_range = pd.date_range(start=d0, end=d1, freq="D", tz=tz, inclusive="left")

        pre_exp = exp_map.loc[pre_id]
        next_exp = exp_map.loc[next_id]

        pre_series = (
            df[df["instrument_id"] == pre_id]
            .set_index(date_col)[price_col]
            .reindex(t_range)
        )
        next_series = (
            df[df["instrument_id"] == next_id]
            .set_index(date_col)[price_col]
            .reindex(t_range)
        )

        pre_weight = (next_exp - (t_range + maturity_td)) / (next_exp - pre_exp)

        seg = pd.DataFrame(
            {
                date_col: t_range,
                "pre_price": pre_series.values,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": next_series.values,
                "next_id": next_id,
                "next_expiration": next_exp,
                "pre_weight": pre_weight.values,
                symbol: pre_weight.values * pre_series.values
                + (1.0 - pre_weight.values) * next_series.values,
            }
        )
        pieces.append(seg)

    result = pd.concat(pieces, ignore_index=True)
    result = result[[date_col, "pre_price", "pre_id", "pre_expiration",
                     "next_price", "next_id", "next_expiration",
                     "pre_weight", symbol]]
    return result


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    *,
    start,
    end,
    maturity_days: int | None = None,
) -> list[dict]:
    if maturity_days is None:
        maturity_days = int(symbol.split(".")[-1])

    df = instrument_df.copy()
    df = df[df["instrument_class"] == "F"]
    df = df[df["expiration"].dt.date >= start]
    df = df[df["ts_recv"].dt.date < end]
    df = df.sort_values("expiration").reset_index(drop=True)

    roll_spec: list[dict] = []

    segments: list[tuple[datetime.date, datetime.date, tuple[int, int]]] = []

    for d in pd.date_range(start=start, end=end, freq="D"):
        current_date = d.date()
        target = current_date + datetime.timedelta(days=maturity_days)

        available = df[df["ts_recv"].dt.date <= current_date]
        if available.empty:
            continue

        exp_dates = available["expiration"].dt.date.values
        ids = available["instrument_id"].values

        pre = None
        nxt = None
        for i in range(len(exp_dates) - 1):
            if exp_dates[i] <= target <= exp_dates[i + 1]:
                pre = int(ids[i])
                nxt = int(ids[i + 1])
                break

        if pre is None or nxt is None:
            continue

        pair = (pre, nxt)

        if not segments:
            segments.append((current_date, current_date, pair))
        else:
            s0, s1, p = segments[-1]
            if pair == p:
                segments[-1] = (s0, current_date, p)
            else:
                segments.append((current_date, current_date, pair))

    if not segments:
        return roll_spec

    for i, (d0, d1, pair) in enumerate(segments):
        if i < len(segments) - 1:
            d1_out = d1 + datetime.timedelta(days=1)
        else:
            d1_out = end
        roll_spec.append(
            {
                "d0": str(d0),
                "d1": str(d1_out),
                "p": str(pair[0]),
                "n": str(pair[1]),
            }
        )

    return roll_spec

