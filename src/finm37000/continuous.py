"""Functions to splice and adjust futures data into continuous data."""

from __future__ import annotations
import datetime as _dt
from typing import Any, Optional
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

# I used Chat GPT to help code the two functions below
def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: _dt.date | str | pd.Timestamp,
    end: _dt.date | str | pd.Timestamp,
) -> list[dict[str, str]]:

    try:
        maturity_days = int(str(symbol).split(".")[-1])
    except Exception as exc: 
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc

    def _to_date(x: _dt.date | str | pd.Timestamp) -> _dt.date:
        if isinstance(x, _dt.date):
            return x
        return pd.to_datetime(x).date()

    start_date = _to_date(start)
    end_date = _to_date(end)

    df = instrument_defs.copy()

    if "instrument_class" in df.columns:
        df = df[df["instrument_class"] == "F"].copy()

    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    df["exp_date"] = df["expiration"].dt.date
    df["recv_date"] = df["ts_recv"].dt.date

    df = df.sort_values("expiration").reset_index(drop=True)

    maturity_delta = _dt.timedelta(days=maturity_days)

    segments: list[dict[str, _dt.date | str]] = []
    last_pair: tuple[int, int] | None = None

    current = start_date
    while current < end_date:
        target = current + maturity_delta

        candidates = df[df["recv_date"] <= current]

        nexts = candidates[candidates["exp_date"] >= target]
        if nexts.empty:
            raise ValueError(f"No suitable 'next' contract on {current}")

        next_row = nexts.sort_values("exp_date").iloc[0]
        next_exp = next_row["exp_date"]

        pres = candidates[candidates["exp_date"] < next_exp]
        if pres.empty:
            raise ValueError(f"No suitable 'pre' contract on {current}")

        pre_row = pres.sort_values("exp_date").iloc[-1]

        pre_id = int(pre_row["instrument_id"])
        next_id = int(next_row["instrument_id"])
        pair = (pre_id, next_id)

        if pair != last_pair:
            segments.append({"d0": current, "p": str(pre_id), "n": str(next_id)})
            last_pair = pair

        current += _dt.timedelta(days=1)

    for i in range(len(segments)):
        if i + 1 < len(segments):
            d1_date = segments[i + 1]["d0"]
        else:
            d1_date = end_date
        segments[i]["d0"] = segments[i]["d0"].isoformat() 
        segments[i]["d1"] = d1_date.isoformat()   

    return segments


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "close",
) -> pd.DataFrame:

    maturity_days = int(str(symbol).split(".")[-1])
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = raw_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    tz = df[date_col].dt.tz

    expirations = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates(subset="instrument_id")
        .set_index("instrument_id")["expiration"]
    )
    expirations = pd.to_datetime(expirations)

    pieces = []

    for spec in roll_spec:
        pre_id = int(spec["p"])
        next_id = int(spec["n"])

        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)

        mask_pre = (
            (df["instrument_id"] == pre_id)
            & (df[date_col] >= d0)
            & (df[date_col] < d1)
        )
        mask_next = (
            (df["instrument_id"] == next_id)
            & (df[date_col] >= d0)
            & (df[date_col] < d1)
        )

        seg_pre = df.loc[mask_pre, [date_col, price_col]].rename(
            columns={price_col: "pre_price"}
        )
        seg_next = df.loc[mask_next, [date_col, price_col]].rename(
            columns={price_col: "next_price"}
        )

        seg = seg_pre.merge(seg_next, on=date_col, how="inner")

        exp_pre = pd.to_datetime(expirations.loc[pre_id])
        exp_next = pd.to_datetime(expirations.loc[next_id])

        seg["pre_id"] = pre_id
        seg["pre_expiration"] = exp_pre
        seg["next_id"] = next_id
        seg["next_expiration"] = exp_next

        seg["pre_weight"] = (exp_next - (seg[date_col] + maturity_delta)) / (
            exp_next - exp_pre
        )

        seg[symbol] = (
            seg["pre_weight"] * seg["pre_price"]
            + (1.0 - seg["pre_weight"]) * seg["next_price"]
        )

        seg = seg[
            [
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
        ]

        pieces.append(seg)

    return pd.concat(pieces, ignore_index=True)
