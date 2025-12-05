from __future__ import annotations
import datetime as _dt
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

try:
    # use package's business-day helper if available
    from .time import us_business_day  # pandas offset (BDay) aliased in package
except Exception:  # fallback
    from pandas.tseries.offsets import BDay as us_business_day  # type: ignore


_CM_REGEX = re.compile(r"^(?P<root>[^.]+)\.cm\.(?P<days>\d+)$")


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

# --- constant-maturity helpers ------------------------------------------------



def _parse_constant_maturity_symbol(symbol: str) -> Tuple[str, int]:
    """
    Parse symbols like 'SR3.cm.182' -> ('SR3', 182).
    """
    m = _CM_REGEX.match(symbol)
    if not m:
        raise ValueError(f"Not a constant-maturity symbol: {symbol!r}")
    return m.group("root"), int(m.group("days"))


def _to_date(x) -> pd.Timestamp:
    # Normalize to midnight date (no tz); tests compare dates (not times)
    ts = pd.Timestamp(x)
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)


def _run_length_windows(dates: List[pd.Timestamp], pairs: List[Tuple[str, str]]):
    """
    Compress per-day (pre_id, next_id) pairs into windows:
      [{"d0": "YYYY-MM-DD", "d1": "YYYY-MM-DD", "p": pre, "n": nxt}, ...]
    where d1 is exclusive.
    """
    if not dates:
        return []
    out = []
    s = 0
    last = pairs[0]
    for i in range(1, len(dates)):
        if pairs[i] != last:
            out.append({
                "d0": dates[s].date().isoformat(),
                "d1": (dates[i]).date().isoformat(),
                "p": last[0],
                "n": last[1],
            })
            s = i
            last = pairs[i]
    # tail
    out.append({
        "d0": dates[s].date().isoformat(),
        "d1": (dates[-1] + pd.Timedelta(days=1)).date().isoformat(),
        "p": last[0],
        "n": last[1],
    })
    return out

# Note: I used ChatGPT to help draft and refine get_roll_spec and constant_maturity_splice.
# I provided the problem statement and tests as prompts, then reviewed and edited the output.

def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: _dt.date | str | pd.Timestamp,
    end: _dt.date | str | pd.Timestamp,
) -> List[Dict[str, str]]:
    try:
        maturity_days = int(str(symbol).split(".")[-1])
    except Exception as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc
    maturity_delta = _dt.timedelta(days=maturity_days)

    def _to_date(x: _dt.date | str | pd.Timestamp) -> _dt.date:
        if isinstance(x, _dt.date):
            return x
        return pd.to_datetime(x).date()

    start_date = _to_date(start)
    end_date = _to_date(end)

    df = instrument_defs.copy()
    if "instrument_class" in df.columns:
        df = df[df["instrument_class"] == "F"].copy()   # drop spreads/others

    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])
    df["exp_date"] = df["expiration"].dt.date
    df["recv_date"] = df["ts_recv"].dt.date
    df = df.sort_values("expiration").reset_index(drop=True)

    segments: List[Dict[str, _dt.date | str]] = []
    last_pair: tuple[int, int] | None = None
    current = start_date

    while current < end_date:
        target = current + maturity_delta

        available = df[df["recv_date"] <= current]
        if available.empty:
            current += _dt.timedelta(days=1)
            continue

        nexts = available[available["exp_date"] >= target]
        if nexts.empty:
            # out-of-range; skip this day
            current += _dt.timedelta(days=1)
            continue
        next_row = nexts.sort_values("exp_date").iloc[0]

        pres = available[available["exp_date"] < next_row["exp_date"]]
        if pres.empty:
            current += _dt.timedelta(days=1)
            continue
        pre_row = pres.sort_values("exp_date").iloc[-1]

        pre_id = int(pre_row["instrument_id"])
        next_id = int(next_row["instrument_id"])
        pair = (pre_id, next_id)

        if pair != last_pair:
            segments.append({"d0": current, "p": str(pre_id), "n": str(next_id)})
            last_pair = pair

        current += _dt.timedelta(days=1)

    roll_spec: List[Dict[str, str]] = []
    for i in range(len(segments)):
        d0_date = segments[i]["d0"]
        if i + 1 < len(segments):
            d1_date = segments[i + 1]["d0"]
        else:
            d1_date = end_date
        roll_spec.append(
            {
                "d0": d0_date.isoformat(),
                "d1": d1_date.isoformat(),
                "p": segments[i]["p"],  # already str
                "n": segments[i]["n"],  # already str
            }
        )
    return roll_spec


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    maturity_days = int(str(symbol).split(".")[-1])
    maturity_td = pd.Timedelta(days=maturity_days)

    df = raw_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    tz = df[date_col].dt.tz

    expirations = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates(subset="instrument_id")
        .set_index("instrument_id")["expiration"]
    )
    expirations = pd.to_datetime(expirations)

    pieces: list[pd.DataFrame] = []

    for spec in roll_spec:
        pre_id = int(spec["p"])
        next_id = int(spec["n"])

        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)

        t_range = pd.date_range(start=d0, end=d1, freq="D", tz=tz, inclusive="left")

        pre_series = (
            df.loc[df["instrument_id"] == pre_id]
              .set_index(date_col)[price_col]
              .reindex(t_range)
        )
        next_series = (
            df.loc[df["instrument_id"] == next_id]
              .set_index(date_col)[price_col]
              .reindex(t_range)
        )

        pre_exp = pd.to_datetime(expirations.loc[pre_id])
        next_exp = pd.to_datetime(expirations.loc[next_id])

        pre_weight = (next_exp - (t_range + maturity_td)) / (next_exp - pre_exp)
        pre_weight = pd.Series(pre_weight.astype(float), index=t_range)


        seg = pd.DataFrame(
            {
                "datetime": t_range,
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

    if not pieces:
        return pd.DataFrame(
            columns=[
                "datetime",
                "pre_price", "pre_id", "pre_expiration",
                "next_price", "next_id", "next_expiration",
                "pre_weight", symbol,
            ]
        )

    out = pd.concat(pieces, ignore_index=True)
    out = out[
        [
            "datetime",
            "pre_price",
            "pre_id",
            "pre_expiration",
            "next_price",
            "next_id",
            "next_expiration",
            "pre_weight",
            symbol,
        ]
    ].reset_index(drop=True)
    return out
