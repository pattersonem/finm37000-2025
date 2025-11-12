"""Functions to splice and adjust futures data into continuous data."""

from typing import Optional

import pandas as pd
import datetime
from typing import List, Dict, Any


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
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> List[Dict[str, str]]:
    try:
        maturity_days = int(symbol.split(".")[-1])
        maturity_delta = pd.Timedelta(days=maturity_days)
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse maturity days from symbol '{symbol}'")

    futures_df = instrument_df[
        instrument_df["instrument_class"] == "F"
    ].copy()
    
    futures_df["expiration_date"] = pd.to_datetime(
        futures_df["expiration"]
    ).dt.date
    futures_df["ts_recv_date"] = pd.to_datetime(
        futures_df["ts_recv"]
    ).dt.date
    
    futures_df = futures_df.sort_values(by="expiration_date")

    exp_dates = pd.to_datetime(futures_df["expiration_date"])
    roll_dates = (exp_dates + pd.Timedelta(days=1)).dt.date
    
    roll_dates = roll_dates[(roll_dates > start) & (roll_dates < end)]

    list_dates = futures_df["ts_recv_date"]
    list_dates = list_dates[(list_dates > start) & (list_dates < end)]

    event_dates = sorted(
        list(set([start] + list(roll_dates) + list(list_dates)))
    )
    
    final_specs: List[Dict[str, str]] = []

    for d0 in event_dates:
        available_contracts = futures_df[
            (futures_df["ts_recv_date"] <= d0)
            & (futures_df["expiration_date"] > d0)
        ]

        if available_contracts.empty:
            continue

        target_maturity_date = d0 + maturity_delta

        before = available_contracts[
            available_contracts["expiration_date"] <= target_maturity_date
        ]
        after = available_contracts[
            available_contracts["expiration_date"] > target_maturity_date
        ]

        if before.empty or after.empty:
            continue

        p_contract = before.iloc[-1]
        n_contract = after.iloc[0]

        current_pair = (
            str(p_contract["instrument_id"]),
            str(n_contract["instrument_id"]),
        )

        if not final_specs or current_pair != (
            final_specs[-1]["p"],
            final_specs[-1]["n"],
        ):
            if final_specs:
                final_specs[-1]["d1"] = d0.strftime("%Y-%m-%d")

            final_specs.append(
                {
                    "d0": d0.strftime("%Y-%m-%d"),
                    "p": current_pair[0],
                    "n": current_pair[1],
                    "d1": end.strftime(
                        "%Y-%m-%d"
                    ),
                }
            )

    return final_specs


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    all_data: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    try:
        maturity_days = int(symbol.split(".")[-1])
        maturity_delta = pd.Timedelta(days=maturity_days)
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse maturity days from symbol '{symbol}'")

    all_data[date_col] = pd.to_datetime(all_data[date_col], utc=True)
    all_data["expiration"] = pd.to_datetime(all_data["expiration"], utc=True)

    segment_dfs = []

    for segment in roll_spec:
        d0 = pd.to_datetime(segment["d0"], utc=True)
        d1 = pd.to_datetime(segment["d1"], utc=True)
        p_id = int(segment["p"])
        n_id = int(segment["n"])

        df_p = all_data[all_data["instrument_id"] == p_id].copy()
        df_n = all_data[all_data["instrument_id"] == n_id].copy()

        df_p = df_p.rename(
            columns={
                price_col: "pre_price",
                "instrument_id": "pre_id",
                "expiration": "pre_expiration",
            }
        )
        df_n = df_n.rename(
            columns={
                price_col: "next_price",
                "instrument_id": "next_id",
                "expiration": "next_expiration",
            }
        )

        merged_df = pd.merge(
            df_p[[date_col, "pre_price", "pre_id", "pre_expiration"]],
            df_n[[date_col, "next_price", "next_id", "next_expiration"]],
            on=date_col,
            how="inner",
        )

        segment_df = merged_df[
            (merged_df[date_col] >= d0) & (merged_df[date_col] < d1)
        ].copy()

        if segment_df.empty:
            continue

        t = segment_df[date_col]
        T_p = segment_df["pre_expiration"]
        T_n = segment_df["next_expiration"]
        
        target_maturity = t + maturity_delta

        numerator = (T_n - target_maturity).dt.total_seconds()
        denominator = (T_n - T_p).dt.total_seconds()
        
        pre_weight = numerator / denominator
        segment_df["pre_weight"] = pre_weight

        segment_df[symbol] = (
            pre_weight * segment_df["pre_price"]
            + (1 - pre_weight) * segment_df["next_price"]
        )

        segment_dfs.append(segment_df)

    if not segment_dfs:
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

    final_df = pd.concat(segment_dfs, ignore_index=True)

    expected_cols = [
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
    
    return final_df[expected_cols]