"""Functions to splice and adjust futures data into continuous data."""

from datetime import date
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

def parse_product_and_maturity(symbol: str) -> tuple[str, int]:
    parsed_symbol = symbol.split('.')
    product = parsed_symbol[0]
    maturity_days = int(parsed_symbol[2])
    return product, maturity_days

def get_roll_spec(symbol: str, instrument_df: pd.DataFrame, start: date, end: date) -> list[dict]:
    roll_periods = []

    # Filter instrument_df for futures only and within the date range
    product, maturity_days = parse_product_and_maturity(symbol)

    futures = instrument_df.copy()
    futures['ts_recv'] = futures['ts_recv'].dt.date
    futures['expiration'] = futures['expiration'].dt.date
    futures = (
        futures[
            (futures['raw_symbol'].str.startswith(product)) &
            (futures['instrument_class'] == 'F') & 
            (futures['ts_recv'] <= end)
        ]
    ).sort_values('expiration')

    d0 = start
    d1 = start

    while d1 < end:
        period = {}
        period['d0'] = d0.isoformat()

        # add maturity_days to d0 to get "target"
        target = d0 + pd.Timedelta(days=maturity_days)
        far_futures = futures[(futures['expiration'] >= target)]
        if far_futures.empty:
            break
        far_future = far_futures[far_futures['ts_recv'] <= d0].iloc[0]

        near_futures = futures[(futures['expiration'] < target)]
        if near_futures.empty:
            break
        near_future = near_futures.iloc[-1]

        # expiration cut off
        d1 = min(end, far_future['expiration'] - pd.Timedelta(days=maturity_days-1))

        # availability cut off
        newly_available = far_futures[
            (far_futures['instrument_id'] < far_future['instrument_id']) &
            (far_futures['ts_recv'] > d0) &
            (far_futures['ts_recv'] < d1)
        ]
        if not newly_available.empty:
            d1 = min(d1, newly_available['ts_recv'].min())

        period['d1'] = d1.isoformat()
        period['p'] = str(near_future['instrument_id'])
        period['n'] = str(far_future['instrument_id'])
        roll_periods.append(period)

        d0 = d1

    return roll_periods


def constant_maturity_splice(symbol: str, roll_spec: list[dict], all_data: pd.DataFrame, date_col: str = "datetime", price_col: str = "price") -> pd.DataFrame:

    _, maturity_days = parse_product_and_maturity(symbol)
    maturity_timedelta = pd.Timedelta(days=maturity_days)

    out_cols = [date_col, "pre_price", "pre_id", "pre_expiration",
                "next_price", "next_id", "next_expiration", "pre_weight", symbol]

    data_by_instr = all_data.copy().groupby("instrument_id")

    segments = []
    for spec in roll_spec:
        d0, d1 = spec['d0'], spec['d1']
        pre = int(spec['p'])
        pre_data_raw = data_by_instr.get_group(pre)
        if pre_data_raw.empty:
            raise ValueError(f"No data for instrument {pre}")

        nex = int(spec['n'])
        nex_data_raw = data_by_instr.get_group(nex)
        if nex_data_raw.empty:
            raise ValueError(f"No data for instrument {nex}")

        pre_data = (
            pre_data_raw[(pre_data_raw[date_col] >= d0) & (pre_data_raw[date_col] < d1)].copy()
        ).rename(columns={
            price_col: "pre_price",
            "instrument_id": "pre_id",
            "expiration": "pre_expiration"
        })

        nex_data = (
            nex_data_raw[(nex_data_raw[date_col] >= d0) & (nex_data_raw[date_col] < d1)].copy()
        ).rename(columns={
            price_col: "next_price",
            "instrument_id": "next_id",
            "expiration": "next_expiration"}
        )

        merged_data = pre_data.merge(
            nex_data,
            on=date_col,
            how='inner'
        )

        merged_data["pre_weight"] = (
            (merged_data['next_expiration'] - (merged_data[date_col] + maturity_timedelta)) /
            (merged_data['next_expiration'] - merged_data['pre_expiration'])
        )
        merged_data[symbol] = (
            merged_data["pre_weight"] * merged_data["pre_price"] +
                (1 - merged_data["pre_weight"]) * merged_data["next_price"]
        )

        segments.append(merged_data[out_cols])

    return pd.concat(segments, ignore_index=True)