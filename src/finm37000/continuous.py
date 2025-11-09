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


def constant_maturity_splice(
    symbol,
    roll_spec,
    all_data,
    date_col="datetime",
    price_col="price",
) -> pd.DataFrame:
    
    raw_sym, _, m = symbol.split('.')
    start = pd.to_datetime(roll_spec[0]['d0'], utc=True)
    end = pd.to_datetime(roll_spec[-1]['d1'], utc = True)
    result = pd.DataFrame()
    result[date_col] = pd.date_range(start, end)[:-1]
    result = result.set_index(date_col)
    result['pre_id'] = 0
    result['next_id'] = 0
    for row in roll_spec:
        result.loc[row['d0']:row['d1'], 'pre_id'] = int(row['p'])
        result.loc[row['d0']:row['d1'], 'next_id'] = int(row['n'])
    all_data_setidx = all_data.copy()
    all_data_setidx[date_col] = pd.to_datetime(all_data_setidx[date_col])
    all_data_setidx = all_data_setidx.set_index(['instrument_id', date_col])
    result[['pre_price', 'pre_expiration']] = all_data_setidx.loc[result.reset_index().set_index(['pre_id', date_col]).index].to_numpy()
    result[['next_price', 'next_expiration']] = all_data_setidx.loc[result.reset_index().set_index(['next_id', date_col]).index].to_numpy()
    result[['pre_price', 'next_price']] = result[['pre_price', 'next_price']].astype('int64')
    result = result.reset_index()
    result['pre_weight'] = (result['next_expiration'] - (result[date_col] + pd.Timedelta(days = int(m)))) / (result['next_expiration'] - result['pre_expiration'])
    result[symbol] = result['pre_weight'] * result['pre_price'] + (1 - result['pre_weight']) * result['next_price']
    return result[[date_col, 'pre_price', 'pre_id', 'pre_expiration', 'next_price', 'next_id', 'next_expiration', 'pre_weight', symbol]]

def get_roll_spec(
    symbol,
    instrument_df, 
    start, 
    end
):
    raw_sym, _, m = symbol.split('.')
    instrument_df = instrument_df[instrument_df['instrument_class'] == 'F'].copy()
    instrument_df['ts_recv'] = pd.to_datetime(instrument_df['ts_recv']).apply(lambda x : x.date())
    instrument_df['expiration'] = pd.to_datetime(instrument_df['expiration'].apply(lambda x : x.date()))
    instrument_df['expiration_shift'] = (instrument_df['expiration'] - pd.Timedelta(days = int(m) - 1)).apply(lambda x : x.date())
    instrument_df = instrument_df.sort_values('expiration')
    result = []
    date = start
    while (date < end) & (end < instrument_df['expiration_shift'].iloc[-1]):
        row1 = instrument_df[(instrument_df['expiration_shift'] <= date) & (instrument_df['ts_recv'] <= date)].iloc[-1]
        end_dates_row = instrument_df[(instrument_df['expiration_shift'] > date)].iloc[0]
        row2 = instrument_df[(instrument_df['expiration_shift'] > date) & (instrument_df['ts_recv'] <= date)].iloc[0]
        date_prev = date
        if end_dates_row['instrument_id'] == row2['instrument_id']:
            date = row2['expiration_shift']
        else:
            date = end_dates_row['ts_recv']
        result.append({
            'd0':str(date_prev), 'd1':str(date), 
            'p':str(row1['instrument_id']), 'n':str(row2['instrument_id'])
        })
    result[0]['d0'] = str(start)
    result[-1]['d1'] = str(end)
    return result
