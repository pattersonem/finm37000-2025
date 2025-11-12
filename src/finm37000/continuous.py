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


def get_roll_spec(symbol, instrument_df, start, end):
    """
    Goal: Return a list where each element is {"d0": "date", "d1": "date", "p": "id", "n": "id"}
    """
    
    # Step 1: Extract maturity_days
    maturity_days = int(symbol.split(".")[-1])
    
    # Step 2: Filter futures contracts
    futures = instrument_df[instrument_df['instrument_class'] == 'F']
    futures = futures[futures['ts_recv'].dt.date <= end]
    futures = futures.sort_values('expiration')
    
    # Step 3: Create date range
    tz = futures['expiration'].dt.tz
    date_range = pd.date_range(start, end, freq='D', tz=tz)
    
    # Step 4: For each day, find contract pairs
    roll_spec = []
    previous_pair = None
    segment_start = start
    
    for current_date in date_range:
        available_futures = futures[futures['ts_recv'] <= current_date]
        target_expiration = current_date + pd.Timedelta(days=maturity_days)
        
        candidates_p = available_futures[available_futures['expiration'] <= target_expiration]
        p = candidates_p.nlargest(1, 'expiration')['instrument_id'].iloc[0] if len(candidates_p) > 0 else None
        
        candidates_n = available_futures[available_futures['expiration'] > target_expiration]
        n = candidates_n.nsmallest(1, 'expiration')['instrument_id'].iloc[0] if len(candidates_n) > 0 else None
        
        current_pair = (p, n)
        
        if current_pair != previous_pair:
            if previous_pair is not None and previous_pair[0] is not None:
                roll_spec.append({
                    "d0": str(segment_start),
                    "d1": str(current_date.date()),
                    "p": str(previous_pair[0]),
                    "n": str(previous_pair[1]),
                })
            segment_start = current_date.date()
            previous_pair = current_pair
    
    # Step 5: Save the last segment (only if segment_start < end)
    if previous_pair is not None and previous_pair[0] is not None and segment_start < end:
        roll_spec.append({
            "d0": str(segment_start),
            "d1": str(end),
            "p": str(previous_pair[0]),
            "n": str(previous_pair[1]),
        })
    
    return roll_spec


def constant_maturity_splice(symbol, roll_spec, all_data, date_col, price_col):
    """
    Goal: Mix prices from two contracts based on roll_spec, return DataFrame with weights
    """
    
    # Step 1: Prepare expiration information
    expirations = all_data[['instrument_id', 'expiration']].drop_duplicates()
    expirations = expirations.set_index('instrument_id')['expiration']
    
    # Step 2: Extract maturity_days
    maturity_days = int(symbol.split(".")[-1])
    maturity_timedelta = pd.Timedelta(days=maturity_days)
    
    tz = expirations.iloc[0].tz if len(expirations) > 0 else None
    
    # Step 3: Process data for each roll_spec segment
    segments = []
    
    for spec in roll_spec:
        d0 = pd.Timestamp(spec['d0'])
        d1 = pd.Timestamp(spec['d1'])
        pre = int(spec['p'])
        nxt = int(spec['n'])
        
        if tz is not None:
            if d0.tz is None:
                d0 = d0.tz_localize(tz)
            if d1.tz is None:
                d1 = d1.tz_localize(tz)
        
        time_range = pd.date_range(d0, d1, inclusive='left', tz=tz)
        
        pre_expiration = expirations[pre]
        next_expiration = expirations[nxt]
        
        pre_weight = (next_expiration - (time_range + maturity_timedelta)) / (
            next_expiration - pre_expiration
        )
        
        # Extract data for these two contracts in this time period from all_data
        pre_data = all_data[all_data['instrument_id'] == pre]
        pre_data = pre_data[pre_data[date_col].isin(time_range)]
        
        next_data = all_data[all_data['instrument_id'] == nxt]
        next_data = next_data[next_data[date_col].isin(time_range)]
        
        # Create result DataFrame with correct column order
        segment_df = pd.DataFrame({
            date_col: time_range,
            'pre_price': pre_data.set_index(date_col).reindex(time_range)[price_col].values,
            'pre_id': pre,
            'pre_expiration': pre_expiration,
            'next_price': next_data.set_index(date_col).reindex(time_range)[price_col].values,
            'next_id': nxt,
            'next_expiration': next_expiration,
            'pre_weight': pre_weight,
        })
        
        # Calculate weighted price
        segment_df[symbol] = segment_df['pre_weight'] * segment_df['pre_price'] + (1 - segment_df['pre_weight']) * segment_df['next_price']
        
        segments.append(segment_df)
    
    # Step 4: Combine all segments
    result = pd.concat(segments, ignore_index=True)
    
    return result
