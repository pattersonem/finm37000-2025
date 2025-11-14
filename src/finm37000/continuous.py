"""Functions to splice and adjust futures data into continuous data."""

import datetime
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
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    """Get roll specification for constant maturity contracts.

    Args:
        symbol: Symbol like "SR3.cm.182" where 182 is the maturity days.
        instrument_df: DataFrame with columns including 'instrument_id', 'expiration',
                       'instrument_class', and 'ts_recv'.
        start: Start date for the roll specification.
        end: End date for the roll specification (inclusive).

    Returns:
        List of dictionaries with keys 'd0', 'd1', 'p', 'n' representing
        date ranges and previous/next instrument IDs.
    """
    # Extract maturity days from symbol (e.g., "SR3.cm.182" -> 182)
    parts = symbol.split(".")
    maturity_days = int(parts[-1])

    # Filter to only futures contracts
    futures_df = instrument_df[instrument_df["instrument_class"] == "F"].copy()

    # Convert expiration to date for comparison (ignore time component)
    futures_df["expiration_date"] = pd.to_datetime(futures_df["expiration"]).dt.date

    roll_spec = []
    current_date = start

    def get_pair(date: datetime.date) -> tuple[str, str] | None:
        """Get the (prev_id, next_id) pair for a given date."""
        target_date = date + datetime.timedelta(days=maturity_days)

        # Filter to contracts that are live (ts_recv <= date)
        if "ts_recv" in futures_df.columns:
            if pd.api.types.is_datetime64_any_dtype(futures_df["ts_recv"]):
                live_contracts = futures_df[
                    pd.to_datetime(futures_df["ts_recv"]).dt.date <= date
                ]
            else:
                live_contracts = futures_df[
                    pd.to_datetime(futures_df["ts_recv"]).dt.date <= date
                ]
        else:
            live_contracts = futures_df

        # Find previous and next contracts that straddle target_date
        # Previous: expiration < target_date (strictly less than)
        # Next: expiration >= target_date (greater than or equal to)
        # This ensures that when target equals an expiration, that contract is "next"
        prev_contracts = live_contracts[
            live_contracts["expiration_date"] < target_date
        ]
        next_contracts = live_contracts[
            live_contracts["expiration_date"] >= target_date
        ]

        if len(prev_contracts) == 0 or len(next_contracts) == 0:
            return None

        # Get the latest previous contract and earliest next contract
        prev_contract = prev_contracts.loc[
            prev_contracts["expiration_date"].idxmax()
        ]
        next_contract = next_contracts.loc[
            next_contracts["expiration_date"].idxmin()
        ]

        prev_id = str(int(prev_contract["instrument_id"]))
        next_id = str(int(next_contract["instrument_id"]))

        return (prev_id, next_id)

    while current_date <= end:
        pair = get_pair(current_date)
        if pair is None:
            current_date += datetime.timedelta(days=1)
            continue

        prev_id, next_id = pair
        d0 = current_date

        # Find when the pair changes
        # The pair changes when target_date > next_expiration_date
        # So we need to find when (date + maturity_days) > next_expiration_date
        # Which means date > (next_expiration_date - maturity_days)
        next_exp_date = futures_df[
            futures_df["instrument_id"] == int(next_id)
        ]["expiration_date"].iloc[0]

        # Calculate the date when target would exceed next expiration
        # target_date = date + maturity_days > next_exp_date
        # date > next_exp_date - maturity_days
        # The period ends on the first date where target > next_exp
        threshold_date = next_exp_date - datetime.timedelta(days=maturity_days)

        # Also check for new contracts becoming live or other changes
        d1 = end  # Default to end (inclusive)

        # The period includes dates where target <= next_exp
        # So it ends on the first date where target > next_exp
        # That's threshold_date + 1
        if threshold_date >= current_date:
            candidate_d1 = threshold_date + datetime.timedelta(days=1)
            # Cap at end (since end is inclusive in the test expectation)
            if candidate_d1 <= end:
                d1 = candidate_d1
            else:
                d1 = end

        # Verify the pair doesn't change earlier due to new contracts becoming live
        check_date = current_date + datetime.timedelta(days=1)
        while check_date < d1 and check_date <= end:
            check_pair = get_pair(check_date)
            if check_pair is None or check_pair != (prev_id, next_id):
                d1 = check_date
                break
            check_date += datetime.timedelta(days=1)

        roll_spec.append(
            {
                "d0": d0.strftime("%Y-%m-%d"),
                "d1": d1.strftime("%Y-%m-%d"),
                "p": prev_id,
                "n": next_id,
            }
        )

        current_date = d1

    return roll_spec


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    """Create constant maturity prices by weighting two contracts.

    Args:
        symbol: Symbol like "SR3.cm.182" where 182 is the maturity days.
        roll_spec: List of dictionaries with 'd0', 'd1', 'p', 'n' keys.
        df: DataFrame with columns including date_col, price_col, 'instrument_id', 'expiration'.
        date_col: Name of the date column.
        price_col: Name of the price column.

    Returns:
        DataFrame with constant maturity prices and metadata.
    """
    # Extract maturity days from symbol
    parts = symbol.split(".")
    maturity_days = int(parts[-1])
    maturity_timedelta = pd.Timedelta(days=maturity_days)

    # Group data by instrument_id
    grouped = df.groupby("instrument_id")

    # Get expirations for all instruments
    expirations = {}
    for instrument_id, group in grouped:
        if "expiration" in group.columns:
            # Get unique expiration (should be same for all rows of same instrument)
            exp = pd.to_datetime(group["expiration"].iloc[0])
            expirations[instrument_id] = exp

    result_segments = []

    for spec in roll_spec:
        d0_str = spec["d0"]
        d1_str = spec["d1"]
        prev_id = int(spec["p"])
        next_id = int(spec["n"])

        # Create date range for this period
        d0 = pd.Timestamp(d0_str)
        d1 = pd.Timestamp(d1_str)
        dates = pd.date_range(start=d0, end=d1, tz=df[date_col].dt.tz, inclusive="left")

        # Get data for previous and next contracts
        prev_group = grouped.get_group(prev_id)
        next_group = grouped.get_group(next_id)

        prev_exp = expirations[prev_id]
        next_exp = expirations[next_id]

        segment_rows = []

        for date in dates:
            # Get prices for this date
            prev_data = prev_group[prev_group[date_col] == date]
            next_data = next_group[next_group[date_col] == date]

            if len(prev_data) == 0 or len(next_data) == 0:
                continue

            prev_price = prev_data[price_col].iloc[0]
            next_price = next_data[price_col].iloc[0]

            # Calculate weight: (next_exp - (date + maturity_days)) / (next_exp - prev_exp)
            target_date = date + maturity_timedelta
            weight = (next_exp - target_date) / (next_exp - prev_exp)

            # Calculate weighted price
            weighted_price = weight * prev_price + (1 - weight) * next_price

            segment_rows.append(
                {
                    "datetime": date,
                    "pre_price": prev_price,
                    "pre_id": prev_id,
                    "pre_expiration": prev_exp,
                    "next_price": next_price,
                    "next_id": next_id,
                    "next_expiration": next_exp,
                    "pre_weight": weight,
                    symbol: weighted_price,
                }
            )

        if segment_rows:
            segment_df = pd.DataFrame(segment_rows)
            result_segments.append(segment_df)

    if not result_segments:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
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
        )

    result = pd.concat(result_segments, ignore_index=True)
    return result
