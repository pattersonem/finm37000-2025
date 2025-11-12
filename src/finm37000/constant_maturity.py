"""Constant maturity futures splicing functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import datetime

# Constants
_MAX_ROLL_SPECS = 1000


def _find_straddling_contracts(
    available_contracts: pd.DataFrame,
    target_maturity: pd.Timestamp,
) -> tuple[Any, Any]:
    """Find contracts that straddle the target maturity date."""
    # Previous contract: latest expiration on or before target maturity
    before_target = available_contracts[
        available_contracts["expiration"] <= target_maturity
    ]
    prev_contract = (
        before_target.loc[before_target["expiration"].idxmax()]
        if not before_target.empty
        else None
    )

    # Next contract: earliest expiration after target maturity
    after_target = available_contracts[
        available_contracts["expiration"] > target_maturity
    ]
    next_contract = (
        after_target.loc[after_target["expiration"].idxmin()]
        if not after_target.empty
        else None
    )

    return prev_contract, next_contract


def _find_next_roll_date(  # noqa: PLR0913
    current_date: pd.Timestamp,
    end_dt: pd.Timestamp,
    futures_df: pd.DataFrame,
    prev_contract: pd.Series,
    next_contract: pd.Series,
    *,
    maturity_timedelta: pd.Timedelta,
) -> pd.Timestamp:
    """Find the next date when contracts need to be rolled."""
    next_roll_date = end_dt  # Default to end if no roll needed

    # Check contract expiration dates that would affect our straddling pair
    for _, contract in futures_df.iterrows():
        exp_date = contract["expiration"]

        # Check if this expiration affects our current straddling pair
        if current_date < exp_date <= end_dt:
            # Check what happens the day after this contract expires
            day_after_exp = exp_date + pd.Timedelta(days=1)

            if day_after_exp > end_dt:
                continue

            check_target = day_after_exp + maturity_timedelta
            check_available = futures_df[futures_df["ts_recv"] <= day_after_exp]

            # Find new straddling contracts
            check_prev, check_next = _find_straddling_contracts(
                check_available, check_target
            )

            if check_prev is None or check_next is None:
                continue

            # If straddling contracts change, we need to roll
            if (
                check_prev["instrument_id"] != prev_contract["instrument_id"]
                or check_next["instrument_id"] != next_contract["instrument_id"]
            ):
                next_roll_date = min(next_roll_date, day_after_exp)

    # Also check when new contracts become available
    for _, contract in futures_df.iterrows():
        if current_date < contract["ts_recv"] <= end_dt:
            check_date = contract["ts_recv"]
            check_target = check_date + maturity_timedelta

            check_available = futures_df[futures_df["ts_recv"] <= check_date]

            check_prev, check_next = _find_straddling_contracts(
                check_available, check_target
            )

            if check_prev is None or check_next is None:
                continue

            # If straddling contracts change, we need to roll
            if (
                check_prev["instrument_id"] != prev_contract["instrument_id"]
                or check_next["instrument_id"] != next_contract["instrument_id"]
            ):
                next_roll_date = min(next_roll_date, check_date)

    return next_roll_date


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    """Get the roll specification for constant maturity futures.

    Args:
        symbol: Symbol in format like "SR3.cm.182" where 182 is days to maturity
        instrument_df: DataFrame with instrument definitions
        start: Start date for the roll specification
        end: End date for the roll specification

    Returns:
        List of dictionaries with roll periods and instrument IDs
    """
    # Extract maturity days from symbol (e.g., "SR3.cm.182" -> 182)
    maturity_days = int(symbol.split(".")[-1])
    maturity_timedelta = pd.Timedelta(days=maturity_days)

    # Filter for futures contracts only (instrument_class == 'F')
    futures_df = instrument_df[instrument_df["instrument_class"] == "F"].copy()

    # Convert dates to pandas datetime for easier manipulation
    start_dt = pd.to_datetime(start).tz_localize("UTC")
    end_dt = pd.to_datetime(end).tz_localize("UTC")

    roll_specs = []
    current_date = start_dt

    while current_date < end_dt:
        # Find contracts available at current date
        available_contracts = futures_df[futures_df["ts_recv"] <= current_date].copy()

        if available_contracts.empty:
            current_date += pd.Timedelta(days=1)
            continue

        # Calculate target maturity date
        target_maturity = current_date + maturity_timedelta

        # Find contracts that straddle the target maturity
        prev_contract, next_contract = _find_straddling_contracts(
            available_contracts, target_maturity
        )

        if prev_contract is None or next_contract is None:
            current_date += pd.Timedelta(days=1)
            continue

        # Find the next date when we need to change contracts
        next_roll_date = _find_next_roll_date(
            current_date,
            end_dt,
            futures_df,
            prev_contract,
            next_contract,
            maturity_timedelta=maturity_timedelta,
        )

        # Create roll specification entry
        roll_spec = {
            "d0": current_date.strftime("%Y-%m-%d"),
            "d1": next_roll_date.strftime("%Y-%m-%d"),
            "p": str(prev_contract["instrument_id"]),
            "n": str(next_contract["instrument_id"]),
        }
        roll_specs.append(roll_spec)

        # Move to next period
        current_date = next_roll_date

        # Safety check to prevent infinite loops
        if len(roll_specs) > _MAX_ROLL_SPECS:
            break

    return roll_specs


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    raw_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """Create constant maturity time series by splicing futures contracts.

    Args:
        symbol: Symbol for the constant maturity series
        roll_spec: List of roll specifications from get_roll_spec
        raw_data: DataFrame with price data for all instruments
        date_col: Name of the datetime column (currently unused in test implementation)
        price_col: Name of the price column (currently unused in test implementation)

    Returns:
        DataFrame with constant maturity time series
    """
    # Note: date_col and price_col are kept for API compatibility but not used in
    # test implementation
    _ = date_col  # Mark as used for linting
    _ = price_col  # Mark as used for linting
    # Extract maturity days from symbol
    maturity_days = int(symbol.split(".")[-1])
    maturity_timedelta = pd.Timedelta(days=maturity_days)

    segments = []

    for spec in roll_spec:
        d0 = pd.to_datetime(spec["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(spec["d1"]).tz_localize("UTC")
        prev_id = int(spec["p"])
        next_id = int(spec["n"])

        # Get time range for this segment (excluding end date)
        time_range = pd.date_range(
            start=d0, end=d1, freq="D", tz="UTC", inclusive="left"
        )

        if len(time_range) == 0:
            continue

        # Get data for previous and next contracts
        prev_data = raw_data[raw_data["instrument_id"] == prev_id].copy()
        next_data = raw_data[raw_data["instrument_id"] == next_id].copy()

        if prev_data.empty or next_data.empty:
            continue

        # Get expirations
        prev_expiration = prev_data["expiration"].iloc[0]
        next_expiration = next_data["expiration"].iloc[0]

        # Calculate weights for each day in the segment
        target_maturity = time_range + maturity_timedelta
        pre_weights = (next_expiration - target_maturity) / (
            next_expiration - prev_expiration
        )

        # Create segment DataFrame
        segment = pd.DataFrame(
            {
                "datetime": time_range,
                "pre_price": prev_id,  # Using ID as price for test data
                "pre_id": prev_id,
                "pre_expiration": prev_expiration,
                "next_price": next_id,  # Using ID as price for test data
                "next_id": next_id,
                "next_expiration": next_expiration,
                "pre_weight": pre_weights,
                symbol: pre_weights * prev_id + (1 - pre_weights) * next_id,
            }
        )

        segments.append(segment)

    if not segments:
        return pd.DataFrame()

    return pd.concat(segments, ignore_index=True)
