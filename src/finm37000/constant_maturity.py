"""Functions to create constant maturity futures contracts."""

import datetime
import pandas as pd


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    """Generate roll specification for constant maturity futures.
    
    Args:
        symbol: Constant maturity symbol (e.g., "SR3.cm.182")
        instrument_df: DataFrame with instrument definitions
        start: Start date for the roll specification
        end: End date for the roll specification
    
    Returns:
        List of dicts with keys d0, d1, p (previous), n (next)
    """
    # Extract maturity days from symbol
    maturity_days = int(symbol.split(".")[-1])
    maturity_delta = pd.Timedelta(days=maturity_days)
    
    # Filter for futures only
    futures = instrument_df[instrument_df["instrument_class"] == "F"].copy()
    futures["expiration"] = pd.to_datetime(futures["expiration"])
    futures = futures.sort_values("expiration").reset_index(drop=True)
    
    roll_spec = []
    current_date = start
    
    while current_date <= end:
        target_maturity = pd.Timestamp(current_date).tz_localize("UTC") + maturity_delta
        
        # Filter instruments that are live at current_date
        ts_recv = pd.to_datetime(futures["ts_recv"])
        if ts_recv.dt.tz is None:
            ts_recv = ts_recv.dt.tz_localize("UTC")
        
        current_ts = pd.Timestamp(current_date)
        if current_ts.tz is None:
            current_ts = current_ts.tz_localize("UTC")
        
        live_instruments = futures[ts_recv <= current_ts]
        
        # Find contracts that straddle the target maturity
        prev_contract = live_instruments[
            live_instruments["expiration"] <= target_maturity
        ].tail(1)
        
        next_contract = live_instruments[
            live_instruments["expiration"] > target_maturity
        ].head(1)
        
        if prev_contract.empty or next_contract.empty:
            current_date += datetime.timedelta(days=1)
            continue
        
        prev_id = str(prev_contract["instrument_id"].iloc[0])
        next_id = str(next_contract["instrument_id"].iloc[0])
        prev_exp = prev_contract["expiration"].iloc[0]
        
        # The pair is valid until the day after prev contract expires
        next_roll_date = prev_exp.date() + datetime.timedelta(days=1)
        
        # Don't go past the end date
        if next_roll_date > end:
            next_roll_date = end + datetime.timedelta(days=1)
        
        # Add new spec or extend existing one
        if not roll_spec or roll_spec[-1]["p"] != prev_id or roll_spec[-1]["n"] != next_id:
            roll_spec.append({
                "d0": current_date.isoformat(),
                "d1": next_roll_date.isoformat(),
                "p": prev_id,
                "n": next_id,
            })
        else:
            roll_spec[-1]["d1"] = next_roll_date.isoformat()
        
        current_date = next_roll_date
    
    return roll_spec


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """Create constant maturity futures prices by interpolating between contracts.
    
    Args:
        symbol: Constant maturity symbol (e.g., "SR3.cm.182")
        roll_spec: List of dicts with keys d0, d1, p (previous), n (next)
        data: DataFrame with price data
        date_col: Name of the date column
        price_col: Name of the price column
    
    Returns:
        DataFrame with interpolated constant maturity prices
    """
    maturity_days = int(symbol.split(".")[-1])
    maturity_delta = pd.Timedelta(days=maturity_days)
    
    result_segments = []
    
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"])
        d1 = pd.Timestamp(spec["d1"])
        prev_id = int(spec["p"])
        next_id = int(spec["n"])
        
        # Get data for both contracts
        prev_data = data[data["instrument_id"] == prev_id].set_index(date_col).sort_index()
        next_data = data[data["instrument_id"] == next_id].set_index(date_col).sort_index()
        
        # Create date range for this segment
        tz = prev_data.index.tz if hasattr(prev_data.index, 'tz') else None
        date_range = pd.date_range(start=d0, end=d1, freq="D", tz=tz, inclusive="left")
        
        # Get prices and expirations for each date
        segment_data = []
        for dt in date_range:
            if dt not in prev_data.index or dt not in next_data.index:
                continue
            
            prev_price = prev_data.loc[dt, price_col]
            next_price = next_data.loc[dt, price_col]
            prev_exp = pd.Timestamp(prev_data.loc[dt, "expiration"])
            next_exp = pd.Timestamp(next_data.loc[dt, "expiration"])
            
            # Calculate target maturity date
            target_maturity = dt + maturity_delta
            
            # Calculate interpolation weight for previous contract
            time_to_next = (next_exp - target_maturity).total_seconds()
            time_between = (next_exp - prev_exp).total_seconds()
            
            if time_between == 0:
                prev_weight = 0.5
            else:
                prev_weight = time_to_next / time_between
            
            # Interpolated price
            interpolated_price = prev_weight * prev_price + (1 - prev_weight) * next_price
            
            segment_data.append({
                date_col: dt,
                "pre_price": prev_price,
                "pre_id": prev_id,
                "pre_expiration": prev_exp,
                "next_price": next_price,
                "next_id": next_id,
                "next_expiration": next_exp,
                "pre_weight": prev_weight,
                symbol: interpolated_price,
            })
        
        if segment_data:
            result_segments.append(pd.DataFrame(segment_data))
    
    return pd.concat(result_segments, ignore_index=True)
