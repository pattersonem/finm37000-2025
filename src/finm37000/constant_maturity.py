"""
Constant maturity futures price construction.
"""

import pandas as pd
from typing import List, Dict
import datetime


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> List[Dict[str, str]]:
    """Generate roll specification for constant maturity futures."""
    
    # Extract maturity days from symbol
    maturity_days = int(symbol.split(".")[-1])
    
    # Filter for futures only
    futures = instrument_df[instrument_df["instrument_class"] == "F"].copy()
    
    # Ensure timezone awareness
    if futures["expiration"].dt.tz is None:
        futures["expiration"] = futures["expiration"].dt.tz_localize("UTC")
    if futures["ts_recv"].dt.tz is None:
        futures["ts_recv"] = futures["ts_recv"].dt.tz_localize("UTC")
    
    # Sort by expiration
    futures = futures.sort_values("expiration").reset_index(drop=True)
    
    roll_spec = []
    current_date = start
    
    # Build a list of all potential roll dates
    roll_dates = set()
    
    for _, contract in futures.iterrows():
        # Roll happens the day AFTER: date + maturity_days > contract_expiration
        # Which means: date > contract_expiration - maturity_days
        # So the first date to roll is: contract_expiration - maturity_days + 1 day
        roll_date_for_expiration = contract["expiration"].date() - datetime.timedelta(days=maturity_days - 1)
        if start < roll_date_for_expiration <= end:
            roll_dates.add(roll_date_for_expiration)
        
        # When does this contract become live?
        live_date = contract["ts_recv"].date()
        if start < live_date <= end:
            roll_dates.add(live_date)
    
    # Sort roll dates
    roll_dates = sorted(roll_dates)
    
    # Add start and end
    all_dates = [start] + roll_dates + [end]
    
    # Remove duplicates and sort
    all_dates = sorted(set(all_dates))
    
    # Generate roll spec for each period, merging consecutive periods with same pair
    i = 0
    while i < len(all_dates) - 1:
        d0 = all_dates[i]
        
        # Find live contracts at d0
        d0_tz = pd.Timestamp(d0).tz_localize("UTC")
        live = futures[futures["ts_recv"] <= d0_tz]
        
        if len(live) == 0:
            i += 1
            continue
        
        # Target maturity at d0
        target = d0_tz + pd.Timedelta(days=maturity_days)
        
        # Find straddling pair
        pre = live[live["expiration"] <= target]
        nxt = live[live["expiration"] > target]
        
        if len(pre) == 0 or len(nxt) == 0:
            i += 1
            continue
        
        pre_contract = pre.iloc[-1]
        nxt_contract = nxt.iloc[0]
        pre_id = int(pre_contract["instrument_id"])
        nxt_id = int(nxt_contract["instrument_id"])
        
        # Find the end of this period by looking ahead
        # Continue while the pair stays the same
        d1 = None
        for j in range(i + 1, len(all_dates)):
            test_date = all_dates[j]
            test_date_tz = pd.Timestamp(test_date).tz_localize("UTC")
            test_live = futures[futures["ts_recv"] <= test_date_tz]
            
            if len(test_live) == 0:
                continue
            
            test_target = test_date_tz + pd.Timedelta(days=maturity_days)
            test_pre = test_live[test_live["expiration"] <= test_target]
            test_nxt = test_live[test_live["expiration"] > test_target]
            
            if len(test_pre) == 0 or len(test_nxt) == 0:
                d1 = test_date
                i = j
                break
            
            test_pre_id = int(test_pre.iloc[-1]["instrument_id"])
            test_nxt_id = int(test_nxt.iloc[0]["instrument_id"])
            
            if test_pre_id != pre_id or test_nxt_id != nxt_id:
                d1 = test_date
                i = j
                break
        
        if d1 is None:
            d1 = all_dates[-1]
            i = len(all_dates)
        
        roll_spec.append({
            "d0": str(d0),
            "d1": str(d1),
            "p": str(pre_id),
            "n": str(nxt_id),
        })
    
    return roll_spec


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """Create constant maturity prices by splicing contracts."""
    
    maturity_days = pd.Timedelta(days=int(symbol.split(".")[-1]))
    
    result_segments = []
    
    for spec in roll_spec:
        d0 = pd.to_datetime(spec["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(spec["d1"]).tz_localize("UTC")
        pre_id = int(spec["p"])
        next_id = int(spec["n"])
        
        # Get data for both contracts
        pre_data = data[data["instrument_id"] == pre_id].copy()
        next_data = data[data["instrument_id"] == next_id].copy()
        
        if len(pre_data) == 0 or len(next_data) == 0:
            continue
        
        # Ensure timezone awareness
        if pre_data[date_col].dt.tz is None:
            pre_data[date_col] = pre_data[date_col].dt.tz_localize("UTC")
        if next_data[date_col].dt.tz is None:
            next_data[date_col] = next_data[date_col].dt.tz_localize("UTC")
        
        # Filter to date range [d0, d1)
        pre_data = pre_data[(pre_data[date_col] >= d0) & (pre_data[date_col] < d1)]
        next_data = next_data[(next_data[date_col] >= d0) & (next_data[date_col] < d1)]
        
        # Merge on date
        merged = pre_data.merge(
            next_data,
            on=date_col,
            how="outer",
            suffixes=("_pre", "_next")
        )
        
        merged = merged.sort_values(date_col).reset_index(drop=True)
        
        if len(merged) == 0:
            continue
        
        # Get expirations
        pre_expiration = merged["expiration_pre"].iloc[0]
        next_expiration = merged["expiration_next"].iloc[0]
        
        # Calculate weights
        target_maturity = merged[date_col] + maturity_days
        pre_weight = (next_expiration - target_maturity) / (next_expiration - pre_expiration)
        
        # Calculate constant maturity price
        cm_price = pre_weight * merged[f"{price_col}_pre"] + (1 - pre_weight) * merged[f"{price_col}_next"]
        
        # Build result
        segment = pd.DataFrame({
            "datetime": merged[date_col],
            "pre_price": merged[f"{price_col}_pre"],
            "pre_id": pre_id,
            "pre_expiration": pre_expiration,
            "next_price": merged[f"{price_col}_next"],
            "next_id": next_id,
            "next_expiration": next_expiration,
            "pre_weight": pre_weight,
            symbol: cm_price,
        })
        
        result_segments.append(segment)
    
    result = pd.concat(result_segments, ignore_index=True)
    
    return result
