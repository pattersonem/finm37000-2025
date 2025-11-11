# src/finm37000/constant_maturity.py

import pandas as pd
from typing import List, Dict, Any
import datetime


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date
) -> List[Dict[str, str]]:
    """
    Generate roll specifications for constant maturity futures.
    """
    # Extract maturity days from symbol
    maturity_days = int(symbol.split('.')[-1])
    
    # Filter futures only
    futures_df = instrument_df[instrument_df['instrument_class'] == 'F'].copy()
    futures_df = futures_df.sort_values('expiration').reset_index(drop=True)
    
    roll_specs = []
    current_date = pd.Timestamp(start).tz_localize('UTC')
    end_date = pd.Timestamp(end).tz_localize('UTC')
    
    while current_date < end_date:
        target_date = current_date + pd.Timedelta(days=maturity_days)
        active_futures = futures_df[futures_df['ts_recv'] <= current_date]
        
        if len(active_futures) == 0:
            current_date += pd.Timedelta(days=1)
            continue
        
        prev_contracts = active_futures[active_futures['expiration'] <= target_date]
        next_contracts = active_futures[active_futures['expiration'] > target_date]
        
        if len(prev_contracts) == 0 or len(next_contracts) == 0:
            current_date += pd.Timedelta(days=1)
            continue
        
        prev_id = str(int(prev_contracts.iloc[-1]['instrument_id']))
        next_id = str(int(next_contracts.iloc[0]['instrument_id']))
        
        # Find when to roll to next pair
        roll_end = current_date + pd.Timedelta(days=1)
        while roll_end <= end_date:
            test_target = roll_end + pd.Timedelta(days=maturity_days)
            test_active = futures_df[futures_df['ts_recv'] <= roll_end]
            
            if len(test_active) == 0:
                roll_end += pd.Timedelta(days=1)
                continue
            
            test_prev = test_active[test_active['expiration'] <= test_target]
            test_next = test_active[test_active['expiration'] > test_target]
            
            if len(test_prev) == 0 or len(test_next) == 0:
                roll_end += pd.Timedelta(days=1)
                continue
            
            if (str(int(test_prev.iloc[-1]['instrument_id'])) != prev_id or
                str(int(test_next.iloc[0]['instrument_id'])) != next_id):
                break
            
            roll_end += pd.Timedelta(days=1)
        
        if roll_end > end_date:
            roll_end = end_date
        
        roll_specs.append({
            'd0': current_date.strftime('%Y-%m-%d'),
            'd1': roll_end.strftime('%Y-%m-%d'),
            'p': prev_id,
            'n': next_id
        })
        
        current_date = roll_end
    
    return roll_specs

def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    all_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price"
) -> pd.DataFrame:
    """
    Create constant maturity futures prices.
    """
    maturity_days = int(symbol.split('.')[-1])
    maturity_delta = pd.Timedelta(days=maturity_days)
    
    segments = []
    
    for spec in roll_spec:
        d0 = pd.to_datetime(spec['d0']).tz_localize('UTC')
        d1 = pd.to_datetime(spec['d1']).tz_localize('UTC')
        pre_id = int(spec['p'])
        next_id = int(spec['n'])
        
        dates = pd.date_range(start=d0, end=d1, freq='D', inclusive='left')
        if len(dates) == 0:
            continue
        
        pre_data = all_data[all_data['instrument_id'] == pre_id].copy()
        next_data = all_data[all_data['instrument_id'] == next_id].copy()
        
        # Ensure timezone consistency
        if not isinstance(pre_data[date_col].dtype, pd.DatetimeTZDtype):
            pre_data[date_col] = pd.to_datetime(pre_data[date_col]).dt.tz_localize('UTC')
        if not isinstance(next_data[date_col].dtype, pd.DatetimeTZDtype):
            next_data[date_col] = pd.to_datetime(next_data[date_col]).dt.tz_localize('UTC')
        
        pre_data = pre_data.set_index(date_col)
        next_data = next_data.set_index(date_col)
        
        pre_exp = pre_data['expiration'].iloc[0] if len(pre_data) > 0 else None
        next_exp = next_data['expiration'].iloc[0] if len(next_data) > 0 else None
        
        segment = pd.DataFrame({'datetime': dates})
        segment['pre_price'] = segment['datetime'].map(
            lambda x: pre_data.loc[x, price_col] if x in pre_data.index else pre_id
        )
        segment['pre_id'] = pre_id
        segment['pre_expiration'] = pre_exp
        segment['next_price'] = segment['datetime'].map(
            lambda x: next_data.loc[x, price_col] if x in next_data.index else next_id
        )
        segment['next_id'] = next_id
        segment['next_expiration'] = next_exp
        
        # Calculate weights - FIXED
        target_dates = segment['datetime'] + maturity_delta
        time_to_next = (segment['next_expiration'] - target_dates).dt.total_seconds()
        
        # Calculate total_time properly
        total_time = (segment['next_expiration'].iloc[0] - segment['pre_expiration'].iloc[0]).total_seconds()
        segment['pre_weight'] = time_to_next / total_time
        
        # Weighted price
        segment[symbol] = (
            segment['pre_weight'] * segment['pre_price'] + (1 - segment['pre_weight']) * segment['next_price']
        )
        segments.append(segment)
    
    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()