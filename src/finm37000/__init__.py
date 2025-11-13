"""A package to support FINM37000."""


from datetime import timedelta
import pandas as pd
from typing import List, Dict, Any

from .agg import (
    make_ohlcv as make_ohlcv,
)
from .continuous import (
    additive_splice as additive_splice,
    multiplicative_splice as multiplicative_splice,
)
from .db_env_util import (
    temp_env as temp_env,
    get_databento_api_key as get_databento_api_key,
)
from .futures import (
    favorite_def_cols as favorite_def_cols,
    get_all_legs_on as get_all_legs_on,
    get_official_stats as get_official_stats,
)
from .time import (
    as_ct as as_ct,
    get_cme_next_session_end as get_cme_next_session_end,
    get_cme_session_end as get_cme_session_end,
    tz_chicago as tz_chicago,
    us_business_day as us_business_day,
)


def get_roll_spec(product: str, instrument_df: pd.DataFrame, start, end):
    """
    Build a roll spec for a Databento-style constant-maturity series.

    Parameters
    ----------
    product : str
        Product string like "SR3.cm.182". The final component is taken
        as the number of maturity days.
    instrument_df : DataFrame
        Must contain at least:
        - instrument_id
        - raw_symbol
        - expiration (datetime64[ns, tz] or naive)
        - instrument_class ("F" for futures, "S" for spreads)
        - ts_recv (datetime64[ns, tz] or naive)
    start, end : date or datetime
        Date range [start, end) over which to compute the spec.

    Returns
    -------
    List[dict]
        Each dict has:
        - d0, d1 : date strings "YYYY-MM-DD" for [d0, d1)
        - p, n   : instrument_id strings for previous/next contract
    """

    # 1. Parse maturity days from product string, e.g. "SR3.cm.182" -> 182
    try:
        maturity_days = int(str(product).split(".")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse maturity days from product '{product}'") from exc

    # Root symbol: "SR3.cm.182" -> "SR3"
    root = str(product).split(".")[0]

    # 2. Normalize start/end to date (allow pd.Timestamp as input)
    if isinstance(start, pd.Timestamp):
        start = start.date()
    if isinstance(end, pd.Timestamp):
        end = end.date()

    # 3. Filter to outright futures for this root symbol
    df = instrument_df.copy()
    df = df[
        (df["instrument_class"] == "F")
        & (df["raw_symbol"].astype(str).str.startswith(root))
    ].copy()
    if df.empty:
        return []

    # 4. Work purely with dates (ignore times and timezones)
    df["exp_date"] = pd.to_datetime(df["expiration"]).dt.date
    df["live_date"] = pd.to_datetime(df["ts_recv"]).dt.date

    df = df.sort_values("exp_date").reset_index(drop=True)

    current = start
    segments = []
    cur_pair = None
    seg_start = None

    # 5. For each date in [start, end), find the pair whose expirations
    #    straddle (date + maturity_days), using only instruments live on that date.
    while current < end:
        # Instruments that exist (are "live") by this date
        avail = df[df["live_date"] <= current]
        pair = (None, None)

        if not avail.empty:
            target = current + timedelta(days=maturity_days)
            avail = avail.sort_values("exp_date")

            # Expiry strictly before target vs on/after target:
            # expiries < target  -> previous
            # expiries >= target -> next
            prev_rows = avail[avail["exp_date"] < target]
            next_rows = avail[avail["exp_date"] >= target]

            if not prev_rows.empty and not next_rows.empty:
                prev_id = int(prev_rows.iloc[-1]["instrument_id"])
                next_id = int(next_rows.iloc[0]["instrument_id"])
                pair = (prev_id, next_id)

        # Compress consecutive dates with the same (p, n) pair into segments
        if pair != cur_pair:
            if cur_pair is not None:
                segments.append((seg_start, current, cur_pair))
            seg_start = current
            cur_pair = pair

        current += timedelta(days=1)

    # Close the last segment
    if cur_pair is not None:
        segments.append((seg_start, end, cur_pair))

    # 6. Format output exactly as the test expects
    result = []
    for d0, d1, (p_id, n_id) in segments:
        if p_id is None or n_id is None:
            # Skip ranges where we couldn't form a valid pair
            continue
        result.append(
            {
                "d0": d0.isoformat(),
                "d1": d1.isoformat(),  # end of half-open interval [d0, d1)
                "p": str(p_id),
                "n": str(n_id),
            }
        )

    return result


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, Any]],
    all_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Create a constant maturity splice from futures data.
    
    Args:
        symbol: The symbol name for the spliced series (e.g., "SR3.cm.182")
        roll_spec: List of roll specifications, each containing:
            - d0: Start date (inclusive)
            - d1: End date (exclusive)
            - p: Previous instrument ID
            - n: Next instrument ID
        all_data: DataFrame containing all instrument data with columns:
            - instrument_id: Identifier for each instrument
            - datetime: Timestamp
            - price: Price data
            - expiration: Expiration timestamp
        date_col: Name of the datetime column (default: "datetime")
        price_col: Name of the price column (default: "price")
    
    Returns:
        DataFrame with columns:
            - datetime
            - pre_price, pre_id, pre_expiration
            - next_price, next_id, next_expiration
            - pre_weight
            - {symbol}: The weighted spliced price
    """
    # Extract maturity days from symbol (e.g., "SR3.cm.182" -> 182 days)
    maturity_days = pd.Timedelta(days=int(symbol.split(".")[-1]))
    
    segments = []
    
    for spec in roll_spec:
        d0 = spec["d0"]
        d1 = spec["d1"]
        pre_id = int(spec["p"])
        nxt_id = int(spec["n"])
        
        # Create date range for this segment (inclusive start, exclusive end)
        time_range = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        
        # Filter data for the two instruments
        pre_data = all_data[all_data["instrument_id"] == pre_id].set_index(date_col)
        nxt_data = all_data[all_data["instrument_id"] == nxt_id].set_index(date_col)
        
        # Get expiration dates
        pre_expiration = pre_data["expiration"].iloc[0]
        nxt_expiration = nxt_data["expiration"].iloc[0]
        
        # Calculate weights for each date in the range
        # Weight is based on how far we are from target maturity
        # pre_weight = (next_expiration - (current_time + maturity_days)) / (next_expiration - pre_expiration)
        pre_weights = (nxt_expiration - (time_range + maturity_days)) / (
            nxt_expiration - pre_expiration
        )
        
        # Get prices for each date
        pre_prices = pre_data.loc[time_range, price_col].values
        nxt_prices = nxt_data.loc[time_range, price_col].values
        
        # Calculate weighted spliced price
        spliced_price = pre_weights * pre_prices + (1 - pre_weights) * nxt_prices
        
        # Create segment dataframe
        segment_df = pd.DataFrame({
            date_col: time_range,
            "pre_price": pre_prices,
            "pre_id": pre_id,
            "pre_expiration": pre_expiration,
            "next_price": nxt_prices,
            "next_id": nxt_id,
            "next_expiration": nxt_expiration,
            "pre_weight": pre_weights,
            symbol: spliced_price,
        })
        
        segments.append(segment_df)
    
    # Concatenate all segments
    result = pd.concat(segments, ignore_index=True)
    
    return result