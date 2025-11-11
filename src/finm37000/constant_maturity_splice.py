import pandas as pd  # Import pandas for data manipulation and time handling

def constant_maturity_splice(
    symbol: str,                       # Synthetic symbol like "SR3.cm.182"
    roll_spec: list[dict[str, str]],   # List of dicts defining roll windows (d0, d1, p, n)
    df: pd.DataFrame,                  # Raw futures data per instrument_id
    date_col: str = "datetime",        # Column name containing datetimes
    price_col: str = "close",          # Column name containing prices
) -> pd.DataFrame:
    """
    Build a constant-maturity synthetic price series by linearly blending
    the 'pre' and 'next' contracts over each roll window in roll_spec.

    The weight on the pre-contract is:
        f_t = (exp_next - (t + maturity_days)) / (exp_next - exp_pre)
    """

    # Try extracting the maturity in days (e.g. 182 from "SR3.cm.182")
    try:
        maturity_days = int(symbol.split(".")[-1])  # Get the integer after the last dot
    except Exception as e:  # If conversion fails
        raise ValueError(f"Symbol must end with days to maturity, got {symbol!r}") from e
    maturity_delta = pd.Timedelta(days=maturity_days)  # Convert days to a pandas Timedelta

    # Ensure datetime column is timezone-aware (UTC)
    if not pd.api.types.is_datetime64tz_dtype(df[date_col]):
        df = df.copy()  # Avoid modifying the original DataFrame
        df[date_col] = pd.to_datetime(df[date_col], utc=True)  # Convert to UTC timestamps

    tz = df[date_col].dt.tz  # Capture the timezone info for later use
    exp_map = (  # Create a Series mapping instrument_id → expiration datetime
        df[["instrument_id", "expiration"]]  # Keep only relevant columns
        .drop_duplicates("instrument_id")    # One expiration per instrument
        .set_index("instrument_id")["expiration"]  # Use instrument_id as index
        .map(pd.to_datetime)  # Convert expiration strings to datetime
    )

    pieces: list[pd.DataFrame] = []  # Initialize a list to collect segment DataFrames

    # Loop over each roll specification entry
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)  # Start of window
        d1 = pd.Timestamp(spec["d1"], tz=tz)  # End of window
        pre_id = int(spec["p"])  # Previous contract instrument ID
        nxt_id = int(spec["n"])  # Next contract instrument ID

        # Build a continuous date range for this window [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz=tz, inclusive="left")

        # Extract the prices for the 'pre' contract during this window
        pre = (
            df[(df["instrument_id"] == pre_id) & (df[date_col] >= d0) & (df[date_col] < d1)]
            [[date_col, price_col]]  # Keep datetime and price columns
            .rename(columns={price_col: "pre_price"})  # Rename price column to pre_price
        )

        # Extract the prices for the 'next' contract during this window
        nxt = (
            df[(df["instrument_id"] == nxt_id) & (df[date_col] >= d0) & (df[date_col] < d1)]
            [[date_col, price_col]]  # Keep datetime and price columns
            .rename(columns={price_col: "next_price"})  # Rename price column to next_price
        )

        # Create a full date grid to ensure all dates exist (even if prices are missing)
        seg = pd.DataFrame({date_col: t})
        # Merge in pre and next contract prices by datetime
        seg = seg.merge(pre, on=date_col, how="left").merge(nxt, on=date_col, how="left")

        # Add identifying info for the pre and next contracts
        seg["pre_id"] = pre_id  # Store previous instrument ID
        seg["next_id"] = nxt_id  # Store next instrument ID
        seg["pre_expiration"] = exp_map.loc[pre_id]  # Expiration date of previous contract
        seg["next_expiration"] = exp_map.loc[nxt_id]  # Expiration date of next contract

        # Compute the linear weight for the pre contract
        seg["pre_weight"] = (
            (seg["next_expiration"] - (seg[date_col] + maturity_delta))  # Time gap between (target maturity) and next expiry
            / (seg["next_expiration"] - seg["pre_expiration"])  # Divide by full time between expirations
        )

        # Compute weighted constant-maturity price: f*pre + (1-f)*next
        seg[symbol] = seg["pre_weight"] * seg["pre_price"] + (1 - seg["pre_weight"]) * seg["next_price"]

        # Keep only the columns (and order) expected by the test suite
        seg = seg[
            [
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
        ]
        # Append this segment to the list of results
        pieces.append(seg)

    # Combine all roll window segments into one DataFrame
    out = pd.concat(pieces, ignore_index=True)

    # The tests pass date_col="datetime", so we leave column names unchanged.
    return out  # Return the complete constant-maturity DataFrame