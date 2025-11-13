import pandas as pd


def _parse_maturity_days(symbol: str) -> int:
    """
    Given a symbol like 'SR3.cm.182', return maturity days (182).
    """
    try:
        return int(symbol.split(".")[-1])
    except Exception:
        raise ValueError(f"Cannot parse maturity days from symbol: {symbol}")


def get_roll_spec(symbol: str, instrument_df: pd.DataFrame, start, end):
    """
    Build roll specification for constant maturity futures.

    Parameters
    ----------
    symbol : str
        Example: "SR3.cm.182"
    instrument_df : DataFrame
        Must contain columns: instrument_id, expiration, ts_recv
    start, end : date
        The date range for which we need a roll schedule.

    Returns
    -------
    list of dicts with keys:
        d0, d1, p, n
    """
    maturity_days = _parse_maturity_days(symbol)
    maturity_offset = pd.Timedelta(days=maturity_days)

    # Use only futures (instrument_class == "F")
    futures = instrument_df[instrument_df["instrument_class"] == "F"].copy()

    # Sort by expiration ascending
    futures = futures.sort_values("expiration")

    result = []
    current_date = pd.to_datetime(start).date()

    # Iterate through contiguous futures pairs: (contract i, contract i+1)
    for i in range(len(futures) - 1):
        pre = futures.iloc[i]
        nxt = futures.iloc[i + 1]

        # The date at which their expirations straddle maturity
        switch_date = (pre["expiration"].date())
        next_exp = nxt["expiration"].date()

        # straddle condition: pre expiration < (d + maturity) <= next expiration
        # Solve for d:
        # pre.exp < d + maturity <= next.exp  =>  d ∈ (pre.exp - maturity, next.exp - maturity]
        d0_candidate = pre["expiration"].date() - maturity_offset
        d1_candidate = nxt["expiration"].date() - maturity_offset

        # Convert to simple dates
        d0 = d0_candidate
        d1 = d1_candidate

        # We only keep overlap with user range
        seg_start = max(current_date, d0)
        seg_end = min(end, d1)

        if seg_start < seg_end:
            result.append(
                {
                    "d0": seg_start.isoformat(),
                    "d1": seg_end.isoformat(),
                    "p": str(int(pre["instrument_id"])),
                    "n": str(int(nxt["instrument_id"])),
                }
            )
            current_date = seg_end

        # Stop when we reach the end of requested range
        if current_date >= end:
            break

    return result


def constant_maturity_splice(
    symbol: str,
    roll_spec,
    all_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
):
    """
    Build a spliced constant-maturity price series.

    Parameters
    ----------
    symbol : str
        e.g., "SR3.cm.182"
    roll_spec : list of dicts
        Output of get_roll_spec
    all_data : DataFrame
        Must contain columns: instrument_id, expiration, date_col, price_col
    """

    # Ensure datetime sorted
    all_data = all_data.copy()
    all_data[date_col] = pd.to_datetime(all_data[date_col])
    all_data["expiration"] = pd.to_datetime(all_data["expiration"])
    all_data = all_data.sort_values([date_col, "instrument_id"])

    output_segments = []

    for r in roll_spec:
        d0 = pd.to_datetime(r["d0"])
        d1 = pd.to_datetime(r["d1"])
        pre = int(r["p"])
        nxt = int(r["n"])

        # slice time range [d0, d1)
        time_idx = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        # Extract pre- and next-contract data
        df_pre = all_data[all_data["instrument_id"] == pre]
        df_next = all_data[all_data["instrument_id"] == nxt]

        df_pre = df_pre.set_index(date_col).reindex(time_idx).reset_index()
        df_next = df_next.set_index(date_col).reindex(time_idx).reset_index()

        df_pre = df_pre.rename(columns={"index": "datetime"})
        df_next = df_next.rename(columns={"index": "datetime"})

        # Compute weight = (next_exp - (t + maturity)) / (next_exp - pre_exp)
        pre_exp = df_pre["expiration"].iloc[0]
        next_exp = df_next["expiration"].iloc[0]

        maturity_days = _parse_maturity_days(symbol)
        maturity_offset = pd.Timedelta(days=maturity_days)

        f = (next_exp - (df_pre["datetime"] + maturity_offset)) / (next_exp - pre_exp)

        seg = pd.DataFrame(
            {
                "datetime": time_idx,
                "pre_price": df_pre[price_col],
                "pre_id": pre,
                "pre_expiration": pre_exp,
                "next_price": df_next[price_col],
                "next_id": nxt,
                "next_expiration": next_exp,
                "pre_weight": f,
                symbol: f * df_pre[price_col] + (1 - f) * df_next[price_col],
            }
        )
        output_segments.append(seg)

    return pd.concat(output_segments, ignore_index=True)
