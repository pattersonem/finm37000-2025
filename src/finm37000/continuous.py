"""Functions to splice and adjust futures data into continuous data."""

from typing import Optional

import pandas as pd
import re
import datetime

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
            adjustment = last_true_value - adjustment_piece[adjust_by].iloc[-1]
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
            adjustment = last_true_value / adjustment_piece[adjust_by].iloc[-1]
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
    """Generate roll specification for constant maturity futures.

    Args:
        symbol: Constant maturity symbol like "SR3.cm.182" where 182 is days to maturity
        instrument_df: DataFrame with columns:
            - instrument_id: unique identifier
            - raw_symbol: symbol name
            - expiration: expiration datetime
            - instrument_class: 'F' for futures, 'S' for spreads
            - ts_recv: timestamp when instrument became available
        start: Start date for the roll specification
        end: End date for the roll specification

    Returns:
        List of roll specs, each with:
            - d0: first date (inclusive) as string "YYYY-MM-DD"
            - d1: one past last date (exclusive) as string "YYYY-MM-DD"
            - p: previous (nearer) contract instrument_id as string
            - n: next (farther) contract instrument_id as string
    """
    # Parse maturity days from symbol
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Cannot parse maturity from symbol '{symbol}'")
    maturity_days = int(m.group(1))

    # Filter to futures only and ensure proper datetime types
    futures = instrument_df[instrument_df["instrument_class"] == "F"].copy()
    futures["expiration"] = pd.to_datetime(futures["expiration"])
    futures["ts_recv"] = pd.to_datetime(futures["ts_recv"])

    # Ensure expiration has timezone (UTC if not specified)
    if futures["expiration"].dt.tz is None:
        futures["expiration"] = futures["expiration"].dt.tz_localize("UTC")
    if futures["ts_recv"].dt.tz is None:
        futures["ts_recv"] = futures["ts_recv"].dt.tz_localize("UTC")

    # Sort by expiration
    futures = futures.sort_values("expiration").reset_index(drop=True)

    # Generate date range (end is exclusive in the output)
    start_dt = pd.Timestamp(start).tz_localize("UTC")
    end_dt = pd.Timestamp(end).tz_localize("UTC")
    # We iterate through dates [start, end), but d1 in output should be end (exclusive)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D", inclusive="left")

    roll_specs = []
    current_pair = None
    segment_start = None

    for date in date_range:
        # Target expiration is date + maturity_days
        target_exp = date + pd.Timedelta(days=maturity_days)

        # Find contracts available at this date
        available = futures[futures["ts_recv"] <= date]

        if len(available) < 2:
            continue

        # Find the pair that straddles target_exp
        # Previous contract: latest expiration <= target_exp
        # Next contract: earliest expiration > target_exp
        prev_contracts = available[available["expiration"] <= target_exp]
        next_contracts = available[available["expiration"] > target_exp]

        if len(prev_contracts) == 0 or len(next_contracts) == 0:
            continue

        prev = prev_contracts.iloc[-1]
        nxt = next_contracts.iloc[0]

        pair = (str(prev["instrument_id"]), str(nxt["instrument_id"]))

        if pair != current_pair:
            # Close previous segment
            if current_pair is not None and segment_start is not None:
                roll_specs.append({
                    "d0": segment_start.strftime("%Y-%m-%d"),
                    "d1": date.strftime("%Y-%m-%d"),
                    "p": current_pair[0],
                    "n": current_pair[1],
                })

            # Start new segment
            current_pair = pair
            segment_start = date

    # Close final segment
    if current_pair is not None and segment_start is not None:
        # The last segment goes up to (but not including) end_dt
        roll_specs.append({
            "d0": segment_start.strftime("%Y-%m-%d"),
            "d1": end_dt.strftime("%Y-%m-%d"),
            "p": current_pair[0],
            "n": current_pair[1],
        })

    return roll_specs


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    instrument_col: str = "instrument_id",
    expiration_col: str = "expiration",
) -> pd.DataFrame:
    """Construct a constant-maturity synthetic futures series.

    Args:
        symbol: Name of the resulting constant-maturity series. Expected to
                encode the maturity in calendar days as in 'SR3.cm.182'.
        roll_spec: List of dicts, each with "d0", "d1", "p", "n":
                   - d0: first date (inclusive) for this segment
                   - d1: one past the last date (exclusive) for this segment
                   - p: 'previous' (nearer) contract instrument_id
                   - n: 'next' (farther) contract instrument_id
        df: Raw futures data with at least date_col, instrument_col,
            price_col, expiration_col.
        date_col: Name of the datetime column in df.
        price_col: Name of the price column in df.
        instrument_col: Name of the instrument id column (default: 'instrument_id').
        expiration_col: Name of the expiration datetime column (default: 'expiration').

    Returns:
        DataFrame with columns:
            - date_col (e.g. 'datetime')
            - 'pre_price', 'pre_id', 'pre_expiration'
            - 'next_price', 'next_id', 'next_expiration'
            - 'pre_weight'
            - symbol (the synthetic constant-maturity price)

        Rows are concatenated in the order implied by roll_spec.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[expiration_col] = pd.to_datetime(df[expiration_col])
    tz = df[date_col].dt.tz

    # Parse constant maturity days from symbol: e.g. "SR3.cm.182" → 182 days
    m = re.search(r"\.cm\.(\d+)$", symbol)
    if not m:
        raise ValueError(f"Cannot parse maturity from symbol '{symbol}'")
    maturity_days = pd.Timedelta(days=int(m.group(1)))

    # Map instrument_id → expiration
    expirations = (
        df[[instrument_col, expiration_col]]
        .drop_duplicates(subset=[instrument_col])
        .set_index(instrument_col)[expiration_col]
    )

    segments: list[pd.DataFrame] = []

    # For each roll segment (d0, d1, p, n), calculate constant-maturity weights
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        pre = int(spec["p"])
        nxt = int(spec["n"])

        # Date range for this segment [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz=tz, inclusive="left")
        if len(t) == 0:
            continue

        T_p = expirations[pre]
        T_n = expirations[nxt]

        # Previous contract weight: f = (T_n - (t + m)) / (T_n - T_p)
        pre_weight = (T_n - (t + maturity_days)) / (T_n - T_p)
        pre_weight = pre_weight.astype("float64")

        # Get actual prices for pre/next contracts from df at each date t
        df_pre = (
            df[df[instrument_col] == pre]
            .sort_values(date_col)
            .set_index(date_col)[price_col]
        )
        df_nxt = (
            df[df[instrument_col] == nxt]
            .sort_values(date_col)
            .set_index(date_col)[price_col]
        )
        pre_price = df_pre.reindex(t).to_numpy()
        next_price = df_nxt.reindex(t).to_numpy()

        # Build segment DataFrame (column order matches test expectations)
        seg = pd.DataFrame(
            {
                date_col: t,
                "pre_price": pre_price,
                "pre_id": pre,
                "pre_expiration": T_p,
                "next_price": next_price,
                "next_id": nxt,
                "next_expiration": T_n,
                "pre_weight": pre_weight.to_numpy(),
            }
        )

        # Constant-maturity synthetic price
        seg[symbol] = (
            seg["pre_weight"] * seg["pre_price"]
            + (1.0 - seg["pre_weight"]) * seg["next_price"]
        )

        segments.append(seg)

    if not segments:
        # Return empty DataFrame with proper schema
        return pd.DataFrame(
            columns=[
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
        )

    result = pd.concat(segments, ignore_index=True)
    return result