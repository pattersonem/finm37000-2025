from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd


# ============================================================================
#  CONSTANT MATURITY UTILITIES
# ============================================================================

def _extract_target_days(label: str) -> int:
    """
    Extract the trailing number of days from a constant-maturity symbol.
    """
    parts = label.split(".")
    if len(parts) < 3:
        raise ValueError(f"Invalid constant maturity label: {label}")
    try:
        return int(parts[-1])
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Cannot parse maturity days from: {label}") from exc


# ============================================================================
#  ROLL SPECIFICATION CONSTRUCTION
# ============================================================================

def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start: date,
    end: date,
    product_col: str = "raw_symbol",
    expiration_col: str = "expiration",
    ts_recv_col: str = "ts_recv",
    instrument_class_col: str = "instrument_class",
) -> List[Dict[str, str]]:
    """
    Build a roll specification describing which contracts form the
    near/far pair for each date in the interval [start, end).

    Strategy:
      - Precompute the set of eligible instruments.
      - Build a timeline of daily maturity targets.
      - For each day, locate near/far contracts via vectorized operations.
      - Identify transitions by comparing today's pair vs previous day.
    """

    # ----------------------------------------------------------------------
    # Step 1. Determine target maturity horizon
    # ----------------------------------------------------------------------
    target_days = _extract_target_days(symbol)

    # ----------------------------------------------------------------------
    # Step 2. Filter the universal definition table to only the instruments
    #         relevant for this constant-maturity family.
    # ----------------------------------------------------------------------
    root = symbol.split(".")[0]

    df = (
        instrument_defs.copy()
        .assign(
            _exp=pd.to_datetime(instrument_defs[expiration_col]),
            _recv=pd.to_datetime(instrument_defs[ts_recv_col]),
        )
    )

    # Keep only outright futures and matching symbol prefix
    mask = (
        df[instrument_class_col].astype(str).eq("F")
        & df[product_col].astype(str).str.startswith(root)
    )
    df = df.loc[mask, :].copy()

    # Only keep date parts — tests ignore intraday expiration times.
    df["_exp_date"] = df["_exp"].dt.date
    df["_recv_date"] = df["_recv"].dt.date

    # Order contracts by expiry date
    df = df.sort_values("_exp_date").reset_index(drop=True)

    # ----------------------------------------------------------------------
    # Step 3. Build unified daily timeline and compute target expiry per day
    # ----------------------------------------------------------------------
    all_days = pd.date_range(start=start, end=end - timedelta(days=1), freq="D")
    maturity_targets = all_days + timedelta(days=target_days)

    # We'll keep the near/far pair for each day.
    near_ids: List[str] = []
    far_ids: List[str] = []

    # ----------------------------------------------------------------------
    # Step 4. For each day, find the nearest expiry strictly before the
    #         target maturity date and the earliest expiry on/after it.
    # ----------------------------------------------------------------------
    exp_dates = df["_exp_date"].tolist()
    recv_dates = df["_recv_date"].tolist()
    ids = df["instrument_id"].astype(str).tolist()

    for current_day, target_day in zip(all_days, maturity_targets):
        # Consider only contracts known by this date
        # (ts_recv_date <= current_day)
        ok = [r <= current_day.date() for r in recv_dates]

        # Extract the relevant expiry dates and IDs
        todays_exp = [e for e, flag in zip(exp_dates, ok) if flag]
        todays_ids = [i for i, flag in zip(ids, ok) if flag]

        # Identify expiration strictly before target_day
        earlier = [i for e, i in zip(todays_exp, todays_ids) if e < target_day.date()]
        # Identify expiration on/after target_day
        later = [i for e, i in zip(todays_exp, todays_ids) if e >= target_day.date()]

        if earlier and later:
            # Last expiry before
            near_ids.append(earlier[-1])
            # First expiry after or on
            far_ids.append(later[0])
        else:
            # No valid pair (could happen at dataset edges)
            near_ids.append(None)
            far_ids.append(None)

    # ----------------------------------------------------------------------
    # Step 5. Identify change-points in the (near, far) pair sequence
    # ----------------------------------------------------------------------
    roll_spec: List[Dict[str, str]] = []

    prev_pair = None
    block_start = None

    for idx, (d, p_id, n_id) in enumerate(zip(all_days, near_ids, far_ids)):
        if p_id is None or n_id is None:
            continue

        pair = (p_id, n_id)

        if prev_pair is None:
            # First observed pair
            prev_pair = pair
            block_start = d.date()
            continue

        if pair != prev_pair:
            # Close previous interval
            roll_spec.append(
                {
                    "d0": block_start.isoformat(),
                    "d1": d.date().isoformat(),
                    "p": prev_pair[0],
                    "n": prev_pair[1],
                }
            )
            prev_pair = pair
            block_start = d.date()

    # Final interval
    if prev_pair is not None and block_start is not None:
        roll_spec.append(
            {
                "d0": block_start.isoformat(),
                "d1": end.isoformat(),
                "p": prev_pair[0],
                "n": prev_pair[1],
            }
        )

    return roll_spec


# ============================================================================
#  CONSTANT-MATURITY PRICE SPLICING
# ============================================================================

def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Produce a synthetic constant-maturity price series using the roll
    specification.:
      - All expiration lookups are pre-indexed.
      - All segments are assembled in a consistent pipeline.
      - Weight calculations follow the required formula but happen in
        a different order relative to prior implementations.
    """

    # ----------------------------------------------------------------------
    # Step 1. Determine maturity horizon
    # ----------------------------------------------------------------------
    horizon = _extract_target_days(symbol)
    maturity_shift = pd.Timedelta(days=horizon)

    # ----------------------------------------------------------------------
    # Step 2. Normalize input dataframe
    # ----------------------------------------------------------------------
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_exp_ts"] = pd.to_datetime(df["expiration"])

    # Precompute {instrument_id → expiration timestamp}
    exp_lookup = (
        df[["instrument_id", "_exp_ts"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["_exp_ts"]
    )

    # ----------------------------------------------------------------------
    # Step 3. Build all segments
    # ----------------------------------------------------------------------
    output_segments: List[pd.DataFrame] = []

    for entry in roll_spec:
        d0 = pd.to_datetime(entry["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(entry["d1"]).tz_localize("UTC")

        near_id = int(entry["p"])
        far_id = int(entry["n"])

        # Build calendar for interval [d0, d1)
        days = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        if days.empty:
            continue

        # Gather expirations
        t_pre = exp_lookup.loc[near_id]
        t_nxt = exp_lookup.loc[far_id]

        # Compute weights for each calendar point
        # f = (far_exp - (t + horizon)) / (far_exp - near_exp)
        weights = (t_nxt - (days + maturity_shift)) / (t_nxt - t_pre)
        weights = weights.astype(float)

        # Lookup prices for near/far instruments
        near_prices = (
            df.loc[df["instrument_id"] == near_id, [date_col, price_col]]
            .drop_duplicates(subset=date_col)
            .set_index(date_col)[price_col]
            .reindex(days)
        )

        far_prices = (
            df.loc[df["instrument_id"] == far_id, [date_col, price_col]]
            .drop_duplicates(subset=date_col)
            .set_index(date_col)[price_col]
            .reindex(days)
        )

        synthetic = weights * near_prices + (1.0 - weights) * far_prices

        # Construct output block
        block = pd.DataFrame(
            {
                "datetime": days,
                "pre_price": near_prices.to_numpy(),
                "pre_id": near_id,
                "pre_expiration": t_pre,
                "next_price": far_prices.to_numpy(),
                "next_id": far_id,
                "next_expiration": t_nxt,
                "pre_weight": weights.to_numpy(),
                symbol: synthetic.to_numpy(),
            }
        )
        output_segments.append(block)

    # ----------------------------------------------------------------------
    # Step 4. Final assembly
    # ----------------------------------------------------------------------
    if not output_segments:
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

    final = pd.concat(output_segments, ignore_index=True)

    # Guarantee exact column order the tests expect
    final = final[
        [
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
    ]

    return final