from __future__ import annotations

import datetime as dt
from typing import Iterable, Mapping, Any, List, Dict

import pandas as pd


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _parse_maturity_days(symbol: str) -> int:
    """
    Extract maturity days from a symbol like "SR3.cm.182" → 182.
    """
    try:
        return int(symbol.split(".")[-1])
    except Exception as exc:
        raise ValueError(f"Cannot parse maturity days from symbol {symbol!r}") from exc


# ---------------------------------------------------------------------
# Roll Specification
# ---------------------------------------------------------------------
def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
) -> List[Dict[str, str]]:
    """
    Construct the roll specification for a constant-maturity future.

    Returns a list of dicts:
      - 'd0': start date (inclusive)
      - 'd1': end date   (exclusive)
      - 'front_id':  id of front (previous) contract
      - 'back_id':   id of back (next) contract
    """
    maturity_days = _parse_maturity_days(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = instrument_df.copy()

    # Use only outright futures
    df = df[df["instrument_class"] == "F"].copy()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    # Use pure dates for comparisons
    df["expiration_date"] = df["expiration"].dt.date
    df["ts_recv_date"] = df["ts_recv"].dt.date

    # Daily iteration over [start, end)
    last_date = end - dt.timedelta(days=1)
    all_dates = pd.date_range(start=start, end=last_date, freq="D")

    df = df.sort_values("expiration")

    roll_list: List[Dict[str, str]] = []
    current_front: str | None = None
    current_back: str | None = None
    block_start: pd.Timestamp | None = None

    for ts in all_dates:
        current_date = ts.date()

        # Contracts available by this date
        available = df[df["ts_recv_date"] <= current_date]
        if available.empty:
            continue

        # Target expiry date (constant maturity target)
        target_date = (ts + maturity_delta).date()

        # Front = last contract expiring BEFORE target
        # Back  = earliest contract expiring ON or AFTER target
        mask_front = available["expiration_date"] < target_date
        mask_back = available["expiration_date"] >= target_date

        front = available[mask_front]
        back = available[mask_back]

        if front.empty or back.empty:
            continue

        front_row = front.sort_values("expiration").iloc[-1]
        back_row = back.sort_values("expiration").iloc[0]

        new_front = str(front_row["instrument_id"])
        new_back = str(back_row["instrument_id"])
        new_pair = (new_front, new_back)

        if current_front is None:
            # First identified roll pair
            current_front, current_back = new_pair
            block_start = ts
        elif new_pair != (current_front, current_back):
            # New pair detected → close previous block
            assert block_start is not None
            roll_list.append(
                {
                    "d0": block_start.strftime("%Y-%m-%d"),
                    "d1": ts.strftime("%Y-%m-%d"),
                    "front_id": current_front,
                    "back_id": current_back,
                }
            )
            current_front, current_back = new_pair
            block_start = ts

    # Close out final block to `end`
    if current_front is not None and block_start is not None:
        roll_list.append(
            {
                "d0": block_start.strftime("%Y-%m-%d"),
                "d1": end.strftime("%Y-%m-%d"),
                "front_id": current_front,
                "back_id": current_back,
            }
        )

    return roll_list


# ---------------------------------------------------------------------
# Splicing into Constant Maturity Series
# ---------------------------------------------------------------------
def constant_maturity_splice(
    symbol: str,
    roll_spec: Iterable[Mapping[str, Any]],
    all_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Build a constant-maturity price series using roll_spec and futures data.

    Output columns:
      - datetime
      - front_price, front_id, front_expiration
      - back_price,  back_id,  back_expiration
      - front_weight
      - <symbol> (final weighted price)
    """
    maturity_days = _parse_maturity_days(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = all_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Map instrument_id → expiration
    expirations = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    # Map instrument_id → price time series
    prices_by_id: Dict[int, pd.Series] = {
        int(inst_id): grp.set_index(date_col)[price_col].sort_index()
        for inst_id, grp in df.groupby("instrument_id")
    }

    segments: List[pd.DataFrame] = []

    for row in roll_spec:
        d0 = row["d0"]
        d1 = row["d1"]
        front_id = int(row["front_id"])
        back_id = int(row["back_id"])

        ts_range = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        front_prices = prices_by_id[front_id].reindex(ts_range)
        back_prices = prices_by_id[back_id].reindex(ts_range)

        front_exp = expirations[front_id]
        back_exp = expirations[back_id]

        # Weight on front contract
        weight_front = (back_exp - (ts_range + maturity_delta)) / (back_exp - front_exp)
        weight_front = pd.Series(weight_front, index=ts_range)

        weighted_price = weight_front * front_prices + (1.0 - weight_front) * back_prices

        segment = pd.DataFrame(
            {
                "datetime": ts_range,
                "front_price": front_prices.values,
                "front_id": front_id,
                "front_expiration": front_exp,
                "back_price": back_prices.values,
                "back_id": back_id,
                "back_expiration": back_exp,
                "front_weight": weight_front.values,
                symbol: weighted_price.values,
            }
        )
        segments.append(segment)

    return pd.concat(segments, ignore_index=True)
