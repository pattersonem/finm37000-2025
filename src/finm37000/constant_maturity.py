from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd


def _parse_constant_maturity_symbol(symbol: str) -> int:
    """
    Extract the target maturity (in days) from a constant-maturity symbol.
    """
    parts = symbol.split(".")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse constant-maturity symbol: {symbol}")
    try:
        return int(parts[-1])
    except ValueError as exc:  # pragma: no cover
        raise ValueError(
            f"Last component of {symbol!r} is not an integer number of days."
        ) from exc


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
    Construct the roll schedule for a constant-maturity futures series.

    Parameters
    ----------
    symbol
        Constant-maturity label such as 'SR3.cm.182'. The trailing integer
        encodes the target maturity in days.
    instrument_defs
        Instrument definition table (Databento-style) containing at least:
        - instrument_id
        - raw_symbol
        - expiration
        - instrument_class
        - ts_recv
    start, end
        Date range over which the constant-maturity series is required.
        The 'end' date is treated as an exclusive bound for the intervals.

    Returns
    -------
    List[Dict[str, str]]
        Each element describes one interval with a fixed (near, far) pair:
        - 'd0': start date, inclusive  (YYYY-MM-DD)
        - 'd1': end date,   exclusive  (YYYY-MM-DD)
        - 'p' : instrument_id for the near/previous contract
        - 'n' : instrument_id for the far/next contract
    """
    target_days = _parse_constant_maturity_symbol(symbol)

    # Work on a copy so we do not mutate the caller's DataFrame.
    df = instrument_defs.copy()

    # Restrict to outright futures for the relevant product prefix.
    product_prefix = symbol.split(".")[0]
    mask = (
        df[instrument_class_col].astype(str).eq("F")
        & df[product_col].astype(str).str.startswith(product_prefix)
    )
    df = df.loc[mask].copy()

    # Normalise date/time fields.
    df[expiration_col] = pd.to_datetime(df[expiration_col])
    df["exp_date"] = df[expiration_col].dt.date

    df[ts_recv_col] = pd.to_datetime(df[ts_recv_col])
    df["ts_recv_date"] = df[ts_recv_col].dt.date

    # Ensure contracts are ordered by expiry so "nearest" is well defined.
    df = df.sort_values("exp_date")

    specs: List[Dict[str, str]] = []

    current_pair: tuple[str, str] | None = None
    current_start: date | None = None

    # Iterate over each calendar date in [start, end).
    day = start
    while day < end:
        # Target maturity date measured from 'day'.
        target_exp_date = day + timedelta(days=target_days)

        # Only use contracts whose definition is known by 'day'.
        live = df.loc[df["ts_recv_date"] <= day]

        before = live.loc[live["exp_date"] < target_exp_date]
        after = live.loc[live["exp_date"] >= target_exp_date]

        if not before.empty and not after.empty:
            # Nearest expiry strictly before target date.
            near_row = before.iloc[-1]
            # Nearest expiry on or after target date.
            far_row = after.iloc[0]

            near_id = str(near_row["instrument_id"])
            far_id = str(far_row["instrument_id"])
            pair = (near_id, far_id)

            if current_pair is None:
                # First valid pair we see.
                current_pair = pair
                current_start = day
            elif pair != current_pair:
                # The (near, far) pair changed; close out previous interval.
                specs.append(
                    {
                        "d0": current_start.isoformat(),
                        "d1": day.isoformat(),
                        "p": current_pair[0],
                        "n": current_pair[1],
                    }
                )
                current_pair = pair
                current_start = day

        # Move to next calendar day.
        day += timedelta(days=1)

    # Close the final interval, if any.
    if current_pair is not None and current_start is not None:
        specs.append(
            {
                "d0": current_start.isoformat(),
                "d1": end.isoformat(),
                "p": current_pair[0],
                "n": current_pair[1],
            }
        )

    return specs


def constant_maturity_splice(
    symbol: str,
    roll_spec: List[Dict[str, str]],
    data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Use a roll specification and per-contract prices to build a
    constant-maturity time series.

    The output schema is chosen to match tests/test_constant_maturity.py:
    columns =
        ["datetime",
         "pre_price", "pre_id", "pre_expiration",
         "next_price", "next_id", "next_expiration",
         "pre_weight",
         symbol]
    """

    maturity_days = _parse_constant_maturity_symbol(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # instrument_id -> expiration (tz-aware)
    exp_map = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    segments: list[pd.DataFrame] = []

    for spec in roll_spec:
        d0 = spec["d0"]
        d1 = spec["d1"]
        pre_id = int(spec["p"])
        nxt_id = int(spec["n"])

        # Daily calendar for this interval, tz-aware, [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        if t.empty:
            continue

        pre_exp = exp_map.loc[pre_id]
        nxt_exp = exp_map.loc[nxt_id]

        # Same weight formula the test uses:
        # f = (nxt_exp - (t + maturity_days)) / (nxt_exp - pre_exp)
        pre_weight = (nxt_exp - (t + maturity_delta)) / (nxt_exp - pre_exp)
        pre_weight = pre_weight.astype(float)

        # Get prices for each leg, aligned to t.
        pre_px = (
            df.loc[df["instrument_id"] == pre_id, [date_col, price_col]]
            .drop_duplicates(subset=date_col)
            .set_index(date_col)[price_col]
            .reindex(t)
        )
        nxt_px = (
            df.loc[df["instrument_id"] == nxt_id, [date_col, price_col]]
            .drop_duplicates(subset=date_col)
            .set_index(date_col)[price_col]
            .reindex(t)
        )

        # Constant-maturity price
        cm_price = pre_weight * pre_px + (1.0 - pre_weight) * nxt_px

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_px.to_numpy(),
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": nxt_px.to_numpy(),
                "next_id": nxt_id,
                "next_expiration": nxt_exp,
                "pre_weight": pre_weight.to_numpy(),
                symbol: cm_price.to_numpy(),
            }
        )

        segments.append(seg)

    if not segments:
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

    result = pd.concat(segments, ignore_index=True)

    # Ensure same column order as the test's expected DataFrame
    result = result[
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

    return result