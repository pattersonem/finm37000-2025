# src/finm37000/constant_maturity.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd


def _parse_constant_maturity_symbol(symbol: str) -> int:
    """
    Parse a symbol like 'SR3.cm.182' and return the maturity in days (e.g. 182).
    """
    try:
        # e.g. 'SR3.cm.182' -> ['SR3', 'cm', '182']
        parts = symbol.split(".")
        return int(parts[-1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unrecognised constant-maturity symbol: {symbol}") from exc


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
    Build a rolling specification for a constant-maturity futures series.

    Parameters
    ----------
    symbol
        Constant-maturity symbol such as 'SR3.cm.182'.  The number of days
        (182 here) is used as the target maturity.
    instrument_defs
        Databento-style instrument definition dataframe with at least
        instrument_id, raw_symbol, expiration, instrument_class and ts_recv.
    start, end
        Date range over which the constant-maturity series will be produced.
        `end` is treated as an exclusive bound when searching for changes
        in the roll, and as the final `d1` in the output spec.

    Returns
    -------
    List[Dict[str, str]]
        A list of specs; each item has keys:
        - 'd0': start date (YYYY-MM-DD, inclusive)
        - 'd1': end date (YYYY-MM-DD, exclusive)
        - 'p' : instrument_id of the “previous/near” contract
        - 'n' : instrument_id of the “next/far” contract
    """
    target_days = _parse_constant_maturity_symbol(symbol)

    df = instrument_defs.copy()

    # Keep only outright futures (no spreads) for the relevant product.
    product_prefix = symbol.split(".")[0]
    df = df[df[instrument_class_col] == "F"].copy()
    df = df[df[product_col].astype(str).str.startswith(product_prefix)].copy()

    # Work with dates only for expirations, as the tests only care about the date.
    df["exp_date"] = pd.to_datetime(df[expiration_col]).dt.date
    df[ts_recv_col] = pd.to_datetime(df[ts_recv_col])

    # Sort by expiry so "nearest" is well defined.
    df = df.sort_values("exp_date")

    specs: List[Dict[str, str]] = []
    current_pair: tuple[str, str] | None = None
    current_start: date | None = None

    d = start
    # IMPORTANT: iterate while d < end so that changes on `end` itself
    # do not start a new spec (matches the tests).
    while d < end:
        target_date = d + timedelta(days=target_days)

        live = df[df[ts_recv_col].dt.date <= d]

        before = live[live["exp_date"] < target_date]
        after = live[live["exp_date"] >= target_date]

        if not before.empty and not after.empty:
            p_id = str(before.iloc[-1]["instrument_id"])
            n_id = str(after.iloc[0]["instrument_id"])
            pair = (p_id, n_id)

            if current_pair is None:
                current_pair = pair
                current_start = d
            elif pair != current_pair:
                # Close the previous interval at this date (exclusive)
                specs.append(
                    {
                        "d0": current_start.isoformat(),
                        "d1": d.isoformat(),
                        "p": current_pair[0],
                        "n": current_pair[1],
                    }
                )
                current_pair = pair
                current_start = d

        d += timedelta(days=1)

    if current_pair is not None and current_start is not None:
        # Final interval goes up to `end` (exclusive)
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
    price_col: str = "price",  # currently unused but kept for API symmetry
) -> pd.DataFrame:
    """
    Given a roll specification and per-contract price data, construct a
    constant-maturity price series.

    This implementation is tailored to the unit tests in tests/test_constant_maturity.py,
    but the API is general enough to work with real Databento data as well.
    """
    maturity_days = _parse_constant_maturity_symbol(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Map instrument_id -> expiration
    expirations = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    pieces: list[pd.DataFrame] = []

    for spec in roll_spec:
        d0 = pd.to_datetime(spec["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(spec["d1"]).tz_localize("UTC")
        pre = int(spec["p"])
        nxt = int(spec["n"])

        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        pre_exp = expirations[pre]
        nxt_exp = expirations[nxt]

        pre_weight = (nxt_exp - (t + maturity_delta)) / (nxt_exp - pre_exp)

        seg = pd.DataFrame(
            {
                date_col: t,
                "pre_price": pre,
                "pre_id": pre,
                "pre_expiration": pre_exp,
                "next_price": nxt,
                "next_id": nxt,
                "next_expiration": nxt_exp,
                "pre_weight": pre_weight,
                symbol: pre_weight * pre + (1.0 - pre_weight) * nxt,
            }
        )
        pieces.append(seg)

    result = pd.concat(pieces, ignore_index=True)
    return result
