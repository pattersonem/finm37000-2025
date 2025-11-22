from __future__ import annotations

import datetime as dt
from typing import Iterable, Mapping, Any, Dict, List

import pandas as pd


# =====================================================================
# Helpers
# =====================================================================

def _extract_days_to_maturity(fut_symbol: str) -> int:
    """
    Given a symbol like 'SR3.cm.182', return the maturity in days (182).
    """
    try:
        return int(fut_symbol.rsplit(".", 1)[-1])
    except Exception as err:
        raise ValueError(f"Unable to interpret maturity days from {fut_symbol}") from err


# =====================================================================
# Roll Schedule Builder
# =====================================================================

def build_roll_schedule(
    fut_symbol: str,
    instruments: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date
) -> List[Dict[str, str]]:
    """
    Compute roll windows for constructing a constant-maturity future series.

    Returns list of:
        {
            "start": str,
            "end": str,
            "lead": contract_id,
            "lag": contract_id
        }
    """
    maturity_days = _extract_days_to_maturity(fut_symbol)
    maturity_offset = pd.Timedelta(days=maturity_days)

    tbl = instruments.copy()

    # Use only outright futures
    tbl = tbl[tbl["instrument_class"] == "F"].copy()
    tbl["exp_ts"] = pd.to_datetime(tbl["expiration"])
    tbl["recv_date"] = pd.to_datetime(tbl["ts_recv"]).dt.date
    tbl["exp_date"] = tbl["exp_ts"].dt.date

    # Generate calendar days for evaluation
    last_eval_day = end_date - dt.timedelta(days=1)
    calendar = pd.date_range(start_date, last_eval_day, freq="D")

    tbl = tbl.sort_values("exp_ts").reset_index(drop=True)

    roll_output: List[Dict[str, str]] = []
    active_lead = None
    active_lag = None
    segment_start = None

    for timestamp in calendar:
        today = timestamp.date()

        # Only contracts visible by today
        available = tbl[tbl["recv_date"] <= today]
        if available.empty:
            continue

        target_exp = (timestamp + maturity_offset).date()

        # Lead: last expiry before target
        # Lag : first expiry on/after target
        lead_mask = available["exp_date"] < target_exp
        lag_mask = available["exp_date"] >= target_exp

        lead_side = available[lead_mask]
        lag_side = available[lag_mask]

        if lead_side.empty or lag_side.empty:
            continue

        new_lead = str(lead_side.sort_values("exp_ts").iloc[-1]["instrument_id"])
        new_lag = str(lag_side.sort_values("exp_ts").iloc[0]["instrument_id"])

        if active_lead is None:
            active_lead, active_lag = new_lead, new_lag
            segment_start = timestamp
            continue

        if (new_lead, new_lag) != (active_lead, active_lag):
            # Close previous segment
            roll_output.append(
                {
                    "start": segment_start.strftime("%Y-%m-%d"),
                    "end": timestamp.strftime("%Y-%m-%d"),
                    "lead": active_lead,
                    "lag": active_lag,
                }
            )
            active_lead, active_lag = new_lead, new_lag
            segment_start = timestamp

    # Close last block
    if segment_start is not None and active_lead is not None:
        roll_output.append(
            {
                "start": segment_start.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "lead": active_lead,
                "lag": active_lag,
            }
        )

    return roll_output


# =====================================================================
# Constant Maturity Construction
# =====================================================================

def splice_constant_maturity(
    fut_symbol: str,
    roll_windows: Iterable[Mapping[str, Any]],
    market_data: pd.DataFrame,
    ts_col: str = "datetime",
    px_col: str = "price",
) -> pd.DataFrame:
    """
    Using roll windows and raw futures data, splice a constant-maturity series.

    Columns in output:
        datetime,
        lead_price, lead_id, lead_expiry,
        lag_price,  lag_id,  lag_expiry,
        lead_weight,
        <symbol>
    """
    maturity_days = _extract_days_to_maturity(fut_symbol)
    maturity_offset = pd.Timedelta(days=maturity_days)

    data = market_data.copy()
    data[ts_col] = pd.to_datetime(data[ts_col])
    data["expiry"] = pd.to_datetime(data["expiration"])

    # Map: id → expiry
    expiry_map = (
        data[["instrument_id", "expiry"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiry"]
    )

    # Map: id → price series
    px_lookup: Dict[int, pd.Series] = {
        int(cid): grp.set_index(ts_col)[px_col].sort_index()
        for cid, grp in data.groupby("instrument_id")
    }

    stitched_segments: List[pd.DataFrame] = []

    for entry in roll_windows:
        seg_start = entry["start"]
        seg_end = entry["end"]
        lead_id = int(entry["lead"])
        lag_id = int(entry["lag"])

        segment_index = pd.date_range(seg_start, seg_end, tz="UTC", inclusive="left")

        lead_px = px_lookup[lead_id].reindex(segment_index)
        lag_px = px_lookup[lag_id].reindex(segment_index)

        lead_exp = expiry_map[lead_id]
        lag_exp = expiry_map[lag_id]

        # Compute weight on the nearer contract
        w_lead = (lag_exp - (segment_index + maturity_offset)) / (lag_exp - lead_exp)
        w_lead = pd.Series(w_lead, index=segment_index)

        blended = w_lead * lead_px + (1 - w_lead) * lag_px

        seg = pd.DataFrame(
            {
                "datetime": segment_index,
                "lead_price": lead_px.values,
                "lead_id": lead_id,
                "lead_expiry": lead_exp,
                "lag_price": lag_px.values,
                "lag_id": lag_id,
                "lag_expiry": lag_exp,
                "lead_weight": w_lead.values,
                fut_symbol: blended.values,
            }
        )
        stitched_segments.append(seg)

    return pd.concat(stitched_segments, ignore_index=True)
