import pandas as pd
from typing import List, Dict
import datetime as dt

def constant_maturity_splice(symbol: str,
                             roll_spec,
                             raw_data: pd.DataFrame,
                             date_col: str = "datetime",
                             price_col: str = "price") -> pd.DataFrame:

    df = raw_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    grouped = df.groupby("instrument_id")

    pieces = []

    for seg in roll_spec:
        d0 = pd.Timestamp(seg["d0"]).tz_localize("UTC")
        d1 = pd.Timestamp(seg["d1"]).tz_localize("UTC")
        pre = int(seg["p"])
        nxt = int(seg["n"])

        mask = (df[date_col] >= d0) & (df[date_col] < d1)
        dt_range = df.loc[mask, date_col].drop_duplicates().sort_values()

        if dt_range.empty:
            continue

        pre_df = grouped.get_group(pre).set_index(date_col)
        nxt_df = grouped.get_group(nxt).set_index(date_col)

        pre_aligned = pre_df.reindex(dt_range)
        nxt_aligned = nxt_df.reindex(dt_range)

        pre_exp = pre_aligned["expiration"].iloc[0]
        nxt_exp = nxt_aligned["expiration"].iloc[0]

        maturity_days = int(symbol.split(".")[-1])
        maturity = pd.Timedelta(days=maturity_days)

        t_plus_m = dt_range + maturity
        pre_weight = (nxt_exp - t_plus_m) / (nxt_exp - pre_exp)

        cm_price = pre_weight * pre_aligned[price_col].values + \
                   (1 - pre_weight) * nxt_aligned[price_col].values

        segment = pd.DataFrame({
            "datetime": dt_range,
            "pre_price": pre_aligned[price_col].values,
            "pre_id": pre,
            "pre_expiration": pre_exp,
            "next_price": nxt_aligned[price_col].values,
            "next_id": nxt,
            "next_expiration": nxt_exp,
            "pre_weight": pre_weight.values,
            symbol: cm_price,
        })

        pieces.append(segment.reset_index(drop=True))

    return pd.concat(pieces, ignore_index=True)

def get_roll_spec(symbol: str, instrument_defs: pd.DataFrame, *, start: dt.date, end: dt.date):
    try:
        maturity_days = int(symbol.rpartition(".")[2])
    except ValueError as exc:
        raise ValueError(f"Cannot parse maturity from symbol {symbol!r}") from exc

    cal_maturity = pd.Timedelta(days=maturity_days)

    df = instrument_defs.loc[instrument_defs["instrument_class"].eq("F")].copy()

    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df["recv_date"] = df["ts_recv"].dt.date
    df["instrument_id"] = df["instrument_id"].astype(int)

    df = df.sort_values(["expiration", "instrument_id"]).reset_index(drop=True)

    def pick_pair_for_date(cur_date: dt.date):
        """
        For a given date, pick the nearest 'previous' and 'next' expiries
        relative to cur_date + cal_maturity, restricted to instruments
        that are live (recv_date <= cur_date).
        """
        live_mask = df["recv_date"].le(cur_date)
        if not live_mask.any():
            return None

        live = df.loc[live_mask]

        cutoff = pd.Timestamp(cur_date, tz="UTC") + cal_maturity

        pre_mask = live["expiration"].le(cutoff)
        nxt_mask = live["expiration"].gt(cutoff)

        if not pre_mask.any() or not nxt_mask.any():
            return None

        pre_row = (
            live.loc[pre_mask, ["expiration", "instrument_id"]]
            .sort_values(["expiration", "instrument_id"])
            .iloc[-1]
        )
        nxt_row = (
            live.loc[nxt_mask, ["expiration", "instrument_id"]]
            .sort_values(["expiration", "instrument_id"])
            .iloc[0]
        )

        return int(pre_row["instrument_id"]), int(nxt_row["instrument_id"])

    specs = []
    current_pair = None
    run_start = None

    num_days = (end - start).days
    for offset in range(num_days + 1):
        cur_date = start + dt.timedelta(days=offset)

        pair = pick_pair_for_date(cur_date)

        if pair is None and current_pair is not None:
            pair = current_pair

        if pair is None and current_pair is None:
            continue

        if pair != current_pair:
            if current_pair is not None and run_start is not None:
                specs.append(
                    {
                        "d0": run_start.isoformat(),
                        "d1": cur_date.isoformat(),
                        "p": str(current_pair[0]),
                        "n": str(current_pair[1]),
                    }
                )

            if cur_date == end:
                current_pair = None
                run_start = None
                break

            current_pair = pair
            run_start = cur_date

    if current_pair is not None and run_start is not None:
        specs.append(
            {
                "d0": run_start.isoformat(),
                "d1": end.isoformat(),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            }
        )

    return [s for s in specs if s.get("p") and s.get("n")]