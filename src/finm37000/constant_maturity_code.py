import pandas as pd
import datetime as dt


import pandas as pd
import datetime as dt


def get_roll_spec(symbol: str, instrument_defs: pd.DataFrame, start, end):
    """
    Build a roll schedule for a constant-maturity symbol like 'SR3.cm.182'.

    For each calendar day between start and end (start inclusive, end exclusive),
    we find the pair of futures contracts whose expirations straddle the
    target date (current day + maturity). Whenever that pair changes,
    we start a new roll segment.
    """
    # --- parse product and maturity ---
    splits = symbol.split(".")
    if len(splits) < 3 or splits[-2] != "cm":
        raise ValueError(f"Unexpected constant-maturity symbol: {symbol}")
    product = splits[0] or ".".join(splits[:-2])
    try:
        maturity_days = int(splits[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse maturity from {symbol}") from exc

    maturity = pd.Timedelta(days=maturity_days)

    # --- normalize start / end to plain dates ---
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if start_date >= end_date:
        raise ValueError("start must be strictly before end")

    # --- prepare instrument definitions ---
    df = instrument_defs.copy()
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df["live_date"] = df["ts_recv"].dt.date

    futures = df[
        (df["instrument_class"] == "F") &
        (df["raw_symbol"].astype(str).str.startswith(product))
    ].sort_values("expiration").reset_index(drop=True)

    if futures.empty:
        raise ValueError(f"No futures found for product {product}")

    # assume all expirations share the same timezone
    tzinfo = futures["expiration"].dt.tz

    # --- walk day by day and record the (pre, next) pair ---
    schedule: list[dict] = []
    current_pair: tuple[int, int] | None = None
    segment_start = start_date

    day = start_date
    while day < end_date:
        live = futures[futures["live_date"] <= day]
        if live.empty:
            raise ValueError(f"No live contracts on {day.isoformat()}")

        anchor = pd.Timestamp(day).tz_localize(tzinfo)
        target = anchor + maturity

        # pick the last expiration <= target as "pre"
        pre_candidates = live[live["expiration"] <= target]
        if pre_candidates.empty:
            pre_row = live.iloc[0]
        else:
            pre_row = pre_candidates.iloc[-1]

        # pick the first expiration >= target as "next"
        next_candidates = live[live["expiration"] >= target]
        if next_candidates.empty:
            next_row = live.iloc[-1]
        else:
            next_row = next_candidates.iloc[0]

        pair = (int(pre_row["instrument_id"]), int(next_row["instrument_id"]))

        if current_pair is None:
            # first day
            current_pair = pair
            segment_start = day
        elif pair != current_pair:
            # close the previous segment ON this change day (inclusive)
            schedule.append(
                {
                    "d0": segment_start.isoformat(),
                    "d1": day.isoformat(),
                    "p": str(current_pair[0]),
                    "n": str(current_pair[1]),
                }
            )
            current_pair = pair
            segment_start = day

        day += dt.timedelta(days=1)

    # close final segment at end_date
    if current_pair is not None:
        schedule.append(
            {
                "d0": segment_start.isoformat(),
                "d1": end_date.isoformat(),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            }
        )

    return schedule


def constant_maturity_splice(
    symbol: str,
    roll_spec: list,
    data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
):
    """
    Blend two adjacent futures contracts into a synthetic constant-maturity
    series. Each roll segment supplies the pair of contract IDs over a
    calendar window. Weighted interpolation is performed using time-to-
    expiration.
    """

    if not roll_spec:
        return pd.DataFrame()

    # parse maturity from symbol
    try:
        maturity_days = int(symbol.split(".")[-1])
    except Exception:
        raise ValueError(f"Cannot parse maturity from {symbol}")
    maturity_offset = pd.Timedelta(days=maturity_days)

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    out_frames = []

    # group raw data by contract id for efficient slicing
    grouped = df.sort_values(date_col).groupby("instrument_id")

    for block in roll_spec:
        start = pd.to_datetime(block["d0"]).tz_localize("UTC")
        end = pd.to_datetime(block["d1"]).tz_localize("UTC")

        pre_id = int(block["p"])
        nxt_id = int(block["n"])

        if pre_id not in grouped.groups or nxt_id not in grouped.groups:
            raise KeyError(f"Missing data for {pre_id} or {nxt_id}")

        left = grouped.get_group(pre_id)
        right = grouped.get_group(nxt_id)

        # restrict each leg to the correct date window
        mask_l = (left[date_col] >= start) & (left[date_col] < end)
        mask_r = (right[date_col] >= start) & (right[date_col] < end)

        seg_l = left.loc[mask_l, [date_col, price_col, "expiration"]].rename(
            columns={price_col: "pre_price", "expiration": "pre_expiration"}
        )
        seg_r = right.loc[mask_r, [date_col, price_col, "expiration"]].rename(
            columns={price_col: "next_price", "expiration": "next_expiration"}
        )

        merged = seg_l.merge(seg_r, on=date_col, how="inner")
        if merged.empty:
            continue

        # compute blend weight
        blend_target = merged[date_col] + maturity_offset
        span = merged["next_expiration"] - merged["pre_expiration"]

        if (span == pd.Timedelta(0)).any():
            raise ValueError("Duplicate expiration dates in roll pair")

        merged["pre_id"] = pre_id
        merged["next_id"] = nxt_id

        merged["pre_weight"] = (merged["next_expiration"] - blend_target) / span
        merged[symbol] = (
            merged["pre_weight"] * merged["pre_price"]
            + (1 - merged["pre_weight"]) * merged["next_price"]
        )

        out_frames.append(
            merged[
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
        )

    if not out_frames:
        return pd.DataFrame()

    return pd.concat(out_frames, ignore_index=True)
