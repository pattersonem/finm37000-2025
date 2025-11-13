import pandas as pd


def _parse_maturity_days(symbol: str) -> int:
    return int(symbol.split(".")[-1])


def get_roll_spec(symbol, instrument_df, start, end):
    """
    Build roll segments like:
    {"d0": "...", "d1": "...", "p": pre_id, "n": next_id}
    """

    maturity_days = _parse_maturity_days(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = instrument_df.copy()
    df = df[df["instrument_class"] == "F"].copy()

    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    df = df[df["ts_recv"].dt.date <= end].copy()
    df = df.sort_values("expiration")

    all_dates = pd.date_range(start=start, end=end, freq="D")

    roll_spec = []
    current_d0 = start
    last_pre = None
    last_next = None

    for date in all_dates:
        t = date
        # FIX 1: Make target tz-aware to match expiration column
        target = (t + maturity_delta).tz_localize("UTC")

        available = df[df["ts_recv"].dt.date <= t.date()]
        after = available[available["expiration"] >= target]
        before = available[available["expiration"] < target]

        if after.empty or before.empty:
            continue

        next_id = after.iloc[0]["instrument_id"]
        pre_id = before.iloc[-1]["instrument_id"]

        if last_pre is None:
            last_pre, last_next = pre_id, next_id
            current_d0 = t.date()
            continue

        if pre_id != last_pre or next_id != last_next:
            roll_spec.append({
                "d0": str(current_d0),
                "d1": str(t.date()),
                "p": str(last_pre),
                "n": str(last_next),
            })
            current_d0 = t.date()
            last_pre, last_next = pre_id, next_id

    if last_pre is not None and current_d0 < end:
        roll_spec.append({
            "d0": str(current_d0),
            "d1": str(end),
            "p": str(last_pre),
            "n": str(last_next),
        })

    return roll_spec


def constant_maturity_splice(
    symbol,
    roll_spec,
    data,
    date_col="datetime",
    price_col="price",
):
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])

    by_id = {k: v for k, v in df.groupby("instrument_id")}

    maturity_days = _parse_maturity_days(symbol)
    maturity_delta = pd.Timedelta(days=maturity_days)

    segments = []

    for r in roll_spec:
        d0 = pd.to_datetime(r["d0"])
        d1 = pd.to_datetime(r["d1"])
        pre = int(r["p"])
        nxt = int(r["n"])

        t_range = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        pre_df = by_id[pre].set_index(date_col).reindex(t_range).reset_index().rename(
            columns={"index": "datetime"}
        )
        nxt_df = by_id[nxt].set_index(date_col).reindex(t_range).reset_index().rename(
            columns={"index": "datetime"}
        )

        # FIX 2: keep int dtype — do NOT cast to float
        pre_price = pre_df[price_col]
        next_price = nxt_df[price_col]

        pre_exp = pd.to_datetime(pre_df["expiration"].iloc[0])
        next_exp = pd.to_datetime(nxt_df["expiration"].iloc[0])

        pre_weight = (next_exp - (t_range + maturity_delta)) / (
            next_exp - pre_exp
        )

        seg = pd.DataFrame({
            "datetime": t_range,
            "pre_price": pre_price,
            "pre_id": pre,
            "pre_expiration": pre_exp,
            "next_price": next_price,
            "next_id": nxt,
            "next_expiration": next_exp,
            "pre_weight": pre_weight,
            symbol: pre_weight * pre_price + (1 - pre_weight) * next_price,
        })

        segments.append(seg)

    return pd.concat(segments, ignore_index=True)
