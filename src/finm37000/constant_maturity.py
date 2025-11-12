import pandas as pd


def _cm_days_from_symbol(symbol):
    try:
        return int(symbol.rsplit(".", 1)[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse constant-maturity days from: {symbol}") from exc


def get_roll_spec(symbol, instrument_defs, *, start, end):
    m_days = _cm_days_from_symbol(symbol)

    df = instrument_defs.copy()
    if "instrument_class" in df:
        df = df[df["instrument_class"] == "F"]

    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    if "ts_recv" in df:
        df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    else:
        df["ts_recv"] = pd.Timestamp("1900-01-01", tz="UTC")

    if not isinstance(start, pd.Timestamp):
        d0 = pd.Timestamp(start).tz_localize("UTC")
    else:
        d0 = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    if not isinstance(end, pd.Timestamp):
        d1 = pd.Timestamp(end).tz_localize("UTC")
    else:
        d1 = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    days = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

    def pair_for_day(day):
        live = df[df["ts_recv"] <= day].sort_values("expiration")
        T = day + pd.Timedelta(days=m_days)
        pre_row = live[live["expiration"] <= T].tail(1)
        nxt_row = live[live["expiration"] > T].head(1)
        if pre_row.empty or nxt_row.empty:
            raise ValueError("Cannot bracket target maturity with available expirations.")
        pre = int(pre_row["instrument_id"].iloc[0])
        nxt = int(nxt_row["instrument_id"].iloc[0])
        return pre, nxt

    spec = []
    if len(days) == 0:
        return spec

    current_p, current_n = pair_for_day(days[0])
    seg_start = days[0]

    for day in days[1:]:
        p, n = pair_for_day(day)
        if (p, n) != (current_p, current_n):
            spec.append(
                {
                    "d0": seg_start.date().isoformat(),
                    "d1": day.date().isoformat(),
                    "p": str(current_p),
                    "n": str(current_n),
                }
            )
            seg_start = day
            current_p, current_n = p, n

    spec.append(
        {
            "d0": seg_start.date().isoformat(),
            "d1": d1.date().isoformat(),
            "p": str(current_p),
            "n": str(current_n),
        }
    )
    return spec


def constant_maturity_splice(symbol, roll_spec, raw_data, *, date_col="datetime", price_col="price"):
    cm_days = _cm_days_from_symbol(symbol)

    df = raw_data.copy()
    df["instrument_id"] = df["instrument_id"].astype(int)
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

    exp_map = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
        .to_dict()
    )

    def series_for(iid):
        return (
            df.loc[df["instrument_id"] == iid, [date_col, price_col]]
            .set_index(date_col)
            .sort_index()[price_col]
        )

    pieces = []
    for r in roll_spec:
        pre_id = int(r["p"])
        nxt_id = int(r["n"])
        tz = "UTC"
        t = pd.date_range(
            start=pd.Timestamp(r["d0"]).tz_localize(tz),
            end=pd.Timestamp(r["d1"]).tz_localize(tz),
            tz=tz,
            inclusive="left",
        )

        pre_exp = pd.Timestamp(exp_map[pre_id]).tz_convert("UTC")
        nxt_exp = pd.Timestamp(exp_map[nxt_id]).tz_convert("UTC")

        pre_prices = series_for(pre_id).reindex(t).ffill().bfill()
        nxt_prices = series_for(nxt_id).reindex(t).ffill().bfill()

        f = (nxt_exp - (t + pd.Timedelta(days=cm_days))) / (nxt_exp - pre_exp)
        f = f.astype("float64")

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_prices.values,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": nxt_prices.values,
                "next_id": nxt_id,
                "next_expiration": nxt_exp,
                "pre_weight": f.values,
                symbol: f.values * pre_prices.values + (1.0 - f.values) * nxt_prices.values,
            }
        )
        pieces.append(seg)

    return pd.concat(pieces, ignore_index=True)