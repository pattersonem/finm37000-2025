import pandas as pd

def constant_maturity_splice(
    symbol: str,
    roll_spec,
    raw_data: pd.DataFrame,
    date_col: str = "datetime",
    price_col: str = "price",
):
    cm = pd.Timedelta(days=int(symbol.rsplit(".", 1)[-1]))
    df = raw_data.copy()

    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

    exp_map = df.groupby("instrument_id")["expiration"].first().to_dict()
    price_map = df.set_index([date_col, "instrument_id"])[price_col]

    segs = []
    for r in roll_spec:
        d0 = pd.to_datetime(r["d0"], utc=True)
        d1 = pd.to_datetime(r["d1"], utc=True)
        pre, nxt = int(r["p"]), int(r["n"])

        t = pd.date_range(d0, d1, tz="UTC", inclusive="left", freq="D")
        pre_exp, nxt_exp = exp_map[pre], exp_map[nxt]
        f = (nxt_exp - (t + cm)) / (nxt_exp - pre_exp)

        idx_pre = pd.MultiIndex.from_product([t, [pre]], names=[date_col, "instrument_id"])
        idx_nxt = pd.MultiIndex.from_product([t, [nxt]], names=[date_col, "instrument_id"])
        pre_px = price_map.reindex(idx_pre).to_numpy()
        nxt_px = price_map.reindex(idx_nxt).to_numpy()

        segs.append(pd.DataFrame({
            "datetime": t,
            "pre_price": pre_px,
            "pre_id": pre,
            "pre_expiration": pre_exp,
            "next_price": nxt_px,
            "next_id": nxt,
            "next_expiration": nxt_exp,
            "pre_weight": f.to_numpy(),
            symbol: f.to_numpy() * pre_px + (1 - f.to_numpy()) * nxt_px,
        }))

    out = pd.concat(segs, ignore_index=True)
    return out
