import pandas as pd

def get_roll_spec(symbol, instrument_defs, *, start, end):
    # get target maturity in days from  symbol  
    maturity = pd.Timedelta(days=int(symbol.split(".")[-1]))
    one_day  = pd.Timedelta(days=1)

    # clean data 
    df = instrument_defs.copy()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"]    = pd.to_datetime(df["ts_recv"])
    if "instrument_class" in df.columns:
        df = df[df["instrument_class"] == "F"]

    # Convert to date-only 
    df["exp_date"]  = df["expiration"].dt.date
    df["recv_date"] = df["ts_recv"].dt.date
    df = df.sort_values("exp_date").reset_index(drop=True)

    # def start/end dates
    d0        = pd.to_datetime(start).date()
    end_date  = pd.to_datetime(end).date()
    specs     = []

    while d0 < end_date:

        # find all contracts available by this date
        avail = df[df["recv_date"] <= d0]
        if avail.empty:
            # if no contract is active yet, move forward one day
            d0 = (pd.to_datetime(d0) + one_day).date()
            continue

        # find the next contract that expires after target maturity date
        target = (pd.to_datetime(d0) + maturity).date()
        nxts   = avail[avail["exp_date"] > target]
        if nxts.empty:
            break
        nxt = nxts.iloc[0]

        # find the previous contract that expires before that one
        pres = avail[avail["exp_date"] < nxt["exp_date"]]
        if pres.empty:
            d0 = (pd.to_datetime(d0) + one_day).date()
            continue
        pre = pres.iloc[-1]

        # default end of segment: day after exp(next) - maturity, clipped to end_date
        roll_boundary = (pd.to_datetime(nxt["exp_date"]) - maturity + one_day).date()
        d1 = min(roll_boundary, end_date)

        # early split only if a new contract becomes live AND changes the (pre,next) pair
        later_recv = df.loc[df["recv_date"] > d0, "recv_date"]
        next_new   = later_recv.min() if not later_recv.empty else None
        if next_new is not None and next_new < d1:
            avail2 = df[df["recv_date"] <= next_new]
            target2 = (pd.to_datetime(next_new) + maturity).date()
            nxt2s = avail2[avail2["exp_date"] > target2]
            if not nxt2s.empty:
                nxt2 = nxt2s.iloc[0]
                pre2s = avail2[avail2["exp_date"] < nxt2["exp_date"]]
                if not pre2s.empty:
                    pre2 = pre2s.iloc[-1]
                    if int(pre2["instrument_id"]) != int(pre["instrument_id"]) or \
                       int(nxt2["instrument_id"]) != int(nxt["instrument_id"]):
                        d1 = next_new  # split early

        if d1 <= d0:
            d0 = (pd.to_datetime(d0) + one_day).date()
            continue

        specs.append({
            "d0": d0.isoformat(),
            "d1": d1.isoformat(),
            "p": str(int(pre["instrument_id"])),
            "n": str(int(nxt["instrument_id"])),
        })
        d0 = d1

    return specs



def _to_utc(series: pd.Series) -> pd.Series:
    """Localize to UTC if tz-naive; otherwise convert to UTC."""
    s = pd.to_datetime(series)
    try:
        return s.dt.tz_convert("UTC")    
    except TypeError:
        return s.dt.tz_localize("UTC")   


def constant_maturity_splice(symbol, roll_spec, all_data, date_col, price_col):
    """
    Blend prices from two contracts across roll periods to create a single constant-maturity price series.
    """
    maturity_days = int(symbol.split(".")[-1])
    maturity = pd.Timedelta(days=maturity_days)

    df = all_data.copy()
    df[date_col] = _to_utc(df[date_col])
    df["expiration"] = _to_utc(df["expiration"])

    exp_map = (df.drop_duplicates("instrument_id").set_index("instrument_id")["expiration"])

    pieces = []
    for seg in roll_spec:
        d0 = pd.to_datetime(seg["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(seg["d1"]).tz_localize("UTC")
        pre_id, nxt_id = int(seg["p"]), int(seg["n"])
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        exp_pre = exp_map.loc[pre_id]
        exp_nxt = exp_map.loc[nxt_id]
        f = (exp_nxt - (t + maturity)) / (exp_nxt - exp_pre)

        pre_price = pre_id
        nxt_price = nxt_id

        seg_df = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_price,
                "pre_id": pre_id,
                "pre_expiration": exp_pre,
                "next_price": nxt_price,
                "next_id": nxt_id,
                "next_expiration": exp_nxt,
                "pre_weight": f,
                symbol: f * pre_price + (1 - f) * nxt_price,
            }
        )
        pieces.append(seg_df)

    return pd.concat(pieces, ignore_index=True)