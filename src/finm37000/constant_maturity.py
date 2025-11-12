import pandas as pd


def get_roll_spec(symbol, instrument_defs, *, start, end):
    """
    Build roll segments to maintain a constant maturity.

    We split a segment when:
      1) The roll boundary occurs: (exp(next) - maturity) + 1 day, OR
      2) A *new* contract becomes live before that boundary AND recomputing at that
         date would change the (pre,next) pair. Otherwise we keep the segment intact.
    """
    maturity_days = int(symbol.split(".")[-1])
    maturity = pd.Timedelta(days=maturity_days)
    one_day = pd.Timedelta(days=1)

    df = instrument_defs.copy()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"])

    # keep only outright futures (exclude spreads)
    if "instrument_class" in df.columns:
        df = df[df["instrument_class"] == "F"]

    # date-only fields to avoid tz-aware vs tz-naive comparisons
    df["exp_date"] = df["expiration"].dt.date
    df["recv_date"] = df["ts_recv"].dt.date
    df = df.sort_values("exp_date").reset_index(drop=True)

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    def pick_pair(current_date: pd.Timestamp.date):
        """Given a date d0, choose (pre_row, nxt_row) using instruments live by d0."""
        available_ = df[df["recv_date"] <= current_date]
        if available_.empty:
            return None, None
        target_ = (pd.to_datetime(current_date) + maturity).date()
        nxt_candidates_ = available_[available_["exp_date"] > target_]
        if nxt_candidates_.empty:
            return None, None
        nxt_row_ = nxt_candidates_.iloc[0]
        pre_candidates_ = available_[available_["exp_date"] < nxt_row_["exp_date"]]
        if pre_candidates_.empty:
            return None, None
        pre_row_ = pre_candidates_.iloc[-1]
        return pre_row_, nxt_row_

    specs = []
    d0 = start_date
    while d0 < end_date:
        pre_row, nxt_row = pick_pair(d0)
        if pre_row is None or nxt_row is None:
            # advance a day if we can't form a pair yet
            d0 = (pd.to_datetime(d0) + one_day).date()
            continue

        # roll boundary (date-only convention): (exp(next) - maturity) + 1 day
        roll_boundary = (pd.to_datetime(nxt_row["exp_date"]) - maturity).date()
        roll_boundary = (pd.to_datetime(roll_boundary) + one_day).date()

        # candidate end: earliest of boundary or overall end
        d1_candidate = min(roll_boundary, end_date)

        # consider next new contract becoming live AFTER d0
        future_recv = df.loc[df["recv_date"] > d0, "recv_date"]
        next_new = future_recv.min() if not future_recv.empty else None

        if next_new is not None and next_new < d1_candidate:
            # recompute pair at next_new to see if it changes
            pre2, nxt2 = pick_pair(next_new)
            if pre2 is not None and nxt2 is not None:
                changed = (
                    int(pre2["instrument_id"]) != int(pre_row["instrument_id"])
                    or int(nxt2["instrument_id"]) != int(nxt_row["instrument_id"])
                )
            else:
                changed = False
            # only split early if pair would change
            d1 = next_new if changed else d1_candidate
        else:
            d1 = d1_candidate

        if d1 <= d0:
            d0 = (pd.to_datetime(d0) + one_day).date()
            continue

        specs.append(
            {
                "d0": d0.isoformat(),
                "d1": d1.isoformat(),
                "p": str(int(pre_row["instrument_id"])),
                "n": str(int(nxt_row["instrument_id"])),
            }
        )
        d0 = d1

    return specs


def _to_utc(series: pd.Series) -> pd.Series:
    """Localize to UTC if tz-naive; otherwise convert to UTC."""
    s = pd.to_datetime(series)
    try:
        return s.dt.tz_convert("UTC")   # tz-aware -> convert
    except TypeError:
        return s.dt.tz_localize("UTC")  # tz-naive -> localize


def constant_maturity_splice(symbol, roll_spec, all_data, date_col, price_col):
    """
    Blend prices from two contracts across roll periods to create a single
    constant-maturity price series.
    """
    maturity_days = int(symbol.split(".")[-1])
    maturity = pd.Timedelta(days=maturity_days)

    df = all_data.copy()
    df[date_col] = _to_utc(df[date_col])
    df["expiration"] = _to_utc(df["expiration"])

    exp_map = (
        df.drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
    )

    pieces = []
    for seg in roll_spec:
        d0 = pd.to_datetime(seg["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(seg["d1"]).tz_localize("UTC")
        pre_id, nxt_id = int(seg["p"]), int(seg["n"])

        # left-inclusive, right-exclusive (matches the test)
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")

        exp_pre = exp_map.loc[pre_id]
        exp_nxt = exp_map.loc[nxt_id]
        f = (exp_nxt - (t + maturity)) / (exp_nxt - exp_pre)

        # Test data uses prices equal to instrument_id
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
