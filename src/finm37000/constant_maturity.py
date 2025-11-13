"""Helpers to build constant-maturity time series."""

import datetime as _dt

import pandas as pd


def _parse_constant_maturity_symbol(symbol):
    parts = symbol.split(".")
    if len(parts) < 3 or parts[-2] != "cm":
        raise ValueError(f"Unrecognized constant-maturity symbol: {symbol}")
    product = ".".join(parts[:-2]) or parts[0]
    try:
        maturity_days = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid maturity in symbol {symbol}") from exc
    return product, pd.Timedelta(days=maturity_days)


def _as_date(value):
    if isinstance(value, _dt.date):
        return value
    return pd.Timestamp(value).date()


def _localize(ts, tz):
    stamp = pd.Timestamp(ts)
    if tz is None:
        return stamp.tz_localize(None) if stamp.tzinfo else stamp
    if stamp.tzinfo is None:
        return stamp.tz_localize(tz)
    return stamp.tz_convert(tz)


def get_roll_spec(symbol, instrument_defs, start, end):
    product, maturity = _parse_constant_maturity_symbol(symbol)
    start_date = _as_date(start)
    end_date = _as_date(end)
    if start_date >= end_date:
        raise ValueError("start must be before end")

    cols = instrument_defs.copy()
    cols["expiration"] = pd.to_datetime(cols["expiration"], utc=True)
    cols["ts_recv"] = pd.to_datetime(cols["ts_recv"], utc=True)
    cols["ts_recv_date"] = cols["ts_recv"].dt.date
    futures = cols[
        (cols["instrument_class"] == "F")
        & cols["raw_symbol"].astype(str).str.startswith(product)
    ].copy()
    if futures.empty:
        raise ValueError(f"No futures found for {product}")
    futures = futures.sort_values("expiration").reset_index(drop=True)
    tz = futures["expiration"].dt.tz

    day = start_date
    current_pair = None
    segment_start = start_date
    out = []
    while day < end_date:
        available = futures[futures["ts_recv_date"] <= day]
        if available.empty:
            raise ValueError(f"No instruments live on {day.isoformat()}")
        target = _localize(day, tz) + maturity
        pre_candidates = available[available["expiration"] <= target]
        if pre_candidates.empty:
            pre_row = available.iloc[0]
        else:
            pre_row = pre_candidates.iloc[-1]
        next_candidates = available[available["expiration"] >= target]
        if next_candidates.empty:
            next_row = available.iloc[-1]
        else:
            next_row = next_candidates.iloc[0]
        pair = (pre_row["instrument_id"], next_row["instrument_id"])
        if current_pair is None:
            current_pair = pair
            segment_start = day
        elif pair != current_pair:
            out.append(
                {
                    "d0": segment_start.isoformat(),
                    "d1": day.isoformat(),
                    "p": str(int(current_pair[0])),
                    "n": str(int(current_pair[1])),
                }
            )
            current_pair = pair
            segment_start = day
        day += _dt.timedelta(days=1)

    if current_pair is not None:
        out.append(
            {
                "d0": segment_start.isoformat(),
                "d1": end_date.isoformat(),
                "p": str(int(current_pair[0])),
                "n": str(int(current_pair[1])),
            }
        )
    return out


def constant_maturity_splice(symbol, roll_spec, raw_data, date_col="datetime", price_col="close"):
    if not roll_spec:
        return pd.DataFrame(
            columns=[
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
        )
    _, maturity = _parse_constant_maturity_symbol(symbol)
    df = raw_data.copy()
    if "instrument_id" not in df or "expiration" not in df:
        raise KeyError("raw_data must include instrument_id and expiration columns")
    df[date_col] = pd.to_datetime(df[date_col])
    df["expiration"] = pd.to_datetime(df["expiration"])
    tz = df[date_col].dt.tz
    grouped = df.sort_values(date_col).groupby("instrument_id")
    pieces = []
    for spec in roll_spec:
        d0 = _localize(spec["d0"], tz)
        d1 = _localize(spec["d1"], tz)
        pre_id = int(spec["p"])
        next_id = int(spec["n"])
        try:
            pre_group = grouped.get_group(pre_id)
            next_group = grouped.get_group(next_id)
        except KeyError as exc:
            raise KeyError(f"Missing data for instrument {exc.args[0]}") from exc
        mask_pre = (pre_group[date_col] >= d0) & (pre_group[date_col] < d1)
        mask_next = (next_group[date_col] >= d0) & (next_group[date_col] < d1)
        pre_piece = pre_group.loc[mask_pre, [date_col, "instrument_id", price_col, "expiration"]].rename(
            columns={
                "instrument_id": "pre_id",
                price_col: "pre_price",
                "expiration": "pre_expiration",
            }
        )
        next_piece = next_group.loc[
            mask_next, [date_col, "instrument_id", price_col, "expiration"]
        ].rename(
            columns={
                "instrument_id": "next_id",
                price_col: "next_price",
                "expiration": "next_expiration",
            }
        )
        merged = pre_piece.merge(next_piece, on=date_col, how="inner")
        if merged.empty:
            continue
        maturity_point = merged[date_col] + maturity
        denom = merged["next_expiration"] - merged["pre_expiration"]
        if (denom == pd.Timedelta(0)).any():
            raise ValueError("Zero day spread between expirations")
        merged["pre_weight"] = (merged["next_expiration"] - maturity_point) / denom
        merged[symbol] = merged["pre_weight"] * merged["pre_price"] + (
            1 - merged["pre_weight"]
        ) * merged["next_price"]
        merged = merged[
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
        pieces.append(merged)
    if not pieces:
        raise ValueError("No data matched the supplied roll specification")
    return pd.concat(pieces, ignore_index=True)
