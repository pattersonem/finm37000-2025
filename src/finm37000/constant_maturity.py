"""Constant-maturity splice utilities."""

from __future__ import annotations

import pandas as pd


def _parse_maturity_days(symbol: str) -> int:
    try:
        return int(str(symbol).split(".")[-1])
    except Exception as exc:
        msg = "Could not parse maturity days from symbol '{sym}'".format(sym=symbol)
        raise ValueError(msg) from exc


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    raw_data: pd.DataFrame,
    *,
    date_col: str = "datetime",
    price_col: str = "close",
) -> pd.DataFrame:
    """Linearly interpolate a fixed-TTM futures price over roll segments.

    For each segment in ``roll_spec``, use the “pre” and “next” instruments
    prices and expirations to compute the weight on the pre leg:
        pre_weight(t) = (exp_next - (t + T)) / (exp_next - exp_pre)
    and the constant-maturity price:
        cm(t) = pre_weight * pre_price + (1 - pre_weight) * next_price

    Args:
        symbol: Constant-maturity symbol ``'<prod>.cm.<days>'``.
            The last token is used for TTM.
        roll_spec: List of dicts with keys ``{'d0','d1','p','n'}``.
        raw_data: DataFrame with at least
            ``['instrument_id', date_col, price_col, 'expiration']``.
        date_col: Timestamp column in ``raw_data``.
        price_col: Price column in ``raw_data``.

    Returns:
        DataFrame with columns:
        ``['datetime','pre_price','pre_id','pre_expiration', 'next_price',
        'next_id','next_expiration','pre_weight', symbol]``.
    """
    maturity_delta = pd.Timedelta(days=_parse_maturity_days(symbol))
    df = raw_data.copy()

    # Normalize to UTC
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df = df.sort_values([date_col, "instrument_id"]).reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    for r in roll_spec:
        d0 = pd.to_datetime(r["d0"], utc=True)
        d1 = pd.to_datetime(r["d1"], utc=True)
        pre_id = int(r["p"])
        nxt_id = int(r["n"])

        # Segment [d0, d1)
        t = pd.date_range(start=d0, end=d1, tz="UTC", inclusive="left")
        if len(t) == 0:
            continue

        # Select and align both legs to grid
        pre = df[(df["instrument_id"] == pre_id) & (df[date_col].isin(t))].copy()
        nxt = df[(df["instrument_id"] == nxt_id) & (df[date_col].isin(t))].copy()
        pre = pre.set_index(date_col).reindex(t)
        nxt = nxt.set_index(date_col).reindex(t)

        pre_price = pre[price_col]
        nxt_price = nxt[price_col]
        pre_exp = pre["expiration"].iloc[0]
        nxt_exp = nxt["expiration"].iloc[0]

        denom = nxt_exp - pre_exp
        pre_weight = (nxt_exp - (t + maturity_delta)) / denom

        cm_price = (
            pre_weight.to_numpy() * pre_price.to_numpy()
            + (1.0 - pre_weight.to_numpy()) * nxt_price.to_numpy()
        )

        seg = pd.DataFrame(
            {
                "datetime": t,
                "pre_price": pre_price.to_numpy(),
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": nxt_price.to_numpy(),
                "next_id": nxt_id,
                "next_expiration": nxt_exp,
                "pre_weight": pre_weight.to_numpy(),
                symbol: cm_price,
            }
        )
        pieces.append(seg)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "datetime",
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

    return pd.concat(pieces, ignore_index=True)
