from __future__ import annotations
from datetime import date, timedelta
import pandas as pd


def get_roll_spec(symbol: str, instrument_defs: pd.DataFrame, *, start: date, end: date) -> list[dict[str,str]]:
    maturity_days = int(symbol.split('.')[-1])
    df = instrument_defs.copy()
    df = df[df['instrument_class']=='F'].copy()
    df['exp_date'] = pd.to_datetime(df['expiration']).dt.date
    df['recv_date'] = pd.to_datetime(df['ts_recv']).dt.date
    df = df.sort_values('exp_date').reset_index(drop=True)

    date_pairs: dict[date,tuple[int,int]] = {}
    d = start
    while d<end:
        live = df[df['recv_date']<=d]
        if len(live) >=2:
            target = d+timedelta(days = maturity_days)
            live = live.sort_values('exp_date')
            exps = live['exp_date'].to_list()
            ids = live['instrument_id'].to_list()
            idx = next((i for i,e in enumerate(exps) if e>=target),None)
            if idx not in (None,0):
                date_pairs[d]=(int(ids[idx-1]),int(ids[idx]))
            d+=timedelta(days=1)
    if not date_pairs:
        return []
    segs: list[dict[str,str]] = []
    dates = sorted(date_pairs.keys())
    seg_start = dates[0]
    pre, post = date_pairs[seg_start]
    prev = seg_start

    for d in dates[1:]:
        pair = date_pairs[d]
        if pair == (pre,post) and d == prev+timedelta(days=1):
            prev = d
        else:
            segs.append(
                {
                    "d0": seg_start.isoformat(),
                    "d1": (prev + timedelta(days=1)).isoformat(),
                    "p": str(pre),
                    "n": str(post),
                }
            )
            seg_start = d
            pre, post = pair
            prev = d
    segs.append(
        {
            "d0": seg_start.isoformat(),
            "d1": (prev + timedelta(days=1)).isoformat(),
            "p": str(pre),
            "n": str(post),
        }
    )
    return segs

def constant_maturity_splice(symbol: str, roll_spec: list[dict[str,str]], raw_data = pd.DataFrame, *, date_col: str = 'datetime',price_col: str = 'price',)->pd.DataFrame:
    maturity_days = int(symbol.split('.')[-1])
    maturity = pd.Timedelta(days = maturity_days)
    grouped = raw_data.groupby('instrument_id')
    expirations = (raw_data.drop_duplicates('instrument_id').set_index('instrument_id')['expiration'].map(pd.to_datetime))
    tz = raw_data[date_col].dt.tz
    pieces: list[pd.DataFrame] = []

    for spec in roll_spec:
        d0 = pd.Timestamp(spec['d0'],tz = tz)
        d1 = pd.Timestamp(spec['d1'],tz = tz)
        pre = int(spec['p'])
        post = int(spec['n'])
        t = pd.date_range(start = d0, end = d1, tz=tz, inclusive = 'left')
        pre_exp = expirations[pre]
        post_exp = expirations[post]
        f = (post_exp-t-maturity)/(post_exp-pre_exp)
        g_pre = grouped.get_group(pre)
        g_post = grouped.get_group(post)

        pre_slice = g_pre[g_pre[date_col].isin(t)].sort_values(date_col)
        post_slice = g_post[g_post[date_col].isin(t)].sort_values(date_col)
        pre_price = pre_slice[price_col].to_numpy()
        post_price = post_slice[price_col].to_numpy()
        seg = pd.DataFrame(
            {
                date_col: t,
                "pre_price": pre_price,
                "pre_id": pre,
                "pre_expiration": pre_exp,
                "next_price": post_price,
                "next_id": post,
                "next_expiration": post_exp,
                "pre_weight": f,
                symbol: f * pre_price + (1.0 - f) * post_price,
            }
        )
        pieces.append(seg)

    return pd.concat(pieces, ignore_index=True)  