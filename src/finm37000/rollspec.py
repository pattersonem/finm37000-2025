from datetime import date, timedelta
import re
import pandas as pd

def get_roll_spec(root_rule: str,
                  instruments: pd.DataFrame,
                  start: date,
                  end: date) -> list[dict[str, str]]:
    m: re.Match[str] | None = re.match(pattern=r'^([A-Z0-9]+)\.cm\.(\d+)$', string=root_rule)
    if not m:
        raise ValueError(f"Unrecognized root_rule format: {root_rule}")
    root, maturity_days = m.group(1), int(m.group(2))

    df: pd.DataFrame = instruments.copy()
    df['expiration'] = pd.to_datetime(arg=df['expiration'], utc=True)
    df['ts_recv'] = pd.to_datetime(arg=df['ts_recv'], utc=True)

    uni: pd.DataFrame = df[
        (df['instrument_class'] == 'F') &
        (df['raw_symbol'].astype(dtype=str).str.startswith(pat=root))
    ].copy()

    if uni.empty:
        return []

    uni['exp_date'] = uni['expiration'].dt.date
    uni['live_date'] = uni['ts_recv'].dt.date

    exp_boundaries = [(e - timedelta(days=maturity_days) + timedelta(days=1))
                      for e in uni['exp_date']]
    live_boundaries = list(uni['live_date'].unique())

    candidates: set[date] = {start, end}
    for d in exp_boundaries + live_boundaries:
        if start < d < end:
            candidates.add(d)

    ordered: list[date] = sorted(candidates)

    def pick_pair(d: date):
        T: date = d + timedelta(days=maturity_days)
        avail: pd.DataFrame = uni[uni['live_date'] <= d]
        if avail.empty:
            return None

        left: pd.DataFrame = avail[avail['exp_date'] <= T]
        right: pd.DataFrame = avail[avail['exp_date'] > T]

        if left.empty or right.empty:
            return None

        p_row = left.sort_values('exp_date').iloc[-1]
        n_row = right.sort_values('exp_date').iloc[0]
        return (str(p_row['instrument_id']), str(n_row['instrument_id']))

    specs = []
    if len(ordered) == 1:
        return specs

    current_pair = None
    seg_start = None

    for i in range(len(ordered) - 1):
        d0: date = ordered[i]
        d1: date = ordered[i + 1]
        pair = pick_pair(d0)

        if pair is None:
            if current_pair is not None and seg_start is not None:
                specs.append({
                    "d0": seg_start.isoformat(),
                    "d1": d0.isoformat(),
                    "p": current_pair[0],
                    "n": current_pair[1],
                })
                current_pair = None
                seg_start = None
            continue

        if current_pair is None:
            current_pair = pair
            seg_start = d0
        elif pair != current_pair:
            specs.append({
                "d0": seg_start.isoformat(),
                "d1": d0.isoformat(),
                "p": current_pair[0],
                "n": current_pair[1],
            })
            current_pair = pair
            seg_start: date = d0

        if i == len(ordered) - 2:
            if current_pair is not None and seg_start is not None:
                specs.append({
                    "d0": seg_start.isoformat(),
                    "d1": d1.isoformat(),
                    "p": current_pair[0],
                    "n": current_pair[1],
                })

    specs = [s for s in specs if s["d0"] < s["d1"]]
    return specs
