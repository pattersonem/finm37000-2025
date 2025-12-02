import datetime as dt
import pandas as pd

def get_roll_spec(symbol: str, instrument_defs: pd.DataFrame, *, start: dt.date, end: dt.date):
    """Pick (pre,next) by target = date + cm_days. Only use FUT rows live on date (ts_recv.date() <= date)."""
    cm = pd.Timedelta(days=int(symbol.rsplit(".", 1)[-1]))
    df = instrument_defs.copy()

    # keep only futures (ignore spreads)
    df = df[df["instrument_class"] == "F"].copy()
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df["recv_date"] = df["ts_recv"].dt.date
    df = df.sort_values(["expiration", "instrument_id"]).reset_index(drop=True)

    exps = list(
        zip(df["instrument_id"].astype(int), df["expiration"].to_list(), df["recv_date"].to_list())
    )

    def pick_pair(d: dt.date):
        x = pd.Timestamp(d, tz="UTC") + cm
        live = [(iid, exp) for iid, exp, rcv in exps if rcv <= d]
        pre = [p for p in live if p[1] <= x]
        nxt = [n for n in live if n[1] >  x]
        if not pre or not nxt:  # incomplete book that day
            return None
        pre_id = max(pre, key=lambda t: t[1])[0]
        nxt_id = min(nxt, key=lambda t: t[1])[0]
        return (pre_id, nxt_id)

    specs, cur_pair, run_start = [], None, None
    d = start
    while d <= end:
        pair = pick_pair(d)

        if pair is None and cur_pair is not None:
            pair = cur_pair

        if pair is None and cur_pair is None:
            d += dt.timedelta(days=1)
            continue

        if pair != cur_pair:
            if cur_pair is not None and run_start is not None:
                specs.append({
                    "d0": run_start.isoformat(),
                    "d1": d.isoformat(),        
                    "p": str(cur_pair[0]),
                    "n": str(cur_pair[1]),
                })
            if d == end:
                cur_pair = None
                run_start = None
                break
                
            # start new run
            cur_pair = pair
            run_start = d
            
        d += dt.timedelta(days=1)
    if cur_pair is not None:
        specs.append({"d0": run_start.isoformat(),
                      "d1": (end).isoformat(),
                      "p": str(cur_pair[0]),
                      "n": str(cur_pair[1])})
    return [s for s in specs if s.get("p") and s.get("n")]
