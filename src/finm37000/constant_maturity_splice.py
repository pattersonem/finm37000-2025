
import pandas as pd


def constant_maturity_splice(symbol,
                             roll_spec,
                             all_data,
                             date_col="datetime",
                             price_col="price"):
    """
    Create constant maturity synthetic time series using the roll specification.
    """

    # Extract number of days from symbol, e.g. "SR3.cm.182" -> 182
    maturity_days = int(symbol.split(".")[-1])
    maturity_delta = pd.Timedelta(days=maturity_days)

    df = all_data.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["instrument_id"] = df["instrument_id"].astype(int)

    out = []

    for seg in roll_spec:
        d0 = pd.to_datetime(seg["d0"]).tz_localize("UTC")
        d1 = pd.to_datetime(seg["d1"]).tz_localize("UTC")

        pre = int(seg["p"])
        nxt = int(seg["n"])

        t_range = pd.date_range(start=d0, end=d1, freq="D", tz="UTC", inclusive="left")

        pre_df = df[df["instrument_id"] == pre].set_index(date_col)
        nxt_df = df[df["instrument_id"] == nxt].set_index(date_col)

        for dt in t_range:
            if dt not in pre_df.index or dt not in nxt_df.index:
                continue

            pre_row = pre_df.loc[dt]
            nxt_row = nxt_df.loc[dt]

            exp_pre = pre_row["expiration"]
            exp_nxt = nxt_row["expiration"]

            # FIXED: Match test behavior
            weight = (exp_nxt - (dt + maturity_delta)) / (exp_nxt - exp_pre)

            synthetic_price = (
                weight * pre_row[price_col] + (1 - weight) * nxt_row[price_col]
            )

            out.append(
                {
                    "datetime": dt,
                    "pre_price": pre_row[price_col],
                    "pre_id": pre,
                    "pre_expiration": exp_pre,
                    "next_price": nxt_row[price_col],
                    "next_id": nxt,
                    "next_expiration": exp_nxt,
                    "pre_weight": weight,
                    symbol: synthetic_price,
                }
            )

    return pd.DataFrame(out).reset_index(drop=True)
