import pandas as pd
import datetime


def extract_instrument_spec(symbol: str) -> tuple[str, int]:
    """Extract base symbol and maturity days from constant-maturity symbol.

    Args:
        symbol (str): Symbol in format "SYMBOL.cm.MATURITY_DAYS" (e.g. "SR3.cm.182")

    Returns:
        tuple[str, int]: Base symbol and maturity days (e.g. ("SR3", 182))
    """
    symbol, _, maturity_days = symbol.split(".")
    maturity_days = int(maturity_days)

    return symbol, maturity_days


def get_roll_spec(
    symbol: str,
    instrument_df: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, str]]:
    """Generate roll specifications for continuous futures contracts.

    Args:
        symbol (str): Constant maturity symbol (e.g. "SR3.cm.182")
        instrument_df (pd.DataFrame): Contract definitions with instrument_id, expiration, ts_recv
        start (datetime.date): Start date for roll schedule
        end (datetime.date): End date for roll schedule

    Returns:
        list[dict[str, str]]: List of roll specs with keys:
            d0: Start date (ISO format)
            d1: End date (ISO format)
            p: Previous contract instrument_id
            n: Next contract instrument_id
    """
    symbol_base, maturity_days = extract_instrument_spec(symbol)
    # data prep
    instrument_df = (
        instrument_df[
            (instrument_df["instrument_class"] == "F")
            & (instrument_df["raw_symbol"].str.startswith(symbol_base))
        ]
        .sort_values("expiration")
        .copy()
    )

    # start building roll spec
    roll_spec_results = []
    roll_spec = None

    dates = pd.date_range(start=start, end=end, inclusive="left").tz_localize("UTC")
    for d in dates:
        target_maturity = d + datetime.timedelta(days=maturity_days)
        df = instrument_df[instrument_df["ts_recv"] <= d]
        p_candidates = df[df["expiration"] <= target_maturity]
        n_candidates = df[df["expiration"] >= target_maturity]

        p = p_candidates.loc[p_candidates["expiration"].idxmax()]
        n = n_candidates.loc[n_candidates["expiration"].idxmin()]

        if roll_spec is not None and (
            roll_spec["p"] == str(p["instrument_id"])
            and roll_spec["n"] == str(n["instrument_id"])
        ):
            roll_spec["d1"] = min(
                (d + datetime.timedelta(days=1)).date().isoformat(), end.isoformat()
            )
        else:
            if roll_spec is not None:
                roll_spec_results.append(roll_spec)
            roll_spec = {
                "d0": d.date().isoformat(),
                "d1": min(
                    (d + datetime.timedelta(days=1)).date().isoformat(), end.isoformat()
                ),
                "p": str(p["instrument_id"]),
                "n": str(n["instrument_id"]),
            }

    if roll_spec is not None:
        roll_spec_results.append(roll_spec)

    return roll_spec_results


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    all_data: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    """Create a constant maturity futures price series by splicing contracts.

    Combines pairs of futures contracts according to roll_spec to create a continuous
    price series with constant time to maturity. Uses time-weighted averaging between
    the previous and next contracts based on their relative distances to the target
    maturity date.

    Args:
        symbol (str): Constant maturity symbol (e.g. "SR3.cm.182")
        roll_spec (list[dict]): Roll specifications from get_roll_spec()
        all_data (pd.DataFrame): Price data with columns: instrument_id, date_col, price_col, expiration
        date_col (str): Name of the datetime column in all_data
        price_col (str): Name of the price column in all_data

    Returns:
        pd.DataFrame: Spliced continuous price series with columns:
            date_col: Trading dates
            pre_price: Previous contract's price
            pre_id: Previous contract ID
            pre_expiration: Previous contract expiry date
            next_price: Next contract's price
            next_id: Next contract ID
            next_expiration: Next contract expiry date
            pre_weight: Weight applied to previous contract (0-1)
            symbol: Resulting constant maturity price
    """
    _, maturity_days = extract_instrument_spec(symbol)
    maturity_days = pd.Timedelta(days=maturity_days)
    # get unique instrument id and expirations
    df = all_data.reset_index().drop(columns=["level_0", "level_1"])
    expirations = (
        df[["instrument_id", "expiration"]]
        .drop_duplicates()
        .set_index("instrument_id")["expiration"]
    )
    df["instrument_id"] = df["instrument_id"].astype(str)

    all_splices = []
    for r in roll_spec:
        df_sub = df[
            df[date_col].between(r["d0"], r["d1"], inclusive="left")
            & df["instrument_id"].isin([r["p"], r["n"]])
        ]
        # merge in prices
        pre_df = df_sub[df_sub["instrument_id"] == r["p"]][
            [date_col, price_col]
        ].drop_duplicates(subset=date_col)
        next_df = df_sub[df_sub["instrument_id"] == r["n"]][
            [date_col, price_col]
        ].drop_duplicates(subset=date_col)
        pre_df = pre_df.rename(columns={price_col: "pre_price"})
        next_df = next_df.rename(columns={price_col: "next_price"})
        splice = pd.merge(pre_df, next_df, on=date_col, how="outer", sort=True)
        # add ids
        splice["pre_id"] = int(r["p"])
        splice["next_id"] = int(r["n"])
        # add expirations
        splice["pre_expiration"] = expirations[int(r["p"])]
        splice["next_expiration"] = expirations[int(r["n"])]
        # compute weights and constant maturity price
        splice["pre_weight"] = (
            splice["next_expiration"] - (splice[date_col] + maturity_days)
        ) / (splice["next_expiration"] - splice["pre_expiration"])
        splice[symbol] = (
            splice["pre_weight"] * splice["pre_price"]
            + (1 - splice["pre_weight"]) * splice["next_price"]
        )
        all_splices.append(splice)

    all_splices = pd.concat(all_splices).reset_index(drop=True)
    all_splices = all_splices[
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
    return all_splices
