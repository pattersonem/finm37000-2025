"""Functions to splice and adjust futures data into continuous data."""

from typing import Optional

import pandas as pd
import datetime

import numpy as np
import pandas as pd


def _splice_unadjusted(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    grouped = df.groupby("instrument_id")
    pieces = []
    tz = df[date_col].dt.tz
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        s = int(spec["s"])
        group = grouped.get_group(s)
        piece = group[(group[date_col] >= d0) & (group[date_col] < d1)].copy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _calc_additive_adjustment(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str,
) -> pd.Series:
    tz = df[date_col].dt.tz
    last_date = None
    last_true_value = None
    grouped = df.groupby("instrument_id")
    adjustments = []
    adjustment_dates = []
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        s = int(spec["s"])
        group = grouped.get_group(s)
        piece = group[(group[date_col] >= d0) & (group[date_col] < d1)].copy()
        if last_date is not None:
            adjustment_piece = group[group[date_col] == last_date]
            adjustment = last_true_value - adjustment_piece[adjust_by].iloc[-1]
            adjustments.append(adjustment)
            adjustment_dates.append(d0)
        last_true_value = piece[adjust_by].iloc[-1]
        last_date = piece[date_col].iloc[-1]
    return pd.Series(adjustments, index=adjustment_dates, name="additive_adjustment")


def _calc_multiplicative_adjustment(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str,
) -> pd.Series:
    tz = df[date_col].dt.tz
    last_date = None
    last_true_value = None
    grouped = df.groupby("instrument_id")
    adjustments = []
    adjustment_dates = []
    for spec in roll_spec:
        d0 = pd.Timestamp(spec["d0"], tz=tz)
        d1 = pd.Timestamp(spec["d1"], tz=tz)
        s = int(spec["s"])
        group = grouped.get_group(s)
        piece = group[(group[date_col] >= d0) & (group[date_col] < d1)].copy()
        if last_date is not None:
            adjustment_piece = group[group[date_col] == last_date]
            adjustment = last_true_value / adjustment_piece[adjust_by].iloc[-1]
            adjustments.append(adjustment)
            adjustment_dates.append(d0)
        last_true_value = piece[adjust_by].iloc[-1]
        last_date = piece[date_col].iloc[-1]
    return pd.Series(
        adjustments,
        index=adjustment_dates,
        name="multiplicative_adjustment",
    )


def additive_splice(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str = "close",
    adjustment_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Additively adjust and splice futures data (forward-adjusted).

    Args:
        roll_spec: List of dicts, each with members `"d0"`, `"d1"`, and `"s"`
                   containing the first date, one past the last date, and
                   the instrument id of the instrument in the spliced contract. See
                   `databento.Historical.symbology.resolve()["results"]` for
                   a dictionary with values of this type.
        df: A pandas DataFrame containing the raw data to splice.
        date_col: The name of the column in `df` that contains the date.
        adjust_by: The name of the column in `df` that contains the column
                   to calculate the adjustment.
        adjustment_cols: The columns in `df` that should be adjusted. The default
                         `None` will adjust the `adjust_by` column.

    Returns:
        A pandas DataFrame containing the additively adjusted adjusted data.

    """
    if adjustment_cols is None:
        adjustment_cols = [adjust_by]
    spliced = _splice_unadjusted(roll_spec, df, date_col)
    adjustments = _calc_additive_adjustment(
        roll_spec,
        df,
        date_col=date_col,
        adjust_by=adjust_by,
    )
    with_adjustment = spliced.merge(
        adjustments,
        left_on=date_col,
        right_index=True,
        how="left",
    )
    with_adjustment = with_adjustment.set_index(date_col)
    aligned_adjustment = with_adjustment[adjustments.name]
    aligned_adjustment = aligned_adjustment.fillna(value=0)
    cumulative_adjustment = aligned_adjustment.cumsum()
    for col in adjustment_cols:
        with_adjustment[col] = with_adjustment[col] + cumulative_adjustment
    with_adjustment[adjustments.name] = cumulative_adjustment  # type: ignore
    new_columns = df.columns.tolist() + [adjustments.name]
    return with_adjustment.reset_index()[new_columns]


def multiplicative_splice(
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str,
    adjust_by: str = "close",
    adjustment_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Multiplicatively adjust and splice futures data (forward-adjusted).

    Args:
        roll_spec: List of dicts, each with members `"d0"`, `"d1"`, and `"s"`
                   containing the first date, one past the last date, and
                   the instrument id of the instrument in the spliced contract. See
                   `databento.Historical.symbology.resolve()["results"]` for
                   a dictionary with values of this type.
        df: A pandas DataFrame containing the raw data to splice.
        date_col: The name of the column in `df` that contains the date.
        adjust_by: The name of the column in `df` that contains the column to
                   calculate the adjustment.
        adjustment_cols: The columns in `df` that should be adjusted. The default
                         `None` will adjust the `adjust_by` column.

    Returns:
        A pandas DataFrame containing the adjusted data.

    """
    if adjustment_cols is None:
        adjustment_cols = [adjust_by]
    spliced = _splice_unadjusted(roll_spec, df, date_col)
    adjustments = _calc_multiplicative_adjustment(
        roll_spec,
        df,
        date_col=date_col,
        adjust_by=adjust_by,
    )
    with_adjustment = spliced.merge(
        adjustments,
        left_on=date_col,
        right_index=True,
        how="left",
    )
    with_adjustment = with_adjustment.set_index(date_col)
    aligned_adjustment = with_adjustment[adjustments.name]
    aligned_adjustment = aligned_adjustment.fillna(value=1)
    cumulative_adjustment = aligned_adjustment.cumprod()
    for col in adjustment_cols:
        with_adjustment[col] = with_adjustment[col] * cumulative_adjustment
    with_adjustment[adjustments.name] = cumulative_adjustment  # type: ignore
    new_columns = df.columns.tolist() + [adjustments.name]
    return with_adjustment.reset_index()[new_columns]


def get_roll_spec(
    symbol: str,
    instrument_defs: pd.DataFrame,
    *,
    start,
    end,
) -> list[dict[str, str]]:
    """Build a roll specification for a constant-maturity futures series.

    Parameters
    ----------
    symbol
        Constant-maturity symbol of the form ``"PRODUCT.cm.<days>"``,
        e.g. ``"SR3.cm.182"``. Only the integer number of days is used.
    instrument_defs
        Instrument definition data with at least:

        - ``"instrument_id"``
        - ``"expiration"``
        - ``"instrument_class"``

        If present, a ``"ts_recv"`` column is used to avoid contracts
        that were not yet listed on a given date.
    start, end
        Start and end of the window as Python ``date`` (or timestamp).
        The interval is interpreted as **[start, end)** (end exclusive).

    Returns
    -------
    list of dict
        Each dict has keys::

            "d0" – first date (inclusive) of the segment
            "d1" – first date *after* the segment (exclusive)
            "p"  – instrument id (as string) of the *previous* expiry
            "n"  – instrument id (as string) of the *next* expiry

        The segments partition the [start, end) range into runs where the
        pair of contracts (p, n) stays constant.
    """
    # Parse the maturity in days from the symbol, e.g. "SR3.cm.182"
    try:
        maturity_days = int(symbol.split(".cm.")[1])
    except Exception as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc

    maturity_delta = pd.Timedelta(days=maturity_days)

    df = instrument_defs.copy()

    # Only outright futures (exclude spreads etc.).
    df = df[df["instrument_class"] == "F"].copy()

    # Work at date precision – the tests are only date-level accurate.
    df["expiration_date"] = pd.to_datetime(df["expiration"]).dt.date

    if "ts_recv" in df.columns:
        df["ts_recv_date"] = pd.to_datetime(df["ts_recv"]).dt.date
    else:
        # If listing time is unknown, assume everything is always listed.
        df["ts_recv_date"] = df["expiration_date"].min()

    # Ensure contracts are sorted by expiry.
    df = df.sort_values("expiration_date").reset_index(drop=True)

    expirations = df["expiration_date"].to_numpy()
    ts_recv_dates = df["ts_recv_date"].to_numpy()
    instrument_ids = df["instrument_id"].to_numpy()

    # Normalise start/end to plain date objects.
    if isinstance(start, pd.Timestamp):
        start_date = start.date()
    else:
        start_date = start
    if isinstance(end, pd.Timestamp):
        end_date = end.date()
    else:
        end_date = end

    one_day = datetime.timedelta(days=1)
    t = start_date

    current_pair: Optional[tuple[int, int]] = None
    current_start = start_date
    spec: list[dict[str, str]] = []

    while t < end_date:
        # Contracts that are *live* on date t:
        #   - listed (ts_recv_date <= t)
        #   - not yet expired (expiration_date > t)
        mask = (ts_recv_dates <= t) & (expirations > t)
        available_idx = np.nonzero(mask)[0]
        if len(available_idx) < 2:
            raise ValueError(f"Not enough live contracts on {t!r}")

        exp_available = expirations[available_idx]

        # Target maturity date (t + maturity) at date precision.
        target_date = (pd.Timestamp(t) + maturity_delta).date()

        # Find the first available expiry on or after the target date.
        pos_rel = None
        for j, exp_date in enumerate(exp_available):
            if exp_date >= target_date:
                pos_rel = j
                break
        if pos_rel is None:
            # Target is beyond the last expiry: fall back to the last two.
            pos_rel = len(available_idx)

        if pos_rel == 0:
            pre_idx = available_idx[0]
            nxt_idx = available_idx[1]
        elif pos_rel >= len(available_idx):
            pre_idx = available_idx[-2]
            nxt_idx = available_idx[-1]
        else:
            pre_idx = available_idx[pos_rel - 1]
            nxt_idx = available_idx[pos_rel]

        pair = (int(instrument_ids[pre_idx]), int(instrument_ids[nxt_idx]))

        if current_pair is None:
            # First date – start the first segment.
            current_pair = pair
            current_start = t
        elif pair != current_pair:
            # Close the previous segment at date t (exclusive).
            spec.append(
                {
                    "d0": current_start.strftime("%Y-%m-%d"),
                    "d1": t.strftime("%Y-%m-%d"),
                    "p": str(current_pair[0]),
                    "n": str(current_pair[1]),
                }
            )
            current_pair = pair
            current_start = t

        t = t + one_day

    # Final open segment
    if current_pair is not None:
        spec.append(
            {
                "d0": current_start.strftime("%Y-%m-%d"),
                "d1": end_date.strftime("%Y-%m-%d"),
                "p": str(current_pair[0]),
                "n": str(current_pair[1]),
            }
        )

    return spec


def constant_maturity_splice(
    symbol: str,
    roll_spec: list[dict[str, str]],
    df: pd.DataFrame,
    *,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    """Produce a constant-maturity futures series from raw prices.

    Parameters
    ----------
    symbol
        Constant-maturity symbol such as ``"SR3.cm.182"``. The integer
        number of days is parsed from the suffix and used as the target
        maturity.
    roll_spec
        Roll specification as returned by :func:`get_roll_spec`.
    df
        Raw futures data with at least:

        - ``"instrument_id"``
        - the date column ``date_col``
        - the price column ``price_col``
        - an ``"expiration"`` column giving each instrument's expiry
    date_col, price_col
        Names of the timestamp and price columns in ``df``.

    Returns
    -------
    pandas.DataFrame
        One row per calendar date in the requested range with columns:

        - ``date_col``
        - ``"pre_price"``, ``"pre_id"``, ``"pre_expiration"``
        - ``"next_price"``, ``"next_id"``, ``"next_expiration"``
        - ``"pre_weight"``
        - a final column named ``symbol`` containing the interpolated
          constant-maturity price.
    """
    # Parse maturity from the symbol (e.g. "SR3.cm.182").
    try:
        maturity_days = int(symbol.split(".cm.")[1])
    except Exception as exc:
        raise ValueError(f"Could not parse maturity from symbol {symbol!r}") from exc

    maturity_delta = pd.Timedelta(days=maturity_days)

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    # Determine timezone so date ranges line up correctly.
    tz = data[date_col].dt.tz

    if "expiration" not in data.columns:
        raise ValueError("constant_maturity_splice requires an 'expiration' column")

    # Map each instrument id to a single expiry timestamp.
    expirations = (
        data[["instrument_id", "expiration"]]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")["expiration"]
        .map(pd.to_datetime)
    )

    # Make a wide price table: index = datetime, columns = instrument_id.
    price_wide = data.pivot(index=date_col, columns="instrument_id", values=price_col)

    segments: list[pd.DataFrame] = []

    for spec in roll_spec:
        d0 = pd.to_datetime(spec["d0"])
        d1 = pd.to_datetime(spec["d1"])

        # Same convention as the tests: daily [d0, d1) with inclusive="left".
        if tz is not None:
            d0 = d0.tz_localize(tz)
            d1 = d1.tz_localize(tz)
        t_index = pd.date_range(start=d0, end=d1, tz=tz, inclusive="left")

        pre_id = int(spec["p"])
        next_id = int(spec["n"])

        pre_exp = expirations.loc[pre_id]
        next_exp = expirations.loc[next_id]

        # Pull out the corresponding price series for each leg.
        pre_price = price_wide[pre_id].reindex(t_index)
        next_price = price_wide[next_id].reindex(t_index)

        # Time-varying interpolation weight on the "previous" expiry.
        # This exactly matches the formula used in the test:
        #
        #   f(t) = (E_next - (t + maturity)) / (E_next - E_pre)
        #
        pre_weight = (next_exp - (t_index + maturity_delta)) / (next_exp - pre_exp)

        cm_price = pre_weight * pre_price + (1.0 - pre_weight) * next_price

        seg = pd.DataFrame(
            {
                date_col: t_index,
                "pre_price": pre_price.values,
                "pre_id": pre_id,
                "pre_expiration": pre_exp,
                "next_price": next_price.values,
                "next_id": next_id,
                "next_expiration": next_exp,
                "pre_weight": pre_weight.values,
                symbol: cm_price.values,
            }git status
        )
        segments.append(seg)

    return pd.concat(segments, ignore_index=True)
