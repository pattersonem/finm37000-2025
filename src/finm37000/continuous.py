"""Functions to splice and adjust futures data into continuous data."""

from datetime import date
from typing import Optional

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
    return pd.Series(adjustments, index=adjustment_dates, name="additive_adjustment")  # type: ignore[no-any-return]


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
    return pd.Series(  # type: ignore[no-any-return]
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
    with_adjustment[adjustments.name] = cumulative_adjustment
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
    with_adjustment[adjustments.name] = cumulative_adjustment
    new_columns = df.columns.tolist() + [adjustments.name]
    return with_adjustment.reset_index()[new_columns]

def parse_product_and_maturity(symbol: str) -> tuple[str, int]:
    parsed_symbol = symbol.split('.')
    product = parsed_symbol[0]
    maturity_days = int(parsed_symbol[2])
    return product, maturity_days

def get_roll_spec(symbol: str, instrument_df: pd.DataFrame, start: date, end: date) -> list[dict]:
    """Compute roll segmentation for a constant-maturity contract.

    The function determines contiguous date segments on [start, end) for which
    a pair of futures contracts (pre, next) should be used to construct a
    constant-maturity series. For a given date d in a segment the chosen pair
    (p, n) satisfies::

        expiration_p <= d + maturity_days < expiration_n

    Additionally, instrument availability (``ts_recv``) is respected: an
    instrument whose ``ts_recv`` is after the segment date is not selected as
    ``p`` or ``n`` until it becomes live.

    Parameters
    ----------
    symbol:
        Constant-maturity symbol in the form ``<PRODUCT>.cm.<days>`` (e.g.
        ``"SR3.cm.182"``). The product prefix and maturity days are parsed
        from this string.
    instrument_df:
        DataFrame containing instrument definitions with at least the
        following columns: ``instrument_id``, ``raw_symbol``, ``expiration``,
        ``instrument_class``, and ``ts_recv``. ``expiration`` and ``ts_recv``
        should be timezone-aware datetimes (the function converts them to
        dates internally).
    start, end:
        Date-like start (inclusive) and end (exclusive) bounds for the
        segmentation. These are python ``date`` objects in the calling tests.

    Returns
    -------
    list[dict]
        A list of dictionaries, each with keys ``d0`` (inclusive start date
        string YYYY-MM-DD), ``d1`` (exclusive end date string YYYY-MM-DD),
        ``p`` (instrument id of the pre contract as string) and ``n`` (instrument
        id of the next contract as string). The segments are ordered by time
        and cover [start, end) (or a prefix of it if there are not enough
        instruments).

    Notes / edge cases
    -------------------
    - If a candidate next contract is not yet live at a given d (its
      ``ts_recv`` > d), the function will skip it until its ``ts_recv`` date
      and may use a later contract as ``n`` in the interim.  The earliest
      event (expiration-driven cutoff or instrument availability) that causes
      the pairing to change becomes the segment boundary.
    - Date arithmetic uses whole days; returned ``d0``/``d1`` are ISO date
      strings (``YYYY-MM-DD``) suitable for building left-inclusive ranges.
    """

    roll_periods = []

    # Filter instrument_df for futures only and within the date range
    product, maturity_days = parse_product_and_maturity(symbol)

    futures = instrument_df.copy()
    futures['ts_recv'] = futures['ts_recv'].dt.date
    futures['expiration'] = futures['expiration'].dt.date
    futures = (
        futures[
            (futures['raw_symbol'].str.startswith(product)) &
            (futures['instrument_class'] == 'F') & 
            (futures['ts_recv'] <= end)
        ]
    ).sort_values('expiration')

    d0 = start
    d1 = start

    while d1 < end:
        period = {}
        period['d0'] = d0.isoformat()

        # add maturity_days to d0 to get "target"
        target = d0 + pd.Timedelta(days=maturity_days)
        far_futures = futures[(futures['expiration'] >= target)]
        if far_futures.empty:
            break
        far_future = far_futures[far_futures['ts_recv'] <= d0].iloc[0]

        near_futures = futures[(futures['expiration'] < target)]
        if near_futures.empty:
            break
        near_future = near_futures.iloc[-1]

        # expiration cut off
        d1 = min(end, far_future['expiration'] - pd.Timedelta(days=maturity_days-1))

        # availability cut off
        newly_available = far_futures[
            (far_futures['instrument_id'] < far_future['instrument_id']) &
            (far_futures['ts_recv'] > d0) &
            (far_futures['ts_recv'] < d1)
        ]
        if not newly_available.empty:
            d1 = min(d1, newly_available['ts_recv'].min())

        period['d1'] = d1.isoformat()
        period['p'] = str(near_future['instrument_id'])
        period['n'] = str(far_future['instrument_id'])
        roll_periods.append(period)

        d0 = d1

    return roll_periods


def constant_maturity_splice(symbol: str, roll_spec: list[dict], all_data: pd.DataFrame, date_col: str = "datetime", price_col: str = "price") -> pd.DataFrame:
    """Create a constant-maturity time series by linear interpolation between legs.

    For each segment in ``roll_spec`` this function joins price rows for the
    two instruments (pre and next) over the segment's date range and computes
    a time-weighted price that corresponds to a contract with the requested
    days-to-maturity.

    The implementation expects ``all_data`` to contain per-instrument rows
    with at least the following columns: ``instrument_id``, the ``date_col``
    (typically a timezone-aware datetime), ``price_col``, and ``expiration``.

    Parameters
    ----------
    symbol:
        Constant-maturity symbol like ``"SR3.cm.182"``. The maturity days are
        parsed from this string and used when computing interpolation weights.
    roll_spec:
        List of segments produced by :func:`get_roll_spec`. Each segment must
        include ``d0``, ``d1``, ``p`` (pre instrument id) and ``n`` (next
        instrument id). Dates in the spec are ISO date strings (``YYYY-MM-DD``)
        and represent left-inclusive / right-exclusive intervals.
    all_data:
        DataFrame containing raw per-instrument price series. Rows are grouped
        by ``instrument_id``; the function will select rows whose
        ``date_col`` is in [d0, d1) for each segment.
    date_col, price_col:
        Column names in ``all_data`` for the datetime index and the price.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing, for each segment, the following
        columns: ``date_col``, ``pre_price``, ``pre_id``, ``pre_expiration``,
        ``next_price``, ``next_id``, ``next_expiration``, ``pre_weight``, and
        a column named by ``symbol`` that contains the interpolated constant
        maturity price.

    Notes
    -----
    - The function performs an inner merge between the pre and next instrument
      rows on the ``date_col``; dates without both legs are dropped for that
      segment.
    - Interpolation weight ``pre_weight`` is computed as::

        (next_expiration - (date + maturity_days)) / (next_expiration - pre_expiration)

      which mirrors typical linear time interpolation to a target maturity.
    - If a segment has missing data for either leg the function raises
      ``ValueError``.
    """

    _, maturity_days = parse_product_and_maturity(symbol)
    maturity_timedelta = pd.Timedelta(days=maturity_days)

    out_cols = [date_col, "pre_price", "pre_id", "pre_expiration",
                "next_price", "next_id", "next_expiration", "pre_weight", symbol]

    data_by_instr = all_data.copy().groupby("instrument_id")

    segments = []
    for spec in roll_spec:
        d0, d1 = spec['d0'], spec['d1']
        pre = int(spec['p'])
        pre_data_raw = data_by_instr.get_group(pre)
        if pre_data_raw.empty:
            raise ValueError(f"No data for instrument {pre}")

        nex = int(spec['n'])
        nex_data_raw = data_by_instr.get_group(nex)
        if nex_data_raw.empty:
            raise ValueError(f"No data for instrument {nex}")

        pre_data = (
            pre_data_raw[(pre_data_raw[date_col] >= d0) & (pre_data_raw[date_col] < d1)].copy()
        ).rename(columns={
            price_col: "pre_price",
            "instrument_id": "pre_id",
            "expiration": "pre_expiration"
        })

        nex_data = (
            nex_data_raw[(nex_data_raw[date_col] >= d0) & (nex_data_raw[date_col] < d1)].copy()
        ).rename(columns={
            price_col: "next_price",
            "instrument_id": "next_id",
            "expiration": "next_expiration"}
        )

        merged_data = pre_data.merge(
            nex_data,
            on=date_col,
            how='inner'
        )

        merged_data["pre_weight"] = (
            (merged_data['next_expiration'] - (merged_data[date_col] + maturity_timedelta)) /
            (merged_data['next_expiration'] - merged_data['pre_expiration'])
        )
        merged_data[symbol] = (
            merged_data["pre_weight"] * merged_data["pre_price"] +
                (1 - merged_data["pre_weight"]) * merged_data["next_price"]
        )

        segments.append(merged_data[out_cols])

    return pd.concat(segments, ignore_index=True)