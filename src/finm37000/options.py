"""Simple option pricing implementations."""

from enum import Enum
from typing import Iterable, Optional, Sequence, cast

import databento as db
import numpy as np
import pandas as pd
import QuantLib as ql  # noqa: N813
from scipy.optimize import root_scalar
from scipy.stats import norm


class OptionType(Enum):
    """Enumeration of option types."""

    CALL = "Call"
    PUT = "Put"


def calc_black_scholes(  # noqa: PLR0913
    S: float | np.ndarray,  # noqa: N803
    K: float | np.ndarray,  # noqa: N803
    T: float | np.ndarray,  # noqa: N803
    vol: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    option_type: OptionType,
) -> float | np.ndarray:
    """Compute Black-Scholes model for European options on equities."""
    f = S * np.exp((r - q) * T)
    d1 = (np.log(f / K) + (vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    discount_factor = np.exp(-r * T)

    cp = 1 if option_type == OptionType.CALL else -1
    return cast(
        "float", discount_factor * cp * (f * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))
    )


def calc_black_scholes_numerical_rho(  # noqa: PLR0913
    S: float | np.ndarray,  # noqa: N803
    K: float | np.ndarray,  # noqa: N803
    T: float | np.ndarray,  # noqa: N803
    vol: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    option_type: OptionType,
    dr: float | np.ndarray = 0.0001,
) -> float | np.ndarray:
    """Compute Black-Scholes rho for European options on equities."""
    up_price = calc_black_scholes(
        S=S, K=K, r=r + dr, T=T, q=q, vol=vol, option_type=option_type
    )
    down_price = calc_black_scholes(
        S=S, K=K, r=r - dr, T=T, q=q, vol=vol, option_type=option_type
    )
    return (up_price - down_price) / (2.0 * dr)


def calc_black(  # noqa: PLR0913
    F: float | np.ndarray,  # noqa: N803
    K: float | np.ndarray,  # noqa: N803
    T: float | np.ndarray,  # noqa: N803
    vol: float | np.ndarray,
    r: float | np.ndarray,
    option_type: OptionType,
) -> float | np.ndarray:
    """Compute Black 76 model for European options on forwards."""
    d1 = (np.log(F / K) + (vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    discount_factor = np.exp(-r * T)

    cp = 1 if option_type == OptionType.CALL else -1
    return cast(
        "float", discount_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))
    )


def calc_black_one_day_theta(  # noqa: PLR0913
    F: float | np.ndarray,  # noqa: N803
    K: float | np.ndarray,  # noqa: N803
    T: float | np.ndarray,  # noqa: N803
    vol: float | np.ndarray,
    r: float | np.ndarray,
    option_type: OptionType,
    dt: float,
) -> float | np.ndarray:
    """Compute Black 76 model for European options on forwards."""
    price_dt = calc_black(
        F=F,
        K=K,
        r=r,
        T=T - dt,
        vol=vol,
        option_type=option_type,
    )
    price = calc_black(
        F=F,
        K=K,
        r=r,
        T=T,
        vol=vol,
        option_type=option_type,
    )
    return price - price_dt


def calc_black_numerical_theta(  # noqa: PLR0913
    F: float | np.ndarray,  # noqa: N803
    K: float | np.ndarray,  # noqa: N803
    T: float | np.ndarray,  # noqa: N803
    vol: float | np.ndarray,
    r: float | np.ndarray,
    option_type: OptionType,
    dt: float,
) -> float | np.ndarray:
    """Use Black model to numerically compute theta."""
    return (
        calc_black_one_day_theta(
            F=F,
            K=K,
            T=T,
            vol=vol,
            r=r,
            option_type=option_type,
            dt=dt,
        )
        / dt
    )


def calc_american_price(  # type: ignore[no-untyped-def] # noqa: ANN201, PLR0913
    future,  # noqa: ANN001
    strike,  # noqa: ANN001
    t,  # noqa: ANN001
    vol,  # noqa: ANN001
    r,  # noqa: ANN001
    option_type,  # noqa: ANN001
):
    """Use QuantLib's Adesi/Whaley model to calculate American prices on futures."""
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot = ql.SimpleQuote(0.0)
    vol_q = ql.SimpleQuote(0.0)

    spot_h = ql.QuoteHandle(spot)
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_q), day_count)
    )

    r_h = ql.RelinkableYieldTermStructureHandle()
    q_h = ql.RelinkableYieldTermStructureHandle()

    dummy_curve = ql.FlatForward(today, 0.0, day_count)
    r_h.linkTo(dummy_curve)
    q_h.linkTo(dummy_curve)

    process = ql.GeneralizedBlackScholesProcess(spot_h, q_h, r_h, vol_h)
    engine = ql.BaroneAdesiWhaleyApproximationEngine(process)

    def calc_one(future_, strike_, t_, vol_, r_, opt_):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN202, PLR0913
        if np.isnan(vol_):
            return np.nan
        spot.setValue(float(future_))
        vol_q.setValue(float(vol_))

        curve = ql.FlatForward(today, r_, day_count)
        r_h.linkTo(curve)
        q_h.linkTo(curve)

        if opt_ == "C":
            opt_type = ql.Option.Call
        elif opt_ == "P":
            opt_type = ql.Option.Put
        else:
            msg = f"Invalid option type: {opt_}"
            raise ValueError(msg)

        payoff = ql.PlainVanillaPayoff(opt_type, strike_)

        maturity = today + int(365 * t_)
        exercise = ql.AmericanExercise(today, maturity)

        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        return option.NPV()

    future_, strike_, t_, vol_, r_, opt_ = np.broadcast_arrays(
        future, strike, t, vol, r, option_type
    )
    flat = zip(
        future_.ravel(),
        strike_.ravel(),
        t_.ravel(),
        vol_.ravel(),
        r_.ravel(),
        opt_.ravel(),
    )
    out = np.array([calc_one(*row) for row in flat])

    return out.reshape(future_.shape)


def calc_american_greeks(  # type: ignore[no-untyped-def] # noqa: ANN201, PLR0913
    future,  # noqa: ANN001
    strike,  # noqa: ANN001
    t,  # noqa: ANN001
    vol,  # noqa: ANN001
    r,  # noqa: ANN001
    option_type,  # noqa: ANN001
):
    """Use QuantLib's Adesi/Whaley model to calculate American greeks on futures."""
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot = ql.SimpleQuote(0.0)
    vol_q = ql.SimpleQuote(0.0)

    spot_h = ql.QuoteHandle(spot)
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_q), day_count)
    )

    r_h = ql.RelinkableYieldTermStructureHandle()
    q_h = ql.RelinkableYieldTermStructureHandle()

    dummy_curve = ql.FlatForward(today, 0.0, day_count)
    r_h.linkTo(dummy_curve)
    q_h.linkTo(dummy_curve)

    process = ql.GeneralizedBlackScholesProcess(spot_h, q_h, r_h, vol_h)
    engine = ql.BaroneAdesiWhaleyApproximationEngine(process)

    def calc_one(future_, strike_, t_, vol_, r_, opt_):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN202, PLR0913
        if np.isnan(vol_):
            return np.nan
        spot.setValue(float(future_))
        vol_q.setValue(float(vol_))

        curve = ql.FlatForward(today, r_, day_count)
        r_h.linkTo(curve)
        q_h.linkTo(curve)

        if opt_ == "C":
            opt_type = ql.Option.Call
        elif opt_ == "P":
            opt_type = ql.Option.Put
        else:
            msg = f"Invalid option type: {opt_}"
            raise ValueError(msg)

        payoff = ql.PlainVanillaPayoff(opt_type, strike_)

        maturity = today + int(365 * t_)
        exercise = ql.AmericanExercise(today, maturity)

        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        # Sadly, deltaForward not provided
        return (
            option.NPV(),
            option.delta(),
            option.vega(),
            option.theta(),
            option.rho() - option.dividendRho(),
        )

    future_, strike_, t_, vol_, r_, opt_ = np.broadcast_arrays(
        future, strike, t, vol, r, option_type
    )
    flat = zip(
        future_.ravel(),
        strike_.ravel(),
        t_.ravel(),
        vol_.ravel(),
        r_.ravel(),
        opt_.ravel(),
    )
    out = np.array([calc_one(*row) for row in flat])

    greek_shape = [future_.shape[0], 5]
    return out.reshape(greek_shape)


def calc_numerical_delta(future, strike, r, t, vol, option_type, dfuture):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN201, PLR0913
    """Compute non-skew adjusted American option deltas."""
    up_price = calc_american_price(
        future=future + dfuture,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
    )
    down_price = calc_american_price(
        future=future - dfuture,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
    )
    return (up_price - down_price) / (2.0 * dfuture)


def calc_one_day_theta(future, strike, r, t, vol, option_type, dt=1 / 365):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN201, PLR0913
    """Compute non-skew adjusted American option one-day time decay.

    This does not seem reliable as implemented, presumably due
    to the granularity of time allowed by the QuantLib implementation.
    """
    price_dt = calc_american_price(
        future=future,
        strike=strike,
        r=r,
        t=t - dt,
        vol=vol,
        option_type=option_type,
    )
    price = calc_american_price(
        future=future,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
    )
    return price - price_dt


def calc_numerical_theta(future, strike, r, t, vol, option_type, dt=1 / 365):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN201, PLR0913
    """Compute non-skew adjusted American option thetas.

    This does not seem reliable as implemented, presumably due
    to the granularity of time allowed by the QuantLib implementation.
    """
    return (
        calc_one_day_theta(
            future=future,
            strike=strike,
            r=r,
            t=t,
            vol=vol,
            option_type=option_type,
            dt=dt,
        )
        / dt
    )


def calc_numerical_vega(future, strike, r, t, vol, option_type, dvol=0.01):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN201, PLR0913
    """Compute non-skew adjusted American option vegas."""
    up_price = calc_american_price(
        future=future, strike=strike, r=r, t=t, vol=vol + dvol, option_type=option_type
    )
    down_price = calc_american_price(
        future=future, strike=strike, r=r, t=t, vol=vol - dvol, option_type=option_type
    )
    return (up_price - down_price) / (2.0 * dvol)


def calc_numerical_rho(future, strike, r, t, vol, option_type, dr=0.0001):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN201, PLR0913
    """Compute non-skew adjusted American option rho."""
    up_price = calc_american_price(
        future=future, strike=strike, r=r + dr, t=t, vol=vol, option_type=option_type
    )
    down_price = calc_american_price(
        future=future, strike=strike, r=r - dr, t=t, vol=vol, option_type=option_type
    )
    return (up_price - down_price) / (2.0 * dr)


def get_options_chain(  # noqa: PLR0913
    parent: str,
    start: pd.Timestamp,
    client: db.Historical,
    underlying: Optional[str],
    days_per_year: float = 365.0,
    instrument_class: Optional[Sequence[str]] = ("C", "P"),
) -> pd.DataFrame:
    """Retrieve the definitions of shared-parent options contracts.

    Parameters
    ----------
    parent : str
        The parent symbol, such as ES.
    start : pd.Timestamp
        The date to obtain the definitions for.
    client: db.Historical
        The Historical client to retrieve databento data.
    underlying : str
        The underlying contract for the option. Default of None
        will return options with any underlying. Note that spreads
        do not come with the `"underlying"` column populated, so
        spreads require this argument to be None.
    days_per_year: float
        The number of days to use to normalize day counts.
    instrument_class: Sequence[str]
        Filter for which instrument type to return. Calls (`"C"`),
        puts (`"P"`), and others such as spreads (`"T"`). Default
        of None will return all definitions.

    Returns:
    -------
    pd.DataFrame

    """
    options_def = client.timeseries.get_range(
        dataset=db.Dataset.GLBX_MDP3,
        schema="definition",
        symbols=f"{parent}.OPT",
        stype_in="parent",
        start=start.date(),
    )

    df = options_def.to_df()
    if underlying is not None:
        df = df[df["underlying"] == underlying]
    if instrument_class is not None:
        df = df[df["instrument_class"].isin(instrument_class)]
    df["years_to_expiration"] = (
        (df["expiration"] - start).dt.total_seconds() / days_per_year / 24 / 60 / 60
    )
    return df.sort_values("strike_price")


def get_top_of_book(
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: db.Historical,
) -> pd.DataFrame:
    """Get the last top-of-book and calculate midprice.

    Parameters
    ----------
    symbols : Iterable[str]
        A collection of symbols to retrieve the midprices for.
    start : pd.Timestamp
        The start time.
    end : pd.Timestamp
        The end time (exclusive).
    client: db.Historical
        The Historical client to retrieve databento data.

    Returns:
    -------
    pd.DataFrame

    """
    price_df = client.timeseries.get_range(
        dataset=db.Dataset.GLBX_MDP3,
        schema="mbp-1",
        symbols=symbols,
        start=start,
        end=end,
    ).to_df()

    price_df = price_df.groupby("symbol").last()
    price_df["bid"] = price_df["bid_px_00"]
    price_df["ask"] = price_df["ask_px_00"]
    price_df["bidq"] = price_df["bid_sz_00"]
    price_df["askq"] = price_df["ask_sz_00"]
    price_df["midprice"] = np.mean(price_df[["bid", "ask"]], axis=1)
    wt = price_df["bidq"] / (price_df["bidq"] + price_df["askq"])
    price_df["weighted_midprice"] = price_df["bid"] * (1.0 - wt) + price_df["ask"] * wt

    cols = ["bid", "ask", "midprice", "bidq", "askq", "weighted_midprice"]
    return price_df[cols]


def imply_european_vol(
    row: pd.Series,
    price_col: str = "midprice",
) -> float:
    """Find the roots of the Black-76 model by varying sigma, implied volatility.

    This function is for use with `pandas.Dataframe.apply`. Each row should contain
    a column for "strike_price", "years_to_expiration", "instrument_class", "midprice",
    and "underlying_price",

    If the optimization fails, `numpy.nan` is returned.

    Parameters
    ----------
    row : pd.Series
        A series of data to process.
    price_col : str, optional
        The name of the column in `row` that contains the price.

    Returns:
    -------
    float | numpy.nan

    """
    target = float(row[price_col])
    option_type = OptionType.CALL if row["instrument_class"] == "C" else OptionType.PUT

    def model_price(vol: float) -> float:
        return cast(
            "float",
            calc_black(
                F=row["underlying_price"],
                K=row["strike_price"],
                T=row["years_to_expiration"],
                vol=vol,
                r=row["interest_rate"],
                option_type=option_type,
            ),
        )

    def f(vol: float) -> float:
        return target - model_price(vol)

    lb = 0.00001
    ub = 4
    f_lb = f(lb)
    f_ub = f(ub)
    if f_ub * f_lb >= 0:
        lower_vol = float(model_price(lb))
        upper_vol = float(model_price(ub))
        print(
            (
                f"Cannot find {option_type} vol between "
                f"{lb=} and {ub=} "
                f"at strike {row['strike_price']}: "
                f"{lower_vol=} {target=} {upper_vol=}"
            )
        )
        print(
            (
                f"  F={row['underlying_price']} "
                f"T={row['years_to_expiration']} "
                f"r={row['interest_rate']} "
                f"mid={row['midprice']}"
            )
        )
        return np.nan
    result = root_scalar(f, bracket=[lb, ub], method="brentq")  # type: ignore[call-overload]
    if result.converged:
        return cast("float", result.root)
    print(
        f"Could not find sigma for {row['raw_symbol']} with midprice {row['midprice']}",
    )
    return np.nan


def imply_american_vols(  # noqa: PLR0913
    option_df: pd.DataFrame,
    futures_price: float,
    risk_free_rate: float,
    min_vol: float = 1e-4,
    max_vol: float = 4.0,
    max_evaluations: int = 200,
    accuracy: float = 1e-6,
    days_per_year: float = 365.0,
) -> dict[str, pd.Series]:
    """Use QuantLib's BAW model to get implied vols."""
    # --- Market setup ---
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(futures_price))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))

    def imply_vol(row: pd.Series, price_col: str = "mid") -> float:
        option_price = row[price_col]
        maturity_date = today + int(days_per_year * row.years_to_expiration)
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if row.instrument_class == "C" else ql.Option.Put,
            row.strike_price,
        )
        exercise = ql.AmericanExercise(today, maturity_date)
        process = ql.GeneralizedBlackScholesProcess(
            spot_handle,
            q_ts,
            r_ts,
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, calendar, 0.2, day_count)
            ),
        )
        engine = ql.BaroneAdesiWhaleyApproximationEngine(process)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        try:
            return cast(
                "float",
                option.impliedVolatility(
                    option_price,
                    process,
                    accuracy,
                    max_evaluations,
                    min_vol,
                    max_vol,
                ),
            )
        except RuntimeError:
            return float("nan")

    ivs = {}
    for col in ["bid", "midprice", "ask", "weighted_midprice"]:
        ivs[f"iv_{col}"] = option_df.apply(imply_vol, axis=1, price_col=col)
    return ivs
