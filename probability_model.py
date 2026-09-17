"""
Probability Model for Nifty/BankNifty Options — Edge Detection Engine
=======================================================================

Mirrors the Polymarket bot's "Model P(UP) vs Fair P(UP) vs Quote" logic,
rebuilt for Indian index options.

Two independent probability estimates are computed for every strike in
the chain, and the GAP between them is the tradeable edge:

  1. MARKET-IMPLIED probability  -> extracted from live option prices
     via the Breeden-Litzenberger (1978) relation. This is "what the
     market thinks", read directly off the option chain, with no
     Black-Scholes assumption baked in (captures skew/smile as-is).

  2. MODEL probability -> your own forecast, built from a HAR-RV
     (Heterogeneous AutoRegressive Realized Volatility) forecaster,
     converted into a probability via a lognormal terminal distribution.

     edge = model_prob - market_implied_prob

This file is a self-contained reference implementation with a synthetic
data demo at the bottom. Swap the synthetic chain/price loaders for
real broker-API data (Kite Connect / Fyers / Upstox) before live use.

Dependencies: numpy, pandas, scipy
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from typing import Optional


# ======================================================================
# 1. BREEDEN-LITZENBERGER: market-implied risk-neutral CDF from prices
# ======================================================================

@dataclass
class OptionChainSnapshot:
    """One expiry's option chain at a point in time."""
    spot: float
    strikes: np.ndarray          # sorted ascending
    call_mid_prices: np.ndarray  # mid = (bid+ask)/2, same order as strikes
    r: float                     # risk-free rate (annualized, e.g. 0.065)
    T: float                     # time to expiry in YEARS (trading-time, see note below)


def breeden_litzenberger_cdf(chain: OptionChainSnapshot,
                              smooth_factor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the risk-neutral CDF P(S_T <= K) at each strike from call prices.

    Breeden-Litzenberger (1978):
        P(S_T > K) = -e^{rT} * dC/dK
        =>  P(S_T <= K) = 1 + e^{rT} * dC/dK

    Call prices must be strictly convex & decreasing in K for a clean
    result — real market quotes are noisy, so we fit a smoothing spline
    to the call-price curve before differentiating. This is the standard
    fix; differentiating raw discrete quotes directly is too noisy to use.

    Returns
    -------
    strikes_fine : np.ndarray  (interpolated grid)
    cdf          : np.ndarray  P(S_T <= K) for each strike in strikes_fine
    """
    K = chain.strikes
    C = chain.call_mid_prices

    # Fit smoothing spline to call price curve C(K). smooth_factor=0 -> exact
    # interpolation; increase it if quotes are noisy (wide bid/ask, thin strikes).
    spline = UnivariateSpline(K, C, k=4, s=smooth_factor)

    # Evaluate on a fine grid for a smoother derivative
    strikes_fine = np.linspace(K.min(), K.max(), 400)
    dC_dK = spline.derivative(n=1)(strikes_fine)

    discount = np.exp(chain.r * chain.T)
    cdf = 1.0 + discount * dC_dK

    # Numerical noise can push this slightly outside [0,1] and it must be
    # monotonic non-decreasing (it's a CDF) — clean both up.
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)

    return strikes_fine, cdf


def market_implied_prob_above(chain: OptionChainSnapshot, strike: float,
                               smooth_factor: float = 0.0) -> float:
    """P(S_T > strike) read off the market-implied CDF. This is your
    'Quote' number — the market-implied probability side of the edge calc."""
    strikes_fine, cdf = breeden_litzenberger_cdf(chain, smooth_factor)
    p_below = np.interp(strike, strikes_fine, cdf)
    return 1.0 - p_below


# ======================================================================
# 2. HAR-RV: your own volatility forecast (this is where the edge lives)
# ======================================================================

def realized_variance(returns: np.ndarray, freq_per_day: int) -> np.ndarray:
    """
    Daily realized variance from intraday returns.
    `returns` = 1D array of intraday log returns, in chronological order.
    `freq_per_day` = number of intraday bars per trading day (e.g. 75 for
    5-min bars over a 6.25hr Nifty session).
    Returns one RV value per day.
    """
    n_days = len(returns) // freq_per_day
    returns = returns[: n_days * freq_per_day]
    r2 = returns.reshape(n_days, freq_per_day) ** 2
    return r2.sum(axis=1)


def har_rv_forecast(daily_rv: np.ndarray, horizon_days: int = 1) -> float:
    """
    HAR-RV (Corsi 2009): forecast tomorrow's realized variance as a linear
    combination of RV averaged over three horizons — daily, weekly, monthly.
    This captures the well-documented long-memory / multi-horizon clustering
    of volatility far better than a single-lag AR model.

        RV_t = c + b_d * RV_{t-1} + b_w * RV_{t-1:t-5}_avg + b_m * RV_{t-1:t-22}_avg

    For a from-scratch fit you'd run OLS on historical (RV_t, RV_d, RV_w, RV_m)
    tuples. Here we fit it directly on the supplied history.

    Returns forecasted daily variance for `horizon_days` ahead (simple
    iterate-forward extension if horizon_days > 1 — for production, fit a
    direct multi-step model instead of iterating).
    """
    if len(daily_rv) < 22:
        raise ValueError("Need at least 22 days of realized variance history for HAR-RV")

    rv = daily_rv.copy()
    X, y = [], []
    for t in range(22, len(rv)):
        rv_d = rv[t - 1]
        rv_w = rv[t - 5:t].mean()
        rv_m = rv[t - 22:t].mean()
        X.append([1.0, rv_d, rv_w, rv_m])
        y.append(rv[t])
    X = np.array(X)
    y = np.array(y)

    # OLS fit: beta = (X'X)^-1 X'y
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Iteratively forecast forward
    history = list(rv)
    for _ in range(horizon_days):
        rv_d = history[-1]
        rv_w = np.mean(history[-5:])
        rv_m = np.mean(history[-22:])
        x_next = np.array([1.0, rv_d, rv_w, rv_m])
        forecast = float(x_next @ beta)
        forecast = max(forecast, 1e-8)  # variance can't be negative
        history.append(forecast)

    return history[-1]  # forecasted variance for the target day


def overnight_jump_adjustment(overnight_returns: np.ndarray) -> float:
    """
    Nifty trades ~6.25hrs/day, 5 days/week — unlike Polymarket's 24/7 crypto
    markets, most macro news (RBI policy, US Fed, global cues, earnings)
    lands when the market is CLOSED. Overnight gap variance must be modeled
    separately and added on top of the intraday HAR-RV forecast, or your
    probability model will systematically underprice tail risk around
    gap opens.

    Returns average overnight variance contribution (simple historical
    average here; consider conditioning on known event days — RBI policy,
    budget, results season — for a sharper estimate).
    """
    return float(np.mean(overnight_returns ** 2))


# ======================================================================
# 3. Convert vol forecast -> model probability (your "Model P(UP)")
# ======================================================================

def model_prob_above(spot: float, strike: float, T_years: float,
                      forecast_variance_annualized: float, r: float = 0.065) -> float:
    """
    Lognormal terminal distribution -> P(S_T > K), using YOUR volatility
    forecast rather than the market's implied vol. This is the direct
    analogue of the Polymarket bot's 'Fair P(UP)' — your model's own view,
    independent of what the market is currently pricing.

    Standard Black-Scholes N(d2) formula, but sigma comes from your HAR-RV
    forecast (+ overnight jump term), not from backing out market IV.
    """
    sigma = np.sqrt(forecast_variance_annualized)
    d2 = (np.log(spot / strike) + (r - 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
    return float(norm.cdf(d2))


def model_call_price(spot: float, strike: float, T_years: float,
                      forecast_variance_annualized: float, r: float = 0.065) -> float:
    """
    Your model's OWN fair value for the vanilla call, using your vol
    forecast in the standard Black-Scholes formula.

    IMPORTANT — why this function exists alongside model_prob_above():
    P(S_T > K) mispricing is NOT the same thing as call-PRICE mispricing.
    A vanilla call's value is E[max(S_T-K, 0)], a convex payoff that
    depends on the whole tail beyond K, not just the probability of
    being above K. Two distributions can agree on P(S>K) while disagreeing
    on E[(S-K)+] if their shapes differ beyond that point.

    Trading a probability edge (as if it were a binary/digital option) on
    a vanilla instrument is a common mistake — see the backtesting harness
    notes. For vanilla calls/puts, compare THIS price to the market's
    quoted premium instead of comparing probabilities.
    """
    sigma = np.sqrt(forecast_variance_annualized)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)
    return float(spot * norm.cdf(d1) - strike * np.exp(-r * T_years) * norm.cdf(d2))


# ======================================================================
# 4. Edge & signal generation
# ======================================================================

@dataclass
class EdgeSignal:
    strike: float
    market_implied_prob: float
    model_prob: float
    edge: float           # meaning depends on mode -- see generate_signals
    signal: str            # "BUY_CALL" / "SELL_CALL" / "NO_TRADE"
    market_price: Optional[float] = None
    model_price: Optional[float] = None


def generate_signals(chain: OptionChainSnapshot,
                      forecast_variance_annualized: float,
                      edge_threshold: float = 0.05,
                      r: float = 0.065,
                      mode: str = "price") -> list[EdgeSignal]:
    """
    For every strike in the chain, compute a mispricing signal.

    mode="price" (default, use this for actual vanilla call/put trading):
        edge = (model_price - market_price) / market_price, a % mispricing
        on the actual tradeable instrument. This is the correct comparison
        for vanilla options, since their value depends on the full tail
        integral, not a single probability point (see model_call_price
        docstring). edge_threshold is then a % (e.g. 0.05 = 5% mispriced).

    mode="prob" (kept for reference / genuinely binary payoffs only, e.g.
        digital options or close-strike call SPREADS that approximate a
        binary payoff — this is the direct analogue of the Polymarket
        bot's 'Model P(UP) vs Quote' comparison):
        edge = model_prob - market_prob, a probability-point gap.
        Do NOT use this mode to size vanilla call/put positions — the
        backtesting harness demonstrates why (well-calibrated probability
        edge, flat-to-negative vanilla P&L, because probability agreement
        doesn't imply price agreement for convex payoffs).
    """
    signals = []
    for K in chain.strikes:
        market_p = market_implied_prob_above(chain, K)
        model_p = model_prob_above(chain.spot, K, chain.T, forecast_variance_annualized, r)
        market_price = float(np.interp(K, chain.strikes, chain.call_mid_prices))
        model_price = model_call_price(chain.spot, K, chain.T, forecast_variance_annualized, r)

        if mode == "price":
            edge = (model_price - market_price) / max(market_price, 1e-6)
        elif mode == "prob":
            edge = model_p - market_p
        else:
            raise ValueError("mode must be 'price' or 'prob'")

        if edge > edge_threshold:
            signal = "BUY_CALL"       # model says instrument is underpriced
        elif edge < -edge_threshold:
            signal = "SELL_CALL"      # model says instrument is overpriced
        else:
            signal = "NO_TRADE"

        signals.append(EdgeSignal(K, market_p, model_p, edge, signal, market_price, model_price))
    return signals


# ======================================================================
# 5. DEMO with synthetic data — replace with real broker-API feeds
# ======================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- synthetic intraday returns to fit HAR-RV (stand-in for 5-min bars) ---
    n_days = 120
    freq_per_day = 75  # ~6.25hr session in 5-min bars
    true_daily_vol = 0.012  # ~1.2% daily vol, roughly Nifty-like
    intraday_returns = rng.normal(0, true_daily_vol / np.sqrt(freq_per_day),
                                   size=n_days * freq_per_day)

    daily_rv = realized_variance(intraday_returns, freq_per_day)
    forecast_var_daily = har_rv_forecast(daily_rv, horizon_days=1)

    overnight_rets = rng.normal(0, 0.004, size=n_days)  # gap risk
    overnight_var = overnight_jump_adjustment(overnight_rets)

    T_days_to_expiry = 3  # e.g. weekly expiry, 3 trading days out
    forecast_var_total = (forecast_var_daily + overnight_var) * T_days_to_expiry
    forecast_var_annualized = forecast_var_total * (252 / T_days_to_expiry)

    print(f"Forecasted daily RV (HAR-RV): {forecast_var_daily:.6e}")
    print(f"Overnight jump variance:      {overnight_var:.6e}")
    print(f"Annualized vol for pricing:   {np.sqrt(forecast_var_annualized):.2%}\n")

    # --- synthetic option chain (stand-in for live broker chain) ---
    spot = 24800.0
    strikes = np.arange(24200, 25400, 100.0)
    T_years = T_days_to_expiry / 252

    # Fabricate call prices using a slightly-off "market" vol so a real edge exists
    market_vol = np.sqrt(forecast_var_annualized) * 1.08  # market pricier vol -> mispricing
    d1 = (np.log(spot / strikes) + (0.065 + 0.5 * market_vol**2) * T_years) / (market_vol * np.sqrt(T_years))
    d2 = d1 - market_vol * np.sqrt(T_years)
    call_prices = spot * norm.cdf(d1) - strikes * np.exp(-0.065 * T_years) * norm.cdf(d2)

    chain = OptionChainSnapshot(spot=spot, strikes=strikes, call_mid_prices=call_prices,
                                 r=0.065, T=T_years)

    signals = generate_signals(chain, forecast_var_annualized, edge_threshold=0.03)

    df = pd.DataFrame([s.__dict__ for s in signals])
    df["market_implied_prob"] = df["market_implied_prob"].round(3)
    df["model_prob"] = df["model_prob"].round(3)
    df["edge"] = df["edge"].round(3)
    print(df.to_string(index=False))
