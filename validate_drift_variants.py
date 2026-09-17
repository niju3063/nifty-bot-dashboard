"""
Drift Diagnostics: Trailing-Drift Variant + Forward-Price Variant
======================================================================

The real-data validation found a stark BUY_CALL/SELL_CALL asymmetry
(25.6% vs 67.3% win rate) traced to the model always pricing options
using the risk-free rate as drift -- correct in theory for risk-neutral
pricing, but only self-consistent if the observed spot's assumed
carry actually matches the market's. Two ways to test/fix that:

  1. TRAILING-DRIFT VARIANT (diagnostic, not a proper pricing model)
     Swaps the fixed r=6.5% assumption for a rolling trailing-realized
     drift estimate. This is NOT theoretically valid for arbitrage-free
     option pricing (real-world drift and risk-neutral drift are
     different things on purpose) -- it exists purely to test whether
     the BUY/SELL asymmetry was caused by the r-assumption specifically.
     If the asymmetry shrinks a lot here, that confirms the diagnosis.

  2. FORWARD-PRICE VARIANT (the actual fix)
     Prices every option off the REAL, market-quoted NIFTY futures price
     instead of spot+assumed-drift. The futures price already embeds
     whatever the market's true cost-of-carry is -- dividend yield,
     funding costs, index arbitrage frictions, all of it -- with no
     assumption required. This is standard practice on real index-
     options desks specifically to avoid the drift-assumption trap.
     Uses the Black-76 formulation (forward-based Black-Scholes).

RUN THIS YOURSELF -- needs a live NSE fetch, same as validate_on_real_data.py.

    python3 validate_drift_variants.py

Requires: fetch_nse_historical.py, probability_model.py, backtest.py,
validate_on_real_data.py (reuses its Trade/BacktestResult/report helpers).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

from fetch_nse_historical import build_historical_dataset, build_historical_dataset_with_futures
from probability_model import har_rv_forecast, generate_signals
from backtest import Trade, BacktestResult
from validate_on_real_data import (
    MIN_HISTORY_DAYS, daily_variance_proxy, find_resolution_date,
    print_report, directional_breakdown, block_bootstrap_pnl,
)


# ======================================================================
# 1. Trailing-drift variant -- reuses the EXISTING model functions,
#    just feeds a different `r` each day instead of the fixed default.
#    (probability_model.py's model_prob_above / model_call_price already
#    take r as a parameter -- this never needed new pricing formulas.)
# ======================================================================

def rolling_trailing_drift(spot_series_values: np.ndarray, i: int, window: int = 60) -> float:
    """Annualized mean log return over the trailing `window` days, as of
    index i. This is what stands in for the risk-free rate in the
    trailing-drift diagnostic -- explicitly NOT a valid risk-neutral
    pricing assumption, just a way to isolate the r-assumption's effect."""
    start = max(0, i - window)
    log_rets = np.diff(np.log(spot_series_values[start:i + 1]))
    if len(log_rets) < 5:
        return 0.065  # not enough trailing data yet -- fall back to the risk-free default
    return float(np.mean(log_rets) * 252)


def run_trailing_drift_validation(start_date: datetime, end_date: datetime, symbol: str = "NIFTY",
                                   edge_threshold: float = 0.03, transaction_cost_pct: float = 0.01,
                                   z_band: float = 1.5, min_days_to_expiry: float = 1.0,
                                   drift_window: int = 60) -> BacktestResult:
    print(f"[Trailing-drift variant] Fetching {symbol}, {start_date.date()} to {end_date.date()}...")
    chains, spot_series = build_historical_dataset(start_date, end_date, symbol=symbol)
    if len(spot_series) < MIN_HISTORY_DAYS + 5:
        raise ValueError(f"Only fetched {len(spot_series)} usable trading days -- need more history.")
    print(f"Fetched {len(spot_series)} trading days, {len(chains)} usable chains.\n")

    dates = list(spot_series.index)
    spot_values = spot_series.values
    daily_var = daily_variance_proxy(spot_series)

    trades = []
    seen_contracts = set()

    for i, date in enumerate(dates):
        if date not in chains or i < MIN_HISTORY_DAYS + 1:
            continue
        chain = chains[date]
        if chain.T * 252 < min_days_to_expiry:
            continue

        trailing_var = daily_var[:i]
        try:
            forecast_var_daily = har_rv_forecast(trailing_var, horizon_days=1)
        except ValueError:
            continue
        forecast_var_annualized = forecast_var_daily * 252

        drift = rolling_trailing_drift(spot_values, i, window=drift_window)
        # Use the trailing drift in place of the fixed risk-free rate --
        # generate_signals already accepts r as a parameter for exactly this.
        signals = generate_signals(chain, forecast_var_annualized, edge_threshold=edge_threshold,
                                    mode="price", r=drift)

        resolve_date = find_resolution_date(date, chain.T, dates)
        if resolve_date is None:
            continue
        spot_at_expiry = float(spot_series.loc[resolve_date])

        sigma = np.sqrt(forecast_var_annualized)
        one_sigma_move = sigma * np.sqrt(chain.T)
        lower_bound = chain.spot * np.exp(-z_band * one_sigma_move)
        upper_bound = chain.spot * np.exp(z_band * one_sigma_move)

        for sig in signals:
            if sig.signal == "NO_TRADE" or not (lower_bound <= sig.strike <= upper_bound):
                continue
            contract_key = (round(sig.strike, 1), resolve_date)
            if contract_key in seen_contracts:
                continue
            seen_contracts.add(contract_key)

            cost = sig.market_price * transaction_cost_pct
            intrinsic = max(spot_at_expiry - sig.strike, 0.0)
            pnl = (intrinsic - sig.market_price - cost) if sig.signal == "BUY_CALL" \
                else (sig.market_price - intrinsic - cost)
            trades.append(Trade(
                entry_day=i, strike=sig.strike, signal=sig.signal, entry_price=sig.market_price,
                model_prob=sig.model_prob, market_prob=sig.market_implied_prob,
                spot_at_expiry=spot_at_expiry, payoff=intrinsic, pnl=pnl,
            ))

    return BacktestResult(trades=trades)


# ======================================================================
# 2. Forward-price variant -- Black-76, priced off the REAL futures
#    close, no drift assumption anywhere.
# ======================================================================

def black76_prob_above(F: float, K: float, T: float, var_annualized: float) -> float:
    """Risk-neutral P(S_T > K) using the observed forward F directly --
    no drift assumption, since F already IS the market's risk-neutral
    expectation of S_T by no-arbitrage construction."""
    sigma = np.sqrt(var_annualized)
    d2 = (np.log(F / K) - 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d2))


def black76_call_price(F: float, K: float, T: float, var_annualized: float, r_discount: float = 0.065) -> float:
    """Black-76 call price off the forward. r_discount here is ONLY a
    time-value-of-money discount rate, not a drift assumption -- a much
    smaller and safer approximation than using r as expected return."""
    sigma = np.sqrt(var_annualized)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(np.exp(-r_discount * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def run_forward_price_validation(start_date: datetime, end_date: datetime, symbol: str = "NIFTY",
                                  edge_threshold: float = 0.03, transaction_cost_pct: float = 0.01,
                                  z_band: float = 1.5, min_days_to_expiry: float = 1.0) -> BacktestResult:
    print(f"[Forward-price variant] Fetching {symbol}, {start_date.date()} to {end_date.date()}...")
    chains, spot_series, futures_series = build_historical_dataset_with_futures(start_date, end_date, symbol=symbol)
    if len(spot_series) < MIN_HISTORY_DAYS + 5:
        raise ValueError(f"Only fetched {len(spot_series)} usable trading days -- need more history.")
    print(f"Fetched {len(spot_series)} trading days, {len(chains)} chains, "
          f"{len(futures_series)} futures observations.\n")

    dates = list(spot_series.index)
    daily_var = daily_variance_proxy(spot_series)

    trades = []
    seen_contracts = set()
    skipped_no_futures = 0

    for i, date in enumerate(dates):
        if date not in chains or i < MIN_HISTORY_DAYS + 1:
            continue
        chain = chains[date]
        if chain.T * 252 < min_days_to_expiry:
            continue
        if date not in futures_series.index:
            skipped_no_futures += 1
            continue
        F = float(futures_series.loc[date])

        trailing_var = daily_var[:i]
        try:
            forecast_var_daily = har_rv_forecast(trailing_var, horizon_days=1)
        except ValueError:
            continue
        forecast_var_annualized = forecast_var_daily * 252

        resolve_date = find_resolution_date(date, chain.T, dates)
        if resolve_date is None:
            continue
        spot_at_expiry = float(spot_series.loc[resolve_date])

        sigma = np.sqrt(forecast_var_annualized)
        one_sigma_move = sigma * np.sqrt(chain.T)
        lower_bound = chain.spot * np.exp(-z_band * one_sigma_move)
        upper_bound = chain.spot * np.exp(z_band * one_sigma_move)

        for strike, market_price in zip(chain.strikes, chain.call_mid_prices):
            if not (lower_bound <= strike <= upper_bound):
                continue

            model_prob = black76_prob_above(F, strike, chain.T, forecast_var_annualized)
            model_price = black76_call_price(F, strike, chain.T, forecast_var_annualized)
            edge = (model_price - market_price) / max(market_price, 1e-6)

            if edge > edge_threshold:
                signal = "BUY_CALL"
            elif edge < -edge_threshold:
                signal = "SELL_CALL"
            else:
                continue

            contract_key = (round(strike, 1), resolve_date)
            if contract_key in seen_contracts:
                continue
            seen_contracts.add(contract_key)

            cost = market_price * transaction_cost_pct
            intrinsic = max(spot_at_expiry - strike, 0.0)
            pnl = (intrinsic - market_price - cost) if signal == "BUY_CALL" \
                else (market_price - intrinsic - cost)
            trades.append(Trade(
                entry_day=i, strike=strike, signal=signal, entry_price=market_price,
                model_prob=model_prob, market_prob=None,
                spot_at_expiry=spot_at_expiry, payoff=intrinsic, pnl=pnl,
            ))

    if skipped_no_futures:
        print(f"Note: {skipped_no_futures} days had no matching futures close and were skipped.\n")

    return BacktestResult(trades=trades)


if __name__ == "__main__":
    end = datetime.now()
    start = end - timedelta(days=400)

    print("=" * 70)
    print("VARIANT 1: TRAILING-DRIFT (diagnostic)")
    print("=" * 70)
    try:
        result1 = run_trailing_drift_validation(start, end, symbol="NIFTY")
        print_report(result1)
        directional_breakdown(result1)
        block_bootstrap_pnl(result1)
    except Exception as e:
        print(f"Trailing-drift variant failed: {e}")

    print("\n\n")
    print("=" * 70)
    print("VARIANT 2: FORWARD-PRICE (the actual fix)")
    print("=" * 70)
    try:
        result2 = run_forward_price_validation(start, end, symbol="NIFTY")
        print_report(result2)
        directional_breakdown(result2)
        block_bootstrap_pnl(result2)
    except Exception as e:
        print(f"Forward-price variant failed: {e}")
