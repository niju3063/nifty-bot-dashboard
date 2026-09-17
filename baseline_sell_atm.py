"""
Baseline Comparison: Sell-ATM-Call vs the Full Model
=========================================================

The decisive test for the variance-risk-premium hypothesis: build the
simplest possible strategy that could produce a similar-looking result
-- sell the at-the-money call every expiry cycle, no HAR-RV, no
Breeden-Litzenberger, no edge threshold, no forecasting of any kind --
and see whether it performs similarly to the full model.

If it does: the model's machinery isn't finding a real mispricing, it's
a complicated way of being short volatility, and all that complexity is
adding nothing beyond what "always sell ATM calls" already captures.

If the full model meaningfully outperforms the naive baseline (higher
Sharpe, better win rate, not just higher total P&L from being a bigger
bet): that's evidence the model IS doing something beyond harvesting
the variance risk premium -- worth a closer look.

Fetches data ONCE and runs both strategies against it, so this doesn't
cost you two separate ~10 minute NSE fetches.

    python3 baseline_sell_atm.py

Requires: fetch_nse_historical.py, probability_model.py, backtest.py,
validate_on_real_data.py.
"""

from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta

from fetch_nse_historical import build_historical_dataset
from probability_model import har_rv_forecast, generate_signals
from backtest import Trade, BacktestResult
from validate_on_real_data import (
    MIN_HISTORY_DAYS, daily_variance_proxy, find_resolution_date,
    print_report, directional_breakdown, block_bootstrap_pnl,
)


def run_atm_sell_baseline(chains: dict, spot_series, dates: list,
                           transaction_cost_pct: float = 0.01,
                           min_days_to_expiry: float = 1.0) -> BacktestResult:
    """
    No model. No forecast. No threshold. Every expiry cycle: find the
    strike closest to spot, sell that call, hold to expiry. This is the
    textbook naive short-volatility strategy -- the thing to beat.
    """
    trades = []
    seen_contracts = set()

    for i, date in enumerate(dates):
        if date not in chains:
            continue
        chain = chains[date]
        if chain.T * 252 < min_days_to_expiry:
            continue

        resolve_date = find_resolution_date(date, chain.T, dates)
        if resolve_date is None:
            continue
        spot_at_expiry = float(spot_series.loc[resolve_date])

        idx_atm = int(np.argmin(np.abs(chain.strikes - chain.spot)))
        atm_strike = float(chain.strikes[idx_atm])
        atm_price = float(chain.call_mid_prices[idx_atm])

        contract_key = (round(atm_strike, 1), resolve_date)
        if contract_key in seen_contracts:
            continue
        seen_contracts.add(contract_key)

        cost = atm_price * transaction_cost_pct
        intrinsic = max(spot_at_expiry - atm_strike, 0.0)
        pnl = atm_price - intrinsic - cost  # SELL_CALL payoff, always

        trades.append(Trade(
            entry_day=i, strike=atm_strike, signal="SELL_CALL", entry_price=atm_price,
            model_prob=0.5, market_prob=0.5,  # no model probability concept here -- placeholder only
            spot_at_expiry=spot_at_expiry, payoff=intrinsic, pnl=pnl,
        ))

    return BacktestResult(trades=trades)


def run_full_model(chains: dict, spot_series, dates: list,
                    edge_threshold: float = 0.03, transaction_cost_pct: float = 0.01,
                    z_band: float = 1.5, min_days_to_expiry: float = 1.0) -> BacktestResult:
    """Same logic as validate_on_real_data.run_validation, but reusing
    already-fetched data instead of fetching again -- avoids a second
    ~10 minute NSE pass just to get a like-for-like comparison."""
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

        signals = generate_signals(chain, forecast_var_annualized, edge_threshold=edge_threshold, mode="price")

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


def print_verdict(baseline: BacktestResult, model: BacktestResult):
    b = baseline.summary()
    m = model.summary()
    print("\n" + "=" * 70)
    print("VERDICT: naive sell-ATM baseline vs the full model")
    print("=" * 70)
    print(f"{'metric':>22} {'baseline (sell ATM)':>22} {'full model':>18}")
    print(f"{'n_trades':>22} {b.get('n_trades', 0):>22} {m.get('n_trades', 0):>18}")
    print(f"{'win_rate':>22} {b.get('win_rate', float('nan')):>21.1%} {m.get('win_rate', float('nan')):>17.1%}")
    print(f"{'total_pnl':>22} {b.get('total_pnl', 0):>22,.0f} {m.get('total_pnl', 0):>18,.0f}")
    print(f"{'sharpe_annualized':>22} {b.get('sharpe_annualized', float('nan')):>22.2f} "
          f"{m.get('sharpe_annualized', float('nan')):>18.2f}")
    print(f"{'max_drawdown':>22} {b.get('max_drawdown', 0):>22,.0f} {m.get('max_drawdown', 0):>18,.0f}")

    if m.get('n_trades', 0) > 0 and b.get('n_trades', 0) > 0:
        sharpe_gap = m['sharpe_annualized'] - b['sharpe_annualized']
        print(f"\nSharpe gap (model - baseline): {sharpe_gap:+.2f}")
        if abs(sharpe_gap) < 0.5:
            print("-> The model's Sharpe is NOT meaningfully different from the naive baseline's.")
            print("   This supports the variance-risk-premium explanation: the model's added")
            print("   complexity (HAR-RV, Breeden-Litzenberger, edge thresholds) isn't finding")
            print("   a real mispricing beyond what 'always sell ATM calls' already captures.")
        elif sharpe_gap > 0:
            print("-> The model meaningfully outperforms the naive baseline on a risk-adjusted")
            print("   basis. Worth investigating further what specifically the model is capturing")
            print("   beyond plain short-volatility exposure.")
        else:
            print("-> The naive baseline actually does BETTER than the full model on a risk-")
            print("   adjusted basis -- the model's added complexity may be net-harmful here.")


if __name__ == "__main__":
    end = datetime.now()
    start = end - timedelta(days=400)

    print(f"Fetching NIFTY data once, {start.date()} to {end.date()} "
          f"(shared by both the baseline and the full model)...\n")
    chains, spot_series = build_historical_dataset(start, end, symbol="NIFTY")
    dates = list(spot_series.index)
    print(f"Fetched {len(spot_series)} trading days, {len(chains)} usable chains.\n")

    print("=" * 70)
    print("BASELINE: sell the ATM call every expiry, no model")
    print("=" * 70)
    baseline_result = run_atm_sell_baseline(chains, spot_series, dates)
    print_report(baseline_result)
    block_bootstrap_pnl(baseline_result)

    print("\n\n")
    print("=" * 70)
    print("FULL MODEL: HAR-RV + Breeden-Litzenberger + price-mode edge")
    print("=" * 70)
    model_result = run_full_model(chains, spot_series, dates)
    print_report(model_result)
    directional_breakdown(model_result)
    block_bootstrap_pnl(model_result)

    print_verdict(baseline_result, model_result)
