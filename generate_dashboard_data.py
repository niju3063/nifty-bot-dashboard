"""
Dashboard Data Generator (Part 1 of 2)
==========================================

Runs the probability model, backtest harness, and adaptive layer on
whatever data you feed the underlying functions, and writes the results
to dashboard_data.json. build_dashboard.py then embeds this JSON into
the HTML template to produce the finished dashboard.

RUN ORDER:
    python3 generate_dashboard_data.py        # writes dashboard_data.json
    python3 generate_paper_trading_data.py    # writes paper_trading_data.json
    python3 build_dashboard.py                # writes nifty-edge-terminal.html

TO USE REAL DATA INSTEAD OF SYNTHETIC:
    Replace the `simulate_underlying_path` / `make_synthetic_chain` calls
    below with real series from fetch_nse_historical.py's
    `build_historical_dataset()` (or broker_data_feed.py for live data).
    Everything downstream (generate_signals, run_backtest, the JSON
    shape) is unchanged either way -- only the data source changes.

Must sit in the same folder as: probability_model.py, backtest.py,
adaptive_model.py.
"""

import json
import numpy as np

from probability_model import realized_variance, har_rv_forecast, overnight_jump_adjustment, generate_signals
from backtest import run_backtest, simulate_underlying_path, make_synthetic_chain

OUTPUT_PATH = "dashboard_data.json"


def downsample_curve(pnls, n=120):
    """Cumulative equity curve, thinned to ~n points so the chart stays
    light regardless of how many trades the backtest produced."""
    equity = np.cumsum(pnls)
    if len(equity) <= n:
        return equity.tolist()
    idx = np.linspace(0, len(equity) - 1, n).astype(int)
    return equity[idx].tolist()


def to_equity_curve(pnl_by_day, n=150):
    days = [d for d, _ in pnl_by_day]
    pnls = np.cumsum([p for _, p in pnl_by_day])
    if len(pnls) <= n:
        return list(zip(days, pnls.tolist()))
    idx = np.linspace(0, len(pnls) - 1, n).astype(int)
    return [(days[i], float(pnls[i])) for i in idx]


def build_regime_shift_data(n_expiries=400, days_per_expiry=3, freq_per_day=75,
                             vol_regime1=0.010, vol_regime2=0.020, market_vol_bias=1.08, seed=7):
    """Re-runs the static-vs-adaptive regime-shift comparison and
    captures the day-by-day cumulative P&L for both, for the chart.
    See adaptive_model.py's run_adaptive_vs_static_demo for the
    narrative version of this same comparison."""
    from adaptive_model import AdaptiveTradingSystem, fit_har_beta, forecast_with_beta
    from pathlib import Path

    rng = np.random.default_rng(seed)
    warmup_days = 30
    total_days = warmup_days + n_expiries * days_per_expiry
    shift_day = total_days // 2
    daily_vols = np.where(np.arange(total_days) < shift_day, vol_regime1, vol_regime2)
    intraday_returns = np.concatenate([
        rng.normal(0, daily_vols[d] / np.sqrt(freq_per_day), freq_per_day)
        for d in range(total_days)
    ])
    log_path = np.cumsum(intraday_returns)
    price_path = 24800.0 * np.exp(log_path)
    daily_close = price_path[freq_per_day - 1::freq_per_day]
    daily_rv_full = realized_variance(intraday_returns, freq_per_day)
    overnight_rets = rng.normal(0, 0.004, size=len(daily_rv_full))

    def run_one(adaptive: bool):
        state_path = "/tmp/_dash_adaptive_state.json"
        Path(state_path).unlink(missing_ok=True)
        system = AdaptiveTradingSystem(state_path=state_path, refit_every_n_expiries=20,
                                        refit_window_days=100) if adaptive else None
        static_beta = None
        day_ptr = warmup_days
        pnl_by_day = []
        for expiry_i in range(n_expiries):
            trailing_rv = daily_rv_full[:day_ptr]
            if len(trailing_rv) < 122:
                day_ptr += days_per_expiry
                continue
            if adaptive:
                system.maybe_refit(trailing_rv)
                beta = np.array(system.state.har_beta)
            else:
                if static_beta is None:
                    static_beta = fit_har_beta(trailing_rv, window=None)
                beta = static_beta
            forecast_var_daily = forecast_with_beta(trailing_rv, beta)
            overnight_var = float(np.mean(overnight_rets[:day_ptr] ** 2))
            T_years = days_per_expiry / 252
            forecast_var_total = (forecast_var_daily + overnight_var) * days_per_expiry
            forecast_var_annualized = forecast_var_total * (252 / days_per_expiry)
            spot_now = daily_close[day_ptr - 1]
            chain = make_synthetic_chain(spot_now, T_years, forecast_var_annualized,
                                          market_vol_bias, seed=seed + expiry_i)
            sigs = generate_signals(chain, forecast_var_annualized, edge_threshold=0.03, mode="price")
            expiry_day_idx = min(day_ptr + days_per_expiry - 1, len(daily_close) - 1)
            spot_at_expiry = daily_close[expiry_day_idx]
            period_pnls = []
            for sig in sigs:
                if sig.signal == "NO_TRADE":
                    continue
                cost = sig.market_price * 0.01
                intrinsic = max(spot_at_expiry - sig.strike, 0.0)
                pnl = (intrinsic - sig.market_price - cost) if sig.signal == "BUY_CALL" \
                    else (sig.market_price - intrinsic - cost)
                period_pnls.append(pnl)
                if adaptive:
                    system.record_outcome(sig.model_prob, spot_at_expiry > sig.strike)
            if adaptive and period_pnls:
                system.record_performance(period_pnls)
            pnl_by_day.append((day_ptr, sum(period_pnls)))
            day_ptr += days_per_expiry
        return pnl_by_day

    static_pnl_by_day = run_one(adaptive=False)
    adaptive_pnl_by_day = run_one(adaptive=True)
    return {
        "shift_day": shift_day,
        "total_days": total_days,
        "static_curve": to_equity_curve(static_pnl_by_day),
        "adaptive_curve": to_equity_curve(adaptive_pnl_by_day),
    }


def main():
    rng = np.random.default_rng(42)

    # ---- 1. Current signal table (price mode) ----
    n_days, freq_per_day = 120, 75
    intraday_returns, daily_close = simulate_underlying_path(n_days, freq_per_day, 24800.0, 0.012, seed=42)
    daily_rv = realized_variance(intraday_returns, freq_per_day)
    forecast_var_daily = har_rv_forecast(daily_rv, horizon_days=1)
    overnight_var = overnight_jump_adjustment(rng.normal(0, 0.004, size=n_days))
    T_days = 3
    forecast_var_annualized = (forecast_var_daily + overnight_var) * T_days * (252 / T_days)

    spot = 24800.0
    chain = make_synthetic_chain(spot, T_days / 252, forecast_var_annualized, market_vol_bias=1.08, seed=42)
    signals = generate_signals(chain, forecast_var_annualized, edge_threshold=0.03, mode="price")

    signals_data = [{
        "strike": float(s.strike),
        "market_prob": round(s.market_implied_prob, 3),
        "model_prob": round(s.model_prob, 3),
        "market_price": round(s.market_price, 2),
        "model_price": round(s.model_price, 2),
        "edge_pct": round(s.edge * 100, 2),
        "signal": s.signal,
    } for s in signals]

    # ---- 2. Backtest equity curves (prob mode vs price mode) ----
    prob_result = run_backtest(n_expiries=200, market_vol_bias=1.08, edge_threshold=0.03, signal_mode="prob", seed=42)
    price_result = run_backtest(n_expiries=200, market_vol_bias=1.08, edge_threshold=0.03, signal_mode="price", seed=42)

    equity_prob = downsample_curve([t.pnl for t in prob_result.trades])
    equity_price = downsample_curve([t.pnl for t in price_result.trades])

    # ---- 3. Calibration table (price mode) ----
    calib_df = price_result.calibration_table(n_buckets=10)
    calibration_data = [{
        "bucket_mid": round(float(row["avg_model_prob"]), 3),
        "realized_freq": round(float(row["realized_freq"]), 3),
        "n": int(row["n"]),
    } for _, row in calib_df.iterrows()]

    # ---- 4. Regime shift comparison ----
    regime_shift = build_regime_shift_data()

    dashboard_data = {
        "signals": signals_data,
        "spot": spot,
        "annualized_vol_pct": round(float(np.sqrt(forecast_var_annualized)) * 100, 2),
        "backtest": {
            "prob_mode": {"equity_curve": equity_prob, "summary": prob_result.summary()},
            "price_mode": {"equity_curve": equity_price, "summary": price_result.summary()},
        },
        "calibration": calibration_data,
        "regime_shift": regime_shift,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dashboard_data, f, separators=(",", ":"), default=str)

    print(f"Wrote {OUTPUT_PATH} ({len(signals_data)} signals, "
          f"{len(price_result.trades)} price-mode trades, "
          f"{len(calibration_data)} calibration buckets)")


if __name__ == "__main__":
    main()
