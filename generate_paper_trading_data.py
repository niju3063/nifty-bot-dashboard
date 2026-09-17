"""
Paper Trading Data Generator (Part 2 of 2)
================================================

Runs the paper-trading kill-switch demo and writes the order-by-order
log to paper_trading_data.json for build_dashboard.py to embed.

TO USE REAL DATA INSTEAD OF SYNTHETIC:
    Swap the synthetic underlying-path / chain generation below for
    real series (fetch_nse_historical.py / broker_data_feed.py) and a
    real signal stream from probability_model.py -- the PaperTradingEngine
    calls (submit / resolve) don't change either way.

Must sit in the same folder as: probability_model.py, backtest.py,
adaptive_model.py, paper_trading.py.
"""

import json
import numpy as np

from probability_model import realized_variance, generate_signals
from backtest import make_synthetic_chain
from adaptive_model import fit_har_beta, forecast_with_beta
from paper_trading import PaperTradingEngine

OUTPUT_PATH = "paper_trading_data.json"
MAX_LOG_ROWS_EMBEDDED = 400  # keeps the HTML file a reasonable size


def trim_log(log_rows, max_rows=MAX_LOG_ROWS_EMBEDDED):
    """Keeps the full run's shape without embedding every single row:
    a chunk from the start, a representative sample from the middle,
    and the full tail (where the kill-switch drama happens)."""
    if len(log_rows) <= max_rows:
        return log_rows
    head = log_rows[:150]
    tail = log_rows[-200:]
    step = max(1, (len(log_rows) - 350) // 50)
    middle = log_rows[150:-200:step][:50]
    return head + middle + tail


def main(max_drawdown=-50_000, max_orders_per_sec=10, seed=7,
         n_expiries=400, days_per_expiry=3, freq_per_day=75,
         vol_regime1=0.010, vol_regime2=0.020, market_vol_bias=1.08):

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

    engine = PaperTradingEngine(max_drawdown=max_drawdown, max_orders_per_sec=max_orders_per_sec,
                                 run_id="NIFTYPAPER")
    static_beta = None
    day_ptr = warmup_days
    log_rows = []
    cumulative = 0.0

    for expiry_i in range(n_expiries):
        trailing_rv = daily_rv_full[:day_ptr]
        if len(trailing_rv) < 122:
            day_ptr += days_per_expiry
            continue
        if static_beta is None:
            static_beta = fit_har_beta(trailing_rv, window=None)
        forecast_var_daily = forecast_with_beta(trailing_rv, static_beta)
        overnight_var = float(np.mean(overnight_rets[:day_ptr] ** 2))
        T_years = days_per_expiry / 252
        forecast_var_total = (forecast_var_daily + overnight_var) * days_per_expiry
        forecast_var_annualized = forecast_var_total * (252 / days_per_expiry)
        spot_now = daily_close[day_ptr - 1]
        chain = make_synthetic_chain(spot_now, T_years, forecast_var_annualized, market_vol_bias,
                                      seed=seed + expiry_i)
        signals = generate_signals(chain, forecast_var_annualized, edge_threshold=0.03, mode="price")
        expiry_day_idx = min(day_ptr + days_per_expiry - 1, len(daily_close) - 1)
        spot_at_expiry = daily_close[expiry_day_idx]

        # Simulated clock -- see paper_trading.py's own demo for why this
        # matters (real wall-clock time falsely trips the rate limiter
        # when a multi-year backtest replays in milliseconds).
        base_time = day_ptr * 86400.0
        for order_idx, sig in enumerate(signals):
            if sig.signal == "NO_TRADE":
                continue
            order = engine.submit(sig, price=sig.market_price, now=base_time + order_idx * 0.2)
            pnl = None
            if order.status == "FILLED":
                cost = sig.market_price * 0.01
                intrinsic = max(spot_at_expiry - sig.strike, 0.0)
                pnl = (intrinsic - sig.market_price - cost) if sig.signal == "BUY_CALL" \
                    else (sig.market_price - intrinsic - cost)
                engine.resolve(order, pnl)
                cumulative += pnl

            log_rows.append({
                "day": day_ptr,
                "strategy_id": order.strategy_id,
                "strike": float(sig.strike),
                "side": sig.signal,
                "price": round(sig.market_price, 2),
                "status": order.status,
                "pnl": round(pnl, 1) if pnl is not None else None,
                "cum_pnl": round(cumulative, 1),
            })

        day_ptr += days_per_expiry
        if engine.kill_switch.tripped:
            break

    summary = engine.summary()
    trimmed = trim_log(log_rows)

    paper_data = {
        "kill_switch_limit": max_drawdown,
        "run_id": "NIFTYPAPER",
        "log": trimmed,
        "summary": summary,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(paper_data, f, separators=(",", ":"))

    print(f"Wrote {OUTPUT_PATH} ({len(trimmed)}/{len(log_rows)} rows embedded)")
    print(f"Kill switch tripped: {summary['kill_switch_tripped']} -- {summary['kill_switch_reason']}")
    print(f"Realized P&L at halt: {summary['realized_pnl']:.0f}")


if __name__ == "__main__":
    main()
