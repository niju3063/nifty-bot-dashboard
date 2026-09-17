"""
Backtesting Harness for the Nifty Options Probability Model
=============================================================

Replays generate_signals() (from probability_model.py) across many
historical days/expiries, simulates entering trades on signals, and
scores the result against what actually happened. This is the step
that turns "plausible model" into "validated edge" before any capital
goes near it.

Three things this harness checks, deliberately kept separate:

  1. P&L        -- did the signals make money after costs?
  2. Calibration -- when the model said 70% probability, did the event
                    actually happen ~70% of the time? (A model can be
                    profitable by luck and miscalibrated, or calibrated
                    and still unprofitable after costs -- check both.)
  3. Drawdown   -- max peak-to-trough equity decline, since this is what
                    actually ends a trading account, not average P&L.

Swap the synthetic path generator for real historical data (via the
broker data-feed layer) before drawing real conclusions.

Dependencies: numpy, pandas, matplotlib (optional, for the equity curve)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from probability_model import (
    OptionChainSnapshot,
    realized_variance,
    har_rv_forecast,
    overnight_jump_adjustment,
    generate_signals,
    EdgeSignal,
)


# ======================================================================
# 1. Synthetic market simulator (replace with real historical data)
# ======================================================================

def simulate_underlying_path(n_days: int, freq_per_day: int, spot0: float,
                              daily_vol: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    GBM-ish synthetic path for the underlying, intraday-resolution.
    Returns (intraday_returns flat array, daily_close_prices array).
    Replace this whole function with a loader that pulls real historical
    candles once you're past the synthetic-validation stage.
    """
    rng = np.random.default_rng(seed)
    intraday_returns = rng.normal(0, daily_vol / np.sqrt(freq_per_day),
                                   size=n_days * freq_per_day)
    log_path = np.cumsum(intraday_returns)
    price_path = spot0 * np.exp(log_path)
    daily_close = price_path[freq_per_day - 1::freq_per_day]
    return intraday_returns, daily_close


def make_synthetic_chain(spot: float, T_years: float, model_var_annualized: float,
                          market_vol_bias: float, strike_step: float = 100.0,
                          n_strikes: int = 12, r: float = 0.065, seed: int = 0) -> OptionChainSnapshot:
    """
    Builds a synthetic option chain priced at (your model vol * bias), i.e.
    the *market* is systematically off from the *true* generating vol by
    `market_vol_bias`. This is what gives the backtest a real edge to find --
    with zero bias the model has nothing to detect and P&L should be ~0
    after costs, which is itself a useful sanity check to run once.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    strikes = spot + strike_step * np.arange(-n_strikes // 2, n_strikes // 2)
    market_vol = np.sqrt(model_var_annualized) * market_vol_bias
    market_vol = max(market_vol, 1e-4)

    d1 = (np.log(spot / strikes) + (r + 0.5 * market_vol**2) * T_years) / (market_vol * np.sqrt(T_years))
    d2 = d1 - market_vol * np.sqrt(T_years)
    call_prices = spot * norm.cdf(d1) - strikes * np.exp(-r * T_years) * norm.cdf(d2)

    # small quote noise, since real bid/ask mids are never perfectly smooth
    call_prices = call_prices * (1 + rng.normal(0, 0.01, size=call_prices.shape))
    call_prices = np.clip(call_prices, 0.05, None)

    return OptionChainSnapshot(spot=spot, strikes=strikes, call_mid_prices=call_prices, r=r, T=T_years)


# ======================================================================
# 2. Trade simulation
# ======================================================================

@dataclass
class Trade:
    entry_day: int
    strike: float
    signal: str
    entry_price: float     # premium paid/received
    model_prob: float
    market_prob: float
    spot_at_expiry: float = None
    payoff: float = None
    pnl: float = None


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: np.ndarray = None

    def summary(self) -> dict:
        pnls = np.array([t.pnl for t in self.trades if t.pnl is not None])
        if len(pnls) == 0:
            return {"n_trades": 0}

        wins = pnls > 0
        equity = np.cumsum(pnls)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd = drawdown.min()

        sharpe = (pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else float("nan")

        return {
            "n_trades": len(pnls),
            "win_rate": wins.mean(),
            "total_pnl": pnls.sum(),
            "avg_pnl_per_trade": pnls.mean(),
            "max_drawdown": max_dd,
            "sharpe_annualized": sharpe,
        }

    def calibration_table(self, n_buckets: int = 10) -> pd.DataFrame:
        """
        Bucket trades by model_prob and compare to actual realized frequency
        of the event (spot > strike at expiry). A well-calibrated model has
        realized_freq ≈ bucket midpoint across all buckets. Systematic
        deviation (e.g. model says 70% but realized is 55%) means the model
        is overconfident and the edge_threshold needs to be wider, or the
        vol forecast itself needs work -- check this BEFORE trusting the P&L
        number, since a miscalibrated model can get lucky over a short backtest.
        """
        rows = []
        for t in self.trades:
            if t.spot_at_expiry is None:
                continue
            realized_above = 1.0 if t.spot_at_expiry > t.strike else 0.0
            rows.append({"model_prob": t.model_prob, "realized_above": realized_above})
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["bucket"] = pd.cut(df["model_prob"], bins=np.linspace(0, 1, n_buckets + 1))
        return df.groupby("bucket", observed=True).agg(
            n=("realized_above", "size"),
            avg_model_prob=("model_prob", "mean"),
            realized_freq=("realized_above", "mean"),
        ).reset_index()


def run_backtest(n_expiries: int = 200, freq_per_day: int = 75, days_per_expiry: int = 3,
                  daily_vol: float = 0.012, market_vol_bias: float = 1.08,
                  edge_threshold: float = 0.03, transaction_cost_pct: float = 0.01,
                  signal_mode: str = "price", seed: int = 42) -> BacktestResult:
    """
    Core loop: for each synthetic expiry cycle --
      1. build trailing RV history, fit HAR-RV
      2. price a synthetic chain (market has a known bias vs. true vol)
      3. generate signals
      4. "enter" every non-NO_TRADE signal at the chain's quoted price
      5. resolve at expiry against the realized spot path
      6. subtract transaction costs (spread + slippage, as a % of premium)

    market_vol_bias=1.0 means market vol == true vol, i.e. NO edge exists --
    run this as a control: total_pnl should hover near zero (net negative
    after costs), which validates the harness isn't fabricating an edge
    out of noise.
    """
    rng_master = np.random.default_rng(seed)
    spot = 24800.0
    warmup_days = 30  # trailing history needed before first HAR-RV fit
    intraday_returns, daily_close = simulate_underlying_path(
        n_days=warmup_days + n_expiries * days_per_expiry,
        freq_per_day=freq_per_day, spot0=spot, daily_vol=daily_vol,
        seed=seed,
    )
    daily_rv_full = realized_variance(intraday_returns, freq_per_day)
    overnight_rets = rng_master.normal(0, 0.004, size=len(daily_rv_full))

    trades: list[Trade] = []
    day_ptr = warmup_days

    for expiry_i in range(n_expiries):
        trailing_rv = daily_rv_full[:day_ptr]
        if len(trailing_rv) < 22:
            day_ptr += days_per_expiry
            continue

        forecast_var_daily = har_rv_forecast(trailing_rv, horizon_days=1)
        overnight_var = float(np.mean(overnight_rets[:day_ptr] ** 2))
        T_years = days_per_expiry / 252
        forecast_var_total = (forecast_var_daily + overnight_var) * days_per_expiry
        forecast_var_annualized = forecast_var_total * (252 / days_per_expiry)

        spot_now = daily_close[day_ptr - 1]
        chain = make_synthetic_chain(spot_now, T_years, forecast_var_annualized,
                                      market_vol_bias, seed=seed + expiry_i)
        signals = generate_signals(chain, forecast_var_annualized, edge_threshold=edge_threshold,
                                    mode=signal_mode)

        expiry_day_idx = min(day_ptr + days_per_expiry - 1, len(daily_close) - 1)
        spot_at_expiry = daily_close[expiry_day_idx]

        for sig in signals:
            if sig.signal == "NO_TRADE":
                continue
            entry_price = float(np.interp(sig.strike, chain.strikes, chain.call_mid_prices))
            cost = entry_price * transaction_cost_pct

            intrinsic_at_expiry = max(spot_at_expiry - sig.strike, 0.0)
            if sig.signal == "BUY_CALL":
                pnl = intrinsic_at_expiry - entry_price - cost
            else:  # SELL_CALL
                pnl = entry_price - intrinsic_at_expiry - cost

            trades.append(Trade(
                entry_day=day_ptr, strike=sig.strike, signal=sig.signal,
                entry_price=entry_price, model_prob=sig.model_prob,
                market_prob=sig.market_implied_prob,
                spot_at_expiry=spot_at_expiry, payoff=intrinsic_at_expiry, pnl=pnl,
            ))

        day_ptr += days_per_expiry

    equity_curve = np.cumsum([t.pnl for t in trades]) if trades else np.array([])
    return BacktestResult(trades=trades, equity_curve=equity_curve)


# ======================================================================
# 3. Demo
# ======================================================================

if __name__ == "__main__":
    print("=== mode='prob' (binary-probability edge, WRONG for vanilla calls) ===")
    prob_result = run_backtest(n_expiries=200, market_vol_bias=1.08, edge_threshold=0.03,
                                signal_mode="prob")
    for k, v in prob_result.summary().items():
        print(f"{k:>20}: {v}")
    print("  -> calibration is excellent (see calibration_table()) but P&L is flat/negative:")
    print("     agreeing on P(S>K) does not mean agreeing on E[(S-K)+]. This is the")
    print("     mistake of porting a binary-market (Polymarket) signal straight onto")
    print("     a convex vanilla-option payoff.\n")

    print("=== mode='price' (correct: compare model's own BS price to market premium) ===")
    price_result = run_backtest(n_expiries=200, market_vol_bias=1.08, edge_threshold=0.03,
                                 signal_mode="price")
    for k, v in price_result.summary().items():
        print(f"{k:>20}: {v}")

    print("\n=== Control: no edge (market vol == true vol), mode='price' ===")
    control = run_backtest(n_expiries=200, market_vol_bias=1.00, edge_threshold=0.03,
                            signal_mode="price")
    for k, v in control.summary().items():
        print(f"{k:>20}: {v}")
