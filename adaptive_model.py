"""
Self-Improving Layer: Rolling Refit + Recalibration + Adaptive Threshold
============================================================================

A static model (fit once on historical data, never touched again) decays
as the market's volatility regime shifts -- realistic over any multi-month
deployment. This module adds three separate, honest adaptation mechanisms.
They're kept separate deliberately, because each fixes a different failure
mode and you want to be able to tell which one is doing the work:

  1. ROLLING HAR-RV REFIT
     Refits the volatility-forecast coefficients on a trailing window
     (not the full history) every N expiries. This is what lets the vol
     forecast track a regime change instead of staying anchored to
     stale data. Trade-off: too short a window = noisy refits; too long
     = slow to adapt. Exposed as a parameter, not hardcoded.

  2. ISOTONIC PROBABILITY RECALIBRATION
     As expiries resolve, (model_prob, realized_outcome) pairs accumulate.
     Isotonic regression fits a monotonic correction curve mapping "raw
     model probability" -> "actual historical frequency of that outcome."
     This catches systematic over/under-confidence WITHOUT touching the
     underlying vol model -- e.g. if the model says 80% and it's actually
     right 65% of the time, every future 80% gets corrected toward 65%,
     no vol-model surgery required.

  3. ADAPTIVE EDGE THRESHOLD
     A simple proportional controller: if rolling win rate is below par,
     raise the threshold (trade less, more selectively); if performance
     is healthy, relax it slightly (trade more). This is NOT a claim of
     reinforcement learning or anything exotic -- it's a bounded feedback
     rule, intentionally simple so its behavior stays predictable.

What this is NOT: a system that discovers new signals on its own, or
learns structurally different strategies over time. It adapts the SAME
model's parameters to changing conditions and corrects its own measurable
miscalibration. That distinction matters -- oversell this and you'll
trust it more than you should.

State (coefficients, calibration curve, threshold, performance history)
persists to a JSON file so learning survives across runs/restarts.

Dependencies: numpy, pandas, scikit-learn (for IsotonicRegression)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from probability_model import (
    OptionChainSnapshot, realized_variance, generate_signals, EdgeSignal,
)
from backtest import simulate_underlying_path, make_synthetic_chain, Trade, BacktestResult

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("adaptive_model")


# ======================================================================
# 1. Persistent state
# ======================================================================

@dataclass
class AdaptiveState:
    har_beta: list = field(default_factory=lambda: [0.0, 0.5, 0.3, 0.2])  # [c, b_d, b_w, b_m]
    calibration_x: list = field(default_factory=list)   # raw model_probs seen so far
    calibration_y: list = field(default_factory=list)   # realized outcomes (0/1) matching them
    edge_threshold: float = 0.03
    performance_history: list = field(default_factory=list)  # per-period dicts
    n_refits: int = 0
    n_recalibrations: int = 0

    def save(self, path: str):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load_or_init(cls, path: str) -> "AdaptiveState":
        if Path(path).exists():
            data = json.loads(Path(path).read_text())
            return cls(**data)
        return cls()


# ======================================================================
# 2. Rolling HAR-RV refit (mechanism #1)
# ======================================================================

def fit_har_beta(daily_rv: np.ndarray, window: Optional[int] = 252) -> np.ndarray:
    """
    OLS fit of HAR-RV coefficients [c, b_daily, b_weekly, b_monthly] on
    a TRAILING window of `window` days (None = use all history, which is
    the static/non-adaptive behavior). A shorter window tracks regime
    shifts faster but with more estimation noise -- 252 days (~1 trading
    year) is a reasonable default, not a tuned optimum.
    """
    rv = daily_rv[-window:] if window else daily_rv
    if len(rv) < 22:
        raise ValueError("Need at least 22 days of RV history to fit HAR-RV")

    X, y = [], []
    for t in range(22, len(rv)):
        X.append([1.0, rv[t - 1], rv[t - 5:t].mean(), rv[t - 22:t].mean()])
        y.append(rv[t])
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def forecast_with_beta(daily_rv_history: np.ndarray, beta: np.ndarray, horizon_days: int = 1) -> float:
    """Forecasts forward using a GIVEN (already-fit) beta -- separated from
    fitting so the hot path (daily forecasting) doesn't refit every call;
    refitting only happens on the cadence set in AdaptiveTradingSystem."""
    history = list(daily_rv_history)
    for _ in range(horizon_days):
        rv_d, rv_w, rv_m = history[-1], np.mean(history[-5:]), np.mean(history[-22:])
        forecast = max(float(np.array([1.0, rv_d, rv_w, rv_m]) @ beta), 1e-8)
        history.append(forecast)
    return history[-1]


# ======================================================================
# 3. Isotonic probability recalibration (mechanism #2)
# ======================================================================

def fit_calibrator(calibration_x: list, calibration_y: list) -> Optional[IsotonicRegression]:
    """Fits a monotonic raw-prob -> corrected-prob mapping from accumulated
    (model_prob, realized_outcome) history. Needs a reasonable sample size
    to be meaningful -- returns None (no correction applied) until then,
    rather than fitting a noisy curve on a handful of points."""
    if len(calibration_x) < 50:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(calibration_x, calibration_y)
    return iso


def apply_calibration(raw_prob: float, calibrator: Optional[IsotonicRegression]) -> float:
    if calibrator is None:
        return raw_prob
    return float(calibrator.predict([raw_prob])[0])


# ======================================================================
# 4. Adaptive edge threshold (mechanism #3)
# ======================================================================

def update_threshold(current_threshold: float, recent_win_rate: float,
                      target_win_rate: float = 0.55, step: float = 0.005,
                      min_threshold: float = 0.01, max_threshold: float = 0.15) -> float:
    """
    Simple bounded proportional controller, deliberately simple so its
    behavior is predictable and auditable:
      - win rate below target -> raise threshold (be more selective)
      - win rate comfortably above target -> lower threshold slightly
        (capture more of a working edge)
    Clamped to [min_threshold, max_threshold] so it can't runaway to
    "never trade" or "trade everything."
    """
    if recent_win_rate < target_win_rate:
        new_threshold = current_threshold + step
    elif recent_win_rate > target_win_rate + 0.10:
        new_threshold = current_threshold - step
    else:
        new_threshold = current_threshold
    return float(np.clip(new_threshold, min_threshold, max_threshold))


# ======================================================================
# 5. The adaptive system -- ties all three mechanisms together
# ======================================================================

class AdaptiveTradingSystem:
    """
    Wraps the static probability_model.py pipeline with the three
    adaptation mechanisms above, refitting/recalibrating on a cadence
    rather than every single day (daily refitting is mostly noise; expiry-
    cycle-level refitting matches how often you actually get new resolved
    outcomes to learn from).
    """

    def __init__(self, state_path: str = "adaptive_state.json",
                 refit_every_n_expiries: int = 20, refit_window_days: int = 252,
                 r: float = 0.065):
        self.state_path = state_path
        self.refit_every_n_expiries = refit_every_n_expiries
        self.refit_window_days = refit_window_days
        self.r = r
        self.state = AdaptiveState.load_or_init(state_path)
        self.calibrator = fit_calibrator(self.state.calibration_x, self.state.calibration_y)
        self._expiries_since_refit = 0

    def maybe_refit(self, daily_rv_history: np.ndarray):
        self._expiries_since_refit += 1
        if self._expiries_since_refit >= self.refit_every_n_expiries:
            new_beta = fit_har_beta(daily_rv_history, window=self.refit_window_days)
            self.state.har_beta = new_beta.tolist()
            self.state.n_refits += 1
            self._expiries_since_refit = 0
            logger.info(f"Refit #{self.state.n_refits}: new HAR-RV beta = {new_beta.round(4)}")

    def record_outcome(self, model_prob_raw: float, realized_above: bool):
        """Feed one resolved expiry's (raw model prob, actual outcome) into
        the calibration history, and periodically refit the calibrator."""
        self.state.calibration_x.append(model_prob_raw)
        self.state.calibration_y.append(1.0 if realized_above else 0.0)
        if len(self.state.calibration_x) % 50 == 0:
            self.calibrator = fit_calibrator(self.state.calibration_x, self.state.calibration_y)
            self.state.n_recalibrations += 1
            logger.info(f"Recalibration #{self.state.n_recalibrations} "
                        f"on {len(self.state.calibration_x)} resolved outcomes.")

    def record_performance(self, period_pnls: list[float]):
        """Feed one period's trade P&Ls in, update the rolling win rate,
        and adapt the edge threshold accordingly."""
        pnls = np.array(period_pnls)
        if len(pnls) == 0:
            return
        win_rate = float((pnls > 0).mean())
        self.state.performance_history.append({"win_rate": win_rate, "pnl": float(pnls.sum())})
        self.state.edge_threshold = update_threshold(self.state.edge_threshold, win_rate)

    def save(self):
        self.state.save(self.state_path)


# ======================================================================
# 6. Demo: regime shift -- static model decays, adaptive model recovers
# ======================================================================

def run_adaptive_vs_static_demo(n_expiries: int = 400, days_per_expiry: int = 3,
                                 freq_per_day: int = 75, vol_regime1: float = 0.010,
                                 vol_regime2: float = 0.020, market_vol_bias: float = 1.08,
                                 seed: int = 7):
    """
    Simulates a volatility REGIME SHIFT halfway through (true daily vol
    doubles from 1.0% to 2.0%, e.g. a stress period) and compares:
      - STATIC: HAR-RV beta fit once at the start, never refit
      - ADAPTIVE: HAR-RV beta refit every 20 expiries on a trailing window

    Both use price-mode signals (the validated correct mode from the
    backtesting harness) and identical synthetic data -- the only
    difference is whether the vol forecast adapts.
    """
    rng = np.random.default_rng(seed)
    warmup_days = 30
    total_days = warmup_days + n_expiries * days_per_expiry
    shift_day = total_days // 2

    # Build a two-regime intraday return series
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
        state_path = "/tmp/_demo_adaptive_state.json"
        Path(state_path).unlink(missing_ok=True)  # fresh state each demo run
        system = AdaptiveTradingSystem(state_path=state_path, refit_every_n_expiries=20,
                                        refit_window_days=100) if adaptive else None
        static_beta = None
        trades = []
        day_ptr = warmup_days

        for expiry_i in range(n_expiries):
            trailing_rv = daily_rv_full[:day_ptr]
            if len(trailing_rv) < 122:  # need enough for at least one HAR fit + window
                day_ptr += days_per_expiry
                continue

            if adaptive:
                system.maybe_refit(trailing_rv)
                beta = np.array(system.state.har_beta)
            else:
                if static_beta is None:
                    static_beta = fit_har_beta(trailing_rv, window=None)  # fit once, all history to date
                beta = static_beta

            forecast_var_daily = forecast_with_beta(trailing_rv, beta)
            overnight_var = float(np.mean(overnight_rets[:day_ptr] ** 2))
            T_years = days_per_expiry / 252
            forecast_var_total = (forecast_var_daily + overnight_var) * days_per_expiry
            forecast_var_annualized = forecast_var_total * (252 / days_per_expiry)

            spot_now = daily_close[day_ptr - 1]
            chain = make_synthetic_chain(spot_now, T_years, forecast_var_annualized,
                                          market_vol_bias, seed=seed + expiry_i)
            signals = generate_signals(chain, forecast_var_annualized, edge_threshold=0.03,
                                        mode="price")

            expiry_day_idx = min(day_ptr + days_per_expiry - 1, len(daily_close) - 1)
            spot_at_expiry = daily_close[expiry_day_idx]

            period_pnls = []
            for sig in signals:
                if sig.signal == "NO_TRADE":
                    continue
                entry_price = sig.market_price
                cost = entry_price * 0.01
                intrinsic = max(spot_at_expiry - sig.strike, 0.0)
                pnl = (intrinsic - entry_price - cost) if sig.signal == "BUY_CALL" \
                    else (entry_price - intrinsic - cost)
                trades.append(Trade(day_ptr, sig.strike, sig.signal, entry_price,
                                     sig.model_prob, sig.market_implied_prob,
                                     spot_at_expiry, intrinsic, pnl))
                period_pnls.append(pnl)

                if adaptive:
                    system.record_outcome(sig.model_prob, spot_at_expiry > sig.strike)

            if adaptive and period_pnls:
                system.record_performance(period_pnls)

            day_ptr += days_per_expiry

        if adaptive:
            system.save()
        return BacktestResult(trades=trades)

    static_result = run_one(adaptive=False)
    adaptive_result = run_one(adaptive=True)

    def split_pre_post(result: BacktestResult):
        pre = [t.pnl for t in result.trades if t.entry_day < shift_day]
        post = [t.pnl for t in result.trades if t.entry_day >= shift_day]
        return np.array(pre), np.array(post)

    print(f"Regime shift at day {shift_day} of {total_days} (vol {vol_regime1:.1%} -> {vol_regime2:.1%})\n")

    for name, result in [("STATIC (fit once, never refit)", static_result),
                          ("ADAPTIVE (rolling refit every 20 expiries)", adaptive_result)]:
        pre, post = split_pre_post(result)
        print(f"=== {name} ===")
        print(f"  Pre-shift:  n={len(pre):4d}  win_rate={np.mean(pre > 0):.1%}  "
              f"total_pnl={pre.sum():>10.1f}")
        print(f"  Post-shift: n={len(post):4d}  win_rate={np.mean(post > 0):.1%}  "
              f"total_pnl={post.sum():>10.1f}")
        print()


if __name__ == "__main__":
    run_adaptive_vs_static_demo()
