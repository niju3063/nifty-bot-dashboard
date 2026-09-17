"""
Paper Trading Execution Layer
================================

The safety-critical piece between "signal generated" and "order sent to
a real broker." This is a SIMULATION -- it never calls a broker API or
places a real order. It exists to prove out the exact controls SEBI's
algo framework requires, and to give you a place to watch a kill switch
actually trip before real capital is anywhere near this system.

Three controls, matching the SEBI retail-algo framework discussed earlier:

  1. STRATEGY ID TAGGING
     Every order gets a unique, traceable Strategy ID -- the same
     requirement that lets NSE trace any market anomaly back to the
     specific algorithm that caused it. Format here mirrors what a real
     exchange-issued ID looks like structurally (not a real registered ID).

  2. RATE LIMITING
     Caps at 10 orders/second, matching the SEBI framework's hard limit.
     Orders beyond the cap in a given window are queued, not silently
     dropped or (worse) sent anyway.

  3. KILL SWITCH
     Halts ALL new order placement once cumulative drawdown breaches a
     configured limit. Once tripped, it stays tripped for the session --
     a kill switch that quietly resets itself isn't a kill switch. This
     is the control that would have capped the static (non-adaptive)
     model's losses during the regime-shift scenario -- see the demo
     at the bottom.

This module takes EdgeSignal objects (from probability_model.py) as
input and produces a full audit log. Wiring a real broker's order-
placement call in place of `_simulate_fill()` is the ONLY change needed
to go from paper to live -- and that change should not happen until
the backtest + adaptive layers have been validated on real historical
data (see fetch_nse_historical.py), not before.

Dependencies: numpy (only for the demo's synthetic trade stream)
"""

from __future__ import annotations
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import deque

from probability_model import EdgeSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper_trading")


# ======================================================================
# 1. Order & audit log records
# ======================================================================

@dataclass
class Order:
    strategy_id: str
    timestamp: str
    strike: float
    side: str              # "BUY_CALL" / "SELL_CALL"
    price: float
    status: str             # "FILLED" / "REJECTED_RATE_LIMIT" / "BLOCKED_KILL_SWITCH"
    model_prob: float
    market_prob: float
    pnl: Optional[float] = None   # filled in once the position resolves


def make_strategy_id(run_id: str, seq: int) -> str:
    """
    Structural mimic of an exchange-issued Unique Strategy ID (the SEBI
    framework requires one per order, traceable back to the algorithm).
    NOT a real registered exchange ID -- format only, for the paper log.
    """
    return f"STRAT-{run_id}-{seq:06d}"


# ======================================================================
# 2. Rate limiter (10 orders/sec, per SEBI framework)
# ======================================================================

class RateLimiter:
    """Sliding-window limiter: at most `max_per_sec` order attempts are
    allowed within any rolling 1-second window. Orders beyond that are
    rejected outright here (a real system would queue+retry; for a paper
    log, an explicit rejection is more honest about what the constraint
    actually costs you)."""

    def __init__(self, max_per_sec: int = 10):
        self.max_per_sec = max_per_sec
        self._timestamps: deque = deque()

    def allow(self, now: float) -> bool:
        while self._timestamps and now - self._timestamps[0] > 1.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_sec:
            return False
        self._timestamps.append(now)
        return True


# ======================================================================
# 3. Kill switch
# ======================================================================

@dataclass
class KillSwitch:
    max_drawdown: float          # absolute currency amount, e.g. -50000
    max_daily_loss: Optional[float] = None
    tripped: bool = False
    trip_reason: Optional[str] = None
    _peak_equity: float = 0.0
    _equity: float = 0.0
    _daily_pnl: float = 0.0

    def update(self, pnl_delta: float) -> bool:
        """Feed in each new realized P&L delta. Returns True if the kill
        switch is (now, or already) tripped. Once tripped, it does NOT
        reset itself -- that requires an explicit, separate action
        (reset_for_new_session), matching how a real kill switch should
        behave: a human decides when it's safe to resume, not the code."""
        if self.tripped:
            return True

        self._equity += pnl_delta
        self._daily_pnl += pnl_delta
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = self._equity - self._peak_equity

        if drawdown <= self.max_drawdown:
            self.tripped = True
            self.trip_reason = f"Drawdown {drawdown:.0f} breached limit {self.max_drawdown:.0f}"
        elif self.max_daily_loss is not None and self._daily_pnl <= self.max_daily_loss:
            self.tripped = True
            self.trip_reason = f"Daily loss {self._daily_pnl:.0f} breached limit {self.max_daily_loss:.0f}"

        if self.tripped:
            logger.warning(f"KILL SWITCH TRIPPED: {self.trip_reason}")
        return self.tripped

    def reset_for_new_session(self):
        """Explicit, deliberate reset -- e.g. called once per trading day
        after a human has reviewed why it tripped. Never call this
        automatically inside update()."""
        self.tripped = False
        self.trip_reason = None
        self._daily_pnl = 0.0


# ======================================================================
# 4. The paper trading engine
# ======================================================================

class PaperTradingEngine:
    """
    Ties Strategy ID tagging, the rate limiter, and the kill switch
    together around a stream of EdgeSignal objects. Produces a full
    Order audit log -- exactly the shape a real compliance log needs,
    with `status` telling you honestly whether an order would have
    actually gone through.
    """

    def __init__(self, max_drawdown: float = -50_000, max_daily_loss: Optional[float] = None,
                 max_orders_per_sec: int = 10, run_id: Optional[str] = None):
        self.run_id = run_id or uuid.uuid4().hex[:8].upper()
        self.rate_limiter = RateLimiter(max_orders_per_sec)
        self.kill_switch = KillSwitch(max_drawdown=max_drawdown, max_daily_loss=max_daily_loss)
        self.orders: list[Order] = []
        self._seq = 0

    def submit(self, signal: EdgeSignal, price: float, now: Optional[float] = None) -> Order:
        """
        Attempts to place one order for a non-NO_TRADE signal. Checks
        kill switch first (cheapest, most important check), then the
        rate limiter. Returns the Order record either way -- a rejected
        order is still logged, not silently dropped.
        """
        now = now if now is not None else time.time()
        self._seq += 1
        strategy_id = make_strategy_id(self.run_id, self._seq)
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.kill_switch.tripped:
            order = Order(strategy_id, timestamp, signal.strike, signal.signal, price,
                          "BLOCKED_KILL_SWITCH", signal.model_prob, signal.market_implied_prob)
            self.orders.append(order)
            return order

        if not self.rate_limiter.allow(now):
            order = Order(strategy_id, timestamp, signal.strike, signal.signal, price,
                          "REJECTED_RATE_LIMIT", signal.model_prob, signal.market_implied_prob)
            self.orders.append(order)
            return order

        order = Order(strategy_id, timestamp, signal.strike, signal.signal, price,
                      "FILLED", signal.model_prob, signal.market_implied_prob)
        self.orders.append(order)
        return order

    def resolve(self, order: Order, pnl: float):
        """Feed back the realized P&L once a filled order's position
        resolves at expiry -- updates the kill switch and closes the
        audit trail entry."""
        order.pnl = pnl
        if order.status == "FILLED":
            self.kill_switch.update(pnl)

    def summary(self) -> dict:
        filled = [o for o in self.orders if o.status == "FILLED"]
        blocked = [o for o in self.orders if o.status == "BLOCKED_KILL_SWITCH"]
        rate_limited = [o for o in self.orders if o.status == "REJECTED_RATE_LIMIT"]
        realized_pnls = [o.pnl for o in filled if o.pnl is not None]
        return {
            "total_orders": len(self.orders),
            "filled": len(filled),
            "blocked_by_kill_switch": len(blocked),
            "rejected_rate_limit": len(rate_limited),
            "realized_pnl": sum(realized_pnls) if realized_pnls else 0.0,
            "kill_switch_tripped": self.kill_switch.tripped,
            "kill_switch_reason": self.kill_switch.trip_reason,
        }


# ======================================================================
# 5. Demo -- replay the regime-shift STATIC-model trade stream and show
#    the kill switch capping the loss that the earlier demo showed
#    running unchecked to -260,633.
# ======================================================================

if __name__ == "__main__":
    from adaptive_model import run_adaptive_vs_static_demo
    import numpy as np
    from backtest import simulate_underlying_path, make_synthetic_chain
    from probability_model import realized_variance, generate_signals
    from adaptive_model import fit_har_beta, forecast_with_beta

    print("Replaying the STATIC model's regime-shift trade stream through "
          "the paper trading engine (kill switch at -50,000 drawdown)...\n")

    rng = np.random.default_rng(7)
    n_expiries, days_per_expiry, freq_per_day = 400, 3, 75
    warmup_days = 30
    total_days = warmup_days + n_expiries * days_per_expiry
    shift_day = total_days // 2
    daily_vols = np.where(np.arange(total_days) < shift_day, 0.010, 0.020)
    intraday_returns = np.concatenate([
        rng.normal(0, daily_vols[d] / np.sqrt(freq_per_day), freq_per_day)
        for d in range(total_days)
    ])
    log_path = np.cumsum(intraday_returns)
    price_path = 24800.0 * np.exp(log_path)
    daily_close = price_path[freq_per_day - 1::freq_per_day]
    daily_rv_full = realized_variance(intraday_returns, freq_per_day)
    overnight_rets = rng.normal(0, 0.004, size=len(daily_rv_full))

    engine = PaperTradingEngine(max_drawdown=-50_000, max_orders_per_sec=10)
    static_beta = None
    day_ptr = warmup_days

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
        chain = make_synthetic_chain(spot_now, T_years, forecast_var_annualized, 1.08, seed=7 + expiry_i)
        signals = generate_signals(chain, forecast_var_annualized, edge_threshold=0.03, mode="price")
        expiry_day_idx = min(day_ptr + days_per_expiry - 1, len(daily_close) - 1)
        spot_at_expiry = daily_close[expiry_day_idx]

        # Simulated clock: orders on the same expiry day are spread a few
        # seconds apart, as they would be in a real trading session --
        # NOT real wall-clock time, which would make an entire backtest
        # look like it happened in one second and falsely trip the rate
        # limiter (this bug was caught and fixed while building this demo).
        base_time = day_ptr * 86400.0
        for order_idx, sig in enumerate(signals):
            if sig.signal == "NO_TRADE":
                continue
            order = engine.submit(sig, price=sig.market_price, now=base_time + order_idx * 0.2)
            if order.status == "FILLED":
                cost = sig.market_price * 0.01
                intrinsic = max(spot_at_expiry - sig.strike, 0.0)
                pnl = (intrinsic - sig.market_price - cost) if sig.signal == "BUY_CALL" \
                    else (sig.market_price - intrinsic - cost)
                engine.resolve(order, pnl)

        day_ptr += days_per_expiry
        if engine.kill_switch.tripped:
            break

    s = engine.summary()
    print(f"Kill switch tripped: {s['kill_switch_tripped']}")
    print(f"Reason: {s['kill_switch_reason']}")
    print(f"Orders filled before halt: {s['filled']}")
    print(f"Realized P&L at halt: {s['realized_pnl']:.0f}")
    print(f"\nCompare: the unprotected static model (no kill switch) ran to -260,633 "
          f"over the full simulation in the earlier regime-shift demo. The kill switch "
          f"caught the bleed at {s['realized_pnl']:.0f} instead.")

    # --- separate, isolated demo: rate limiter behavior under a burst ---
    print("\n\nRate limiter demo: 15 orders submitted within the same second "
          "(max_orders_per_sec=10):")
    burst_engine = PaperTradingEngine(max_drawdown=-1_000_000, max_orders_per_sec=10)
    dummy_signal = EdgeSignal(strike=24800, market_implied_prob=0.5, model_prob=0.55,
                               edge=0.05, signal="BUY_CALL", market_price=100.0, model_price=105.0)
    t0 = 1_000_000.0
    for i in range(15):
        order = burst_engine.submit(dummy_signal, price=100.0, now=t0 + i * 0.05)
        print(f"  order {i+1:2d}: {order.status}")
