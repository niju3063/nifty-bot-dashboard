"""
Daily Forward Paper-Trading Scheduler
=========================================

IMPORTANT -- read this before running:

Nothing in this pipeline has demonstrated a real, tradeable edge. What
IS demonstrated: a naive "sell the ATM call every expiry" baseline
showed a strong Sharpe (4.38) over one 13-month backtest window that
happened to contain no volatility shock. That is the ONLY strategy this
scheduler paper-trades -- the HAR-RV / Breeden-Litzenberger edge-
detection model showed NO benefit over that baseline and is not used
here. Automating a disproven model would just automate losing money
faster.

This scheduler does NOT place real orders. It fetches the latest
available NSE data, decides what the sell-ATM strategy would do today,
logs it as a PAPER order through paper_trading.py's engine (Strategy ID,
rate limiter, kill switch all active), and tracks FORWARD performance
-- decisions made without already knowing the outcome, which is the one
thing a backtest can never give you. State persists in paper_state.json
so each day builds on the last.

RUN THIS ONCE PER TRADING DAY, after NSE publishes the day's bhavcopy
(typically evening IST). Running it twice on the same trading day is
safe -- it won't double-enter a position it already holds for that day's
contract, but it also won't do anything new until the next trading day's
data is available.

    python3 run_daily_paper_trade.py

To actually run this daily, add a cron job (macOS/Linux), e.g. to run
every weekday at 7pm local time:
    crontab -e
    0 19 * * 1-5 cd /path/to/nifty-bot && /usr/bin/python3 run_daily_paper_trade.py >> paper_trade_log.txt 2>&1

Requires: fetch_nse_historical.py, paper_trading.py in the same folder.
"""

from __future__ import annotations
import json
import dataclasses
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from fetch_nse_historical import get_nse_session, download_fo_bhavcopy, normalize_fo_columns, _nearest_expiry
from paper_trading import PaperTradingEngine, KillSwitch, RateLimiter, Order, make_strategy_id

STATE_PATH = "paper_state.json"
MAX_DRAWDOWN = -20_000       # paper-capital kill switch limit -- adjust to whatever you'd actually risk
TRANSACTION_COST_PCT = 0.01
MIN_DAYS_TO_EXPIRY = 1.0     # don't enter same-day (0DTE) contracts -- see earlier discussion on why


# ======================================================================
# 1. Fetch the most recent available trading day's chain (not a range --
#    this is a single-day lookup, walking back a few days if today's
#    file isn't published yet).
# ======================================================================

def fetch_latest_chain(symbol: str = "NIFTY", max_lookback_days: int = 5):
    """
    Returns (as_of_date, spot, expiry_date, strikes, close_prices) for the
    most recent trading day NSE has actually published -- which may not
    be "today" if you run this before the evening bhavcopy is out, or on
    a weekend/holiday. Returns None if nothing usable is found within
    the lookback window.
    """
    session = get_nse_session()
    d = datetime.now()
    for _ in range(max_lookback_days):
        if d.weekday() < 5:
            raw = download_fo_bhavcopy(d, session)
            normalized = normalize_fo_columns(raw)
            if not normalized.empty:
                expiry_str = _nearest_expiry(normalized, symbol, d)
                if expiry_str:
                    subset = normalized[
                        (normalized["symbol"].astype(str).str.upper() == symbol.upper())
                        & (normalized["expiry"].astype(str).str.contains(expiry_str, na=False))
                        & (normalized["option_type"].astype(str).str.upper() == "CE")
                    ].sort_values("strike")
                    if len(subset) >= 5:
                        spot = float(subset["underlying_close"].iloc[0])
                        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
                        return (d.date(), spot, expiry_date,
                                subset["strike"].to_numpy(dtype=float),
                                subset["close"].to_numpy(dtype=float))
        d -= timedelta(days=1)
    return None


# ======================================================================
# 2. Persistent state
# ======================================================================

def load_state(path: str) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {
        "kill_switch": dataclasses.asdict(KillSwitch(max_drawdown=MAX_DRAWDOWN)),
        "open_positions": [],   # list of {strategy_id, entry_date, strike, expiry_date, entry_price}
        "trade_log": [],        # list of resolved trades
        "last_entry_date": None,
        "run_id": None,
    }


def save_state(path: str, state: dict):
    Path(path).write_text(json.dumps(state, indent=2, default=str))


# ======================================================================
# 3. Daily run
# ======================================================================

def run_daily(symbol: str = "NIFTY"):
    state = load_state(STATE_PATH)
    kill_switch = KillSwitch(**state["kill_switch"])
    run_id = state["run_id"] or make_strategy_id("PAPER", 0).split("-")[1]  # reuse or mint a run id

    print(f"=== Daily paper trade run: {datetime.now().date()} ===\n")

    fetched = fetch_latest_chain(symbol)
    if fetched is None:
        print("Could not fetch any usable recent chain data (weekend/holiday/NSE issue). "
              "Nothing to do today -- try again tomorrow.")
        return
    as_of_date, spot, expiry_date, strikes, close_prices = fetched
    print(f"Latest available data: {as_of_date} (spot={spot:.1f}, nearest expiry={expiry_date.date()})\n")

    # --- 3a. Resolve any open positions whose expiry has arrived ---
    still_open = []
    newly_resolved = []
    for pos in state["open_positions"]:
        pos_expiry = datetime.strptime(pos["expiry_date"], "%Y-%m-%d")
        if pos_expiry.date() <= as_of_date:
            if pos_expiry.date() < as_of_date:
                print(f"  Note: position expiring {pos_expiry.date()} is resolving late "
                      f"(latest data is {as_of_date}) -- using today's spot as the best "
                      f"available proxy for the settlement price.")
            cost = pos["entry_price"] * TRANSACTION_COST_PCT
            intrinsic = max(spot - pos["strike"], 0.0)
            pnl = pos["entry_price"] - intrinsic - cost  # SELL_CALL payoff
            kill_switch.update(pnl)
            record = {**pos, "resolve_date": str(as_of_date), "spot_at_expiry": spot, "pnl": round(pnl, 2)}
            newly_resolved.append(record)
            state["trade_log"].append(record)
            print(f"  RESOLVED: strike {pos['strike']:.0f}, entered {pos['entry_date']}, "
                  f"pnl = {pnl:+.1f}")
        else:
            still_open.append(pos)
    state["open_positions"] = still_open

    if not newly_resolved:
        print("  No positions resolved today.")

    # --- 3b. Kill switch check ---
    if kill_switch.tripped:
        print(f"\n  KILL SWITCH TRIPPED: {kill_switch.trip_reason}")
        print("  No new positions will be entered. Call reset_for_new_session() deliberately "
              "in a Python shell if you've reviewed why it tripped and want to resume:")
        print(f"    from run_daily_paper_trade import load_state, save_state, STATE_PATH")
        print(f"    from paper_trading import KillSwitch")
        print(f"    s = load_state(STATE_PATH); ks = KillSwitch(**s['kill_switch'])")
        print(f"    ks.reset_for_new_session(); s['kill_switch'] = __import__('dataclasses').asdict(ks)")
        print(f"    save_state(STATE_PATH, s)")
    else:
        # --- 3c. Enter a new position, if not already holding one for
        #          this expiry, and if it's not too close to expiry ---
        already_holding = any(
            datetime.strptime(p["expiry_date"], "%Y-%m-%d").date() == expiry_date.date()
            for p in state["open_positions"]
        )
        days_to_expiry = (expiry_date.date() - as_of_date).days
        already_ran_today = state["last_entry_date"] == str(as_of_date)

        if already_ran_today:
            print(f"\n  Already made today's entry decision for {as_of_date} -- skipping "
                  f"(safe to run this script more than once per day).")
        elif already_holding:
            print(f"\n  Already holding a position for the {expiry_date.date()} expiry -- "
                  f"no new entry today.")
        elif days_to_expiry < MIN_DAYS_TO_EXPIRY:
            print(f"\n  Nearest expiry is only {days_to_expiry} day(s) out -- skipping "
                  f"(0DTE-like, excluded per the validation findings).")
        else:
            idx_atm = int(np.argmin(np.abs(strikes - spot)))
            atm_strike = float(strikes[idx_atm])
            atm_price = float(close_prices[idx_atm])

            new_position = {
                "strategy_id": make_strategy_id(run_id, len(state["trade_log"]) + len(state["open_positions"]) + 1),
                "entry_date": str(as_of_date),
                "strike": atm_strike,
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "entry_price": atm_price,
            }
            state["open_positions"].append(new_position)
            print(f"\n  NEW PAPER ORDER: SELL_CALL strike={atm_strike:.0f} @ {atm_price:.2f} "
                  f"(strategy_id={new_position['strategy_id']}, expiry={expiry_date.date()})")

        state["last_entry_date"] = str(as_of_date)

    # --- 3d. Save state ---
    state["kill_switch"] = dataclasses.asdict(kill_switch)
    state["run_id"] = run_id
    save_state(STATE_PATH, state)

    # --- 3e. Summary ---
    resolved_pnls = [t["pnl"] for t in state["trade_log"]]
    print(f"\n--- Summary ---")
    print(f"  Open positions: {len(state['open_positions'])}")
    print(f"  Resolved trades to date: {len(resolved_pnls)}")
    if resolved_pnls:
        print(f"  Cumulative forward P&L: {sum(resolved_pnls):+.1f}")
        print(f"  Forward win rate: {np.mean(np.array(resolved_pnls) > 0):.1%}")
    print(f"  Kill switch: {'TRIPPED' if kill_switch.tripped else 'armed'}")


if __name__ == "__main__":
    run_daily()
