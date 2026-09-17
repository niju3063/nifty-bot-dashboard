"""
Paper Trading Status Viewer
================================

Read-only. Shows what's currently open and what's already resolved,
without fetching anything from NSE or changing paper_state.json in any
way. Run this any time you want to check status without waiting for
the next scheduled daily run.

    python3 show_status.py
"""

import json
from pathlib import Path
from datetime import datetime

STATE_PATH = "paper_state.json"


def main():
    if not Path(STATE_PATH).exists():
        print(f"No {STATE_PATH} found yet -- run run_daily_paper_trade.py first.")
        return

    state = json.loads(Path(STATE_PATH).read_text())
    today = datetime.now().date()

    print(f"=== Paper Trading Status (as of {today}) ===\n")

    ks = state["kill_switch"]
    status = "TRIPPED" if ks["tripped"] else "ARMED"
    print(f"Kill switch: {status}")
    if ks["tripped"]:
        print(f"  Reason: {ks['trip_reason']}")
    print(f"  Cumulative equity: {ks['_equity']:+.1f}  (peak: {ks['_peak_equity']:+.1f}, "
          f"limit: {ks['max_drawdown']:.0f})\n")

    positions = state["open_positions"]
    print(f"Open positions ({len(positions)}):")
    if not positions:
        print("  (none)")
    for p in positions:
        expiry = datetime.strptime(p["expiry_date"], "%Y-%m-%d").date()
        days_left = (expiry - today).days
        print(f"  SELL_CALL strike={p['strike']:.0f} @ {p['entry_price']:.2f}  "
              f"entered {p['entry_date']}  expiry {p['expiry_date']} "
              f"({days_left} day(s) away)  [{p['strategy_id']}]")

    log = state["trade_log"]
    print(f"\nResolved trades ({len(log)}):")
    if not log:
        print("  (none yet)")
    for t in log:
        result = "WIN" if t["pnl"] > 0 else "LOSS"
        print(f"  {t['entry_date']} -> {t['resolve_date']}  strike={t['strike']:.0f}  "
              f"pnl={t['pnl']:+.1f}  [{result}]")

    if log:
        pnls = [t["pnl"] for t in log]
        wins = sum(1 for p in pnls if p > 0)
        print(f"\nForward track record: {wins}/{len(pnls)} wins "
              f"({wins/len(pnls):.1%}), cumulative P&L = {sum(pnls):+.1f}")


if __name__ == "__main__":
    main()
