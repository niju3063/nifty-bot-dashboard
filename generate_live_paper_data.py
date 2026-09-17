"""
Live Paper Trading Data Adapter
====================================

Reads your REAL paper_state.json (produced by run_daily_paper_trade.py --
your actual forward paper-trading history, not a backtest) and converts
it into the same JSON shape build_dashboard.py already knows how to embed
as the Paper Trading panel's data. This REPLACES the canned synthetic
regime-shift demo that panel used to show with your genuine forward
track record.

Run this in place of generate_paper_trading_data.py in the pipeline --
it writes to the same output filename so build_dashboard.py needs no
changes.

    python3 generate_live_paper_data.py

Requires: paper_state.json to exist (run run_daily_paper_trade.py at
least once first). If it doesn't exist yet, writes an empty-but-valid
placeholder so the dashboard still builds cleanly.
"""

import json
from pathlib import Path
from datetime import datetime

STATE_PATH = "paper_state.json"
OUTPUT_PATH = "paper_trading_data.json"


def main():
    if not Path(STATE_PATH).exists():
        print(f"No {STATE_PATH} found yet -- writing an empty placeholder so the "
              f"dashboard still builds. Run run_daily_paper_trade.py first for real data.")
        placeholder = {
            "kill_switch_limit": -20000,
            "run_id": "PAPER",
            "log": [],
            "summary": {"total_orders": 0, "filled": 0, "blocked_by_kill_switch": 0,
                        "rejected_rate_limit": 0, "realized_pnl": 0.0,
                        "kill_switch_tripped": False, "kill_switch_reason": None},
            "is_live": True,
        }
        Path(OUTPUT_PATH).write_text(json.dumps(placeholder, separators=(",", ":")))
        return

    state = json.loads(Path(STATE_PATH).read_text())
    ks = state["kill_switch"]
    log_rows = []
    cumulative = 0.0

    # Resolved trades first, in chronological order
    for t in state["trade_log"]:
        cumulative += t["pnl"]
        log_rows.append({
            "day": t["entry_date"],
            "strategy_id": t["strategy_id"],
            "strike": t["strike"],
            "side": "SELL_CALL",
            "price": t["entry_price"],
            "status": "FILLED",
            "pnl": round(t["pnl"], 1),
            "cum_pnl": round(cumulative, 1),
        })

    # Then any still-open positions -- shown with pnl=null (not yet resolved)
    for p in state["open_positions"]:
        log_rows.append({
            "day": p["entry_date"],
            "strategy_id": p["strategy_id"],
            "strike": p["strike"],
            "side": "SELL_CALL",
            "price": p["entry_price"],
            "status": "OPEN",
            "pnl": None,
            "cum_pnl": round(cumulative, 1),
        })

    realized_pnls = [t["pnl"] for t in state["trade_log"]]
    summary = {
        "total_orders": len(log_rows),
        "filled": len(state["trade_log"]),
        "blocked_by_kill_switch": 0,  # not tracked separately in the daily scheduler's log
        "rejected_rate_limit": 0,
        "realized_pnl": round(sum(realized_pnls), 1) if realized_pnls else 0.0,
        "kill_switch_tripped": ks["tripped"],
        "kill_switch_reason": ks["trip_reason"],
    }

    data = {
        "kill_switch_limit": ks["max_drawdown"],
        "run_id": state.get("run_id", "PAPER"),
        "log": log_rows,
        "summary": summary,
        "is_live": True,
        "generated_at": datetime.now().isoformat(),
    }

    Path(OUTPUT_PATH).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUTPUT_PATH}: {len(state['trade_log'])} resolved, "
          f"{len(state['open_positions'])} open, cumulative P&L = {summary['realized_pnl']}")


if __name__ == "__main__":
    main()
