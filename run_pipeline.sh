#!/bin/bash
#
# Full Automated Pipeline: paper trade -> dashboard -> publish
# ==================================================================
#
# Runs unattended via launchd (see com.niftybot.dailyupdate.plist).
# Designed to fail SAFELY: if any step errors, it logs the failure and
# stops rather than pushing a broken or partial dashboard. The kill
# switch and trading state are never touched by a failed run -- only
# run_daily_paper_trade.py's own internal logic modifies paper_state.json,
# and it's idempotent/safe to re-run.
#
# SETUP (one-time, see the full instructions you were given separately):
#   1. This script must live in your git repo folder, alongside all the
#      .py files.
#   2. The repo must already be connected to a GitHub remote with push
#      access configured (SSH key or credential helper -- something that
#      doesn't require typing a password, since this runs unattended).
#   3. GitHub Pages must be enabled on the repo, serving from the /docs
#      folder on the main branch.
#   4. Make this file executable: chmod +x run_pipeline.sh
#
# WHAT IT DOES, IN ORDER:
#   1. Daily paper-trading decision (the actual trading logic)
#   2. Regenerate dashboard data (methodology/backtest -- synthetic,
#      unchanged run to run, but harmless to regenerate)
#   3. Regenerate live paper-trading data (YOUR real forward history)
#   4. Regenerate market context (real VIX/PCR/max pain)
#   5. Build the final HTML
#   6. Copy it into docs/index.html (GitHub Pages' expected location)
#   7. Commit and push -- this is the step that actually makes the
#      public page update, with no human involved.

set -e  # stop immediately on any error -- never push a partial/broken state

cd "$(dirname "$0")"  # always run from this script's own folder, regardless of how launchd invokes it

LOG_FILE="pipeline_log.txt"
echo "=== Pipeline run: $(date) ===" >> "$LOG_FILE"

run_step() {
    echo "--- $1 ---" >> "$LOG_FILE"
    if ! python3 "$1" >> "$LOG_FILE" 2>&1; then
        echo "FAILED at $1 -- stopping, nothing pushed. See $LOG_FILE for details." >> "$LOG_FILE"
        exit 1
    fi
}

run_step "run_daily_paper_trade.py"
run_step "generate_dashboard_data.py"
run_step "generate_live_paper_data.py"
run_step "generate_market_context_data.py"
run_step "build_dashboard.py"

mkdir -p docs
cp nifty-edge-terminal.html docs/index.html

# Only commit if something actually changed -- avoids empty commits on
# days where the market was closed and nothing new happened.
if git diff --quiet docs/ paper_state.json 2>/dev/null; then
    echo "No changes to publish today." >> "$LOG_FILE"
else
    git add docs/ paper_state.json
    git commit -m "Automated update: $(date '+%Y-%m-%d %H:%M')" >> "$LOG_FILE" 2>&1
    git push >> "$LOG_FILE" 2>&1
    echo "Pushed update at $(date)" >> "$LOG_FILE"
fi

echo "=== Pipeline run complete: $(date) ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
