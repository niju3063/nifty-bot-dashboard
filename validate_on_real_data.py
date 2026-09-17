"""
Real-Data Validation
========================

Everything up to this point (probability_model.py, backtest.py,
adaptive_model.py) has only been proven correct against SYNTHETIC data
-- it proves the code works, not that Nifty options actually behave
this way. This script is the actual test: same signal logic, same
scoring, fed real NSE historical chains instead.

RUN THIS YOURSELF -- it needs a live network call to NSE, which this
sandbox can't make. Requires fetch_nse_historical.py in the same folder.

    python3 validate_on_real_data.py

--------------------------------------------------------------------
IMPORTANT LIMITATION: bhavcopy only has daily CLOSE prices, not
intraday bars. probability_model.py's HAR-RV was built for intraday
realized variance (many bars/day). Real historical intraday data
requires either your broker's historical API (chargeable, needs Kite
Connect) or a paid data vendor.

Until then, this script uses a documented, standard substitute:
close-to-close daily squared log returns as a single daily variance
observation, and fits a "HAR-vol" style model on THAT instead of
proper intraday RV. This is noisier and less accurate than true
intraday RV -- flagged explicitly here and in the printed output --
but it's the honest free option and still tests the actual model
architecture (edge detection, calibration, price vs probability mode)
against real market prices.

Once you have Kite Connect running (broker_data_feed.py), swap this
daily-proxy variance for real intraday RV via realized_variance() from
probability_model.py, and this script's logic barely changes.
--------------------------------------------------------------------

Expiry resolution here is APPROXIMATE: chain.T (years to expiry) is
inverted back to a calendar date and matched to the nearest available
spot observation on or after it, since OptionChainSnapshot doesn't
carry the expiry date directly. Good enough for validation; don't
treat it as exact settlement-day precision.
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fetch_nse_historical import build_historical_dataset
from probability_model import har_rv_forecast, generate_signals
from backtest import Trade, BacktestResult

MIN_HISTORY_DAYS = 22  # HAR-RV's own minimum, from probability_model.py


def daily_variance_proxy(spot_series: pd.Series) -> np.ndarray:
    """Close-to-close squared log returns -- see module docstring for
    why this substitutes for real intraday realized variance here."""
    log_rets = np.diff(np.log(spot_series.values))
    return log_rets ** 2


def find_resolution_date(as_of_date, T_years: float, available_dates: list) -> "datetime | None":
    """Inverts chain.T back to an approximate expiry calendar date and
    finds the nearest fetched date on/after it. Returns None if the
    expiry hasn't resolved within the fetched window yet (can't score
    that trade)."""
    expiry_approx = as_of_date + timedelta(days=T_years * 365.25)
    future = [d for d in available_dates if d >= expiry_approx]
    return min(future) if future else None


def run_validation(start_date: datetime, end_date: datetime, symbol: str = "NIFTY",
                    edge_threshold: float = 0.03, transaction_cost_pct: float = 0.01,
                    z_band: float = 1.5, min_days_to_expiry: float = 1.0) -> BacktestResult:
    """
    z_band restricts trading to strikes within +/- z_band standard
    deviations of spot, using the MODEL'S OWN vol forecast and the
    option's actual time-to-expiry -- not a flat percentage of spot.

    This matters more than the earlier flat-percentage version did: for
    a weekly option with only a few days left, one standard deviation of
    price movement can be under 2% of spot. A flat 8% band is then 4+
    standard deviations wide, meaning nearly every strike in it is
    already "certain" regardless of any real skew or liquidity issue --
    that's a scaling mismatch, not a mispricing. Scoring against sigma
    fixes this at any expiry horizon.

    min_days_to_expiry skips chains where the nearest expiry is the
    SAME DAY (0DTE-like): near-zero time-to-expiry makes even small
    moneyness look near-binary by construction, which is a genuine
    option-pricing feature, not something this filter should paper
    over -- it's a structurally different animal from a multi-day trade
    and is excluded here rather than silently distorted.

    Contracts are also deduplicated by (strike, resolve_date): the same
    real option shows up in the chain on every day leading up to its
    expiry, so without deduplication a single persistent signal on one
    contract gets logged as several "trades" -- inflating both the
    trade count and the Sharpe ratio by counting one real bet multiple
    times. Only the first day a contract triggers is counted.
    """
    print(f"Fetching real NSE data for {symbol}, {start_date.date()} to {end_date.date()}...")
    chains, spot_series = build_historical_dataset(start_date, end_date, symbol=symbol)

    if len(spot_series) < MIN_HISTORY_DAYS + 5:
        raise ValueError(
            f"Only fetched {len(spot_series)} usable trading days -- need at least "
            f"{MIN_HISTORY_DAYS + 5}. Either widen the date range, or NSE's format/URL "
            f"has changed again (see fetch_nse_historical.py's module docstring for "
            f"the current known URL and how to check for a newer one)."
        )
    print(f"Fetched {len(spot_series)} trading days, {len(chains)} usable option chain snapshots.\n")

    dates = list(spot_series.index)
    daily_var = daily_variance_proxy(spot_series)  # daily_var[i] ~ variance realized between dates[i] and dates[i+1]

    trades = []
    skipped_no_expiry_yet = 0
    skipped_out_of_band = 0
    skipped_too_close_to_expiry = 0
    skipped_duplicate_contract = 0
    one_sigma_moves = []
    seen_contracts = set()  # (strike, resolve_date) -- see docstring on double-counting

    for i, date in enumerate(dates):
        if date not in chains:
            continue
        if i < MIN_HISTORY_DAYS + 1:
            continue  # not enough trailing history for HAR-RV yet

        chain = chains[date]
        if chain.T * 252 < min_days_to_expiry:
            skipped_too_close_to_expiry += 1
            continue

        trailing_var = daily_var[:i]
        try:
            forecast_var_daily = har_rv_forecast(trailing_var, horizon_days=1)
        except ValueError:
            continue
        forecast_var_annualized = forecast_var_daily * 252  # simple annualization of the daily proxy

        signals = generate_signals(chain, forecast_var_annualized, edge_threshold=edge_threshold, mode="price")

        resolve_date = find_resolution_date(date, chain.T, dates)
        if resolve_date is None:
            skipped_no_expiry_yet += len([s for s in signals if s.signal != "NO_TRADE"])
            continue
        spot_at_expiry = float(spot_series.loc[resolve_date])

        sigma = np.sqrt(forecast_var_annualized)
        one_sigma_move = sigma * np.sqrt(chain.T)  # fractional move, e.g. 0.018 = 1.8% of spot
        one_sigma_moves.append(one_sigma_move)
        lower_bound = chain.spot * np.exp(-z_band * one_sigma_move)
        upper_bound = chain.spot * np.exp(z_band * one_sigma_move)

        for sig in signals:
            if sig.signal == "NO_TRADE":
                continue
            if not (lower_bound <= sig.strike <= upper_bound):
                skipped_out_of_band += 1
                continue

            contract_key = (round(sig.strike, 1), resolve_date)
            if contract_key in seen_contracts:
                skipped_duplicate_contract += 1
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

    if skipped_no_expiry_yet:
        print(f"Note: {skipped_no_expiry_yet} signals near the end of the window couldn't be "
              f"scored yet -- their expiry falls after your fetched date range ends.\n")
    if skipped_too_close_to_expiry:
        print(f"Note: {skipped_too_close_to_expiry} days were within {min_days_to_expiry:.0f} day(s) "
              f"of expiry and skipped (0DTE-like, structurally different -- see docstring).\n")
    if skipped_out_of_band:
        print(f"Note: {skipped_out_of_band} signals were outside the +/-{z_band} standard-deviation "
              f"band and excluded.\n")
    if skipped_duplicate_contract:
        print(f"Note: {skipped_duplicate_contract} signals were the SAME contract (strike+expiry) "
              f"already entered on an earlier day, and were skipped to avoid counting one real bet "
              f"as several trades.\n")
    if one_sigma_moves:
        arr = np.array(one_sigma_moves)
        print(f"One-sigma move across the run: min={arr.min():.2%}  median={np.median(arr):.2%}  "
              f"max={arr.max():.2%} of spot -- this is what the z_band multiplies.\n")

    return BacktestResult(trades=trades)


def print_report(result: BacktestResult):
    summary = result.summary()
    if summary.get("n_trades", 0) == 0:
        print("No trades were generated. Either the edge_threshold is too strict for the "
              "fetched window, or too few chain snapshots were successfully built -- check "
              "the fetch log above for warnings.")
        return

    print("=== REAL-DATA VALIDATION RESULT ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")

    if summary["win_rate"] > 0.80 or summary["sharpe_annualized"] > 3.0:
        print("\n*** SANITY CHECK: this win rate / Sharpe is implausibly high for liquid")
        print("    index options. Before trusting this, check whether trades are ")
        print("    concentrated in deep ITM/OTM strikes (see the calibration table's")
        print("    extreme buckets) -- that usually means the flat-volatility model is")
        print("    missing the real market's volatility skew, not finding genuine edge.")
        print("    Try tightening z_band (e.g. to 1.0) and re-running. ***\n")

    print("\nCalibration (model_prob bucket vs realized frequency):")
    calib = result.calibration_table()
    if not calib.empty:
        print(calib.to_string(index=False))
    else:
        print("  (not enough resolved trades yet to build a calibration table)")

    print("\n--- How to read this against the synthetic-data results ---")
    print("Synthetic backtest (price-mode) showed: ~70% win rate, positive total P&L,")
    print("well-calibrated probabilities. If real data shows something close to that,")
    print("the model architecture is worth taking further. If win rate is near 50% and")
    print("P&L is roughly zero after costs, that's the honest null result: the specific")
    print("mispricing this model looks for may not exist in real Nifty options at this")
    print("edge_threshold -- which is a normal, useful outcome to learn before risking capital.")


def directional_breakdown(result: BacktestResult):
    """Splits P&L by BUY_CALL vs SELL_CALL. If almost all the profit sits
    on one side, that's a sign the 'edge' is really a directional bet in
    disguise (e.g. selling calls that only looked rich because the vol
    model's risk-neutral drift assumption didn't match how the market
    actually moved), not a genuine volatility-mispricing edge."""
    print("\nDirectional breakdown (BUY_CALL vs SELL_CALL):")
    for side in ["BUY_CALL", "SELL_CALL"]:
        pnls = np.array([t.pnl for t in result.trades if t.signal == side])
        if len(pnls) == 0:
            print(f"  {side}: no trades")
            continue
        print(f"  {side}: n={len(pnls):4d}  win_rate={np.mean(pnls > 0):.1%}  "
              f"total_pnl={pnls.sum():>10.0f}  avg={pnls.mean():>7.1f}")


def block_bootstrap_pnl(result: BacktestResult, n_boot: int = 2000, seed: int = 0):
    """
    Trades resolving on the same expiry date are NOT independent -- they
    all depend on the same underlying outcome. A naive Sharpe ratio
    treats every trade as an independent observation, which overstates
    confidence. This resamples whole expiry-cycle CLUSTERS (grouped by
    shared spot_at_expiry, since same-expiry trades share that value)
    rather than individual trades, giving an honest confidence interval
    that respects the real correlation structure.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for t in result.trades:
        groups[round(t.spot_at_expiry, 2)].append(t.pnl)
    blocks = list(groups.values())
    n_blocks = len(blocks)

    rng = np.random.default_rng(seed)
    totals = np.array([
        sum(sum(blocks[i]) for i in rng.integers(0, n_blocks, size=n_blocks))
        for _ in range(n_boot)
    ])

    print(f"\nBlock bootstrap ({n_blocks} independent expiry clusters, {n_boot} resamples):")
    print(f"  5th percentile total P&L:  {np.percentile(totals, 5):>10.0f}")
    print(f"  median total P&L:          {np.percentile(totals, 50):>10.0f}")
    print(f"  95th percentile total P&L: {np.percentile(totals, 95):>10.0f}")
    print(f"  P(total P&L <= 0):         {np.mean(totals <= 0):.1%}")
    print("  -> if that last number isn't comfortably low (well under 20%), the point")
    print("     estimate above is not strong evidence of a real edge yet -- it's still")
    print("     plausibly noise given how few independent expiry cycles you have.")


if __name__ == "__main__":
    end = datetime.now()
    start = end - timedelta(days=400)  # ~1 year of trading days (~250), for a real read on
                                        # whether any edge survives more than a handful of
                                        # independent expiry cycles. At ~1.5s/day this run
                                        # takes roughly 8-12 minutes -- that's expected, not stuck.
    try:
        result = run_validation(start, end, symbol="NIFTY")
        print_report(result)
        directional_breakdown(result)
        block_bootstrap_pnl(result)
    except Exception as e:
        print(f"\nValidation run failed: {e}", file=sys.stderr)
        print("\nCommon causes: NSE blocked the requests (try again in a few minutes, or "
              "check your IP isn't rate-limited), the URL format changed (see "
              "fetch_nse_historical.py's docstring), or the date range has too many "
              "holidays/weekends with no data.", file=sys.stderr)
