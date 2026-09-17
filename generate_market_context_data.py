"""
Market Context Data Generator
==================================

Produces market_context_data.json: real India VIX, PCR, and Max Pain
series, plus the model's own realized-vol forecast for comparison
against VIX (this is what visualizes the variance-risk-premium finding
on the dashboard -- if VIX consistently runs above realized/forecast
vol, that gap is the premium a short-vol strategy harvests).

These are CONTEXT indicators, not validated trading signals -- nothing
in this pipeline has found a real edge, and adding more charts doesn't
change that. Presented as market conditions to observe, same as you'd
read any financial news dashboard.

Requires a live NSE fetch (same as the other generator scripts).
Run alongside generate_dashboard_data.py / generate_paper_trading_data.py.

    python3 generate_market_context_data.py
"""

import json
import numpy as np
from datetime import datetime, timedelta

from fetch_nse_historical import build_historical_dataset_with_context, fetch_vix_series
from probability_model import realized_variance, har_rv_forecast

OUTPUT_PATH = "market_context_data.json"


def main(days_back=400, symbol="NIFTY"):
    end = datetime.now()
    start = end - timedelta(days=days_back)

    print(f"Fetching market context data for {symbol}, {start.date()} to {end.date()}...")
    chains, spot_series, context_df = build_historical_dataset_with_context(start, end, symbol=symbol)
    vix_series = fetch_vix_series(start, end)

    if len(spot_series) < 30:
        raise ValueError(f"Only fetched {len(spot_series)} days -- need more history for a useful context panel.")

    # Rolling realized-vol forecast (same HAR-RV as the rest of the pipeline),
    # annualized, aligned to the same dates as VIX for direct comparison.
    log_rets = np.diff(np.log(spot_series.values))
    daily_var = log_rets ** 2
    dates = list(spot_series.index)

    realized_vol_pct = {}
    for i in range(22, len(dates)):
        try:
            forecast_var = har_rv_forecast(daily_var[:i], horizon_days=1)
            realized_vol_pct[dates[i]] = float(np.sqrt(forecast_var * 252)) * 100
        except ValueError:
            continue

    # Align all three series to common dates for a clean chart
    common_dates = sorted(set(realized_vol_pct.keys()) & set(vix_series.index))
    vix_vs_realized = [
        {"date": str(d), "vix": round(float(vix_series.loc[d]), 2),
         "realized_vol_forecast": round(realized_vol_pct[d], 2)}
        for d in common_dates
    ]

    pcr_series = [
        {"date": str(d), "pcr": None if pd_isna(row["pcr"]) else round(float(row["pcr"]), 3)}
        for d, row in context_df.iterrows()
    ] if not context_df.empty else []

    max_pain_vs_spot = [
        {"date": str(d), "max_pain": None if pd_isna(row["max_pain"]) else float(row["max_pain"]),
         "spot": float(spot_series.loc[d]) if d in spot_series.index else None}
        for d, row in context_df.iterrows()
    ] if not context_df.empty else []

    avg_gap = None
    if vix_vs_realized:
        gaps = [row["vix"] - row["realized_vol_forecast"] for row in vix_vs_realized]
        avg_gap = round(float(np.mean(gaps)), 2)

    data = {
        "vix_vs_realized": vix_vs_realized,
        "pcr_series": pcr_series,
        "max_pain_vs_spot": max_pain_vs_spot,
        "avg_vix_minus_realized_vol": avg_gap,
        "latest_pcr": pcr_series[-1]["pcr"] if pcr_series else None,
        "latest_vix": vix_vs_realized[-1]["vix"] if vix_vs_realized else None,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"Wrote {OUTPUT_PATH}: {len(vix_vs_realized)} VIX-vs-realized points, "
          f"{len(pcr_series)} PCR points, avg VIX-minus-realized gap = {avg_gap}")


def pd_isna(x):
    import pandas as pd
    return pd.isna(x)


if __name__ == "__main__":
    main()
