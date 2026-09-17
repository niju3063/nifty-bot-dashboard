"""
Live Broker Data-Fetching Layer — Zerodha Kite Connect
=========================================================

Pulls the two data feeds the probability model needs:
  1. Historical intraday candles  -> for HAR-RV volatility forecasting
  2. Live option chain snapshots  -> for Breeden-Litzenberger extraction

Produces OptionChainSnapshot objects directly consumable by
probability_model.generate_signals().

This file is DATA-FETCHING ONLY — no order placement. Wiring this into
live order execution is a separate, higher-stakes module and should only
happen after the backtesting harness has validated the model on real
historical data (not just the synthetic data used so far).

--------------------------------------------------------------------
SEBI algo-framework notes that affect this layer specifically
(full framework: static IP, unique Strategy ID, daily 2FA, 10 ops/sec
 — see order-execution layer, not covered here):

  - Kite's historical API has its own rate limits (separate from the
    10 orders/sec order-placement cap) — this file backs off on 429s.
  - Kite session tokens expire daily; `authenticate()` below assumes
    you complete the login-flow redirect once per day and hand back
    the request_token (this is the "daily 2FA" requirement in
    practice for a personal, non-empanelled API setup).
--------------------------------------------------------------------

Setup required before this runs:
    pip install kiteconnect
    Kite Connect API key + secret (from developers.kite.trade)
    A completed login flow to obtain a request_token (manual, once/day)

Swap KiteDataFeed for an equivalent Fyers/Upstox wrapper if you're on
a different broker — the OptionChainSnapshot output contract is the
same either way, so nothing downstream (probability_model.py,
backtest.py) needs to change.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from probability_model import OptionChainSnapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("broker_data_feed")

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None  # allows this file to be imported/read without the package installed


# ======================================================================
# 1. Authentication
# ======================================================================

@dataclass
class KiteCredentials:
    api_key: str
    api_secret: str
    access_token: Optional[str] = None  # filled in after authenticate()


def authenticate(creds: KiteCredentials, request_token: str) -> KiteCredentials:
    """
    Completes the Kite login flow for the day.

    Manual step required first (do this once per trading day, per SEBI's
    daily-2FA requirement — refresh-token-only sessions are no longer
    supported under the April 2026 framework):
        1. Visit: KiteConnect(api_key=creds.api_key).login_url()
        2. Log in, get redirected with `request_token` in the URL
        3. Pass that request_token here

    Returns updated creds with access_token populated. Cache the
    access_token for the rest of the day; a fresh request_token is
    needed again tomorrow.
    """
    if KiteConnect is None:
        raise ImportError("pip install kiteconnect first")

    kite = KiteConnect(api_key=creds.api_key)
    session_data = kite.generate_session(request_token, api_secret=creds.api_secret)
    creds.access_token = session_data["access_token"]
    logger.info("Kite session authenticated for today.")
    return creds


def get_kite_client(creds: KiteCredentials) -> "KiteConnect":
    if KiteConnect is None:
        raise ImportError("pip install kiteconnect first")
    if not creds.access_token:
        raise ValueError("Call authenticate() first — no access_token set.")
    kite = KiteConnect(api_key=creds.api_key)
    kite.set_access_token(creds.access_token)
    return kite


# ======================================================================
# 2. Historical intraday candles -> feeds HAR-RV
# ======================================================================

def fetch_historical_candles(kite: "KiteConnect", instrument_token: int,
                              from_date: datetime, to_date: datetime,
                              interval: str = "5minute", max_retries: int = 3) -> pd.DataFrame:
    """
    Pulls historical OHLC candles for realized-variance calculation.

    Kite's historical API caps each request to ~60 days for minute-level
    intervals, so long lookbacks are chunked automatically here.

    Returns a DataFrame with columns [date, open, high, low, close, volume],
    sorted ascending by date.
    """
    chunk_days = 60
    all_candles = []
    chunk_start = from_date

    while chunk_start < to_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), to_date)

        for attempt in range(max_retries):
            try:
                candles = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=chunk_start, to_date=chunk_end,
                    interval=interval,
                )
                all_candles.extend(candles)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"historical_data failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"historical_data failed after {max_retries} retries "
                                f"for chunk {chunk_start} - {chunk_end}")

        chunk_start = chunk_end
        time.sleep(0.35)  # stay well under Kite's historical-API rate limit

    df = pd.DataFrame(all_candles)
    if df.empty:
        raise ValueError("No candle data returned — check instrument_token / date range.")
    return df.sort_values("date").reset_index(drop=True)


def candles_to_log_returns(df: pd.DataFrame) -> np.ndarray:
    """Close-to-close log returns, in chronological order, ready for
    probability_model.realized_variance()."""
    closes = df["close"].to_numpy()
    return np.diff(np.log(closes))


# ======================================================================
# 3. Live option chain -> feeds Breeden-Litzenberger extraction
# ======================================================================

def get_instrument_dump(kite: "KiteConnect", exchange: str = "NFO") -> pd.DataFrame:
    """
    Kite requires a full instrument-list dump to map (underlying, strike,
    expiry, option-type) -> the tradingsymbol/instrument_token needed for
    quotes. Cache this locally and refresh once a day (it's a big file
    and doesn't change intraday).
    """
    instruments = kite.instruments(exchange)
    return pd.DataFrame(instruments)


def get_option_chain_snapshot(kite: "KiteConnect", instrument_df: pd.DataFrame,
                               underlying: str, expiry: str,
                               spot: float, r: float = 0.065) -> OptionChainSnapshot:
    """
    Builds an OptionChainSnapshot for a given underlying + expiry, ready
    to feed straight into probability_model.generate_signals().

    Parameters
    ----------
    underlying : "NIFTY" or "BANKNIFTY"
    expiry     : "2026-09-25" (ISO date string, matches Kite's expiry field)
    spot       : current spot price (pull separately via kite.ltp on the
                 index instrument, e.g. "NSE:NIFTY 50")
    """
    chain_rows = instrument_df[
        (instrument_df["name"] == underlying)
        & (instrument_df["expiry"].astype(str) == expiry)
        & (instrument_df["instrument_type"] == "CE")
    ].sort_values("strike")

    if chain_rows.empty:
        raise ValueError(f"No CE contracts found for {underlying} expiry {expiry} "
                          f"— check the expiry date format matches instrument_df.")

    tradingsymbols = [f"NFO:{s}" for s in chain_rows["tradingsymbol"]]

    # Kite's quote() call is capped at 500 instruments per request — fine
    # for a single expiry's chain, chunk it if pulling several expiries at once.
    quotes = kite.quote(tradingsymbols)

    strikes, mid_prices = [], []
    for _, row in chain_rows.iterrows():
        key = f"NFO:{row['tradingsymbol']}"
        q = quotes.get(key)
        if q is None:
            continue
        bid = q["depth"]["buy"][0]["price"] if q["depth"]["buy"] else None
        ask = q["depth"]["sell"][0]["price"] if q["depth"]["sell"] else None
        if bid and ask and bid > 0 and ask > 0:
            strikes.append(row["strike"])
            mid_prices.append((bid + ask) / 2)
        elif q.get("last_price", 0) > 0:
            # fallback to LTP for illiquid strikes with no live depth —
            # flag these, since LTP can be stale on thin strikes
            logger.warning(f"No live depth for strike {row['strike']}, using LTP fallback.")
            strikes.append(row["strike"])
            mid_prices.append(q["last_price"])

    if len(strikes) < 5:
        raise ValueError("Too few liquid strikes returned to build a usable chain — "
                          "check market hours / expiry liquidity.")

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    T_years = max((expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600), 1e-6)

    return OptionChainSnapshot(
        spot=spot,
        strikes=np.array(strikes, dtype=float),
        call_mid_prices=np.array(mid_prices, dtype=float),
        r=r,
        T=T_years,
    )


def get_index_spot(kite: "KiteConnect", index_symbol: str = "NSE:NIFTY 50") -> float:
    """Live spot price for the underlying index."""
    quote = kite.ltp([index_symbol])
    return float(quote[index_symbol]["last_price"])


# ======================================================================
# 4. Example wiring (requires real credentials — will not run as-is)
# ======================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nThis module requires real Kite Connect credentials and a live \n"
          "market session to execute — it's a template, not a runnable demo.\n"
          "Wiring sketch:\n\n"
          "    creds = KiteCredentials(api_key='...', api_secret='...')\n"
          "    creds = authenticate(creds, request_token='...')  # once/day\n"
          "    kite = get_kite_client(creds)\n\n"
          "    instrument_df = get_instrument_dump(kite)\n"
          "    spot = get_index_spot(kite, 'NSE:NIFTY 50')\n"
          "    chain = get_option_chain_snapshot(kite, instrument_df,\n"
          "                                       underlying='NIFTY',\n"
          "                                       expiry='2026-09-25',\n"
          "                                       spot=spot)\n\n"
          "    hist = fetch_historical_candles(kite, instrument_token=256265,\n"
          "        from_date=datetime.now()-timedelta(days=90), to_date=datetime.now())\n"
          "    log_rets = candles_to_log_returns(hist)\n\n"
          "    # -> feed `chain` and `log_rets` into probability_model.py /\n"
          "    #    backtest.py exactly as the synthetic versions were used.")
