"""
Free NSE Historical Data Fetcher — for offline model validation
===================================================================

Downloads NSE's public daily F&O "bhavcopy" archive (free, no API key,
no broker subscription needed) and reconstructs:
  1. Daily option chain snapshots (OptionChainSnapshot, same shape the
     live Kite layer produces) -- for testing Breeden-Litzenberger
  2. Daily underlying index closes -- for testing HAR-RV

Use this BEFORE paying for Kite Connect's ₹500/month API subscription:
run the model + backtest against real historical data first, and only
move to the live broker layer once this offline pass shows a real edge.

--------------------------------------------------------------------
IMPORTANT — NSE's file format/URLs change periodically. As of this
writing (Sept 2026):
  - Current ("UDIFF") FO bhavcopy:
      https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
  - Legacy pre-May-2024 FO bhavcopy (for older history):
      https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MON}/fo{DD}{MON}{YYYY}bhav.csv.zip
  - NSE has announced a further move toward .DAT-format extranet
    dissemination rolling out July-Oct 2026, so expect this to need
    updating again. If both URLs below start failing, check
    https://www.nseindia.com/all-reports-derivatives for whatever the
    current file link is and update FO_BHAVCOPY_URL_UDIFF accordingly.

  - This limitation is exactly why the flexible column-detector below
    exists: rather than hardcoding exact header names, it pattern-matches
    for symbol/expiry/strike/option-type/price columns so small header
    tweaks don't break the whole parser.

  - Bhavcopy gives daily CLOSE prices only, not live bid/ask -- so the
    OptionChainSnapshot built here uses close price as a "mid" proxy.
    This is an approximation, particularly for illiquid strikes with
    stale closes. Fine for validating the model's shape and calibration
    offline; the live Kite layer's real bid/ask mid is what you'd use
    once trading for real.
--------------------------------------------------------------------

Dependencies: requests, pandas, numpy
"""

from __future__ import annotations
import io
import time
import zipfile
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd

from probability_model import OptionChainSnapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_nse_historical")

FO_BHAVCOPY_URL_UDIFF = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
FO_BHAVCOPY_URL_LEGACY = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year}/{mon}/fo{dd}{mon}{year}bhav.csv.zip"
INDEX_CLOSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{date_str}.csv"

# Format changed here -- legacy URL for dates before this, UDIFF after.
UDIFF_CUTOVER_DATE = datetime(2024, 7, 8)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# ======================================================================
# 1. Session + download (NSE requires warmed-up cookies from the main
#    site before archive downloads succeed -- hitting the archive URL
#    cold, without visiting nseindia.com first, gets rejected)
# ======================================================================

def get_nse_session() -> requests.Session:
    """Warms up cookies by visiting the main site first, as NSE's
    anti-scraping layer requires a valid session before archive
    downloads work."""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(1)  # brief pause mimics natural browsing, reduces block risk
    return session


def download_fo_bhavcopy(date: datetime, session: Optional[requests.Session] = None,
                          max_retries: int = 3) -> pd.DataFrame:
    """
    Downloads and unzips one day's F&O bhavcopy, returns the raw CSV as
    a DataFrame (unparsed column names -- normalize with
    `normalize_fo_columns()` before use, since NSE's exact header names
    have changed across format versions).
    """
    session = session or get_nse_session()
    date_str = date.strftime("%Y%m%d")

    if date >= UDIFF_CUTOVER_DATE:
        url = FO_BHAVCOPY_URL_UDIFF.format(date_str=date_str)
    else:
        url = FO_BHAVCOPY_URL_LEGACY.format(
            year=date.year, mon=date.strftime("%b").upper(), dd=date.strftime("%d"))

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 500:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        return pd.read_csv(f)
            else:
                logger.warning(f"{date.date()}: HTTP {resp.status_code} or empty body "
                                f"(likely a holiday/weekend, or NSE changed the URL again)")
                return pd.DataFrame()
        except zipfile.BadZipFile:
            logger.warning(f"{date.date()}: response wasn't a valid zip -- "
                            f"NSE may have changed the file format. Check "
                            f"https://www.nseindia.com/all-reports-derivatives")
            return pd.DataFrame()
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"{date.date()}: request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"{date.date()}: failed after {max_retries} retries.")
    return pd.DataFrame()


def download_index_close(date: datetime, session: Optional[requests.Session] = None) -> Optional[dict]:
    """Downloads the daily index-close file and returns {index_name: close}
    for the indices present (includes NIFTY 50, NIFTY BANK, etc.)."""
    session = session or get_nse_session()
    date_str = date.strftime("%d%m%Y")
    try:
        resp = session.get(INDEX_CLOSE_URL.format(date_str=date_str), timeout=15)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        name_col = _find_column(df, ["index name", "index_name"])
        close_col = _find_column(df, ["closing index value", "close"])
        if name_col is None or close_col is None:
            return None
        return dict(zip(df[name_col].str.strip(), df[close_col]))
    except Exception as e:
        logger.warning(f"{date.date()}: index close fetch failed ({e})")
        return None


# ======================================================================
# 2. Flexible column normalization (survives NSE header-naming changes)
# ======================================================================

def _find_column(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    """Case-insensitive substring match against column names -- used
    instead of exact-name lookups so minor header renames don't break
    everything downstream."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for kw in keywords:
        for lower_name, original in cols_lower.items():
            if kw in lower_name:
                return original
    return None


def normalize_fo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps whatever NSE's current column names are onto a fixed internal
    schema: [symbol, expiry, strike, option_type, close, underlying_close].
    Logs what it matched so a silent mismatch (NSE renamed something and
    the wrong column got matched) is easy to spot, rather than failing
    quietly downstream.

    NOTE: this keeps BOTH option rows and futures rows (futures rows have
    a blank/zero strike and blank/"XX" option_type -- see
    extract_futures_close(), which relies on exactly that to find them
    in the same normalized frame, no second download needed).
    """
    if df.empty:
        return df

    mapping = {
        "symbol": _find_column(df, ["tckrsymb", "symbol"]),
        "expiry": _find_column(df, ["xpry", "expiry"]),
        "strike": _find_column(df, ["strkpric", "strike_pr", "strike"]),
        "option_type": _find_column(df, ["optntp", "option_typ", "opttype"]),
        "close": _find_column(df, ["clspric", "close"]),
        "underlying_close": _find_column(df, ["undrlygpric", "underlying"]),
        "open_interest": _find_column(df, ["opnintrst", "open_int", "openinterest"]),
        "volume": _find_column(df, ["ttltradgvol", "contracts", "tradgvol"]),
    }

    # open_interest and volume are OPTIONAL -- only used for the market-context
    # panel (PCR, max pain), not the core option-pricing pipeline, so a miss on
    # those two shouldn't trigger the "NSE changed the schema" warning.
    required_keys = ["symbol", "expiry", "strike", "option_type", "close", "underlying_close"]
    missing = [k for k in required_keys if mapping.get(k) is None]
    if missing:
        logger.warning(f"Could not find columns for: {missing}. "
                        f"Available columns were: {list(df.columns)}. "
                        f"NSE likely changed the schema -- inspect and update "
                        f"the keyword lists in normalize_fo_columns().")
    if mapping.get("open_interest") is None or mapping.get("volume") is None:
        logger.debug("open_interest/volume columns not found -- PCR and max pain "
                     "will be unavailable for this day, but core chain data is unaffected.")

    out = pd.DataFrame()
    for key, col in mapping.items():
        if col is not None:
            out[key] = df[col]

    logger.debug(f"Column match: { {k: v for k, v in mapping.items() if v} }")
    return out


# ======================================================================
# 3. Build OptionChainSnapshot from a normalized day's data
# ======================================================================

def build_chain_snapshot(normalized_df: pd.DataFrame, symbol: str, expiry_str: str,
                          as_of_date: datetime, r: float = 0.065) -> Optional[OptionChainSnapshot]:
    """
    Filters the day's F&O data down to one underlying + one expiry's
    call chain, and builds an OptionChainSnapshot -- the same object
    generate_signals() in probability_model.py consumes directly.

    NOTE: uses daily CLOSE as the "mid price" (bhavcopy has no bid/ask).
    See module docstring -- treat this as an approximation for offline
    validation, not a substitute for live quotes.
    """
    if normalized_df.empty:
        return None

    chain = normalized_df[
        (normalized_df["symbol"].astype(str).str.upper() == symbol.upper())
        & (normalized_df["expiry"].astype(str).str.contains(expiry_str, na=False))
        & (normalized_df["option_type"].astype(str).str.upper() == "CE")
    ].sort_values("strike")

    if chain.empty or len(chain) < 5:
        return None

    spot = float(chain["underlying_close"].iloc[0]) if "underlying_close" in chain else None
    if spot is None or spot <= 0:
        logger.warning(f"{as_of_date.date()}: no usable underlying_close -- "
                        f"fetch it separately via download_index_close() instead.")
        return None

    expiry_dt = pd.to_datetime(chain["expiry"].iloc[0])
    T_years = max((expiry_dt - as_of_date).total_seconds() / (365.25 * 24 * 3600), 1e-6)

    return OptionChainSnapshot(
        spot=spot,
        strikes=chain["strike"].to_numpy(dtype=float),
        call_mid_prices=chain["close"].to_numpy(dtype=float),
        r=r,
        T=T_years,
    )


# ======================================================================
# 3a. Market context: Put-Call Ratio and Max Pain (both derived from the
#     SAME bhavcopy file already downloaded -- no new data source needed)
# ======================================================================

def compute_pcr_and_max_pain(normalized_df: pd.DataFrame, symbol: str,
                              expiry_str: str) -> tuple:
    """
    Put-Call Ratio (by open interest) and Max Pain for one expiry, both
    standard, widely-cited option-chain context metrics -- NOT validated
    trading signals. Treat them as market sentiment/positioning context,
    the same way you'd read India VIX: informative about current
    conditions, not a proven edge (nothing in this pipeline has found one).

    PCR = total put OI / total call OI. Conventionally read as
    "elevated PCR (>1) suggests more hedging/bearish positioning via
    puts; low PCR suggests more call positioning" -- a common heuristic,
    not a backtested rule here.

    Max Pain = the strike where option WRITERS collectively lose the
    least (equivalently, where total option buyer payoff is minimized)
    at expiry. The theory that price gravitates toward max pain is
    widely discussed and genuinely contested among practitioners --
    presented here as a data point to observe, not a claim it's true.

    Returns (pcr, max_pain_strike) -- either may be None if open_interest
    wasn't available in this day's file (see normalize_fo_columns).
    """
    if "open_interest" not in normalized_df.columns:
        return None, None

    subset = normalized_df[
        (normalized_df["symbol"].astype(str).str.upper() == symbol.upper())
        & (normalized_df["expiry"].astype(str).str.contains(expiry_str, na=False))
    ].copy()
    if subset.empty:
        return None, None

    subset["oi"] = pd.to_numeric(subset["open_interest"], errors="coerce").fillna(0)
    subset["strike_num"] = pd.to_numeric(subset["strike"], errors="coerce")
    opt_type = subset["option_type"].astype(str).str.upper().str.strip()

    calls = subset[opt_type == "CE"]
    puts = subset[opt_type == "PE"]
    if calls.empty or puts.empty:
        return None, None  # need both sides for a meaningful PCR/max pain

    total_call_oi = calls["oi"].sum()
    total_put_oi = puts["oi"].sum()
    pcr = float(total_put_oi / total_call_oi) if total_call_oi > 0 else None

    strikes = sorted(set(calls["strike_num"].dropna()) | set(puts["strike_num"].dropna()))
    if not strikes:
        return pcr, None

    call_oi_by_strike = calls.groupby("strike_num")["oi"].sum()
    put_oi_by_strike = puts.groupby("strike_num")["oi"].sum()

    total_pain = {}
    for candidate in strikes:
        pain = 0.0
        for k, oi in call_oi_by_strike.items():
            pain += max(candidate - k, 0.0) * oi   # call writers' loss if settle > k
        for k, oi in put_oi_by_strike.items():
            pain += max(k - candidate, 0.0) * oi   # put writers' loss if settle < k
        total_pain[candidate] = pain

    max_pain_strike = min(total_pain, key=total_pain.get)
    return pcr, float(max_pain_strike)


# ======================================================================
# 3b. India VIX (real market fear gauge, published daily by NSE as part
#     of the same index-close file already used elsewhere)
# ======================================================================

def fetch_vix_series(start_date: datetime, end_date: datetime, pause_sec: float = 1.0) -> pd.Series:
    """
    Pulls India VIX closes over a date range from NSE's daily index-close
    archive (the same file download_index_close() reads from -- India VIX
    is published there alongside NIFTY 50 and other indices). This is a
    REAL, market-quoted volatility gauge -- unlike this pipeline's own
    HAR-RV forecast, VIX reflects options-implied expected volatility as
    priced by the market right now. Plotting the two together is exactly
    what visualizes the variance-risk-premium finding: if VIX consistently
    runs above realized/forecast vol, that gap is the premium being
    harvested by short-volatility strategies (see the sell-ATM baseline).
    """
    session = get_nse_session()
    records = {}
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            closes = download_index_close(current, session)
            if closes:
                vix_key = next((k for k in closes if "vix" in k.lower()), None)
                if vix_key:
                    records[current.date()] = closes[vix_key]
            time.sleep(pause_sec)
        current += timedelta(days=1)
    series = pd.Series(records).sort_index()
    logger.info(f"Fetched {len(series)} India VIX observations from {start_date.date()} "
                f"to {end_date.date()}.")
    return series


# ======================================================================
# 4. Batch loop across a date range -> chains + spot series
# ======================================================================

def build_historical_dataset(start_date: datetime, end_date: datetime,
                              symbol: str = "NIFTY", target_expiry: Optional[str] = None,
                              pause_sec: float = 1.5) -> tuple[dict, pd.Series]:
    """
    Loops over trading days in [start_date, end_date], downloading each
    day's bhavcopy and building a chain snapshot (for whichever expiry
    is nearest, if target_expiry isn't pinned).

    Returns:
        chains: {date -> OptionChainSnapshot}
        spot_series: pd.Series of daily closes, indexed by date -- feed
                      into probability_model.realized_variance() /
                      har_rv_forecast() the same way the synthetic data was used.

    `pause_sec` between requests is deliberate -- NSE rate-limits/blocks
    aggressive scraping; this is a free public archive, not a paid API,
    so be a considerate citizen of it.
    """
    session = get_nse_session()
    chains = {}
    spot_records = {}

    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # skip weekends; holidays return empty and are skipped naturally
            raw = download_fo_bhavcopy(current, session)
            normalized = normalize_fo_columns(raw)

            if not normalized.empty:
                expiry_str = target_expiry or _nearest_expiry(normalized, symbol, current)
                if expiry_str:
                    snap = build_chain_snapshot(normalized, symbol, expiry_str, current)
                    if snap:
                        chains[current.date()] = snap
                        spot_records[current.date()] = snap.spot

            time.sleep(pause_sec)
        current += timedelta(days=1)

    spot_series = pd.Series(spot_records).sort_index()
    logger.info(f"Built {len(chains)} chain snapshots and {len(spot_series)} spot "
                f"observations from {start_date.date()} to {end_date.date()}.")
    return chains, spot_series


def _nearest_expiry(normalized_df: pd.DataFrame, symbol: str, as_of_date: datetime) -> Optional[str]:
    """Picks the nearest upcoming expiry for `symbol` on this day, so a
    batch run doesn't need every expiry hardcoded in advance."""
    subset = normalized_df[normalized_df["symbol"].astype(str).str.upper() == symbol.upper()]
    if subset.empty:
        return None
    expiries = pd.to_datetime(subset["expiry"], errors="coerce").dropna()
    future = expiries[expiries >= as_of_date]
    if future.empty:
        return None
    return future.min().strftime("%Y-%m-%d")


# ======================================================================
# 5a. Futures close extraction (for forward-price pricing -- avoids ever
#     having to ASSUME a drift, since the real quoted futures price
#     already embeds the market's actual cost-of-carry)
# ======================================================================

def extract_futures_close(normalized_df: pd.DataFrame, symbol: str,
                           as_of_date: datetime) -> Optional[float]:
    """
    Finds the near-month index futures close price from the SAME day's
    already-downloaded, already-normalized bhavcopy -- no second network
    call needed, since futures and options for one symbol/day ship in
    one file.

    Futures rows are identified the standard bhavcopy way: strike is
    blank/zero and option_type is blank/"XX" (documented NSE convention,
    both legacy and UDIFF formats). NSE's index futures are monthly, not
    weekly, so this is the nearest-month future -- a standard, liquid
    proxy for the risk-neutral forward even when pricing a weekly
    option, since intra-month basis differences are small relative to
    the overall price level.

    Returns None if no futures row is found (rare, but don't silently
    fall back to something else -- the caller should skip that day
    rather than use a stale/wrong forward).
    """
    subset = normalized_df[normalized_df["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if subset.empty:
        return None

    strike_num = pd.to_numeric(subset["strike"], errors="coerce")
    opt_type = subset["option_type"].astype(str).str.upper().str.strip()
    is_futures = (strike_num.isna() | (strike_num == 0)) & (opt_type.isin(["", "XX", "NAN"]) | opt_type.isna())
    futures_rows = subset[is_futures].copy()
    if futures_rows.empty:
        return None

    futures_rows["expiry_dt"] = pd.to_datetime(futures_rows["expiry"], errors="coerce")
    futures_rows = futures_rows.dropna(subset=["expiry_dt"])
    future_only = futures_rows[futures_rows["expiry_dt"] >= as_of_date]
    if future_only.empty:
        return None

    nearest = future_only.loc[future_only["expiry_dt"].idxmin()]
    close = pd.to_numeric(nearest["close"], errors="coerce")
    return float(close) if pd.notna(close) and close > 0 else None


def build_historical_dataset_with_futures(start_date: datetime, end_date: datetime,
                                           symbol: str = "NIFTY", target_expiry: Optional[str] = None,
                                           pause_sec: float = 1.5) -> tuple[dict, pd.Series, pd.Series]:
    """
    Same as build_historical_dataset(), but also captures each day's
    near-month futures close alongside the option chain and spot --
    kept as a separate function (rather than changing
    build_historical_dataset's return shape) so existing callers of the
    original function are untouched.

    Returns:
        chains: {date -> OptionChainSnapshot}          (same as before)
        spot_series: pd.Series of daily index closes    (same as before)
        futures_series: pd.Series of daily near-month futures closes,
                         indexed by date -- feed into the forward-price
                         (Black-76 style) pricing variant instead of
                         assuming a drift.
    """
    session = get_nse_session()
    chains = {}
    spot_records = {}
    futures_records = {}

    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            raw = download_fo_bhavcopy(current, session)
            normalized = normalize_fo_columns(raw)

            if not normalized.empty:
                expiry_str = target_expiry or _nearest_expiry(normalized, symbol, current)
                if expiry_str:
                    snap = build_chain_snapshot(normalized, symbol, expiry_str, current)
                    if snap:
                        chains[current.date()] = snap
                        spot_records[current.date()] = snap.spot

                fut_close = extract_futures_close(normalized, symbol, current)
                if fut_close is not None:
                    futures_records[current.date()] = fut_close

            time.sleep(pause_sec)
        current += timedelta(days=1)

    spot_series = pd.Series(spot_records).sort_index()
    futures_series = pd.Series(futures_records).sort_index()
    logger.info(f"Built {len(chains)} chain snapshots, {len(spot_series)} spot observations, "
                f"and {len(futures_series)} futures observations from {start_date.date()} "
                f"to {end_date.date()}.")
    return chains, spot_series, futures_series


def build_historical_dataset_with_context(start_date: datetime, end_date: datetime,
                                           symbol: str = "NIFTY", target_expiry: Optional[str] = None,
                                           pause_sec: float = 1.5) -> tuple:
    """
    Same as build_historical_dataset(), but also computes daily PCR and
    max pain from the SAME already-downloaded file (no extra network
    calls -- open interest for both CE and PE rows is right there in the
    bhavcopy we're already fetching for the option chain).

    Returns:
        chains: {date -> OptionChainSnapshot}
        spot_series: pd.Series of daily index closes
        context_df: pd.DataFrame indexed by date, columns ["pcr", "max_pain"]
                     (either may be NaN for a given day if OI columns
                     weren't found in that day's file)
    """
    session = get_nse_session()
    chains = {}
    spot_records = {}
    context_records = {}

    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            raw = download_fo_bhavcopy(current, session)
            normalized = normalize_fo_columns(raw)

            if not normalized.empty:
                expiry_str = target_expiry or _nearest_expiry(normalized, symbol, current)
                if expiry_str:
                    snap = build_chain_snapshot(normalized, symbol, expiry_str, current)
                    if snap:
                        chains[current.date()] = snap
                        spot_records[current.date()] = snap.spot

                    pcr, max_pain = compute_pcr_and_max_pain(normalized, symbol, expiry_str)
                    context_records[current.date()] = {"pcr": pcr, "max_pain": max_pain}

            time.sleep(pause_sec)
        current += timedelta(days=1)

    spot_series = pd.Series(spot_records).sort_index()
    context_df = pd.DataFrame(context_records).T.sort_index() if context_records else pd.DataFrame()
    logger.info(f"Built {len(chains)} chain snapshots and {len(context_records)} PCR/max-pain "
                f"observations from {start_date.date()} to {end_date.date()}.")
    return chains, spot_series, context_df


# ======================================================================
# 5. Offline structural test (validates parsing logic without a live
#    network call, since this sandbox can't reach nseindia.com) --
#    run this yourself against the real download on your own machine.
# ======================================================================

def _structural_selftest():
    """Fabricates a bhavcopy-shaped DataFrame matching NSE's documented
    UDIFF column names, and checks the normalize/build pipeline handles
    it correctly. This does NOT touch the network -- it only proves the
    parsing logic is sound, ahead of pointing it at a real download."""
    fake = pd.DataFrame({
        "TckrSymb": ["NIFTY"] * 6,
        "XpryDt": ["2026-09-25"] * 6,
        "StrkPric": [24600, 24700, 24800, 24900, 25000, 25100],
        "OptnTp": ["CE"] * 6,
        "ClsPric": [280.5, 210.2, 150.8, 102.4, 65.3, 38.9],
        "UndrlygPric": [24800.0] * 6,
    })
    normalized = normalize_fo_columns(fake)
    snap = build_chain_snapshot(normalized, "NIFTY", "2026-09-25", datetime(2026, 9, 16))
    assert snap is not None, "self-test failed: no snapshot built"
    assert len(snap.strikes) == 6, "self-test failed: wrong strike count"
    assert snap.spot == 24800.0, "self-test failed: wrong spot"
    print("Structural self-test PASSED -- parsing/normalization logic is sound.")
    print(f"  Snapshot: spot={snap.spot}, strikes={snap.strikes}, T={snap.T:.4f} yrs")


if __name__ == "__main__":
    print(__doc__)
    _structural_selftest()

    print("\n\nTo run the REAL download on your own machine:\n")
    print("    from datetime import datetime")
    print("    chains, spot_series = build_historical_dataset(")
    print("        start_date=datetime(2026, 8, 1), end_date=datetime(2026, 9, 15),")
    print("        symbol='NIFTY')")
    print("\n    # Feed spot_series into HAR-RV:")
    print("    log_returns = np.diff(np.log(spot_series.values))")
    print("    # Feed each chains[date] into probability_model.generate_signals()")
    print("    # exactly like the synthetic OptionChainSnapshot was used.")
