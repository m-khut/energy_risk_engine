import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import warnings


# If more than this many consecutive NaNs appear, we warn.


MAX_CONSECUTIVE_NAN = 3
 
# Default tickers used across the project
DEFAULT_TICKERS = {
    "energy_etf": "XLE",       # SPDR Energy Select Sector ETF
    "crude_proxy": "CL=F",     # WTI Crude Oil Futures (front month)
    "oil_etf": "USO",          # United States Oil Fund ETF
    "volatility": "^VIX",      # CBOE Volatility Index
}

def load_price_data(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    price_col: str = "Close",
) -> pd.Series:
    

    # yfinance returns a DataFrame: Open, High, Low, Close, Volume
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
 
    # Guard: empty download means bad ticker or no data in range
    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start} and {end}. "
            f"Check that the ticker is valid on Yahoo Finance."
        )
 
    # yfinance >=0.2 may return MultiIndex columns when downloading single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
 
    # Guard: confirm the column we want actually exists
    if price_col not in raw.columns:
        raise ValueError(
            f"Column '{price_col}' not found. Available: {list(raw.columns)}"
        )
 
    prices = raw[price_col].copy()
 
    # yfinance sometimes returns timezone-aware index; strip it for simplicity
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
 
    # Run validation checks (prints warnings, does not crash)
    _validate_prices(prices, ticker)
 
    return prices
 
 
def load_multi_asset(
    tickers: list,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    
    frames = {}
    for ticker in tickers:
        try:
            frames[ticker] = load_price_data(ticker, start=start, end=end)
        except ValueError as e:
            # Don't crash the whole load if one ticker fails
            warnings.warn(f"Skipping {ticker}: {e}")
 
    if not frames:
        raise ValueError("All tickers failed to load. Check your inputs.")
 
    # concat aligns on index automatically
    df = pd.concat(frames, axis=1)
    df.columns = list(frames.keys())
 
    # Drop rows where ALL columns are NaN
    df = df.dropna(how="all")
 
    print(f"[loader] Loaded {len(df)} trading days for: {list(df.columns)}")
    print(f"[loader] Date range: {df.index[0].date()} to {df.index[-1].date()}")
 
    return df

def _validate_prices(prices: pd.Series, ticker: str) -> None:
    """
    Run sanity checks on a price series and print warnings (does not raise).
 
    Checks:
        1. NaN values present?
        2. More than MAX_CONSECUTIVE_NAN consecutive NaNs?
        3. Non-positive prices (data corruption)?
        4. Large date gaps > 5 calendar days?
    """
 
    n_nan = prices.isna().sum()
 
    if n_nan > 0:
        pct = 100 * n_nan / len(prices)
        warnings.warn(
            f"[{ticker}] {n_nan} NaN values ({pct:.1f}% of series). "
            f"Call clean_prices() before computing returns."
        )
 
        # Detect consecutive NaN runs using cumsum trick:
        # Each time we see a non-NaN, the cumsum increments, creating groups.
        # Within each group, summing isna() gives the consecutive NaN count.
        consecutive = (
            prices.isna()
            .astype(int)
            .groupby((~prices.isna()).cumsum())
            .sum()
            .max()
        )
        if consecutive > MAX_CONSECUTIVE_NAN:
            warnings.warn(
                f"[{ticker}] {consecutive} consecutive NaN values. "
                f"Consider shortening date range or using a different ticker."
            )
 
    # Non-positive prices = data error (stocks can't trade at 0 or negative)
    non_positive = (prices <= 0).sum()
    if non_positive > 0:
        warnings.warn(
            f"[{ticker}] {non_positive} non-positive price values. Likely data error."
        )
 
    # Large gaps: anything > 5 calendar days between observations
    date_diffs = prices.index.to_series().diff().dt.days.dropna()
    large_gaps = date_diffs[date_diffs > 5]
    if not large_gaps.empty:
        for gap_date, gap_days in large_gaps.items():
            warnings.warn(
                f"[{ticker}] {int(gap_days)}-day gap ending {gap_date.date()}."
            )
 
 
def clean_prices(prices: pd.Series) -> pd.Series:
    n_before = prices.isna().sum()
    prices = prices.ffill()   # carry last price forward
    prices = prices.dropna()  # remove any leading NaNs (no prior price exists)
    n_after = prices.isna().sum()
 
    if n_before > 0:
        print(f"[clean] Filled {n_before} NaN values. Remaining: {n_after}.")
 
    return prices
