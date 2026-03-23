import pandas as pd
import numpy as np
from typing import Optional
from loader import *
from returns import *

TRADING_DAYS_PER_YEAR = 252   # market standard
EWMA_LAMBDA = 0.94            # RiskMetrics standard for daily data

# Rolling volatility
 
def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    
    vol = returns.rolling(window=window).std()
 
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        vol.name = f"rolling_vol_{window}d_ann"
    else:
        vol.name = f"rolling_vol_{window}d"
 
    return vol
 
 
def multi_window_volatility(
    returns: pd.Series,
    windows: list = [21, 63, 252],
    annualize: bool = True,
) -> pd.DataFrame:
    
    results = {}
    for w in windows:
        results[f"vol_{w}d"] = rolling_volatility(returns, window=w, annualize=annualize)
 
    return pd.DataFrame(results)

# EWMA volatility (RiskMetrics standard)

def ewma_volatility(
    returns: pd.Series,
    lam: float = EWMA_LAMBDA,
    annualize: bool = True,
) -> pd.Series:
    
    # We compute EWMA variance, then take sqrt for std deviation
    # pandas ewm(com) with com = lambda/(1-lambda) implements the RiskMetrics formula
    # com = "center of mass" — a different parameterization of the same decay
    com = lam / (1 - lam)
 
    # squared returns = proxy for variance (assumes mean return ≈ 0, standard practice)
    squared_returns = returns ** 2
 
    # ewm().mean() computes the exponentially weighted mean of squared returns
    ewma_variance = squared_returns.ewm(com=com, adjust=False).mean()
 
    # Variance → standard deviation
    ewma_vol = np.sqrt(ewma_variance)
 
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        ewma_vol.name = "ewma_vol_ann"
    else:
        ewma_vol.name = "ewma_vol"
 
    return ewma_vol

# Volatility diagnostics

def vol_regime_summary(
    returns: pd.Series,
    window: int = 21,
) -> dict:
    
    vol = rolling_volatility(returns, window=window, annualize=True)
    vol = vol.dropna()
 
    return {
        "vol_5th_pct":  round(vol.quantile(0.05), 4),
        "vol_25th_pct": round(vol.quantile(0.25), 4),
        "vol_median":   round(vol.median(), 4),
        "vol_75th_pct": round(vol.quantile(0.75), 4),
        "vol_95th_pct": round(vol.quantile(0.95), 4),
        "vol_max":      round(vol.max(), 4),
        "vol_min":      round(vol.min(), 4),
    }
 
 
def realized_vs_ewma(
    returns: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    
    realized = rolling_volatility(returns, window=window, annualize=True)
    ewma = ewma_volatility(returns, annualize=True)
 
    df = pd.DataFrame({
        "realized_vol": realized,
        "ewma_vol":     ewma,
    }).dropna()
 
    # Forecast error: how much did EWMA miss?
    df["forecast_error"] = df["realized_vol"] - df["ewma_vol"]
 
    return df
