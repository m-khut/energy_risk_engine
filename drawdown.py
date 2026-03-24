import pandas as pd
import numpy as np
from typing import Optional
from loader import *
from returns import *
from volatility import *
from var import *
import warnings

def drawdown_series(prices: pd.Series) -> pd.Series:

    if prices.isna().any():
        raise ValueError("Price series contains NaN. Clean before calling drawdown.")
    
    # Running maximum: the highest price seen up to each date
    running_max = prices.cummax()
 
    # Drawdown = percentage below the running maximum
    dd = (prices - running_max) / running_max
    dd.name = "drawdown"
    return dd


def max_drawdown(prices: pd.Series) -> dict:
    
    dd = drawdown_series(prices)
 
    # Find the trough (worst drawdown point)
    trough_idx = dd.idxmin()
    max_dd_value = dd.min()
 
    # Find the peak: latest date BEFORE trough where drawdown was 0
    pre_trough = dd[:trough_idx]
    peak_candidates = pre_trough[pre_trough == 0]
 
    if peak_candidates.empty:
        # If no zero-drawdown point, the peak is the first observation
        peak_date = prices.index[0]
    else:
        peak_date = peak_candidates.index[-1]
 
    # Find recovery: first date AFTER trough where drawdown returns to 0
    post_trough = dd[trough_idx:]
    recovery_candidates = post_trough[post_trough >= 0]

    if recovery_candidates.empty:
        recovery_date = None
        recovery_days = None
    else:
        recovery_date = recovery_candidates.index[0]
        recovery_days = (recovery_date - trough_idx).days
 
    duration_days = (trough_idx - peak_date).days
 
    return {
        "max_drawdown":   round(max_dd_value, 4),
        "peak_date":      peak_date.date(),
        "trough_date":    trough_idx.date(),
        "recovery_date":  recovery_date.date() if recovery_date else None,
        "duration_days":  duration_days,
        "recovery_days":  recovery_days,
    }

def drawdown_periods(
    prices: pd.Series,
    min_drawdown: float = -0.05,
    top_n: int = 5,
) -> pd.DataFrame:
    
    dd = drawdown_series(prices)
 
    # Identify drawdown periods: contiguous stretches where drawdown < 0
    # We do this by finding transitions: 0 → negative and negative → 0
    in_drawdown = dd < min_drawdown
 
    # Find start and end of each drawdown episode
    transitions = in_drawdown.astype(int).diff()
    starts = transitions[transitions == 1].index   # 0→1 transition
    ends = transitions[transitions == -1].index    # 1→0 transition

    # Handle edge cases: series starts or ends in drawdown
    if len(starts) == 0:
        return pd.DataFrame(columns=["peak_date", "trough_date", "max_drawdown", "duration_days"])
 
    if len(ends) == 0 or (len(starts) > 0 and starts[-1] > ends[-1] if len(ends) > 0 else True):
        ends = ends.append(pd.Index([dd.index[-1]]))
 
    periods = []
    for start in starts:
        # Find the next end after this start
        future_ends = ends[ends > start]
        if future_ends.empty:
            end = dd.index[-1]
        else:
            end = future_ends[0]
 
        window = dd[start:end]
        worst = window.min()
        trough = window.idxmin()
 
        periods.append({
            "peak_date":    start.date(),
            "trough_date":  trough.date(),
            "end_date":     end.date(),
            "max_drawdown": round(worst, 4),
            "duration_days": (trough - start).days,
        })
 
    df = pd.DataFrame(periods)
    if df.empty:
        return df
 
    df = df.sort_values("max_drawdown").head(top_n).reset_index(drop=True)
    return df

def calmar_ratio(
    prices: pd.Series,
    returns: pd.Series,
) -> float:

    ann_return = returns.mean() * 252
    max_dd = max_drawdown(prices)["max_drawdown"]
 
    if max_dd == 0:
        return np.inf
 
    return round(ann_return / abs(max_dd), 4)