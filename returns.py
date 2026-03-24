import pandas as pd
import numpy as np
import warnings
from typing import Tuple
from loader import load_price_data, clean_prices


def simple_returns(prices: pd.Series) -> pd.Series:

    _check_prices(prices)
    # pct_change() computes (current - previous) / previous
    # dropna() removes the first NaN (no return for first day)
    returns = prices.pct_change().dropna()
    returns.name = "simple_returns"
    return returns
 
 
def log_returns(prices: pd.Series) -> pd.Series:

    _check_prices(prices)
    # np.log(P_t / P_{t-1}) = np.log(P_t) - np.log(P_{t-1})
    # shift(1) aligns P_{t-1} with row t
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = "log_returns"
    return returns
 
 
def excess_returns(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.05
) -> pd.Series:
    
    # Convert annual rate to daily: (1 + r_annual)^(1/252) - 1
    # 252 = standard number of trading days per year
    daily_rf = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess = returns - daily_rf
    excess.name = "excess_returns"
    return excess

def return_summary(returns: pd.Series) -> dict:

    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
 
    return {
        "mean_daily":        round(returns.mean(), 6),
        "std_daily":         round(returns.std(), 6),
        "annualized_return": round(ann_return, 4),
        "annualized_vol":    round(ann_vol, 4),
        # Sharpe: excess return per unit of risk (assumes rf=0 here for simplicity)
        "sharpe_ratio":      round(ann_return / ann_vol, 4) if ann_vol != 0 else np.nan,
        # Skewness: negative = left tail heavier (typical for equities/energy)
        "skewness":          round(returns.skew(), 4),
        # Excess kurtosis: 0 = normal. Positive = fat tails. Energy often > 3.
        "kurtosis":          round(returns.kurtosis(), 4),
        "min_return":        round(returns.min(), 6),
        "max_return":        round(returns.max(), 6),
        "n_observations":    len(returns),
    }

def worst_days(returns: pd.Series, n: int = 10) -> pd.DataFrame:
     
    worst = returns.nsmallest(n).reset_index()
    worst.columns = ["date", "return"]
    worst["rank"] = range(1, len(worst) + 1)
    return worst

def detect_return_gaps(prices: pd.Series, threshold_days: int = 5) -> pd.Series:

    diffs = prices.index.to_series().diff().dt.days
    gaps = diffs[diffs > threshold_days]
    
    if gaps.empty:
            print("[returns] No suspicious gaps detected.")
    else:
        print(f"[returns] {len(gaps)} gap(s) detected exceeding {threshold_days} days:")
        for date, days in gaps.items():
            print(f"  {date.date()}: {int(days)}-day gap")

    return gaps


def _check_prices(prices: pd.Series) -> None:
    """
    Guard against common input errors before computing returns.
    Raises ValueError for hard failures; warns for soft issues.
    """
 
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(
            "prices must have a DatetimeIndex. "
            "Call load_price_data() which returns correctly indexed Series."
        )
 
    n_nan = prices.isna().sum()
    if n_nan > 0:
        raise ValueError(
            f"{n_nan} NaN values in price series. "
            f"Call clean_prices() from data/loader.py before computing returns."
        )
 
    if len(prices) < 30:
        warnings.warn(
            f"Only {len(prices)} price observations. "
            f"VaR and volatility estimates will be unreliable."
        )

# spy = load_price_data("SPY", "2020-01-01", "2026-03-23")
# spy_clean = clean_prices(spy)
# spy_returns = simple_returns(spy_clean)
# spy_summary = return_summary(spy_returns)
# print("SPY Return Summary:")
# for key, value in spy_summary.items():
#     print(f"  {key}: {value}") 

# spy_worst = worst_days(spy_returns, n=20)
# print("\nSPY Worst 10 Days:")
# print(spy_worst)