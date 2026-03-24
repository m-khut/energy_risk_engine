import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
from loader import *
from returns import *
from volatility import *
import warnings

# Historical Simulation VaR

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    _validate_returns(returns, confidence)
 
    # quantile(1 - confidence) = the left tail cutoff
    # e.g. at 95% confidence, we take the 5th percentile
    var_threshold = returns.quantile(1 - confidence)
 
    # A "breach" = a day where the actual return was worse than VaR
    breaches = returns[returns < var_threshold]
 
    return {
        "var_threshold":  round(var_threshold, 6),
        "confidence":     confidence,
        "breach_dates":   breaches.index.tolist(),
        "breach_count":   len(breaches),
        "breach_rate":    round(len(breaches) / len(returns), 4),
        "n_observations": len(returns),
        "method":         "historical_simulation",
    }
 
 
def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    _validate_returns(returns, confidence)
 
    var_result = historical_var(returns, confidence)
    var_threshold = var_result["var_threshold"]
 
    # Tail returns = all returns below the VaR threshold
    tail_returns = returns[returns < var_threshold]
 
    if len(tail_returns) == 0:
        warnings.warn("No returns breach the VaR threshold. CVaR is undefined.")
        cvar = np.nan
    else:
        cvar = tail_returns.mean()
 
    return {
        "var_threshold":       round(var_threshold, 6),
        "cvar":                round(cvar, 6),
        "confidence":          confidence,
        "tail_returns":        tail_returns,
        "n_tail_observations": len(tail_returns),
        "method":              "historical_simulation",
    }
 
 # Parametric VaR (Gaussian assumption)

def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    _validate_returns(returns, confidence)
 
    mu = returns.mean()
    sigma = returns.std()
 
    # stats.norm.ppf = percent-point function (inverse CDF)
    # ppf(1 - confidence) gives the z-score at the left tail
    z_score = stats.norm.ppf(1 - confidence)
 
    # VaR = mean return + z_score * std
    # z_score is negative for left tail, so this is mean MINUS |z| * std
    var_threshold = mu + z_score * sigma
 
    return {
        "var_threshold": round(var_threshold, 6),
        "fitted_mean":   round(mu, 6),
        "fitted_std":    round(sigma, 6),
        "z_score":       round(z_score, 4),
        "confidence":    confidence,
        "method":        "parametric_gaussian",
    }
 
# VaR comparison

def var_comparison(
    returns: pd.Series,
    confidence_levels: list = [0.95, 0.99],
) -> pd.DataFrame:
    rows = []
    for conf in confidence_levels:
        h_var = historical_var(returns, conf)["var_threshold"]
        p_var = parametric_var(returns, conf)["var_threshold"]
        diff = h_var - p_var
        diff_pct = (diff / abs(p_var)) * 100 if p_var != 0 else np.nan
 
        rows.append({
            "confidence":     conf,
            "historical_var": h_var,
            "parametric_var": p_var,
            # Negative difference means historical is MORE negative (worse)
            # than parametric — this is the fat-tail signature
            "difference":     round(diff, 6),
            "diff_pct":       round(diff_pct, 2),
        })
 
    df = pd.DataFrame(rows).set_index("confidence")
    return df

# Rolling VaR

def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    
    # Apply quantile over a rolling window
    # quantile(1 - confidence) at each step

    rolling = returns.rolling(window=window).quantile(1 - confidence)
    rolling.name = f"rolling_var_{confidence:.0%}_{window}d"
    return rolling

# Backtest: Kupiec Test

def kupiec_test(
    returns: pd.Series,
    confidence: float = 0.95,
    window: Optional[int] = None,
) -> dict:
    
    if window is not None:
        returns = returns.iloc[-window:]
 
    var_result = historical_var(returns, confidence)
    var_threshold = var_result["var_threshold"]
 
    n = len(returns)
    x = (returns < var_threshold).sum()   # observed breaches
    p = 1 - confidence                    # expected breach rate
    p_hat = x / n                         # observed breach rate
 
    # Likelihood ratio statistic
    # Guard against log(0) with small epsilon
    eps = 1e-10
    p_hat_safe = max(min(p_hat, 1 - eps), eps)
 
    lr = -2 * (
        x * np.log(p / p_hat_safe) +
        (n - x) * np.log((1 - p) / (1 - p_hat_safe))
    )
 
    # Chi-squared p-value with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr, df=1)
 
    # Reject H0 if p-value < 0.05 (model is mis-specified at 95% confidence)
    reject = p_value < 0.05
 
    if reject:
        verdict = (
            f"FAIL: Model rejected at 95% significance. "
            f"Observed {x} breaches ({p_hat:.1%}) vs expected {p:.1%}."
        )
    else:
        verdict = (
            f"PASS: Model not rejected. "
            f"Observed {x} breaches ({p_hat:.1%}) consistent with {p:.1%} expected."
        )
 
    return {
        "expected_breach_rate": round(p, 4),
        "observed_breach_rate": round(p_hat, 4),
        "n_breaches":           int(x),
        "n_observations":       n,
        "lr_statistic":         round(lr, 4),
        "p_value":              round(p_value, 4),
        "reject_h0":            reject,
        "verdict":              verdict,
    }

# Internal helpers
def _validate_returns(returns: pd.Series, confidence: float) -> None:
    """Guard against bad inputs before computing risk metrics."""
 
    if returns.isna().any():
        raise ValueError(
            "NaN values in returns. Clean your data before calling VaR functions."
        )
    if not (0 < confidence < 1):
        raise ValueError(
            f"confidence must be between 0 and 1. Got: {confidence}"
        )
    if len(returns) < 30:
        warnings.warn(
            f"Only {len(returns)} observations. VaR estimates will be unreliable. "
            f"Use at least 252 observations (1 trading year) for production."
        )