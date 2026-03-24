import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
from loader import *
from returns import *
from volatility import *
from var import *
from drawdown import *
from diagnostics import *
import warnings

# Rolling correlation

def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 63,
) -> pd.Series:

    # Align both series to shared dates first
    aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
 
    corr = aligned["a"].rolling(window=window).corr(aligned["b"])
    corr.name = f"rolling_corr_{window}d"
    return corr
 
def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()

# Crisis correlation analysis

def period_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    start: str,
    end: str,
    label: str = "period",
) -> dict:
    
    aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    window = aligned.loc[start:end]
 
    if len(window) < 10:
        raise ValueError(
            f"Only {len(window)} observations in {start} to {end}. "
            f"Need at least 10 for meaningful correlation."
        )
 
    corr = window["a"].corr(window["b"])
 
    return {
        "period_label":    label,
        "correlation":     round(corr, 4),
        "n_observations":  len(window),
        "start":           start,
        "end":             end,
    }

def correlation_breakdown_analysis(
    returns_a: pd.Series,
    returns_b: pd.Series,
    crisis_periods: Optional[list] = None,
) -> pd.DataFrame:
    
    results = []
 
    # Full sample baseline
    aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    full_corr = aligned["a"].corr(aligned["b"])
    results.append({
        "period_label":   "FULL_SAMPLE",
        "correlation":    round(full_corr, 4),
        "n_observations": len(aligned),
    })
 
    # Each crisis period
    for period in crisis_periods:
        try:
            result = period_correlation(
                returns_a, returns_b,
                start=period["start"],
                end=period["end"],
                label=period["label"],
            )
            results.append(result)
        except (ValueError, KeyError) as e:
            print(f"[correlation] Skipping {period['label']}: {e}")
 
    df = pd.DataFrame(results)
    return df


# Correlation stress scenario

def apply_correlation_stress(
    returns_df: pd.DataFrame,
    stressed_corr: Optional[float] = None,
    crisis_period: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    
    returns_df = returns_df.dropna()
    n_assets = returns_df.shape[1]
 
    if stressed_corr is not None:
        # Build a stressed correlation matrix: diag = 1, off-diag = stressed_corr
        stressed_mat = np.full((n_assets, n_assets), stressed_corr)
        np.fill_diagonal(stressed_mat, 1.0)
        target_corr = pd.DataFrame(
            stressed_mat,
            index=returns_df.columns,
            columns=returns_df.columns,
        )
 
    elif crisis_period is not None:
        start, end = crisis_period
        crisis_returns = returns_df.loc[start:end]
        target_corr = crisis_returns.corr()
    else:
        raise ValueError("Provide either stressed_corr or crisis_period.")
 
    # Cholesky decomposition of target correlation matrix
    # P = L @ L.T where L is lower triangular
    try:
        L_target = np.linalg.cholesky(target_corr.values)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Stressed correlation matrix is not positive definite. "
            "Reduce stressed_corr or use a smaller crisis window."
        )
 
    # Current correlation structure
    current_corr = returns_df.corr().values
    try:
        L_current = np.linalg.cholesky(current_corr)
    except np.linalg.LinAlgError:
        raise ValueError("Current returns correlation matrix is not positive definite.")
 
    # Decorrelate returns, then re-correlate with stressed structure
    # Transform: X_stressed = X_decorrelated @ L_target.T
    # Decorrelate: X_decorr = X @ inv(L_current).T
    standardized = (returns_df - returns_df.mean()) / returns_df.std()
    decorrelated = standardized.values @ np.linalg.inv(L_current).T
    stressed = decorrelated @ L_target.T
 
    # Rescale back to original marginal distribution
    stressed_df = pd.DataFrame(
        stressed * returns_df.std().values + returns_df.mean().values,
        index=returns_df.index,
        columns=returns_df.columns,
    )
 
    return stressed_df
 
