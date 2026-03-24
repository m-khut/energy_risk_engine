import pandas as pd
import numpy as np
from typing import Optional
from loader import *
from returns import *
from volatility import *
from var import *
from drawdown import *
import warnings

# Distribution tests
 
def jarque_bera_test(returns: pd.Series) -> dict:
    jb_stat, p_value = stats.jarque_bera(returns.dropna())
    reject = p_value < 0.05
 
    interpretation = (
        "REJECT normality: returns have significant skewness and/or fat tails. "
        "Gaussian VaR is likely to underestimate risk."
        if reject else
        "CANNOT REJECT normality at 5% level. Gaussian VaR may be appropriate."
    )
 
    return {
        "jb_statistic":      round(jb_stat, 4),
        "p_value":           round(p_value, 6),
        "reject_normality":  reject,
        "interpretation":    interpretation,
    }

def distribution_stats(returns: pd.Series) -> dict:
    clean = returns.dropna()
    sigma = clean.std()
 
    # Fraction of returns beyond ±3 sigma (fat tail signature)
    fat_tail_ratio = (np.abs(clean) > 3 * sigma).mean()
 
    # Historical CVaR at 95% (quick version for diagnostics)
    var_95 = clean.quantile(0.05)
    cvar_95 = clean[clean < var_95].mean()
 
    return {
        # First moment: average daily return
        "mean":           round(clean.mean(), 6),
        # Second moment: spread of returns
        "std":            round(sigma, 6),
        # Third moment: asymmetry (negative = left-tailed)
        "skewness":       round(clean.skew(), 4),
        # Fourth moment: tail heaviness (0 = normal baseline)
        "excess_kurtosis": round(clean.kurtosis(), 4),
        # Fraction of days beyond ±3 sigma (should be ~0.27% if normal)
        "fat_tail_ratio": round(fat_tail_ratio, 4),
        # Jarque-Bera p-value
        "jb_p_value":     round(stats.jarque_bera(clean)[1], 6),
        # VaR and CVaR for quick reference
        "var_95":         round(var_95, 6),
        "cvar_95":        round(cvar_95, 6),
        "n_observations": len(clean),
        # Range
        "min":            round(clean.min(), 6),
        "max":            round(clean.max(), 6),
    }

def tail_ratio(returns: pd.Series, sigma_threshold: float = 3.0) -> dict:
    clean = returns.dropna()
    sigma = clean.std()
    mu = clean.mean()
 
    left_tail = (clean < mu - sigma_threshold * sigma).mean()
    right_tail = (clean > mu + sigma_threshold * sigma).mean()
 
    # Under normality, P(|X - mu| > 3*sigma) ≈ 0.0027
    expected_normal = 2 * (1 - stats.norm.cdf(sigma_threshold))
 
    return {
        "left_tail_pct":      round(left_tail, 4),
        "right_tail_pct":     round(right_tail, 4),
        "total_tail_pct":     round(left_tail + right_tail, 4),
        "expected_normal_pct": round(expected_normal, 4),
        "tail_excess_ratio":   round((left_tail + right_tail) / expected_normal, 2),
        "sigma_threshold":    sigma_threshold,
    }

def rolling_kurtosis(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    
    kurt = returns.rolling(window=window).kurt()
    kurt.name = f"rolling_kurtosis_{window}d"
    return kurt

def print_diagnostic_report(returns: pd.Series, ticker: str = "Asset") -> None:
    stats_dict = distribution_stats(returns)
    jb = jarque_bera_test(returns)
    tails = tail_ratio(returns)
 
    print("=" * 60)
    print(f"DISTRIBUTION DIAGNOSTIC REPORT: {ticker}")
    print("=" * 60)
 
    print("\n[1] RETURN MOMENTS")
    print(f"  Mean daily return : {stats_dict['mean']:.4%}")
    print(f"  Std deviation     : {stats_dict['std']:.4%}")
    print(f"  Skewness          : {stats_dict['skewness']:.4f}  (0 = symmetric)")
    print(f"  Excess Kurtosis   : {stats_dict['excess_kurtosis']:.4f}  (0 = normal)")
 
    print("\n[2] NORMALITY TEST (Jarque-Bera)")
    print(f"  JB Statistic: {jb['jb_statistic']:.2f}")
    print(f"  p-value     : {jb['p_value']:.6f}")
    print(f"  Verdict     : {jb['interpretation']}")
 
    print("\n[3] FAT TAIL ANALYSIS (±3 sigma)")
    print(f"  Observed tail frequency : {tails['total_tail_pct']:.2%}")
    print(f"  Expected (Gaussian)     : {tails['expected_normal_pct']:.2%}")
    print(f"  Tail excess ratio       : {tails['tail_excess_ratio']:.1f}x Gaussian")
    print(f"  Left tail               : {tails['left_tail_pct']:.2%}")
    print(f"  Right tail              : {tails['right_tail_pct']:.2%}")
 
    print("\n[4] RISK SUMMARY")
    print(f"  95% Historical VaR : {stats_dict['var_95']:.4%}")
    print(f"  95% Historical CVaR: {stats_dict['cvar_95']:.4%}")
 
    print("\n[5] MODEL RISK IMPLICATIONS")
    if stats_dict["excess_kurtosis"] > 2:
        print("  !! Significant fat tails: Gaussian VaR underestimates risk.")
    if stats_dict["skewness"] < -0.5:
        print("  !! Negative skew: crashes are more extreme than rallies.")
    if tails["tail_excess_ratio"] > 2:
        print(f"  !! Tails are {tails['tail_excess_ratio']}x fatter than Gaussian.")
        print("     Expected Shortfall (CVaR) is required, not just VaR.")
 
    print("=" * 60)
