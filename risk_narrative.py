#### CELL 1: Configuration and imports ####
 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
 
# Our modules
from loader import load_price_data, clean_prices, load_multi_asset
from returns import log_returns, simple_returns, return_summary, worst_days
from volatility import rolling_volatility, ewma_volatility, multi_window_volatility
from var import (historical_var, historical_cvar, parametric_var,
                      var_comparison, rolling_var, kupiec_test)
from drawdown import drawdown_series, max_drawdown, drawdown_periods
from diagnostics import distribution_stats, jarque_bera_test, tail_ratio, print_diagnostic_report
from correlation import rolling_correlation, correlation_breakdown_analysis
 
# ---- Plot style ----
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary":   "#1f3a5f",    # dark navy
    "secondary": "#c0392b",    # deep red (loss / risk)
    "accent":    "#e67e22",    # amber (warning)
    "neutral":   "#7f8c8d",    # grey
    "positive":  "#27ae60",    # green (gain)
}
FIG_SIZE = (14, 5)
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "figures")
os.makedirs(SAVE_DIR, exist_ok=True)
 
def save_fig(name: str):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")
 
# ---- Configuration ----
TICKER = input("Enter ticker (e.g., XLE, CL=F): ").strip()
START_DATE = input("Enter start date (YYYY-MM-DD): ").strip()
END_DATE = None         # None = today
 
print(f"Config: {TICKER} | {START_DATE} to {END_DATE or 'today'}")
 
 
#### CELL 2: Load data ####
 
prices = load_price_data(TICKER, start=START_DATE, end=END_DATE)
prices = clean_prices(prices)
 
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(prices.index, prices.values, color=COLORS["primary"], linewidth=1.5, label=TICKER)
ax.set_title(f"{TICKER} Closing Prices", fontsize=14, fontweight="bold")
ax.set_ylabel("Price (USD)")
ax.set_xlabel("")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
# Shade the 2020 crash period for reference
ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-01"),
           alpha=0.15, color=COLORS["secondary"], label="2020 crash")
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-07-01"),
           alpha=0.15, color=COLORS["accent"], label="2022 energy shock")
ax.axvspan(pd.Timestamp("2026-02-25"), pd.Timestamp("2026-04-25"),
           alpha=0.15, color=COLORS["accent"], label="2026 Iran Crisis")
ax.legend()
save_fig(f"{TICKER}_01_prices")
plt.show()


#### CELL 3: Returns - simple vs log ####
 
rets_log = log_returns(prices)
rets_simple = simple_returns(prices)
 
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
 
axes[0].plot(rets_log.index, rets_log.values, color=COLORS["primary"],
             linewidth=0.8, alpha=0.9, label="Log returns")
axes[0].axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
axes[0].set_title(f"{TICKER} Daily Log Returns", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Log Return")
 
axes[1].hist(rets_log.values, bins=80, color=COLORS["primary"],
             alpha=0.7, density=True, label="Observed")
# Overlay fitted normal for comparison
x = np.linspace(rets_log.min(), rets_log.max(), 300)
mu, sigma = rets_log.mean(), rets_log.std()
axes[1].plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS["secondary"],
             linewidth=2, label="Fitted normal")
axes[1].set_title("Return Distribution vs Normal", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Daily Log Return")
axes[1].set_ylabel("Density")
axes[1].legend()
 
plt.tight_layout()
save_fig(f"{TICKER}_02_returns")
plt.show()
 
# Print summary
summary = return_summary(rets_log)
print("\nReturn Summary:")
for k, v in summary.items():
    print(f"  {k:<22}: {v}")


#### CELL 4: Volatility - rolling vs EWMA ####
 
vol_21 = rolling_volatility(rets_log, window=21)
vol_63 = rolling_volatility(rets_log, window=63)
ewma_vol = ewma_volatility(rets_log)
 
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(vol_21.index, vol_21.values * 100, color=COLORS["accent"],
        linewidth=1.2, alpha=0.9, label="21d Rolling Vol")
ax.plot(vol_63.index, vol_63.values * 100, color=COLORS["primary"],
        linewidth=1.5, label="63d Rolling Vol")
ax.plot(ewma_vol.index, ewma_vol.values * 100, color=COLORS["secondary"],
        linewidth=1.5, linestyle="--", label="EWMA Vol (λ=0.94)")
 
ax.set_title(f"{TICKER} Volatility: Rolling vs EWMA (Annualized %)", fontsize=13, fontweight="bold")
ax.set_ylabel("Annualized Volatility (%)")
ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-01"),
           alpha=0.12, color=COLORS["secondary"])
ax.legend()
save_fig(f"{TICKER}_03_volatility")
plt.show()


#### CELL 5: VaR - historical vs parametric ####
 
var_comp = var_comparison(rets_log, confidence_levels=[0.95, 0.99])
print("\nVaR Comparison (Historical vs Parametric):")
print(var_comp.to_string())
 
# Rolling VaR time series
roll_var = rolling_var(rets_log, window=252, confidence=0.95)
 
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
 
# Top: P&L distribution with VaR lines
h_var_95 = historical_var(rets_log, 0.95)
p_var_95 = parametric_var(rets_log, 0.95)
 
axes[0].hist(rets_log.values, bins=80, color=COLORS["primary"],
             alpha=0.6, density=True, label="Return distribution")
axes[0].axvline(h_var_95["var_threshold"], color=COLORS["secondary"],
                linewidth=2.5, linestyle="-", label=f"Hist VaR 95%: {h_var_95['var_threshold']:.2%}")
axes[0].axvline(p_var_95["var_threshold"], color=COLORS["accent"],
                linewidth=2.5, linestyle="--", label=f"Param VaR 95%: {p_var_95['var_threshold']:.2%}")
axes[0].set_title("Historical vs Parametric VaR - Fat Tail Comparison", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Daily Log Return")
axes[0].legend()
 
# Bottom: Rolling VaR over time
axes[1].fill_between(roll_var.index, roll_var.values * 100, 0,
                     where=roll_var.notna(), alpha=0.3, color=COLORS["secondary"])
axes[1].plot(roll_var.index, roll_var.values * 100, color=COLORS["secondary"],
             linewidth=1.2, label="252d Rolling VaR 95%")
axes[1].set_title("Rolling VaR (252-day window) - Endogenous Risk in Action", fontsize=12, fontweight="bold")
axes[1].set_ylabel("VaR (%)")
axes[1].legend()
axes[1].axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-01"),
                alpha=0.12, color=COLORS["accent"])
 
plt.tight_layout()
save_fig(f"{TICKER}_04_var")
plt.show()


#### CELL 6: VaR backtest ####
 
kup = kupiec_test(rets_log, 0.95)
print("\nKupiec Test Result:")
for k, v in kup.items():
    print(f"  {k:<25}: {v}")
 
# Plot breaches
h_var = historical_var(rets_log, 0.95)
breach_mask = rets_log < h_var["var_threshold"]
 
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(rets_log.index, rets_log.values * 100, color=COLORS["primary"],
        linewidth=0.8, alpha=0.7, label="Daily returns")
ax.scatter(rets_log[breach_mask].index,
           rets_log[breach_mask].values * 100,
           color=COLORS["secondary"], s=25, zorder=5,
           label=f"VaR breaches ({kup['n_breaches']} days)")
ax.axhline(h_var["var_threshold"] * 100, color=COLORS["secondary"],
           linewidth=2, linestyle="--",
           label=f"VaR 95% threshold: {h_var['var_threshold']:.2%}")
ax.set_title(f"{TICKER} VaR Backtest - Breach Days Highlighted", fontsize=13, fontweight="bold")
ax.set_ylabel("Daily Return (%)")
ax.legend()
save_fig(f"{TICKER}_05_var_backtest")
plt.show()
 
 
#### CELL 7: Drawdown ####
 
dd = drawdown_series(prices)
dd_stats = max_drawdown(prices)
 
print("\nMax Drawdown Analysis:")
for k, v in dd_stats.items():
    print(f"  {k:<18}: {v}")
 
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
 
# Price with drawdown shading
axes[0].plot(prices.index, prices.values, color=COLORS["primary"], linewidth=1.5)
axes[0].fill_between(prices.index, prices.values, prices.cummax(),
                     where=prices < prices.cummax(),
                     alpha=0.3, color=COLORS["secondary"], label="Underwater")
axes[0].set_title(f"{TICKER} Price - Underwater Periods", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Price (USD)")
axes[0].legend()
 
# Drawdown curve
axes[1].fill_between(dd.index, dd.values * 100, 0,
                     color=COLORS["secondary"], alpha=0.5)
axes[1].plot(dd.index, dd.values * 100, color=COLORS["secondary"], linewidth=1)
axes[1].axhline(dd_stats["max_drawdown"] * 100, color="black",
                linewidth=1.2, linestyle="--",
                label=f"Max DD: {dd_stats['max_drawdown']:.1%}")
axes[1].set_title("Drawdown Over Time (%)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend()
 
plt.tight_layout()
save_fig(f"{TICKER}_06_drawdown")
plt.show()


#### CELL 8: Distribution diagnostics ####
 
print_diagnostic_report(rets_log, ticker=TICKER)
 
# QQ plot: if points follow the line, distribution is normal.
# Fat tails appear as S-curve deviations at the extremes.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
# QQ plot
(osm, osr), (slope, intercept, r) = stats.probplot(rets_log.values, dist="norm")
axes[0].scatter(osm, osr, alpha=0.4, color=COLORS["primary"], s=10, label="Observed")
axes[0].plot(osm, slope * np.array(osm) + intercept,
             color=COLORS["secondary"], linewidth=2, label="Normal line")
axes[0].set_title("QQ Plot - Normality Check\n(S-curve = fat tails)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Theoretical quantiles (Normal)")
axes[0].set_ylabel("Sample quantiles")
axes[0].legend()
 
# Rolling kurtosis
from diagnostics import rolling_kurtosis
roll_kurt = rolling_kurtosis(rets_log, window=63)
axes[1].plot(roll_kurt.index, roll_kurt.values, color=COLORS["primary"], linewidth=1.2)
axes[1].axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=1, label="Normal (0)")
axes[1].axhline(3, color=COLORS["secondary"], linestyle="--", linewidth=1, label="Threshold (3)")
axes[1].set_title("Rolling Excess Kurtosis (63d)\n(Spikes = fat-tail periods)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Excess Kurtosis")
axes[1].legend()
 
plt.tight_layout()
save_fig(f"{TICKER}_07_diagnostics")
plt.show()
 
 
#### CELL 9: Correlation analysis (multi-asset) ####
 
# Load second asset for comparison
TICKER_B = "USO"    # Oil ETF - should correlate with XLE
prices_b = load_price_data(TICKER_B, start=START_DATE, end=END_DATE)
prices_b = clean_prices(prices_b)
rets_b = log_returns(prices_b)
 
# Rolling correlation
roll_corr = rolling_correlation(rets_log, rets_b, window=63)
 
# Crisis-period breakdown
breakdown = correlation_breakdown_analysis(rets_log, rets_b)
print(f"\nCorrelation breakdown ({TICKER} vs {TICKER_B}):")
print(breakdown.to_string(index=False))
 
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.plot(roll_corr.index, roll_corr.values, color=COLORS["primary"],
        linewidth=1.2, label=f"63d Rolling Corr ({TICKER} vs {TICKER_B})")
ax.axhline(roll_corr.mean(), color=COLORS["neutral"],
           linestyle="--", linewidth=1, label=f"Mean: {roll_corr.mean():.2f}")
ax.axhspan(0.8, 1.0, alpha=0.08, color=COLORS["secondary"])
ax.set_ylim(-0.2, 1.05)
ax.set_title(f"Rolling Correlation: {TICKER} vs {TICKER_B}", fontsize=13, fontweight="bold")
ax.set_ylabel("Correlation")
ax.legend()
ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-01"),
           alpha=0.12, color=COLORS["accent"])
save_fig(f"{TICKER}_08_correlation")
plt.show()
 
print("\n[Done] All figures saved to outputs/figures/")
print("Run: python reports/generate_report.py --ticker XLE --start 2018-01-01")
 