# ================================
# ENERGY RISK DASHBOARD (FULL)
# ================================

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Add project root to path to import your modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Your custom modules ---
from loader import load_price_data, clean_prices
from returns import log_returns
from volatility import rolling_volatility, ewma_volatility
from var import rolling_var, historical_var, parametric_var
from drawdown import drawdown_series, max_drawdown
from correlation import rolling_correlation

# --- Plotting style ---
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary":   "#1f3a5f",
    "secondary": "#c0392b",
    "accent":    "#e67e22",
    "neutral":   "#7f8c8d",
    "positive":  "#27ae60",
}
FIG_SIZE = (14, 5)

# --- Input ---
TICKER = input("Enter main ticker (e.g., XLE, SPY): ").strip()
START_DATE = input("Enter start date (YYYY-MM-DD): ").strip()
END_DATE = None   # None = today
TICKER_B = input("Enter secondary ticker (e.g., CL=F, USO): ").strip()  # Secondary asset for correlation

# --- Load data ---
prices = clean_prices(load_price_data(TICKER, start=START_DATE, end=END_DATE))
rets = log_returns(prices)

prices_b = clean_prices(load_price_data(TICKER_B, start=START_DATE, end=END_DATE))
rets_b = log_returns(prices_b)

# --- Derived metrics ---
vol_21 = rolling_volatility(rets, 21)
vol_63 = rolling_volatility(rets, 63)
ewma_vol = ewma_volatility(rets)
roll_var = rolling_var(rets, 252, 0.95)
dd = drawdown_series(prices)
roll_corr = rolling_correlation(rets, rets_b, 63)

h_var = historical_var(rets, 0.95)
breach_mask = rets < h_var["var_threshold"]

# ================================
# DASHBOARD 1: Core Risk
# ================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 2)

# Price
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(prices.index, prices.values, color=COLORS["primary"])
ax1.set_title(f"{TICKER} Price")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Returns
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(rets.index, rets.values, color=COLORS["primary"], linewidth=0.7)
ax2.axhline(0, linestyle="--", color=COLORS["neutral"])
ax2.set_title("Daily Returns")

# Volatility
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(vol_21.index, vol_21*100, color=COLORS["accent"], label="21d Rolling")
ax3.plot(vol_63.index, vol_63*100, color=COLORS["primary"], label="63d Rolling")
ax3.plot(ewma_vol.index, ewma_vol*100, color=COLORS["secondary"], linestyle="--", label="EWMA")
ax3.set_title("Volatility (%)")
ax3.legend()

# Rolling VaR
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(roll_var.index, roll_var*100, color=COLORS["secondary"])
ax4.set_title("252d Rolling VaR 95%")

# Underwater chart
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(prices.index, prices.values, color=COLORS["primary"])
ax5.fill_between(prices.index, prices.values, prices.cummax(),
                 where=prices<prices.cummax(), color=COLORS["secondary"], alpha=0.3)
ax5.set_title("Underwater / Capital at Risk")

# Drawdown curve
ax6 = fig.add_subplot(gs[3, 0])
ax6.fill_between(dd.index, dd*100, 0, color=COLORS["secondary"], alpha=0.5)
ax6.plot(dd.index, dd*100, color=COLORS["secondary"], linewidth=1)
ax6.set_title("Drawdown (%)")

# VaR breaches
ax7 = fig.add_subplot(gs[3, 1])
ax7.plot(rets.index, rets*100, color=COLORS["primary"], alpha=0.6)
ax7.scatter(rets[breach_mask].index, rets[breach_mask]*100,
            color=COLORS["secondary"], s=15)
ax7.axhline(h_var["var_threshold"]*100, color=COLORS["secondary"], linestyle="--")
ax7.set_title("VaR 95% Breaches")

plt.tight_layout()
plt.show()

# ================================
# DASHBOARD 2: Diagnostics
# ================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2)

# Return distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(rets.values, bins=80, density=True, alpha=0.6, color=COLORS["primary"])
x = np.linspace(rets.min(), rets.max(), 300)
mu, sigma = rets.mean(), rets.std()
ax1.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS["secondary"], linewidth=2)
ax1.set_title("Return Distribution vs Normal")

# QQ plot
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(rets.values, dist="norm", plot=ax2)
ax2.set_title("QQ Plot (Fat Tails)")

# Rolling vol regime
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(vol_21.index, vol_21*100, color=COLORS["accent"])
ax3.set_title("Volatility Regime (21d)")

# Rolling VaR regime
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(roll_var.index, roll_var*100, color=COLORS["secondary"])
ax4.set_title("Rolling VaR (252d)")

# Rolling correlation
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(roll_corr.index, roll_corr.values, color=COLORS["primary"])
ax5.axhline(roll_corr.mean(), linestyle="--", color=COLORS["neutral"])
ax5.set_title(f"Rolling Correlation ({TICKER} vs {TICKER_B})")

plt.tight_layout()
plt.show()