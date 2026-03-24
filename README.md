Project - Energy Risk Engine

Build a production-style risk analytics engine for energy markets that computes VaR, CVaR, volatility regimes, and drawdown with explicit model limitation documentation

VaR · CVaR · EWMA Volatility · Drawdown · Diagnostics ·
Kupiec Backtest · Correlation · CLI Report Generator

Methodology

- **Historical VaR:** Empirical quantile of log return distribution. No distributional assumption.
- **Parametric VaR:** Gaussian assumption. Used as comparison baseline only.
- **CVaR / Expected Shortfall:** Mean of returns beyond VaR threshold. Required by Basel III.
- **EWMA Volatility:** RiskMetrics (1994) standard, λ=0.94. Weights recent returns exponentially.

## Project Structure
```
energy-risk-engine/
├── data/
│   └── [[loader.py]]          # yfinance pull, validation, cleaning
├── risk/
│   ├── [[returns.py]]         # log/simple returns, diagnostics
│   ├── [[volatility.py]]      # rolling, EWMA, regime summary
│   ├── [[var.py]]             # Historical + Parametric VaR, CVaR, Kupiec
│   ├── [[drawdown.py]]        # max drawdown, periods, Calmar
│   ├── [[diagnostics.py]]     # Jarque-Bera, kurtosis, fat tail tests
│   └── [[correlation.py]]     # rolling corr, crisis breakdown
├── reports/
│   ├── [[generate_report.py]] # CLI risk report generator
│   └── output/            # generated reports
├── notebooks/
│   └── [[risk_nerrative.py]]  # interview walkthrough notebook
├── outputs/figures/       # publication-quality plots
├── [[Mode Risk Memo]]          # model validation memo (SR 11-7 format)
└── [[requirements.txt]]
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a full risk report for XLE
python reports/generate_report.py --ticker XLE --start 2018-01-01

# Save the report to file
python reports/generate_report.py --ticker CL=F --start 2015-01-01 --save

# Run the narrative notebook (Jupyter or as script)
python notebooks/risk_narrative.py
