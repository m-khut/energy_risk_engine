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
├── [[loader.py]]          # yfinance pull, validation, cleaning
├── [[returns.py]]         # log/simple returns, diagnostics
├── [[volatility.py]]      # rolling, EWMA, regime summary
├── [[var.py]]             # Historical + Parametric VaR, CVaR, Kupiec
├── [[drawdown.py]]        # max drawdown, periods, Calmar
├── [[diagnostics.py]]     # Jarque-Bera, kurtosis, fat tail tests
├── [[correlation.py]]     # rolling corr, crisis breakdown
├── [[generate_report.py]] # CLI risk report generator
    └── output/            # generated reports
├── [[risk_nerrative.py]]  # interview walkthrough notebook
├── outputs/figures/       # publication-quality plots
├── [[Mode Risk Memo]]     # model validation memo (SR 11-7 format)
```

## How to Run

```bash
# Generate a full risk report for XLE
python generate_report.py --ticker XLE --start 2018-01-01

# Save the report to file
python generate_report.py --ticker CL=F --start 2015-01-01 --save

# Run the narrative notebook (Jupyter or as script)
python risk_narrative.py
```

## Requirements
yfinance>=0.2.28
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0


