**Energy Risk Engine - Project 1** **Prepared by:** **Meet Khut**
**Review Standard:** Modeled on Federal Reserve SR 11-7 Supervisory Guidance on Model Risk Management


## 1. Model Purpose and Scope

This engine computes market risk metrics for energy sector assets (ETFs, futures, spot proxies) using publicly available daily closing price data from Yahoo Finance. Metrics produced include:

- **Value at Risk (VaR):** Historical simulation and parametric (Gaussian) methods at 95% and 99% confidence
- **Conditional Value at Risk (CVaR / Expected Shortfall):** Mean loss in the tail beyond VaR
- **Volatility:** Rolling window (21d, 63d) and EWMA (λ=0.94, RiskMetrics standard)
- **Drawdown:** Maximum drawdown, duration, recovery
- **Distributional diagnostics:** Jarque-Bera test, kurtosis, fat-tail ratio

**Intended use:** Research, skill development, and portfolio monitoring for energy sector exposures. **Not intended for:** Live trading decisions, regulatory capital reporting, or client risk disclosure.

---

## 2. Methodology Summary

### 2.1 Historical Simulation VaR

Returns the empirical quantile of historical daily log returns at confidence level α. No distributional assumption is imposed. VaR = Q(1-α) of the observed return distribution.

### 2.2 Parametric Gaussian VaR

Fits a normal distribution (mean, std) to observed returns. VaR = μ - z(α) · σ where z(α) is the Gaussian quantile at confidence level α.

### 2.3 EWMA Volatility

Implements the RiskMetrics (1994) exponentially weighted moving average: 
σ²_t = λ · σ²_{t-1} + (1-λ) · r²_{t-1}, with λ = 0.94 (daily standard). 
Designed to weight recent returns more heavily than equally-weighted rolling std.

### 2.4 Kupiec Backtest

Tests whether the observed VaR breach frequency is statistically consistent with the model's confidence level. Uses a likelihood-ratio test under H₀ that the true breach probability = (1 - confidence).

---

## 3. Key Assumptions

|#|Assumption|Where Used|Consequence if Wrong|
|---|---|---|---|
|1|Past return distribution represents future risk|All historical methods|Fails completely in regime changes (2020, 2022)|
|2|Returns are i.i.d. (no autocorrelation)|Annualization, VaR|Volatility clustering means this is false for energy|
|3|Returns are normally distributed|Parametric VaR only|Systematically underestimates tail risk (see §4.1)|
|4|λ = 0.94 is appropriate for energy assets|EWMA volatility|Sub-optimal decay rate for high-vol, fat-tailed assets|
|5|Log returns are additive|Multi-period P&L|Breaks down for large return magnitudes (>10% daily)|
|6|Yahoo Finance data is accurate|All metrics|Corporate actions, roll dates (futures), errors not corrected|

---

## 4. Known Limitations

### 4.1 Fat Tails and Non-Normality

Energy asset returns exhibit significant excess kurtosis (typically 2–6 above normal baseline) and negative skew. The parametric Gaussian VaR systematically underestimates tail risk. For XLE (2018–2024):

- Observed tail frequency: ~1.5–2% at ±3σ
- Expected under normality: 0.27%
- **Implication:** Parametric VaR should be treated as a lower bound, not the primary estimate.

### 4.2 Volatility Clustering

Volatility in energy markets exhibits persistence: high-vol periods cluster together. None of the models in this engine explicitly capture this (GARCH would be required). EWMA partially addresses this through exponential weighting, but does not model mean-reversion in volatility.

### 4.3 Endogenous Risk

Rolling VaR using historical simulation rises _after_ a crash event enters the window and falls _after_ it exits. This creates pro-cyclical risk estimation - VaR signals high risk precisely when the portfolio is already under stress.

### 4.4 Correlation Instability

Correlation estimates are computed on full-sample or rolling windows. During the 2020 COVID crash, correlations across energy sector assets spiked toward 1.0. The correlation module documents this but the VaR engine does not dynamically adjust for correlation breakdowns.

### 4.5 Futures Roll Risk

WTI crude (CL=F from Yahoo Finance) represents the front-month futures contract. At roll dates, there is a price discontinuity that appears as a return outlier. This has not been corrected in the data pipeline.

### 4.6 No Liquidity Adjustment

All metrics assume that positions can be liquidated at closing prices. During extreme events (2020 crash), bid-ask spreads in energy ETFs widened significantly. Actual realized losses can exceed modeled VaR for this reason alone.

---

## 5. Conditions Under Which the Model Fails

1. **Regime change:** A structural shift in energy markets (new supply technology, demand destruction) makes historical return distributions obsolete. The model has no mechanism to detect or adjust for this.
    
2. **Liquidity crisis:** VaR assumes mark-to-market losses. In an actual liquidation scenario, slippage and market impact can multiply the modeled loss by 2–5x for large positions.
    
3. **Correlation breakdown:** Portfolio-level VaR computed from asset-level VaRs implicitly assumes stable correlation. Crisis periods violate this.
    
4. **Very short history:** Models calibrated on <252 trading days are statistically unreliable. The Kupiec test has low power at small sample sizes.
    
5. **Futures-specific events:** Negative oil price event (April 2020, WTI front-month to -$37/barrel) is not representable in a log-return framework (log of negative number undefined). The data pipeline would need specific handling for such events.
    

---

## 6. Recommended Validation Approach

|Validation Type|Method|Frequency|
|---|---|---|
|Backtest (frequency)|Kupiec test at 95% and 99%|Monthly|
|Backtest (clustering)|Christoffersen test|Quarterly|
|Distributional fit|Jarque-Bera + QQ plot|Quarterly|
|Parameter sensitivity|Rerun with λ = 0.91, 0.97|Annual|
|Out-of-sample test|Reserve 2 years as holdout|Model build|
|Stress comparison|Compare modeled vs 2020 actual losses|Ad hoc|

---

## 7. What "Correct Code ≠ Safe Model" Means Here

The implementation is mathematically correct. The Kupiec test is implemented per the original Kupiec (1995) specification. EWMA uses the RiskMetrics standard. The issue is not the code - it is the assumptions embedded in the models themselves.

A model that passes code review and unit tests can still:

- Underestimate tail risk because of distributional assumptions
- Fail in a crisis because historical data doesn't contain the relevant scenario
- Give a false sense of precision (VaR to 4 decimal places) when the true uncertainty is ±50%

**The primary risk of this engine is not a bug. It is overconfidence in the output.**

---

_This memo should be read alongside the model implementation before any use of these metrics for decision-making._