import argparse
import os
import sys
from datetime import date
 
# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from loader import load_price_data, clean_prices
from returns import log_returns, return_summary, worst_days
from volatility import rolling_volatility, ewma_volatility, vol_regime_summary
from var import historical_var, historical_cvar, parametric_var, var_comparison, kupiec_test
from drawdown import max_drawdown, drawdown_series
from diagnostics import distribution_stats, jarque_bera_test, tail_ratio
 
 # Report builder
 
def build_report(ticker: str, start: str, end: str = None) -> str:
    lines = []
    sep = "=" * 65
    thin = "-" * 65
 
    def add(text=""):
        lines.append(text)

    # Data loading

    add(sep)
    add(f"  ENERGY RISK ENGINE — MARKET RISK REPORT")
    add(sep)
    add(f"  Ticker    : {ticker}")
    add(f"  Start     : {start}")
    add(f"  End       : {end or str(date.today())}")
    add(f"  Generated : {date.today()}")
    add(sep)
 
    try:
        prices = load_price_data(ticker, start=start, end=end)
        prices = clean_prices(prices)
        rets = log_returns(prices)
    except Exception as e:
        add(f"ERROR: Could not load data. {e}")
        return "\n".join(lines)
 
    n_obs = len(rets)
    add(f"\n  Observations: {n_obs} trading days")
    add(f"  Price range : ${prices.min():.2f} — ${prices.max():.2f}")

    # Section 1: Return statistics

    add(f"\n{thin}")
    add("  [1] RETURN STATISTICS")
    add(thin)
 
    summary = return_summary(rets)
    add(f"  Mean daily return   : {summary['mean_daily']:.4%}")
    add(f"  Daily volatility    : {summary['std_daily']:.4%}")
    add(f"  Annualized return   : {summary['annualized_return']:.2%}")
    add(f"  Annualized vol      : {summary['annualized_vol']:.2%}")
    add(f"  Sharpe ratio        : {summary['sharpe_ratio']:.3f}  (rf = 0)")
    add(f"  Skewness            : {summary['skewness']:.4f}")
    add(f"  Excess kurtosis     : {summary['kurtosis']:.4f}  (0 = normal)")

    # Section 2: VaR and CVaR

    add(f"\n{thin}")
    add("  [2] VALUE AT RISK (VaR) AND EXPECTED SHORTFALL (CVaR)")
    add(thin)
 
    h_var_95 = historical_var(rets, 0.95)
    h_var_99 = historical_var(rets, 0.99)
    h_cvar_95 = historical_cvar(rets, 0.95)
    h_cvar_99 = historical_cvar(rets, 0.99)
    p_var_95 = parametric_var(rets, 0.95)
 
    add(f"  Historical VaR  95% : {h_var_95['var_threshold']:.4%} per day")
    add(f"  Historical VaR  99% : {h_var_99['var_threshold']:.4%} per day")
    add(f"  Parametric VaR  95% : {p_var_95['var_threshold']:.4%} per day")
    add(f"  Historical CVaR 95% : {h_cvar_95['cvar']:.4%} per day")
    add(f"  Historical CVaR 99% : {h_cvar_99['cvar']:.4%} per day")
 
    gap = h_var_95['var_threshold'] - p_var_95['var_threshold']
    add(f"\n  Fat-tail gap (hist - param @ 95%): {gap:.4%}")
    add(f"  Interpretation: {'Fat tails present — Gaussian understates risk.' if abs(gap) > 0.003 else 'Tails close to Gaussian.'}")
 
    # Section 3: VaR backtest

    add(f"\n{thin}")
    add("  [3] VAR BACKTEST (KUPIEC TEST, 95%)")
    add(thin)
 
    kup = kupiec_test(rets, 0.95)
    add(f"  Expected breach rate : {kup['expected_breach_rate']:.1%}")
    add(f"  Observed breach rate : {kup['observed_breach_rate']:.1%}")
    add(f"  Breach count         : {kup['n_breaches']} / {kup['n_observations']} days")
    add(f"  LR statistic         : {kup['lr_statistic']:.4f}")
    add(f"  p-value              : {kup['p_value']:.4f}")
    add(f"  Result               : {kup['verdict']}")
 
    # Section 4: Volatility

    add(f"\n{thin}")
    add("  [4] VOLATILITY ANALYSIS")
    add(thin)
 
    vol_21 = rolling_volatility(rets, 21).dropna()
    vol_63 = rolling_volatility(rets, 63).dropna()
    ewma = ewma_volatility(rets).dropna()
 
    add(f"  Current 21d rolling vol : {vol_21.iloc[-1]:.2%} (annualized)")
    add(f"  Current 63d rolling vol : {vol_63.iloc[-1]:.2%} (annualized)")
    add(f"  Current EWMA vol        : {ewma.iloc[-1]:.2%} (annualized)")
 
    vol_summary = vol_regime_summary(rets, 21)
    add(f"\n  Vol distribution (21d rolling, annualized):")
    add(f"    5th pct  : {vol_summary['vol_5th_pct']:.2%}")
    add(f"    Median   : {vol_summary['vol_median']:.2%}")
    add(f"    95th pct : {vol_summary['vol_95th_pct']:.2%}")
    add(f"    Max      : {vol_summary['vol_max']:.2%}")
 
    # Section 5: Drawdown

    add(f"\n{thin}")
    add("  [5] DRAWDOWN ANALYSIS")
    add(thin)
 
    dd_stats = max_drawdown(prices)
    add(f"  Max drawdown    : {dd_stats['max_drawdown']:.2%}")
    add(f"  Peak date       : {dd_stats['peak_date']}")
    add(f"  Trough date     : {dd_stats['trough_date']}")
    add(f"  Duration        : {dd_stats['duration_days']} calendar days (peak to trough)")
    add(f"  Recovery date   : {dd_stats['recovery_date'] or 'Not yet recovered'}")
 
    # Section 6: Distribution diagnostics

    add(f"\n{thin}")
    add("  [6] DISTRIBUTION DIAGNOSTICS")
    add(thin)
 
    jb = jarque_bera_test(rets)
    tails = tail_ratio(rets)
 
    add(f"  Jarque-Bera p-value     : {jb['p_value']:.6f}")
    add(f"  Normality rejected      : {jb['reject_normality']}")
    add(f"  Observed tail frequency : {tails['total_tail_pct']:.2%}  (±3 sigma)")
    add(f"  Gaussian expectation    : {tails['expected_normal_pct']:.2%}")
    add(f"  Tail excess ratio       : {tails['tail_excess_ratio']:.1f}x Gaussian")
 
    # Section 7: Worst days

    add(f"\n{thin}")
    add("  [7] WORST 10 RETURN DAYS")
    add(thin)
    add(f"  {'Rank':<6} {'Date':<14} {'Return':<12}")
    add(f"  {'-'*4:<6} {'-'*12:<14} {'-'*10:<12}")
 
    wd = worst_days(rets, n=10)
    for _, row in wd.iterrows():
        add(f"  {int(row['rank']):<6} {str(row['date'].date()):<14} {row['return']:.4%}")
 
    # Section 8: Model risk memo

    add(f"\n{thin}")
    add("  [8] MODEL RISK LIMITATIONS")
    add(thin)
    add("  1. Historical VaR assumes past distribution = future distribution; fails during regime changes (2020, 2022).")
    add(f"  2. Gaussian VaR underestimates tail risk; fat tails are {tails['tail_excess_ratio']:.1f}x more frequent than Gaussian.")
    add("  3. Rolling VaR reacts to past events, not future ones (endogenous risk).")
    add("  3. Rolling VaR reacts to past events, not future ones (endogenous risk).")
    add("  4. No volatility clustering model (GARCH) — vol persistence is ignored.")
    add("  5. All metrics assume continuous liquid markets. Energy can gap violently.")
    add("  6. Correlation structure is assumed constant — breaks down in crises.")
    add("  7. VaR says nothing about losses BEYOND the threshold.")
    add("     Always pair with CVaR/Expected Shortfall.")
 
    add(f"\n{sep}")
    add("  END OF REPORT")
    add(sep)
 
    return "\n".join(lines)

# CLI entry point

def main():
    parser = argparse.ArgumentParser(
        description="Energy Risk Engine — CLI Market Risk Report Generator"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Yahoo Finance ticker symbol (e.g. XLE, CL=F, USO)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to reports/output/ directory"
    )
 
    args = parser.parse_args()
 
    # Build the report
    report = build_report(args.ticker, args.start, args.end)
 
    # Print to terminal
    print(report)
 
    # Optionally save
    if args.save:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
 
        # Sanitize ticker for filename (CL=F → CL_F)
        safe_ticker = args.ticker.replace("=", "_").replace("^", "")
        filename = f"{safe_ticker}_risk_report_{date.today()}.txt"
        filepath = os.path.join(output_dir, filename)
 
        with open(filepath, "w") as f:
            f.write(report)
 
        print(f"\n[saved] Report saved to: {filepath}")
 
 
if __name__ == "__main__":
    main()
 

