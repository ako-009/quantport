"""
Backtest Overfitting & Deflated Sharpe Ratio
=============================================
Bailey & López de Prado (2014)
"The Deflated Sharpe Ratio: Correcting for Selection Bias,
 Backtest Overfitting and Non-Normality"
Journal of Portfolio Management

Three components:
1. Overfitting Simulation   — 1000 random strategies, show best is pure luck
2. Deflated Sharpe Ratio    — probability your Sharpe is real given N trials
3. Minimum Track Record     — days of live trading needed before trusting result
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from typing import Tuple, List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf: float = 0.065 / 252) -> float:
    """Annualized Sharpe Ratio."""
    excess = returns - rf
    return float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skewness: float,
    excess_kurtosis: float
) -> float:
    """
    PSR = P(SR* > benchmark_SR)

    Corrects for non-normality of returns using skewness and kurtosis.
    Bailey & López de Prado (2014), Eq. 3.

    Parameters
    ----------
    observed_sr       : Sharpe ratio observed in backtest (annualized)
    benchmark_sr      : SR we want to beat (usually 0 or Sharpe of buy-and-hold)
    n_obs             : number of daily observations
    skewness          : skewness of daily returns
    excess_kurtosis   : excess kurtosis of daily returns (kurtosis - 3)
    """
    # De-annualize SR for the formula (works on daily scale)
    sr_daily      = observed_sr   / np.sqrt(252)
    bench_daily   = benchmark_sr  / np.sqrt(252)

    # Variance of SR estimator (Lo 2002 correction for non-normality)
    var_sr = (1 - skewness * sr_daily +
              (excess_kurtosis / 4) * sr_daily**2) / (n_obs - 1)

    if var_sr <= 0:
        return 1.0 if sr_daily > bench_daily else 0.0

    z = (sr_daily - bench_daily) / np.sqrt(var_sr)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    trial_srs: List[float],       # SRs from all N trials/strategies tested
    n_obs: int,
    skewness: float,
    excess_kurtosis: float
) -> Dict:
    """
    DSR = PSR with benchmark = E[max SR | N trials, T observations]

    The expected maximum SR from N independent trials is approximated
    by the formula in Bailey & López de Prado (2014), Eq. 8:

    E[max SR] ≈ (1 - γ) Z^{-1}(1 - 1/N) + γ Z^{-1}(1 - 1/(N·e))

    where γ = Euler-Mascheroni constant ≈ 0.5772

    Parameters
    ----------
    observed_sr  : SR of the strategy you want to validate
    trial_srs    : list of SRs from ALL strategies you tested (including observed)
    n_obs        : number of daily return observations
    skewness     : skewness of daily returns of observed strategy
    excess_kurtosis : excess kurtosis of daily returns
    """
    N = len(trial_srs)
    if N == 0:
        return {"dsr": np.nan, "psr": np.nan, "e_max_sr": np.nan,
                "n_trials": 0, "verdict": "Insufficient data"}

    # Expected maximum SR from N independent trials
    gamma_em  = 0.5772156649   # Euler-Mascheroni constant
    e_max_sr  = ((1 - gamma_em) * norm.ppf(1 - 1/N) +
                  gamma_em      * norm.ppf(1 - 1/(N * np.e)))

    # De-annualize
    e_max_sr_daily = e_max_sr / np.sqrt(252)

    # PSR with E[max SR] as benchmark
    sr_daily  = observed_sr / np.sqrt(252)
    var_sr    = (1 - skewness * sr_daily +
                 (excess_kurtosis / 4) * sr_daily**2) / max(n_obs - 1, 1)

    if var_sr <= 0:
        psr = 1.0 if sr_daily > e_max_sr_daily else 0.0
    else:
        z   = (sr_daily - e_max_sr_daily) / np.sqrt(var_sr)
        psr = float(norm.cdf(z))

    # Verdict
    if psr >= 0.95:   verdict = "✅ Very likely genuine (p ≥ 95%)"
    elif psr >= 0.75: verdict = "🟡 Possibly genuine (75–95%)"
    elif psr >= 0.50: verdict = "🟠 Uncertain (50–75%)"
    else:             verdict = "🔴 Likely overfitted (p < 50%)"

    return {
        "dsr":        round(psr, 4),
        "psr":        round(psr, 4),
        "e_max_sr":   round(e_max_sr, 4),
        "n_trials":   N,
        "observed_sr":round(observed_sr, 4),
        "verdict":    verdict,
    }


def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    excess_kurtosis: float,
    target_psr: float = 0.95
) -> float:
    """
    Minimum number of daily observations needed so that
    PSR(observed_sr) ≥ target_psr.

    Inverts the PSR formula for n_obs.
    Bailey & López de Prado (2014), Eq. 6.

    Returns minimum track record length in DAYS.
    """
    sr_daily    = observed_sr  / np.sqrt(252)
    bench_daily = benchmark_sr / np.sqrt(252)
    z_target    = norm.ppf(target_psr)

    # From PSR formula: z = (sr - bench) / sqrt(var_sr)
    # var_sr = (1 - skew*sr + (kurt/4)*sr^2) / (n-1)
    # Solve for n:
    numerator   = 1 - skewness * sr_daily + (excess_kurtosis / 4) * sr_daily**2
    denominator = ((sr_daily - bench_daily) / z_target) ** 2

    if denominator <= 0:
        return np.inf

    n_min = 1 + numerator / denominator
    return float(max(n_min, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM STRATEGY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_strategies(
    prices: pd.DataFrame,
    n_strategies: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate N random momentum/mean-reversion strategies on real price data.

    Strategy types:
    - Momentum:        buy if price > rolling_mean(w), else sell
    - Mean-reversion:  buy if price < rolling_mean(w), else sell
    - Random signal:   random ±1 signal

    Returns DataFrame with one column per strategy (daily returns).
    """
    rng = np.random.default_rng(seed)
    returns_all = prices.pct_change().dropna()

    # Use first ticker or equal-weight
    if returns_all.shape[1] == 1:
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices.mean(axis=1)

    raw_returns = price_series.pct_change().dropna()
    n_days      = len(raw_returns)

    strategy_returns = {}

    for i in range(n_strategies):
        stype  = rng.choice(["momentum", "mean_reversion", "random"])
        window = int(rng.integers(5, 120))

        if stype == "random":
            signal = pd.Series(
                rng.choice([-1, 1], size=n_days),
                index=raw_returns.index
            )
        else:
            roll = price_series.rolling(window).mean().dropna()
            common = raw_returns.index.intersection(roll.index)
            raw_r  = raw_returns.loc[common]
            roll   = roll.loc[common]

            if stype == "momentum":
                signal = np.sign(price_series.loc[common] - roll)
            else:
                signal = -np.sign(price_series.loc[common] - roll)

            # Align
            raw_returns_aligned = raw_r
            strategy_ret = signal.shift(1).fillna(0) * raw_returns_aligned
            strategy_returns[f"s{i}"] = strategy_ret
            continue

        strategy_ret = signal.shift(1).fillna(0) * raw_returns
        strategy_returns[f"s{i}"] = strategy_ret

    return pd.DataFrame(strategy_returns).dropna()


def compute_strategy_sharpes(
    strategy_returns: pd.DataFrame,
    rf: float = 0.065 / 252
) -> pd.Series:
    """Compute annualized Sharpe for each strategy column."""
    excess = strategy_returns.subtract(rf)
    mean   = excess.mean()
    std    = excess.std()
    return (mean / std * np.sqrt(252)).replace([np.inf, -np.inf], np.nan).dropna()


def full_overfitting_report(
    prices: pd.DataFrame,
    portfolio_returns: pd.Series,
    rf: float = 0.065,
    n_strategies: int = 1000,
    benchmark_sr: float = 0.0,
    target_psr: float = 0.95,
) -> Dict:
    """
    Complete overfitting analysis:
    1. Generate N random strategies
    2. Compute Sharpe distribution
    3. Apply Deflated Sharpe to portfolio
    4. Compute minimum track record length

    Returns all results in a single dict.
    """
    # Portfolio stats
    rf_daily   = rf / 252
    port_sr    = sharpe_ratio(portfolio_returns, rf_daily)
    port_skew  = float(skew(portfolio_returns))
    port_kurt  = float(kurtosis(portfolio_returns))  # excess kurtosis
    n_obs      = len(portfolio_returns)

    # Random strategies
    strat_ret  = generate_random_strategies(prices, n_strategies)
    strat_srs  = compute_strategy_sharpes(strat_ret, rf_daily)

    # Best random strategy
    best_sr    = float(strat_srs.max())
    best_idx   = strat_srs.idxmax()
    best_ret   = strat_ret[best_idx] if best_idx in strat_ret.columns else pd.Series()

    # Percentile of portfolio SR among random strategies
    pct_rank   = float((strat_srs < port_sr).mean() * 100)

    # DSR
    dsr_result = deflated_sharpe_ratio(
        observed_sr=port_sr,
        trial_srs=strat_srs.tolist(),
        n_obs=n_obs,
        skewness=port_skew,
        excess_kurtosis=port_kurt,
    )

    # PSR vs benchmark=0
    psr_vs_zero = probabilistic_sharpe_ratio(
        observed_sr=port_sr,
        benchmark_sr=benchmark_sr,
        n_obs=n_obs,
        skewness=port_skew,
        excess_kurtosis=port_kurt,
    )

    # Minimum track record
    mtrl_days  = minimum_track_record_length(
        observed_sr=port_sr,
        benchmark_sr=benchmark_sr,
        skewness=port_skew,
        excess_kurtosis=port_kurt,
        target_psr=target_psr,
    )
    mtrl_years = mtrl_days / 252

    return {
        # Portfolio
        "port_sr":          round(port_sr,  4),
        "port_skew":        round(port_skew, 4),
        "port_kurt":        round(port_kurt, 4),
        "n_obs":            n_obs,

        # Random strategies
        "n_strategies":     n_strategies,
        "strat_srs":        strat_srs,
        "strat_ret":        strat_ret,
        "best_random_sr":   round(best_sr, 4),
        "best_random_ret":  best_ret,
        "pct_rank":         round(pct_rank, 2),

        # DSR
        "dsr":              dsr_result,
        "psr_vs_zero":      round(psr_vs_zero, 4),

        # Track record
        "mtrl_days":        round(mtrl_days, 1),
        "mtrl_years":       round(mtrl_years, 2),
        "target_psr":       target_psr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n, na = 756, 5
    idx    = pd.date_range("2021-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.0004, 0.015, (n, na)), axis=0),
        index=idx, columns=["AAPL","MSFT","GOOGL","JPM","GLD"]
    )
    port_ret = prices.pct_change().dropna().mean(axis=1)

    print("Running overfitting analysis (1000 strategies)...")
    result = full_overfitting_report(prices, port_ret, n_strategies=500)

    print(f"\nPortfolio SR       : {result['port_sr']}")
    print(f"Best Random SR     : {result['best_random_sr']}")
    print(f"Portfolio Percentile: {result['pct_rank']:.1f}%")
    print(f"DSR (prob genuine) : {result['dsr']['dsr']}")
    print(f"Verdict            : {result['dsr']['verdict']}")
    print(f"Min Track Record   : {result['mtrl_days']:.0f} days ({result['mtrl_years']:.1f} years)")
    print(f"PSR vs SR=0        : {result['psr_vs_zero']}")
    print("\nOK ✓")
