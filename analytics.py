"""
QuantPort Analytics Engine
==========================
Markowitz optimization, risk metrics, factor model, stress testing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
from typing import List, Dict, Tuple, Optional


# ─── DATA ────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.dropna(how="all").ffill().dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annualized_stats(returns: pd.DataFrame, rf: float = 0.065) -> pd.DataFrame:
    mu    = returns.mean() * 252
    sigma = returns.std()  * np.sqrt(252)
    sharpe = (mu - rf) / sigma

    def max_dd(col):
        cum = (1 + col).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    def sortino(col):
        downside = col[col < 0].std() * np.sqrt(252)
        return ((col.mean() * 252) - rf) / downside if downside > 0 else np.nan

    mdd  = returns.apply(max_dd)
    sort = returns.apply(sortino)
    sk   = returns.apply(skew)
    kurt = returns.apply(kurtosis)

    return pd.DataFrame({
        "Ann. Return":     mu,
        "Ann. Volatility": sigma,
        "Sharpe":          sharpe,
        "Sortino":         sort,
        "Max Drawdown":    mdd,
        "Skewness":        sk,
        "Kurtosis":        kurt,
    })


# ─── OPTIMIZATION ────────────────────────────────────────────────────────────

def portfolio_performance(w, mu, cov, rf=0.065):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sr  = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sr


def max_sharpe_portfolio(mu, cov, rf=0.065,
                          sector_caps: Optional[Dict[int, float]] = None) -> np.ndarray:
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if sector_caps:
        for idx_list, cap in sector_caps.items():
            constraints.append({
                "type": "ineq",
                "fun": lambda w, il=idx_list, c=cap: c - w[il]
            })
    res = minimize(lambda w: -portfolio_performance(w, mu, cov, rf)[2],
                   np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n, constraints=constraints,
                   options={"ftol":1e-12,"maxiter":1000})
    return res.x


def min_volatility_portfolio(mu, cov) -> np.ndarray:
    n = len(mu)
    res = minimize(lambda w: portfolio_performance(w, mu, cov)[1],
                   np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"ftol":1e-12,"maxiter":1000})
    return res.x


def max_sortino_portfolio(returns: pd.DataFrame, rf=0.065) -> np.ndarray:
    """Maximize Sortino ratio (penalizes only downside vol)."""
    n = returns.shape[1]
    def neg_sortino(w):
        pr = (returns * w).sum(axis=1)
        ann_ret = pr.mean() * 252
        downside = pr[pr < 0].std() * np.sqrt(252)
        return -(ann_ret - rf) / downside if downside > 0 else 0.0
    res = minimize(neg_sortino, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"ftol":1e-10,"maxiter":1000})
    return res.x


def efficient_frontier(mu, cov, n_points=80) -> pd.DataFrame:
    w_mv  = min_volatility_portfolio(mu, cov)
    r_min = portfolio_performance(w_mv, mu, cov)[0]
    r_max = float(mu.max()) * 0.98
    n = len(mu)
    results = []
    for target in np.linspace(r_min, r_max, n_points):
        res = minimize(
            lambda w: portfolio_performance(w, mu, cov)[1],
            np.ones(n)/n, method="SLSQP", bounds=[(0,1)]*n,
            constraints=[
                {"type":"eq","fun":lambda w: w.sum()-1},
                {"type":"eq","fun":lambda w, r=target: portfolio_performance(w, mu, cov)[0]-r}
            ], options={"ftol":1e-10,"maxiter":500}
        )
        if res.success:
            r, v, s = portfolio_performance(res.x, mu, cov)
            results.append({"Return":r,"Volatility":v,"Sharpe":s})
    return pd.DataFrame(results)


def random_portfolios(mu, cov, n=5000, rf=0.065) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(len(mu)))
        r, v, s = portfolio_performance(w, mu, cov, rf)
        rows.append({"Return":r,"Volatility":v,"Sharpe":s})
    return pd.DataFrame(rows)


# ─── RISK METRICS ────────────────────────────────────────────────────────────

def portfolio_returns_series(w, returns: pd.DataFrame) -> pd.Series:
    return (returns * w).sum(axis=1)


def historical_var(pr: pd.Series, conf=0.95) -> float:
    return float(np.percentile(pr, (1-conf)*100))


def historical_cvar(pr: pd.Series, conf=0.95) -> float:
    var = historical_var(pr, conf)
    return float(pr[pr <= var].mean())


def parametric_var(pr: pd.Series, conf=0.95) -> float:
    return float(norm.ppf(1-conf, pr.mean(), pr.std()))


def cornish_fisher_var(pr: pd.Series, conf=0.95) -> float:
    """Modified VaR using Cornish-Fisher expansion for non-normal returns."""
    z  = norm.ppf(1-conf)
    s  = skew(pr)
    k  = kurtosis(pr)   # excess kurtosis
    z_cf = (z + (z**2-1)*s/6 + (z**3-3*z)*k/24 - (2*z**3-5*z)*s**2/36)
    return float(pr.mean() + z_cf * pr.std())


def monte_carlo_var(pr: pd.Series, conf=0.95, n_sim=10000, horizon=1) -> float:
    rng = np.random.default_rng(0)
    sim = rng.normal(pr.mean()*horizon, pr.std()*np.sqrt(horizon), n_sim)
    return float(np.percentile(sim, (1-conf)*100))


def drawdown_series(pr: pd.Series) -> pd.Series:
    cum = (1+pr).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def rolling_beta(pr: pd.Series, bm: pd.Series, window=60) -> pd.Series:
    return pr.rolling(window).cov(bm) / bm.rolling(window).var()


def compute_all_risk(w, returns: pd.DataFrame,
                     benchmark="^GSPC", conf=0.95) -> Dict:
    pr = portfolio_returns_series(w, returns)
    try:
        bm = yf.download(benchmark, period="5y",
                         auto_adjust=True, progress=False)["Close"].squeeze()
        bm = bm.pct_change().dropna()
        common = pr.index.intersection(bm.index)
        rb = rolling_beta(pr.loc[common], bm.loc[common])
    except Exception:
        rb = pd.Series(dtype=float)

    dd = drawdown_series(pr)
    ann_ret = pr.mean() * 252
    ann_vol = pr.std()  * np.sqrt(252)
    return {
        "daily_returns": pr,
        "hist_var":      historical_var(pr, conf),
        "hist_cvar":     historical_cvar(pr, conf),
        "param_var":     parametric_var(pr, conf),
        "cf_var":        cornish_fisher_var(pr, conf),
        "mc_var":        monte_carlo_var(pr, conf),
        "drawdown":      dd,
        "rolling_beta":  rb,
        "ann_return":    ann_ret,
        "ann_vol":       ann_vol,
        "sharpe":        (ann_ret - 0.065) / ann_vol,
        "sortino":       (ann_ret - 0.065) / (pr[pr<0].std()*np.sqrt(252)),
        "max_drawdown":  dd.min(),
        "calmar":        ann_ret / abs(dd.min() + 1e-9),
        "skewness":      float(skew(pr)),
        "kurtosis":      float(kurtosis(pr)),
    }


# ─── TRANSACTION COST REBALANCING ────────────────────────────────────────────

def rebalancing_backtest(
    returns: pd.DataFrame,
    target_weights: np.ndarray,
    rebal_freq: int = 21,        # days between rebalances
    transaction_cost: float = 0.001,  # 10 bps per trade
    initial_value: float = 1_000_000
) -> pd.DataFrame:
    """
    Simulates portfolio with periodic rebalancing and transaction costs.
    Returns daily portfolio value series.
    """
    n_days, n_assets = returns.shape
    values   = [initial_value]
    weights  = target_weights.copy()
    costs_paid = [0.0]
    turnover_list = [0.0]

    for t in range(1, n_days):
        # Drift weights with today's returns
        ret_t  = returns.iloc[t].values
        drifted = weights * (1 + ret_t)
        port_val = values[-1] * (1 + (weights * ret_t).sum())

        # Normalize drifted weights
        drifted_w = drifted / drifted.sum()

        # Rebalance?
        if t % rebal_freq == 0:
            turnover = np.abs(drifted_w - target_weights).sum() / 2
            cost     = port_val * turnover * transaction_cost
            port_val -= cost
            weights   = target_weights.copy()
            costs_paid.append(cost)
            turnover_list.append(turnover)
        else:
            weights = drifted_w
            costs_paid.append(0.0)
            turnover_list.append(0.0)

        values.append(port_val)

    df = pd.DataFrame({
        "portfolio_value": values,
        "transaction_cost": costs_paid,
        "turnover": turnover_list,
    }, index=[returns.index[0]] + list(returns.index[1:]))
    df["daily_return"] = df["portfolio_value"].pct_change()
    df["cum_costs"]    = df["transaction_cost"].cumsum()
    return df


# ─── FACTOR MODEL ────────────────────────────────────────────────────────────

def compute_factor_model(
    returns: pd.DataFrame,
    rf: float = 0.065 / 252
) -> pd.DataFrame:
    """
    3-Factor model (market, size proxy, value proxy) via OLS regression.
    Uses SPY as market, IWM-SPY spread as size, IVE-IVW spread as value.
    Returns per-asset: Alpha, Beta_market, Beta_size, Beta_value, R²
    """
    try:
        factor_tickers = ["SPY", "IWM", "IVE", "IVW"]
        fprices = yf.download(factor_tickers, period="5y",
                              auto_adjust=True, progress=False)["Close"]
        fret = fprices.pct_change().dropna()

        mkt   = fret["SPY"] - rf
        size  = fret["IWM"] - fret["SPY"]   # small minus large proxy
        value = fret["IVE"] - fret["IVW"]   # value minus growth proxy

        common = returns.index.intersection(mkt.index)
        X = np.column_stack([
            np.ones(len(common)),
            mkt.loc[common].values,
            size.loc[common].values,
            value.loc[common].values,
        ])

        results = []
        for ticker in returns.columns:
            y = returns.loc[common, ticker].values - rf
            beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ss_res = ((y - y_hat)**2).sum()
            ss_tot = ((y - y.mean())**2).sum()
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            results.append({
                "Ticker":      ticker,
                "Alpha (Ann.)": beta[0] * 252,
                "Beta (Mkt)":  beta[1],
                "Beta (Size)": beta[2],
                "Beta (Value)":beta[3],
                "R²":          r2,
            })
        return pd.DataFrame(results).set_index("Ticker")
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# ─── STRESS TESTING ──────────────────────────────────────────────────────────

CRISIS_PERIODS = {
    "GFC 2008-09":         ("2008-09-01", "2009-03-31"),
    "Euro Debt 2011":      ("2011-07-01", "2011-10-31"),
    "China Crash 2015":    ("2015-06-01", "2015-09-30"),
    "COVID Crash 2020":    ("2020-02-01", "2020-04-30"),
    "Rate Shock 2022":     ("2022-01-01", "2022-12-31"),
    "SVB Crisis 2023":     ("2023-03-01", "2023-05-31"),
}


def stress_test(
    weights: np.ndarray,
    returns: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """
    For each crisis period, compute portfolio return, max drawdown,
    worst day, and recovery time (days to new high).
    """
    pr = portfolio_returns_series(weights, returns)
    rows = []

    for crisis, (start, end) in CRISIS_PERIODS.items():
        try:
            mask = (pr.index >= start) & (pr.index <= end)
            if mask.sum() < 5:
                continue
            crisis_pr = pr[mask]
            total_ret  = (1 + crisis_pr).prod() - 1
            worst_day  = crisis_pr.min()
            vol        = crisis_pr.std() * np.sqrt(252)

            # Max drawdown in crisis
            cum  = (1 + crisis_pr).cumprod()
            mdd  = ((cum - cum.cummax()) / cum.cummax()).min()

            # Recovery: days from end of crisis to new high (in full series)
            post = pr[pr.index > end]
            cum_full = (1 + pr).cumprod()
            pre_peak = cum_full[pr.index <= end].max()
            recovery_days = None
            for i, (dt, val) in enumerate(cum_full[pr.index > end].items()):
                if val >= pre_peak:
                    recovery_days = i
                    break

            rows.append({
                "Crisis":          crisis,
                "Period Return":   f"{total_ret*100:.1f}%",
                "Max Drawdown":    f"{mdd*100:.1f}%",
                "Worst Day":       f"{worst_day*100:.2f}%",
                "Ann. Vol":        f"{vol*100:.1f}%",
                "Recovery (days)": str(recovery_days) if recovery_days is not None else "N/A",
            })
        except Exception:
            continue

    return pd.DataFrame(rows).set_index("Crisis") if rows else pd.DataFrame()


def scenario_shock(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: List[str],
    scenarios: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Apply custom return shocks to assets and compute portfolio impact.
    scenarios: dict of {name: {ticker: shock_pct}}
    """
    if scenarios is None:
        scenarios = {
            "Equity -20% (crash)":  {t: -0.20 for t in tickers if t not in ["TLT","GLD"]},
            "Rates +200bps":        {"TLT": -0.15} if "TLT" in tickers else {},
            "Gold +15% (safe haven)":{"GLD": 0.15} if "GLD" in tickers else {},
            "Tech selloff -30%":    {t: -0.30 for t in tickers if t in ["AAPL","MSFT","GOOGL","AMZN","NVDA","META"]},
            "Dollar rally":         {t: -0.05 for t in tickers},
        }

    rows = []
    base_ret = float(weights @ mu)

    for name, shocks in scenarios.items():
        shocked_mu = mu.copy()
        for ticker, shock in shocks.items():
            if ticker in tickers:
                idx = tickers.index(ticker)
                shocked_mu[idx] += shock
        shocked_ret = float(weights @ shocked_mu)
        impact = shocked_ret - base_ret
        rows.append({
            "Scenario":       name,
            "Base Return":    f"{base_ret*100:.2f}%",
            "Shocked Return": f"{shocked_ret*100:.2f}%",
            "Impact":         f"{impact*100:.2f}%",
            "Direction":      "🔴 Loss" if impact < 0 else "🟢 Gain",
        })

    return pd.DataFrame(rows).set_index("Scenario")
