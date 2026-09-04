"""
Walk-Forward VaR Backtest with Basel III Statistical Tests
==========================================================
Kupiec POF | Christoffersen IT | Joint LR Test
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from typing import Dict
from analytics import (
    portfolio_returns_series,
    historical_var, parametric_var,
    monte_carlo_var, cornish_fisher_var
)


def walk_forward_var(
    pr: pd.Series,
    confidence: float = 0.95,
    train_window: int = 252,
    method: str = "historical"
) -> pd.DataFrame:
    results = []
    for t in range(train_window, len(pr)):
        train  = pr.iloc[t - train_window: t]
        actual = pr.iloc[t]
        if   method == "historical":    var = historical_var(train, confidence)
        elif method == "parametric":    var = parametric_var(train, confidence)
        elif method == "cornish_fisher":var = cornish_fisher_var(train, confidence)
        elif method == "montecarlo":    var = monte_carlo_var(train, confidence, n_sim=2000)
        else: raise ValueError(method)

        results.append({
            "date":          pr.index[t],
            "actual_return": actual,
            "predicted_var": var,
            "breach":        int(actual < var),
            "exceedance":    actual - var if actual < var else 0.0,
        })
    return pd.DataFrame(results).set_index("date")


def kupiec_pof(breaches: pd.Series, conf: float) -> Dict:
    T, N  = len(breaches), int(breaches.sum())
    p0    = 1 - conf
    p_hat = N / T if T > 0 else 0
    eps   = 1e-10
    ph    = np.clip(p_hat, eps, 1-eps)
    try:
        lr = -2*(N*np.log(p0/ph) + (T-N)*np.log((1-p0)/(1-ph)))
        pv = 1 - chi2.cdf(lr, 1)
    except Exception:
        lr, pv = np.nan, np.nan
    return {"T":T,"N":N,"p0":p0,"p_hat":round(p_hat,5),
            "LR":round(lr,4),"p_value":round(pv,4),
            "calibrated": pv >= 0.05 if not np.isnan(pv) else None}


def christoffersen_it(breaches: pd.Series) -> Dict:
    b = breaches.values
    n00 = ((b[:-1]==0)&(b[1:]==0)).sum()
    n01 = ((b[:-1]==0)&(b[1:]==1)).sum()
    n10 = ((b[:-1]==1)&(b[1:]==0)).sum()
    n11 = ((b[:-1]==1)&(b[1:]==1)).sum()
    eps = 1e-10
    p01 = n01/(n00+n01+eps)
    p11 = n11/(n10+n11+eps)
    p   = (n01+n11)/(n00+n01+n10+n11+eps)
    try:
        ll1 = (n00*np.log(1-p01+eps)+n01*np.log(p01+eps)+
               n10*np.log(1-p11+eps)+n11*np.log(p11+eps))
        ll0 = ((n00+n10)*np.log(1-p+eps)+(n01+n11)*np.log(p+eps))
        lr  = -2*(ll0-ll1)
        pv  = 1-chi2.cdf(lr,1)
    except Exception:
        lr, pv = np.nan, np.nan
    return {"p01":round(p01,5),"p11":round(p11,5),
            "LR":round(lr,4),"p_value":round(pv,4),
            "clustered": bool(p11>p01)}


def full_backtest_report(pr: pd.Series, confidence=0.95,
                          train_window=252) -> Dict:
    methods = ["historical","parametric","cornish_fisher","montecarlo"]
    report  = {}
    for m in methods:
        df  = walk_forward_var(pr, confidence, train_window, m)
        b   = df["breach"]
        pof = kupiec_pof(b, confidence)
        it  = christoffersen_it(b)
        try:
            lr_j = pof["LR"] + it["LR"]
            pv_j = 1 - chi2.cdf(lr_j, 2)
        except Exception:
            lr_j, pv_j = np.nan, np.nan

        N = int(b.sum())
        zone = ("🟢 Green" if N<=4 else "🟡 Yellow" if N<=9 else "🔴 Red")

        report[m] = {
            "df":           df,
            "n_obs":        len(df),
            "n_breaches":   N,
            "breach_rate":  round(N/len(df), 5),
            "expected_rate":1-confidence,
            "basel_zone":   zone,
            "kupiec":       pof,
            "christoffersen":it,
            "joint":        {"LR":round(lr_j,4),"p_value":round(pv_j,4),
                             "calibrated": pv_j>=0.05 if not np.isnan(pv_j) else None},
            "mean_exceedance": round(df.loc[df["breach"]==1,"exceedance"].mean(),6) if N>0 else 0.0,
        }
    return report
