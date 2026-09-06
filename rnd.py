"""
Breeden-Litzenberger Implied Risk-Neutral Distribution
=======================================================
Breeden & Litzenberger (1978), Journal of Business 51(4)

Core insight:
    The risk-neutral probability density function is the second derivative
    of the call price with respect to strike price:

    q(K) = e^(rT) * d²C/dK²

The catch: second derivatives amplify noise, so we must smooth in
implied-volatility space before differentiating — otherwise we get
negative probabilities.

Pipeline:
    1. Fetch real option chain (calls) from yfinance
    2. Compute implied volatility for each strike
    3. Fit smooth cubic spline to IV smile
    4. Reconstruct smooth call prices from smoothed IV
    5. Take second derivative numerically → risk-neutral density
    6. Normalize to ensure it integrates to 1
    7. Compare to Black-Scholes (Gaussian) assumption
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import brentq
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol(C_market: float, S: float, K: float, T: float, r: float,
                tol: float = 1e-6) -> Optional[float]:
    """
    Invert Black-Scholes to get implied volatility via Brent's method.
    Returns None if no solution found.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if C_market < intrinsic - tol:
        return None
    try:
        iv = brentq(
            lambda sigma: bs_call_price(S, K, T, r, sigma) - C_market,
            1e-6, 10.0, xtol=tol, maxiter=200
        )
        return float(iv)
    except Exception:
        return None


def bs_risk_neutral_density(S: float, T: float, r: float, sigma: float,
                             K_range: np.ndarray) -> np.ndarray:
    """
    Analytical risk-neutral density under Black-Scholes (log-normal).
    This is the GAUSSIAN assumption we compare against.
    """
    F = S * np.exp(r * T)  # forward price
    density = np.zeros_like(K_range, dtype=float)
    for i, K in enumerate(K_range):
        if K > 0:
            d2 = (np.log(F / K) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
            density[i] = norm.pdf(d2) / (K * sigma * np.sqrt(T))
    return density


# ─────────────────────────────────────────────────────────────────────────────
# OPTION CHAIN FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_option_chain(ticker: str = "SPY",
                       expiry_index: int = 2) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Fetch real call option chain from yfinance.
    Returns (calls_df, spot_price, risk_free_rate, time_to_expiry_years)
    """
    tk      = yf.Ticker(ticker)
    spot    = float(tk.fast_info["lastPrice"])
    exps    = tk.options

    if not exps:
        raise ValueError(f"No options available for {ticker}")

    # Pick expiry — default index 2 gives ~30-45 days out
    exp_idx = min(expiry_index, len(exps) - 1)
    expiry  = exps[exp_idx]

    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()

    # Parse time to expiry
    from datetime import datetime
    exp_date = datetime.strptime(expiry, "%Y-%m-%d")
    T = max((exp_date - datetime.now()).days / 365, 1/365)

    # Clean: keep calls with volume and valid bid/ask
    calls = calls[calls["impliedVolatility"] > 0.01]
    calls = calls[calls["impliedVolatility"] < 5.0]
    calls = calls[(calls["bid"] > 0) | (calls["lastPrice"] > 0)]

    # Use mid-price where available, else last price
    calls["mid"] = np.where(
        (calls["bid"] > 0) & (calls["ask"] > 0),
        (calls["bid"] + calls["ask"]) / 2,
        calls["lastPrice"]
    )
    calls = calls[calls["mid"] > 0.01].reset_index(drop=True)

    # Risk-free rate (approximate with 3-month T-bill proxy)
    r = 0.053  # ~5.3%, approximate Sep 2026 US rate

    return calls, spot, r, T, expiry


# ─────────────────────────────────────────────────────────────────────────────
# BREEDEN-LITZENBERGER CORE
# ─────────────────────────────────────────────────────────────────────────────

def extract_iv_smile(calls: pd.DataFrame, S: float, T: float, r: float,
                     moneyness_range: Tuple[float, float] = (0.75, 1.25)
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract implied volatility smile from option chain.
    Filters to reasonable moneyness range to avoid deep ITM/OTM noise.
    """
    strikes = calls["strike"].values
    prices  = calls["mid"].values

    ivs     = []
    valid_K = []

    for K, C in zip(strikes, prices):
        moneyness = K / S
        if not (moneyness_range[0] <= moneyness <= moneyness_range[1]):
            continue
        iv = implied_vol(C, S, K, T, r)
        if iv is not None and 0.02 <= iv <= 3.0:
            ivs.append(iv)
            valid_K.append(K)

    return np.array(valid_K), np.array(ivs)


def smooth_iv_smile(strikes: np.ndarray, ivs: np.ndarray,
                    n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit smooth spline to IV smile.
    Smoothing is CRITICAL — raw IV data is noisy, and second derivative
    of noisy data gives garbage (including negative probabilities).
    """
    if len(strikes) < 4:
        raise ValueError("Not enough strikes for smoothing")

    # Sort
    idx     = np.argsort(strikes)
    K_sort  = strikes[idx]
    iv_sort = ivs[idx]

    # Smoothing spline (s parameter controls smoothness)
    # Higher s = smoother = less noise in second derivative
    s_param = len(strikes) * 0.001
    spline  = UnivariateSpline(K_sort, iv_sort, k=4, s=s_param)

    K_fine  = np.linspace(K_sort[0], K_sort[-1], n_points)
    iv_fine = spline(K_fine)

    # Ensure positive IV
    iv_fine = np.maximum(iv_fine, 0.001)

    return K_fine, iv_fine


def breeden_litzenberger(S: float, T: float, r: float,
                          K_fine: np.ndarray, iv_fine: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract risk-neutral density via Breeden-Litzenberger (1978).

    q(K) = e^(rT) * d²C/dK²

    We compute call prices from smoothed IV, then take the
    numerical second derivative.

    Returns (strikes, risk_neutral_density)
    """
    # Reconstruct call prices from smoothed IV
    call_prices = np.array([
        bs_call_price(S, K, T, r, iv)
        for K, iv in zip(K_fine, iv_fine)
    ])

    # Numerical second derivative: d²C/dK²
    # Using central difference: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    dK      = K_fine[1] - K_fine[0]
    d2C_dK2 = np.gradient(np.gradient(call_prices, dK), dK)

    # Risk-neutral density: q(K) = e^(rT) * d²C/dK²
    rnd = np.exp(r * T) * d2C_dK2

    # Remove negative values (numerical artifacts at tails)
    rnd = np.maximum(rnd, 0)

    # Normalize so density integrates to 1
    area = np.trapezoid(rnd, K_fine)
    if area > 0:
        rnd = rnd / area

    return K_fine, rnd


def compute_rnd_stats(K: np.ndarray, rnd: np.ndarray, S: float) -> Dict:
    """
    Compute statistics of the risk-neutral distribution.
    """
    # Expected value (risk-neutral mean)
    mean = np.trapezoid(K * rnd, K)

    # Variance
    variance = np.trapezoid((K - mean)**2 * rnd, K)
    std       = np.sqrt(max(variance, 0))

    # Skewness (negative = left tail = market fear)
    skewness  = np.trapezoid(((K - mean)/std)**3 * rnd, K) if std > 0 else 0

    # Excess kurtosis (positive = fat tails)
    kurt      = np.trapezoid(((K - mean)/std)**4 * rnd, K) - 3 if std > 0 else 0

    # Tail probabilities
    # P(S_T < 0.9 * S) — probability of >10% drop
    mask_drop10 = K < 0.90 * S
    p_drop10    = np.trapezoid(rnd[mask_drop10], K[mask_drop10]) if mask_drop10.any() else 0

    # P(S_T < 0.8 * S) — probability of >20% drop (crash)
    mask_drop20 = K < 0.80 * S
    p_drop20    = np.trapezoid(rnd[mask_drop20], K[mask_drop20]) if mask_drop20.any() else 0

    # P(S_T > 1.1 * S) — probability of >10% rally
    mask_rally  = K > 1.10 * S
    p_rally10   = np.trapezoid(rnd[mask_rally], K[mask_rally]) if mask_rally.any() else 0

    return {
        "rnd_mean":       round(mean, 2),
        "rnd_std":        round(std, 2),
        "rnd_skewness":   round(float(skewness), 4),
        "rnd_kurtosis":   round(float(kurt), 4),
        "p_drop_10pct":   round(float(p_drop10) * 100, 2),
        "p_drop_20pct":   round(float(p_drop20) * 100, 2),
        "p_rally_10pct":  round(float(p_rally10) * 100, 2),
        "implied_vol_atm": None,  # filled later
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def full_rnd_pipeline(ticker: str = "SPY",
                       expiry_index: int = 2) -> Dict:
    """
    Complete Breeden-Litzenberger pipeline:
    1. Fetch option chain
    2. Extract IV smile
    3. Smooth IV
    4. Extract RND
    5. Compare to Black-Scholes
    6. Compute tail probabilities

    Returns dict with all results for dashboard display.
    """
    # Step 1: Fetch
    calls, S, r, T, expiry = fetch_option_chain(ticker, expiry_index)

    # Step 2: Extract IV smile
    K_raw, iv_raw = extract_iv_smile(calls, S, T, r)

    if len(K_raw) < 5:
        raise ValueError(f"Only {len(K_raw)} valid strikes found. Try a different expiry.")

    # ATM IV
    atm_idx = np.argmin(np.abs(K_raw - S))
    atm_iv  = float(iv_raw[atm_idx])

    # Step 3: Smooth
    K_smooth, iv_smooth = smooth_iv_smile(K_raw, iv_raw)

    # Step 4: Breeden-Litzenberger RND
    K_rnd, rnd = breeden_litzenberger(S, T, r, K_smooth, iv_smooth)

    # Step 5: Black-Scholes Gaussian comparison (using ATM IV)
    bs_density = bs_risk_neutral_density(S, T, r, atm_iv, K_rnd)
    area_bs    = np.trapezoid(bs_density, K_rnd)
    if area_bs > 0:
        bs_density = bs_density / area_bs

    # Step 6: Stats
    stats = compute_rnd_stats(K_rnd, rnd, S)
    stats["implied_vol_atm"] = round(atm_iv * 100, 2)

    # BS tail probabilities for comparison
    bs_stats = compute_rnd_stats(K_rnd, bs_density, S)

    return {
        # Raw data
        "ticker":       ticker,
        "spot":         S,
        "expiry":       expiry,
        "T":            T,
        "r":            r,
        "atm_iv":       atm_iv,

        # IV smile
        "K_raw":        K_raw,
        "iv_raw":       iv_raw,
        "K_smooth":     K_smooth,
        "iv_smooth":    iv_smooth,

        # Densities
        "K_rnd":        K_rnd,
        "rnd":          rnd,
        "bs_density":   bs_density,

        # Stats
        "rnd_stats":    stats,
        "bs_stats":     bs_stats,

        # Option chain
        "calls":        calls,
        "n_strikes":    len(K_raw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FALLBACK (when market is closed / no internet)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_rnd_pipeline(ticker: str = "SPY",
                            S: float = 500.0,
                            T: float = 30/365,
                            r: float = 0.053,
                            atm_vol: float = 0.18,
                            skew: float = -0.35,
                            kurt_adj: float = 0.15) -> Dict:
    """
    Generate synthetic option chain with realistic vol smile for demo.
    Used when market is closed or options data unavailable.
    Produces a realistic left-skewed distribution mimicking SPY options.
    """
    # Generate strikes
    K_range = np.linspace(S * 0.78, S * 1.22, 60)

    # Vol smile: negative skew (puts expensive) + slight smile
    moneyness   = np.log(K_range / S)
    iv_raw      = atm_vol + skew * moneyness + kurt_adj * moneyness**2
    iv_raw      = np.maximum(iv_raw, 0.05)

    # Smooth
    K_smooth, iv_smooth = smooth_iv_smile(K_range, iv_raw, n_points=300)

    # RND
    K_rnd, rnd  = breeden_litzenberger(S, T, r, K_smooth, iv_smooth)

    # BS comparison
    bs_density  = bs_risk_neutral_density(S, T, r, atm_vol, K_rnd)
    area_bs     = np.trapezoid(bs_density, K_rnd)
    if area_bs > 0:
        bs_density = bs_density / area_bs

    stats    = compute_rnd_stats(K_rnd, rnd, S)
    bs_stats = compute_rnd_stats(K_rnd, bs_density, S)
    stats["implied_vol_atm"] = round(atm_vol * 100, 2)

    return {
        "ticker":       ticker,
        "spot":         S,
        "expiry":       "Synthetic (30-day)",
        "T":            T,
        "r":            r,
        "atm_iv":       atm_vol,
        "K_raw":        K_range,
        "iv_raw":       iv_raw,
        "K_smooth":     K_smooth,
        "iv_smooth":    iv_smooth,
        "K_rnd":        K_rnd,
        "rnd":          rnd,
        "bs_density":   bs_density,
        "rnd_stats":    stats,
        "bs_stats":     bs_stats,
        "calls":        pd.DataFrame(),
        "n_strikes":    len(K_range),
        "is_synthetic": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing synthetic RND pipeline...")
    result = synthetic_rnd_pipeline("SPY", S=500, atm_vol=0.18, skew=-0.35)

    print(f"\nTicker      : {result['ticker']}")
    print(f"Spot        : ${result['spot']}")
    print(f"Expiry      : {result['expiry']}")
    print(f"ATM IV      : {result['atm_iv']*100:.1f}%")
    print(f"RND Mean    : ${result['rnd_stats']['rnd_mean']}")
    print(f"RND Skew    : {result['rnd_stats']['rnd_skewness']}")
    print(f"RND Kurt    : {result['rnd_stats']['rnd_kurtosis']}")
    print(f"P(drop>10%) : {result['rnd_stats']['p_drop_10pct']}%")
    print(f"P(drop>20%) : {result['rnd_stats']['p_drop_20pct']}%")
    print(f"P(rally>10%): {result['rnd_stats']['p_rally_10pct']}%")
    print(f"\nBS P(drop>10%): {result['bs_stats']['p_drop_10pct']}%")
    print(f"BS P(drop>20%): {result['bs_stats']['p_drop_20pct']}%")
    print(f"\nRND integrates to: {np.trapezoid(result['rnd'], result['K_rnd']):.4f}")
    print("\nALL OK ✓")