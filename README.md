# QuantPort — Portfolio Risk & Optimization Dashboard

**Abhishek Kumar Ojha | IIT Kharagpur | 22CY23003**

An end-to-end quantitative finance dashboard implementing institutional-grade
portfolio optimization, multi-method VaR, Basel III backtesting, factor analysis,
stress testing, and transaction cost simulation on real market data.

---

## Features (7 Tabs)

| Tab | Methods |
|---|---|
| 🎯 **Optimization** | Markowitz efficient frontier, Max-Sharpe (SLSQP), Min-Volatility, Max-Sortino; 5,000 Monte Carlo portfolios |
| ⚠️ **Risk Analytics** | Historical VaR/CVaR, Parametric VaR, Cornish-Fisher VaR (non-normal correction), Monte Carlo VaR (GBM); Sharpe, Sortino, Calmar; skewness/kurtosis |
| 📉 **Drawdown** | Underwater equity curves (4 portfolios), max drawdown, cumulative return comparison |
| 🔄 **Rolling Metrics** | Rolling Beta vs benchmark, rolling Sharpe, rolling annualized volatility (configurable window) |
| 💸 **Rebalancing** | Transaction cost simulation, cost sensitivity analysis (0–50 bps), turnover tracking, cost-adjusted portfolio value |
| 🧪 **VaR Backtest** | Walk-forward validation; **Kupiec POF Test**, **Christoffersen Independence Test**, **Joint LR Test** (Basel III regulatory framework); 4 VaR methods compared |
| 🔥 **Stress Test** | Historical crisis performance (GFC/COVID/Rate Shock/SVB); custom scenario shocks; **3-Factor Model** (Fama-French style) alpha/beta decomposition |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/quantport.git
cd quantport
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` → configure tickers → click **Run Analysis**.

---

## Methods

### Portfolio Optimization
- **Mean-Variance (Markowitz 1952):** minimize `w'Σw` subject to `w'μ = target`, `Σw = 1`, `w ≥ 0`
- **Max-Sharpe:** maximize `(μ_p - r_f) / σ_p` via SLSQP (scipy)
- **Min-Volatility:** minimize `σ_p` with full-investment constraint
- **Max-Sortino:** maximize `(μ_p - r_f) / σ_downside` — penalizes only downside risk

### Risk Metrics
| Metric | Formula |
|---|---|
| Historical VaR | `Quantile(R, 1-α)` |
| Historical CVaR | `E[R | R ≤ VaR_α]` |
| Parametric VaR | `μ + σ·Φ⁻¹(1-α)` |
| Cornish-Fisher VaR | Modified z-score using skewness & kurtosis |
| Monte Carlo VaR | GBM simulation, 10K paths, 1-day horizon |
| Calmar Ratio | `Ann.Return / |Max Drawdown|` |
| Sortino Ratio | `(Ann.Return - r_f) / Downside Vol` |

### VaR Backtesting (Basel III)
Walk-forward protocol: predict VaR on day t using data from `[t-W, t-1]`, compare to `R_t`.

| Test | H₀ | Statistic |
|---|---|---|
| Kupiec POF | Breach rate = `1-α` | LR ~ χ²(1) |
| Christoffersen IT | Breaches are independent | LR ~ χ²(1) |
| Joint LR | Both simultaneously | LR ~ χ²(2) |

**Basel traffic light:** ≤4 breaches (250-day) = Green; 5-9 = Yellow; ≥10 = Red.

### Factor Model
3-factor OLS regression using SPY (market), IWM-SPY (size/SMB), IVE-IVW (value/HML) as proxies.
Returns per-asset: `Alpha (annualized)`, `β_market`, `β_size`, `β_value`, `R²`.

### Stress Testing
Historical crisis periods: GFC 2008-09, Euro Debt 2011, China Crash 2015,
COVID 2020, Rate Shock 2022, SVB Crisis 2023.
Reports: total return, max drawdown, worst day, annualized vol, recovery days.

### Transaction Costs
Simulates periodic rebalancing with configurable:
- Transaction cost (bps per unit of turnover)
- Rebalance frequency (monthly/quarterly/annual)
- Tracks cumulative costs, turnover, cost-adjusted portfolio value

---

## CV Bullet

```
Built QuantPort — institutional-grade quant finance dashboard on 11 global equities:
Markowitz efficient frontier, Max-Sharpe/Min-Vol/Max-Sortino optimization (SLSQP),
4-method VaR/CVaR suite (Historical, Parametric, Cornish-Fisher, Monte Carlo GBM),
walk-forward backtesting with Kupiec POF + Christoffersen IT + Joint LR tests (Basel III),
transaction cost simulation, Fama-French style 3-factor model, historical stress testing
across 6 crisis periods (GFC/COVID/Rate Shock) — deployed as 7-tab Streamlit app on real yfinance data
```

---

## File Structure

```
quantport/
├── analytics.py     # Core: optimization, risk, factor model, stress test
├── backtest.py      # Walk-forward VaR validation + Basel III tests
├── app.py           # Streamlit dashboard (7 tabs)
├── requirements.txt
└── README.md
```

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance.*
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives.*
- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review.*
- Fama, E. & French, K. (1993). Common Risk Factors in Returns on Stocks and Bonds. *Journal of Financial Economics.*
- Basel Committee on Banking Supervision (2016). Minimum Capital Requirements for Market Risk.
