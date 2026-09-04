"""
QuantPort v3 — Portfolio Risk & Optimization Dashboard
Abhishek Kumar Ojha | IIT Kharagpur | 22CY23003
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm as spnorm, skew, kurtosis

from analytics import (
    fetch_prices, compute_returns, annualized_stats,
    portfolio_performance, max_sharpe_portfolio,
    min_volatility_portfolio, max_sortino_portfolio,
    efficient_frontier, random_portfolios,
    portfolio_returns_series, compute_all_risk,
    historical_var, historical_cvar,
    cornish_fisher_var, monte_carlo_var,
    drawdown_series, rebalancing_backtest,
    compute_factor_model, stress_test, scenario_shock,
    CRISIS_PERIODS,
)
from backtest import full_backtest_report
from overfitting import full_overfitting_report

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantPort", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.section-header {
    color:#7c3aed;font-size:18px;font-weight:700;
    margin:20px 0 10px 0;border-bottom:2px solid #7c3aed;padding-bottom:4px;
}
.kpi-box {
    background:#1e1e2e;border-radius:8px;padding:14px 18px;
    margin:4px 0;border-left:4px solid #7c3aed;
}
</style>
""", unsafe_allow_html=True)

DARK = "#0f0f1a"

def dfig(rows=1, cols=1, **kw):
    fig, ax = plt.subplots(rows, cols, **kw)
    fig.patch.set_facecolor(DARK)
    axes = ax.flat if hasattr(ax, "flat") else [ax]
    for a in axes:
        a.set_facecolor(DARK)
        a.tick_params(colors="white")
        a.spines[:].set_color("#333355")
    return fig, ax

TAB_COLORS = plt.cm.tab20.colors

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ QuantPort v3")
    st.caption("Portfolio Risk & Optimization")
    st.divider()

    DEFAULT = "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, GS, JNJ, XOM, TLT, GLD"
    ticker_input = st.text_area("📋 Tickers", value=DEFAULT,
        help="Yahoo Finance symbols. NSE: RELIANCE.NS, TCS.NS")
    period     = st.selectbox("📅 Period", ["2y","3y","5y"], index=2)
    rf_rate    = st.slider("🏦 Risk-Free Rate (%)", 0.0, 10.0, 6.5, 0.5) / 100
    confidence = st.slider("📊 VaR Confidence (%)", 90, 99, 95) / 100
    n_sim      = st.selectbox("🎲 MC Portfolios", [1000,3000,5000], index=1)
    benchmark  = st.selectbox("📌 Benchmark", ["^GSPC","^NSEI","^DJI","^IXIC"])
    tcost      = st.slider("💸 Transaction Cost (bps)", 0, 50, 10) / 10000
    rebal_freq = st.selectbox("🔄 Rebalance Frequency",
                               ["Monthly (21d)","Quarterly (63d)","Annual (252d)"],
                               index=0)
    rebal_days = {"Monthly (21d)":21,"Quarterly (63d)":63,"Annual (252d)":252}[rebal_freq]
    st.divider()
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    if run_btn:
        st.session_state["analysis_run"] = True
    if st.button("🔄 Reset", key="reset_btn"):
        st.session_state["analysis_run"] = False
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 QuantPort — Portfolio Risk & Optimization")
st.caption("Markowitz · Sortino · VaR/CVaR · Basel III Backtest · Factor Model · Stress Testing · Transaction Costs")

if not st.session_state.get("analysis_run", False):
    st.info("👈 Configure your portfolio in the sidebar and click **Run Analysis**.")
    st.markdown("""
| Tab | Methods |
|---|---|
| 🎯 Optimization | Efficient Frontier, Max-Sharpe, Min-Vol, Max-Sortino, Monte Carlo (5000 portfolios) |
| ⚠️ Risk Analytics | Hist/Param/Cornish-Fisher/MC VaR & CVaR, Return distribution, MC paths |
| 📉 Drawdown | Underwater equity curves, Calmar ratio, Cumulative return comparison |
| 🔄 Rolling Metrics | Rolling Beta vs benchmark, Sharpe, Volatility (configurable window) |
| 💸 Rebalancing | Transaction cost simulation, Turnover analysis, Cost-adjusted returns |
| 🧪 VaR Backtest | Walk-forward validation, Kupiec POF, Christoffersen IT, Joint LR (Basel III) |
| 🔥 Stress Test | GFC/COVID/Rate shock scenarios, Factor model (3-factor), Custom shocks |
""")
    st.stop()

# ── Load & Compute ────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if len(tickers) < 2:
    st.error("Enter at least 2 tickers."); st.stop()

@st.cache_data(show_spinner=False)
def load_data(tickers_tuple, period, rf_rate):
    from analytics import fetch_prices, compute_returns, annualized_stats
    prices  = fetch_prices(list(tickers_tuple), period=period)
    tickers_clean = [c for c in tickers_tuple if c in prices.columns]
    prices  = prices[list(tickers_clean)]
    returns = compute_returns(prices)
    stats   = annualized_stats(returns, rf=rf_rate)
    return prices, returns, stats, tickers_clean

@st.cache_data(show_spinner=False)
def run_optimization(tickers_tuple, period, rf_rate, n_sim):
    from analytics import (fetch_prices, compute_returns,
                           max_sharpe_portfolio, min_volatility_portfolio,
                           max_sortino_portfolio, efficient_frontier, random_portfolios)
    import numpy as np
    prices  = fetch_prices(list(tickers_tuple), period=period)
    tickers_clean = [c for c in tickers_tuple if c in prices.columns]
    prices  = prices[list(tickers_clean)]
    returns = compute_returns(prices)
    mu  = (returns.mean() * 252).values
    cov = (returns.cov()  * 252).values
    w_ms  = max_sharpe_portfolio(mu, cov, rf=rf_rate)
    w_mv  = min_volatility_portfolio(mu, cov)
    w_so  = max_sortino_portfolio(returns, rf=rf_rate)
    w_eq  = np.ones(len(tickers_clean)) / len(tickers_clean)
    ef_df = efficient_frontier(mu, cov)
    mc_df = random_portfolios(mu, cov, n=n_sim, rf=rf_rate)
    return w_ms, w_mv, w_so, w_eq, ef_df, mc_df, mu, cov

with st.spinner("📡 Fetching market data..."):
    try:
        prices, returns, stats, tickers = load_data(
            tuple(tickers), period, rf_rate
        )
    except Exception as e:
        st.error(f"Data error: {e}"); st.stop()

if prices.empty or len(tickers) < 2:
    st.error("Not enough data. Check tickers."); st.stop()

mu  = (returns.mean() * 252).values
cov = (returns.cov()  * 252).values

with st.spinner("⚙️ Optimizing portfolios..."):
    w_ms, w_mv, w_so, w_eq, ef_df, mc_df, mu, cov = run_optimization(
        tuple(tickers), period, rf_rate, n_sim
    )

PORT_LABELS  = ["Max-Sharpe","Min-Vol","Max-Sortino","Equal-Weight"]
PORT_WEIGHTS = [w_ms, w_mv, w_so, w_eq]
PORT_COLORS  = ["#f59e0b","#10b981","#a78bfa","#60a5fa"]

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "🎯 Optimization","⚠️ Risk Analytics","📉 Drawdown",
    "🔄 Rolling Metrics","💸 Rebalancing","🧪 VaR Backtest",
    "🔥 Stress Test","🎰 Overfitting"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — OPTIMIZATION
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Efficient Frontier & Portfolio Optimization</div>',
                unsafe_allow_html=True)

    perfs = [portfolio_performance(w, mu, cov, rf_rate) for w in PORT_WEIGHTS]
    cols  = st.columns(4)
    for col, (r,v,s), lbl in zip(cols, perfs, PORT_LABELS):
        with col:
            st.markdown(f"**{lbl}**")
            st.metric("Return",     f"{r*100:.2f}%")
            st.metric("Volatility", f"{v*100:.2f}%")
            st.metric("Sharpe",     f"{s:.3f}")
    st.divider()

    # Efficient frontier
    fig, ax = dfig(figsize=(11,5.5))
    sc = ax.scatter(mc_df["Volatility"]*100, mc_df["Return"]*100,
                    c=mc_df["Sharpe"], cmap="plasma", alpha=0.35, s=5)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Sharpe Ratio", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    if not ef_df.empty:
        ax.plot(ef_df["Volatility"]*100, ef_df["Return"]*100,
                "w-", lw=2.5, label="Efficient Frontier", zorder=5)

    markers = ["*","D","^","o"]
    sizes   = [250,180,160,140]
    for (r,v,s), lbl, col, mk, sz in zip(perfs, PORT_LABELS, PORT_COLORS, markers, sizes):
        ax.scatter(v*100, r*100, s=sz, c=col, marker=mk, zorder=10,
                   label=f"{lbl} (SR={s:.2f})")

    for i,t in enumerate(tickers):
        ax.scatter(np.sqrt(cov[i,i])*100, mu[i]*100,
                   s=60, c="#ef4444", zorder=8)
        ax.annotate(t, (np.sqrt(cov[i,i])*100, mu[i]*100),
                    xytext=(5,3), textcoords="offset points",
                    color="white", fontsize=7)

    ax.set_xlabel("Annualized Volatility (%)", color="white")
    ax.set_ylabel("Annualized Return (%)", color="white")
    ax.set_title("Markowitz Efficient Frontier", color="white",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()

    st.divider()
    wdf = pd.DataFrame({
        lbl+"%": (w*100).round(2)
        for lbl,w in zip(PORT_LABELS, PORT_WEIGHTS)
    }, index=tickers)
    st.dataframe(wdf.style.background_gradient(cmap="YlOrRd"), width="stretch")

    fig2, axes2 = plt.subplots(1,4, figsize=(14,3.5))
    fig2.patch.set_facecolor(DARK)
    for ax2, w, lbl, col in zip(axes2, PORT_WEIGHTS, PORT_LABELS, PORT_COLORS):
        ax2.set_facecolor(DARK)
        mask = w > 0.005
        ax2.pie(w[mask], labels=np.array(tickers)[mask],
                autopct="%1.0f%%", colors=TAB_COLORS[:mask.sum()],
                textprops={"color":"white","fontsize":7})
        ax2.set_title(lbl, color="white", fontsize=9, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig2); plt.close()


# ════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYTICS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Value-at-Risk & Expected Shortfall</div>',
                unsafe_allow_html=True)

    pc = st.radio("Portfolio", PORT_LABELS, horizontal=True, key="risk_port")
    w_sel = PORT_WEIGHTS[PORT_LABELS.index(pc)]

    with st.spinner("Computing risk..."):
        risk = compute_all_risk(w_sel, returns, benchmark, confidence)
    pr = risk["daily_returns"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Hist VaR",        f"{risk['hist_var']*100:.3f}%")
    c2.metric("Hist CVaR",       f"{risk['hist_cvar']*100:.3f}%")
    c3.metric("Parametric VaR",  f"{risk['param_var']*100:.3f}%")
    c4.metric("Cornish-Fisher",  f"{risk['cf_var']*100:.3f}%")
    c5.metric("MC VaR",          f"{risk['mc_var']*100:.3f}%")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ann. Return",   f"{risk['ann_return']*100:.2f}%")
    c2.metric("Ann. Vol",      f"{risk['ann_vol']*100:.2f}%")
    c3.metric("Sharpe",        f"{risk['sharpe']:.3f}")
    c4.metric("Sortino",       f"{risk['sortino']:.3f}")
    c5.metric("Calmar",        f"{risk['calmar']:.3f}")

    c1,c2 = st.columns(2)
    c1.metric("Skewness", f"{risk['skewness']:.3f}",
              help="Negative = left tail (bad); positive = right tail")
    c2.metric("Excess Kurtosis", f"{risk['kurtosis']:.3f}",
              help=">0 means fat tails → VaR underestimates real risk")

    st.divider()

    # Return distribution
    fig, ax = dfig(figsize=(10, 4.5))
    ax.hist(pr*100, bins=80, color="#7c3aed", alpha=0.7, density=True)
    x = np.linspace(pr.min()*100, pr.max()*100, 300)
    ax.plot(x, spnorm.pdf(x, pr.mean()*100, pr.std()*100),
            "w--", lw=1.5, label="Normal fit")

    var_lines = [
        (risk["hist_var"]*100,  f"Hist VaR {confidence*100:.0f}%", "#ef4444"),
        (risk["hist_cvar"]*100, f"CVaR",                           "#f97316"),
        (risk["param_var"]*100, "Param VaR",                       "#facc15"),
        (risk["cf_var"]*100,    "Cornish-Fisher VaR",              "#a78bfa"),
    ]
    for val, lbl, col in var_lines:
        ax.axvline(val, color=col, lw=1.8, linestyle="--",
                   label=f"{lbl}: {val:.3f}%")

    ax.set_xlabel("Daily Return (%)", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("Return Distribution with Risk Thresholds",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8)
    ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()

    # MC paths
    st.markdown("#### Monte Carlo Paths — 30-Day Horizon")
    rng   = np.random.default_rng(7)
    paths = np.column_stack([
        np.cumprod(1 + rng.normal(pr.mean(), pr.std(), 30)) - 1
        for _ in range(400)
    ])
    fig, ax = dfig(figsize=(10, 4))
    ax.plot(paths*100, alpha=0.10, color="#7c3aed", lw=0.7)
    for pct, col, lbl in [(5,"#ef4444","5th pct"),(50,"white","Median"),(95,"#10b981","95th pct")]:
        ax.plot(np.percentile(paths*100, pct, axis=1), color=col, lw=2, label=lbl)
    ax.set_xlabel("Days", color="white"); ax.set_ylabel("Cum. Return (%)", color="white")
    ax.set_title("Monte Carlo Simulation (GBM)", color="white",
                 fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1e1e2e", labelcolor="white"); ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════
# TAB 3 — DRAWDOWN
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Drawdown & Underwater Equity Curves</div>',
                unsafe_allow_html=True)

    fig, axes = dfig(4, 1, figsize=(11, 12), sharex=True)
    for ax, w, lbl, col in zip(axes, PORT_WEIGHTS, PORT_LABELS, PORT_COLORS):
        pr_s = portfolio_returns_series(w, returns)
        dd   = drawdown_series(pr_s)
        ax.fill_between(dd.index, dd*100, 0, color=col, alpha=0.35)
        ax.plot(dd.index, dd*100, color=col, lw=1.2, label=lbl)
        ax.axhline(dd.min()*100, color="red", lw=1, linestyle="--",
                   label=f"Max DD: {dd.min()*100:.1f}%")
        ax.set_ylabel("Drawdown (%)", color="white", fontsize=9)
        ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
        ax.grid(True, alpha=0.12)
    axes[-1].set_xlabel("Date", color="white")
    fig.suptitle("Underwater Equity Curves", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Cumulative Returns")
    fig, ax = dfig(figsize=(11, 4))
    for w, lbl, col in zip(PORT_WEIGHTS, PORT_LABELS, PORT_COLORS):
        cum = (1 + portfolio_returns_series(w, returns)).cumprod()*100 - 100
        ax.plot(cum.index, cum, color=col, lw=1.8, label=lbl)
    ax.axhline(0, color="white", lw=0.6, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_title("Cumulative Returns Comparison", color="white",
                 fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1e1e2e", labelcolor="white"); ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════
# TAB 4 — ROLLING METRICS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Rolling Risk Metrics</div>',
                unsafe_allow_html=True)

    window = st.slider("Rolling Window (days)", 20, 120, 60, 10)
    risk_ms = compute_all_risk(w_ms, returns, benchmark, confidence)
    pr_s    = risk_ms["daily_returns"]
    rb      = risk_ms["rolling_beta"]

    fig, axes = dfig(3, 1, figsize=(11, 10), sharex=True)

    if not rb.empty:
        axes[0].plot(rb.index, rb, color="#f59e0b", lw=1.4, alpha=0.8,
                     label=f"Rolling Beta ({window}d)")
        axes[0].axhline(1.0, color="white", lw=0.8, linestyle="--", label="β=1")
        axes[0].axhline(0.0, color="gray",  lw=0.6, linestyle=":")
    axes[0].set_ylabel("Beta", color="white")
    axes[0].set_title("Rolling Beta vs Benchmark", color="white")
    axes[0].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[0].grid(True, alpha=0.12)

    roll_sr = ((pr_s.rolling(window).mean()*252 - rf_rate) /
               (pr_s.rolling(window).std()*np.sqrt(252)))
    axes[1].plot(roll_sr.index, roll_sr, color="#a78bfa", lw=1.4,
                 label=f"Rolling Sharpe ({window}d)")
    axes[1].axhline(0, color="red",   lw=0.8, linestyle="--")
    axes[1].axhline(1, color="green", lw=0.8, linestyle="--", label="SR=1")
    axes[1].set_ylabel("Sharpe Ratio", color="white")
    axes[1].set_title("Rolling Sharpe Ratio", color="white")
    axes[1].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[1].grid(True, alpha=0.12)

    roll_v = pr_s.rolling(window).std()*np.sqrt(252)*100
    axes[2].plot(roll_v.index, roll_v, color="#34d399", lw=1.4,
                 label=f"Rolling Vol ({window}d)")
    axes[2].fill_between(roll_v.index, roll_v, alpha=0.15, color="#34d399")
    axes[2].set_ylabel("Volatility (%)", color="white")
    axes[2].set_xlabel("Date", color="white")
    axes[2].set_title("Rolling Annualized Volatility", color="white")
    axes[2].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[2].grid(True, alpha=0.12)

    plt.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════
# TAB 5 — REBALANCING + TRANSACTION COSTS
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Transaction Cost Impact & Rebalancing Analysis</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    Simulating portfolio rebalancing with:
    - **Transaction cost:** {tcost*10000:.0f} bps per unit of turnover
    - **Rebalance frequency:** {rebal_freq}
    - **Initial value:** $1,000,000
    """)

    with st.spinner("Simulating rebalancing..."):
        rb_results = {}
        for lbl, w in zip(PORT_LABELS, PORT_WEIGHTS):
            rb_results[lbl] = rebalancing_backtest(
                returns, w,
                rebal_freq=rebal_days,
                transaction_cost=tcost,
                initial_value=1_000_000
            )

    # Summary table
    rows = []
    for lbl, df_rb in rb_results.items():
        final_val   = df_rb["portfolio_value"].iloc[-1]
        total_cost  = df_rb["cum_costs"].iloc[-1]
        total_ret   = (final_val / 1_000_000 - 1) * 100
        avg_turnover= df_rb["turnover"][df_rb["turnover"]>0].mean() * 100
        rows.append({
            "Portfolio":    lbl,
            "Final Value":  f"${final_val:,.0f}",
            "Total Return": f"{total_ret:.1f}%",
            "Total Costs":  f"${total_cost:,.0f}",
            "Cost Drag":    f"{total_cost/1e6*100:.2f}%",
            "Avg Turnover": f"{avg_turnover:.1f}%" if not np.isnan(avg_turnover) else "N/A",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Portfolio"), width="stretch")
    st.divider()

    # Portfolio value over time
    fig, axes = dfig(2, 1, figsize=(11, 8), sharex=True)

    for (lbl, df_rb), col in zip(rb_results.items(), PORT_COLORS):
        axes[0].plot(df_rb.index, df_rb["portfolio_value"]/1e6,
                     color=col, lw=1.6, label=lbl)
    axes[0].set_ylabel("Portfolio Value ($M)", color="white")
    axes[0].set_title("Portfolio Value Over Time (with Transaction Costs)",
                      color="white", fontsize=12, fontweight="bold")
    axes[0].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[0].grid(True, alpha=0.12)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.1f}M"))

    for (lbl, df_rb), col in zip(rb_results.items(), PORT_COLORS):
        axes[1].plot(df_rb.index, df_rb["cum_costs"]/1000,
                     color=col, lw=1.4, label=lbl)
    axes[1].set_ylabel("Cumulative Costs ($K)", color="white")
    axes[1].set_xlabel("Date", color="white")
    axes[1].set_title("Cumulative Transaction Costs", color="white",
                      fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[1].grid(True, alpha=0.12)

    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Cost sensitivity
    st.markdown("#### Cost Sensitivity Analysis")
    cost_range = [0, 5, 10, 20, 30, 50]
    sens_rows  = []
    for bps in cost_range:
        df_s = rebalancing_backtest(returns, w_ms, rebal_days, bps/10000, 1e6)
        final = df_s["portfolio_value"].iloc[-1]
        sens_rows.append({
            "Cost (bps)": bps,
            "Final Value": f"${final:,.0f}",
            "Return": f"{(final/1e6-1)*100:.1f}%",
            "Total Cost": f"${df_s['cum_costs'].iloc[-1]:,.0f}",
        })
    st.dataframe(pd.DataFrame(sens_rows).set_index("Cost (bps)"), width="stretch")


# ════════════════════════════════════════════════════════════════
# TAB 6 — VaR BACKTEST
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Walk-Forward VaR Backtest — Basel III Validation</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Walk-forward protocol:** VaR predicted daily using only past data.
    Next day's actual loss is compared to the prediction.
    Expected breach rate = **1 - confidence level** (e.g. 5% for 95% VaR).

    | Test | What it checks |
    |---|---|
    | **Kupiec POF** | Is breach *frequency* correct? (binomial LR, chi²(1)) |
    | **Christoffersen IT** | Are breaches *independent*? (no clustering, chi²(1)) |
    | **Joint LR** | Both simultaneously (chi²(2)) — Basel III standard |
    """)

    bt_port = st.radio("Portfolio", PORT_LABELS, horizontal=True, key="bt_p")
    train_w = st.slider("Training Window (days)", 120, 504, 252, 21)
    w_bt    = PORT_WEIGHTS[PORT_LABELS.index(bt_port)]

    with st.spinner("Running walk-forward backtest (4 methods)..."):
        pr_bt  = portfolio_returns_series(w_bt, returns)
        report = full_backtest_report(pr_bt, confidence, train_w)

    # Summary table
    st.markdown("### Summary")
    rows = []
    for method, res in report.items():
        rows.append({
            "Method":       method.replace("_"," ").title(),
            "Obs":          str(res["n_obs"]),
            "Breaches":     str(res["n_breaches"]),
            "Breach Rate":  f"{res['breach_rate']*100:.2f}%",
            "Expected":     f"{res['expected_rate']*100:.1f}%",
            "Basel":        res["basel_zone"],
            "Kupiec p":     str(res["kupiec"]["p_value"]),
            "Christof p":   str(res["christoffersen"]["p_value"]),
            "Joint p":      str(res["joint"]["p_value"]),
            "Calibrated":   "✅" if res["kupiec"]["calibrated"] else "❌",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Method"), width="stretch")
    st.info("p > 0.05 → fail to reject H₀ → model is well-calibrated ✅")

    sel = st.selectbox("Detail view", [m.replace("_"," ").title() for m in report.keys()])
    res = report[sel.replace(" ","_").lower()]
    df_bt = res["df"]

    # Actual vs VaR + cumulative breaches
    fig, axes = dfig(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(df_bt.index, df_bt["actual_return"]*100,
                 color="#60a5fa", lw=0.8, alpha=0.7, label="Daily Return")
    axes[0].plot(df_bt.index, df_bt["predicted_var"]*100,
                 color="#f59e0b", lw=1.5, linestyle="--", label="Predicted VaR")
    bd = df_bt[df_bt["breach"]==1]
    axes[0].scatter(bd.index, bd["actual_return"]*100,
                    color="#ef4444", s=20, zorder=5,
                    label=f"Breach (n={len(bd)})")
    axes[0].set_ylabel("Return (%)", color="white")
    axes[0].set_title(f"Walk-Forward VaR — {sel}", color="white",
                      fontsize=12, fontweight="bold")
    axes[0].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[0].grid(True, alpha=0.12)
    axes[0].axhline(0, color="white", lw=0.4, linestyle=":")

    cum_b = df_bt["breach"].cumsum()
    exp_l = [(i+1)*(1-confidence) for i in range(len(df_bt))]
    axes[1].plot(df_bt.index, cum_b,  color="#ef4444", lw=2, label="Actual breaches")
    axes[1].plot(df_bt.index, exp_l,  color="#10b981", lw=1.5, linestyle="--", label="Expected")
    axes[1].fill_between(df_bt.index, cum_b, exp_l,
                         where=cum_b>exp_l, alpha=0.15, color="#ef4444")
    axes[1].fill_between(df_bt.index, cum_b, exp_l,
                         where=cum_b<=exp_l, alpha=0.15, color="#10b981")
    axes[1].set_ylabel("Cumulative Breaches", color="white")
    axes[1].set_xlabel("Date", color="white")
    axes[1].set_title("Cumulative Breaches vs Expected", color="white")
    axes[1].legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    axes[1].grid(True, alpha=0.12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Monthly breach bar
    st.markdown("### Monthly Breach Clustering")
    df_bt2 = df_bt.copy(); df_bt2["month"] = df_bt2.index.to_period("M")
    monthly = df_bt2.groupby("month")["breach"].sum().reset_index()
    monthly["ms"] = monthly["month"].astype(str)
    fig, ax = dfig(figsize=(11, 2.8))
    bc = ["#ef4444" if c>0 else "#1e3a2e" for c in monthly["breach"]]
    ax.bar(range(len(monthly)), monthly["breach"].values, color=bc, width=0.7)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["ms"].values, rotation=45, ha="right",
                       fontsize=7, color="white")
    ax.set_ylabel("Breaches", color="white")
    ax.set_title("Monthly Breach Count", color="white",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.1, axis="y"); st.pyplot(fig); plt.close()

    # Test detail columns
    st.markdown("### Statistical Test Details")
    kup = res["kupiec"]; chr_ = res["christoffersen"]; jnt = res["joint"]
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Kupiec POF Test** — Frequency")
        st.metric("p-value",      kup["p_value"])
        st.metric("LR statistic", kup["LR"])
        st.metric("Observed rate",f"{kup['p_hat']*100:.2f}%")
        (st.success if kup["calibrated"] else st.error)(
            "✅ Calibrated" if kup["calibrated"] else "❌ Miscalibrated")
    with c2:
        st.markdown("**Christoffersen IT** — Independence")
        st.metric("p-value",                    chr_["p_value"])
        st.metric("P(breach|breach yesterday)", chr_["p11"])
        st.metric("P(breach|no breach)",        chr_["p01"])
        (st.warning if chr_["clustered"] else st.success)(
            "⚠️ Clustering detected" if chr_["clustered"] else "✅ No clustering")
    with c3:
        st.markdown("**Joint LR Test** — Basel III")
        st.metric("p-value",          jnt["p_value"])
        st.metric("LR stat (χ²(2))", jnt["LR"])
        (st.success if jnt["calibrated"] else st.error)(
            "✅ Passes Basel test" if jnt["calibrated"] else "❌ Fails Basel test")


# ════════════════════════════════════════════════════════════════
# TAB 7 — STRESS TEST + FACTOR MODEL
# ════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">Stress Testing & Factor Analysis</div>',
                unsafe_allow_html=True)

    st_port = st.radio("Portfolio", PORT_LABELS, horizontal=True, key="st_p")
    w_st    = PORT_WEIGHTS[PORT_LABELS.index(st_port)]

    # Historical stress test
    st.markdown("### 📅 Historical Crisis Performance")
    with st.spinner("Running stress scenarios..."):
        stress_df = stress_test(w_st, returns, tickers)

    if not stress_df.empty:
        st.dataframe(stress_df, width="stretch")

        # Bar chart of crisis returns
        crisis_rets = []
        for crisis,(start,end) in CRISIS_PERIODS.items():
            pr_full = portfolio_returns_series(w_st, returns)
            mask    = (pr_full.index>=start)&(pr_full.index<=end)
            if mask.sum() > 0:
                crisis_rets.append((crisis, (1+pr_full[mask]).prod()-1))

        if crisis_rets:
            labels_c, vals_c = zip(*crisis_rets)
            colors_c = ["#ef4444" if v<0 else "#10b981" for v in vals_c]
            fig, ax = dfig(figsize=(10, 3.5))
            ax.barh(range(len(labels_c)), [v*100 for v in vals_c],
                    color=colors_c, alpha=0.8)
            ax.set_yticks(range(len(labels_c)))
            ax.set_yticklabels(labels_c, color="white", fontsize=9)
            ax.set_xlabel("Total Return (%)", color="white")
            ax.axvline(0, color="white", lw=0.8)
            ax.set_title("Portfolio Return During Historical Crises",
                         color="white", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.12, axis="x")
            st.pyplot(fig); plt.close()
    else:
        st.info("Not enough historical data for crisis periods with this date range.")

    st.divider()

    # Scenario shocks
    st.markdown("### ⚡ Custom Scenario Analysis")
    shock_df = scenario_shock(w_st, mu, cov, tickers)
    if not shock_df.empty:
        st.dataframe(shock_df, width="stretch")

    st.divider()

    # Factor model
    st.markdown("### 📐 3-Factor Model (Fama-French Style)")
    st.caption("Alpha, Market Beta, Size (SMB proxy), Value (HML proxy) via OLS regression")
    with st.spinner("Running factor regressions..."):
        factor_df = compute_factor_model(returns, rf=rf_rate/252)

    if "Error" not in factor_df.columns:
        st.dataframe(
            factor_df.style.format("{:.4f}")
                     .background_gradient(cmap="RdYlGn",   subset=["Alpha (Ann.)"])
                     .background_gradient(cmap="RdYlGn_r", subset=["Beta (Mkt)"]),
            width="stretch"
        )

        # Alpha bar chart
        fig, ax = dfig(figsize=(10, 3.5))
        alphas = factor_df["Alpha (Ann.)"].values
        cols_a = ["#10b981" if a>0 else "#ef4444" for a in alphas]
        ax.bar(factor_df.index, alphas*100, color=cols_a, alpha=0.8)
        ax.axhline(0, color="white", lw=0.8)
        ax.set_ylabel("Alpha (Ann. %)", color="white")
        ax.set_xticklabels(factor_df.index, rotation=45, ha="right", color="white")
        ax.set_title("Annualized Alpha per Asset (3-Factor Model)",
                     color="white", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.12, axis="y")
        st.pyplot(fig); plt.close()
    else:
        st.warning("Factor data unavailable (network issue). Factor model requires SPY, IWM, IVE, IVW data.")

    st.divider()

    # Asset Analysis (correlation + normalized prices)
    st.markdown("### 📊 Correlation Matrix")
    corr = returns.corr()
    fig, ax = dfig(figsize=(max(6,len(tickers)), max(5,len(tickers)-1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size":8}, cbar_kws={"shrink":0.8})
    ax.set_title("Return Correlation Matrix", color="white",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    st.pyplot(fig); plt.close()

    st.markdown("### 📈 Normalized Price Performance (Base=100)")
    fig, ax = dfig(figsize=(11, 4))
    norm_p = prices/prices.iloc[0]*100
    for i, col in enumerate(norm_p.columns):
        ax.plot(norm_p.index, norm_p[col], lw=1.5, label=col,
                color=TAB_COLORS[i%20])
    ax.axhline(100, color="white", lw=0.6, linestyle="--")
    ax.set_ylabel("Indexed Price", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_title("Normalized Price Performance", color="white",
                 fontsize=11, fontweight="bold")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()

# ── Footer ───────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════
# TAB 8 — BACKTEST OVERFITTING & DEFLATED SHARPE RATIO
# ════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-header">Backtest Overfitting & Deflated Sharpe Ratio</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **The core problem:** If you test 1,000 strategies on the same data and keep the best,
    that best Sharpe is almost entirely luck — not skill. The more strategies you test,
    the more impressive the winner looks, even with purely random rules.

    **Bailey & López de Prado (2014)** formalize this with three tools:

    | Tool | What it answers |
    |---|---|
    | **Overfitting Simulation** | What Sharpe can random strategies achieve on this data? |
    | **Deflated Sharpe Ratio (DSR)** | Given N trials, what's the probability *your* Sharpe is real? |
    | **Min Track Record Length** | How many days of live trading before you can trust the result? |
    """)

    ov_port = st.radio("Portfolio to analyse",
                       PORT_LABELS, horizontal=True, key="ov_port")
    n_strat = st.select_slider("Number of random strategies",
                                options=[100, 250, 500, 1000], value=500)
    bench_sr = st.slider("Benchmark Sharpe (SR₀)", 0.0, 2.0, 0.0, 0.1,
                          help="SR you want to beat. 0 = 'better than nothing'")
    target_psr = st.slider("Target PSR confidence", 0.80, 0.99, 0.95, 0.01)

    w_ov = PORT_WEIGHTS[PORT_LABELS.index(ov_port)]

    with st.spinner(f"Running {n_strat} random strategies + Deflated Sharpe..."):
        pr_ov  = portfolio_returns_series(w_ov, returns)
        result = full_overfitting_report(
            prices, pr_ov,
            rf=rf_rate,
            n_strategies=n_strat,
            benchmark_sr=bench_sr,
            target_psr=target_psr,
        )

    dsr = result["dsr"]

    # ── Top KPIs ──────────────────────────────────────────────
    st.divider()
    st.markdown("### Key Results")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Portfolio Sharpe",    result["port_sr"],
              help="Annualized Sharpe of selected portfolio")
    c2.metric("Best Random Sharpe",  result["best_random_sr"],
              help=f"Best SR from {n_strat} purely random strategies")
    c3.metric("Portfolio Percentile",f"{result['pct_rank']:.1f}%",
              help="Where portfolio SR ranks among random strategies")
    c4.metric("Deflated SR (prob)",  f"{dsr['dsr']*100:.1f}%",
              help="Probability this Sharpe is genuine after correcting for N trials")
    c5.metric("Min Track Record",    f"{result['mtrl_years']:.1f} yrs",
              help="Live trading needed before Sharpe is statistically trustworthy")

    # Verdict banner
    verdict = dsr["verdict"]
    if "✅" in verdict:
        st.success(f"**Verdict:** {verdict}")
    elif "🟡" in verdict:
        st.warning(f"**Verdict:** {verdict}")
    else:
        st.error(f"**Verdict:** {verdict}")

    st.caption(f"Expected max SR from {n_strat} random trials: **{dsr['e_max_sr']:.3f}** "
               f"| Portfolio SR: **{result['port_sr']}** "
               f"| PSR vs SR=0: **{result['psr_vs_zero']*100:.1f}%**")

    st.divider()

    # ── Plot 1: Sharpe distribution of random strategies ──────
    st.markdown("### 🎲 Sharpe Distribution of Random Strategies")
    st.caption("If your portfolio SR is buried in this distribution, your backtest is likely overfitted.")

    srs = result["strat_srs"]
    fig, ax = dfig(figsize=(11, 4.5))
    ax.hist(srs, bins=60, color="#7c3aed", alpha=0.7,
            edgecolor="none", density=True, label=f"{n_strat} random strategies")

    # Portfolio SR line
    ax.axvline(result["port_sr"], color="#f59e0b", lw=2.5, linestyle="-",
               label=f"Your portfolio SR = {result['port_sr']:.3f}")

    # Best random
    ax.axvline(result["best_random_sr"], color="#ef4444", lw=2, linestyle="--",
               label=f"Best random SR = {result['best_random_sr']:.3f}")

    # E[max SR]
    ax.axvline(dsr["e_max_sr"], color="#10b981", lw=1.8, linestyle=":",
               label=f"E[max SR | {n_strat} trials] = {dsr['e_max_sr']:.3f}")

    # Shade overfitting zone
    x_shade = srs[srs >= result["port_sr"]]
    if len(x_shade) > 0:
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 2],
                          result["port_sr"], srs.max(),
                          alpha=0.12, color="#ef4444", label="Strategies beating your SR")

    ax.set_xlabel("Annualized Sharpe Ratio", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title(
        f"Sharpe Distribution: {n_strat} Random Strategies vs Your Portfolio\n"
        f"Portfolio ranks at {result['pct_rank']:.1f}th percentile",
        color="white", fontsize=12, fontweight="bold"
    )
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()

    # ── Plot 2: DSR as function of N trials ───────────────────
    st.markdown("### 📉 How DSR Degrades with More Trials")
    st.caption("The more strategies you test, the less trustworthy any single 'best' result becomes.")

    trial_range = np.arange(1, min(n_strat + 1, 1001), 10)
    from scipy.stats import norm as spnorm2
    from overfitting import deflated_sharpe_ratio as dsr_fn

    dsr_curve = []
    for n_t in trial_range:
        fake_srs = [result["port_sr"]] * n_t  # same SR repeated
        d = dsr_fn(result["port_sr"], fake_srs,
                   result["n_obs"], result["port_skew"], result["port_kurt"])
        dsr_curve.append(d["dsr"])

    fig, ax = dfig(figsize=(11, 3.8))
    ax.plot(trial_range, [d*100 for d in dsr_curve],
            color="#a78bfa", lw=2, label="DSR (prob genuine)")
    ax.axhline(95, color="#10b981", lw=1.5, linestyle="--", label="95% threshold")
    ax.axhline(50, color="#ef4444", lw=1,   linestyle=":",  label="50% (coin flip)")
    ax.axvline(n_strat, color="#f59e0b", lw=1.5, linestyle="--",
               label=f"Your trial count = {n_strat}")
    ax.fill_between(trial_range, [d*100 for d in dsr_curve], 0,
                    alpha=0.12, color="#a78bfa")
    ax.set_xlabel("Number of Strategies Tested", color="white")
    ax.set_ylabel("DSR (%)", color="white")
    ax.set_title("Deflated Sharpe Ratio vs Number of Trials",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.13)
    ax.set_ylim(0, 105)
    st.pyplot(fig); plt.close()

    # ── Plot 3: Min Track Record vs SR ────────────────────────
    st.markdown("### ⏱️ Minimum Track Record Length")
    st.caption("Higher SR requires less live data. Lower SR needs years before it's statistically trustworthy.")

    from overfitting import minimum_track_record_length as mtrl_fn
    sr_range   = np.linspace(0.3, 3.0, 100)
    mtrl_curve = [
        mtrl_fn(sr, bench_sr, result["port_skew"],
                result["port_kurt"], target_psr) / 252
        for sr in sr_range
    ]

    fig, ax = dfig(figsize=(11, 3.8))
    ax.plot(sr_range, mtrl_curve, color="#34d399", lw=2,
            label=f"Min years needed (PSR ≥ {target_psr*100:.0f}%)")
    ax.axvline(result["port_sr"], color="#f59e0b", lw=2, linestyle="--",
               label=f"Your SR = {result['port_sr']:.3f}")
    ax.axhline(result["mtrl_years"], color="#ef4444", lw=1.5, linestyle=":",
               label=f"Your min track record = {result['mtrl_years']:.1f} yrs")
    ax.fill_between(sr_range, mtrl_curve, 0, alpha=0.1, color="#34d399")
    ax.set_xlabel("Sharpe Ratio", color="white")
    ax.set_ylabel("Min Track Record (years)", color="white")
    ax.set_title("Minimum Live Track Record Before Trusting Sharpe",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(max(mtrl_curve)*1.1, 20))
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.13)
    st.pyplot(fig); plt.close()

    # ── Summary table ─────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Full Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Portfolio Properties**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Sharpe Ratio", "Skewness", "Excess Kurtosis",
                       "Observations", "PSR vs SR=0"],
            "Value":  [str(result["port_sr"]), str(result["port_skew"]),
                       str(result["port_kurt"]), str(result["n_obs"]),
                       f"{result['psr_vs_zero']*100:.1f}%"]
        }).set_index("Metric"), width="stretch")

    with col2:
        st.markdown("**Deflated Sharpe Analysis**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Strategies Tested", "E[Max SR | N trials]",
                       "DSR (prob genuine)", "Verdict",
                       "Min Track Record"],
            "Value":  [str(dsr["n_trials"]), str(dsr["e_max_sr"]),
                       f"{dsr['dsr']*100:.1f}%", dsr["verdict"],
                       f"{result['mtrl_years']:.1f} years"]
        }).set_index("Metric"), width="stretch")

    # ── Interpretation box ────────────────────────────────────
    st.divider()
    st.markdown("### 💡 How to Interpret This")
    st.info(f"""
**Your portfolio SR = {result['port_sr']:.3f}**

When tested against {n_strat} random strategies on the same data:
- The best random strategy achieved SR = **{result['best_random_sr']:.3f}**
- Your portfolio ranks at the **{result['pct_rank']:.1f}th percentile** of random strategies
- After correcting for {n_strat} trials, DSR = **{dsr['dsr']*100:.1f}%** probability genuine

**{dsr['verdict']}**

To be 95% confident this SR is real with **live trading**, you need at least **{result['mtrl_years']:.1f} years** of out-of-sample data.

*Reference: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management*
    """)

st.divider()
st.caption("**QuantPort v3** | Abhishek Kumar Ojha | IIT Kharagpur | 22CY23003 "
           "| Python · scipy · yfinance · Streamlit | github.com/YOUR_USERNAME/quantport")
