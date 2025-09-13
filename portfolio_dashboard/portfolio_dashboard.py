import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta, datetime
import numpy as np
import io

# Optional PDF support
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# -------------------------- Page Setup --------------------------
st.set_page_config(
    page_title="Smart Portfolio Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- Styles --------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .risk-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .performance-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .sector-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        margin: 0.3rem 0;
        text-align: center;
        font-weight: bold;
    }
    .alert-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #ffd700;
    }
    .economic-tile {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #222;
        margin: 0.5rem 0;
        text-align: center;
        border: 2px solid #ff9500;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------- Export Helpers --------------------------
def create_csv_export(metrics, portfolio, portfolio_value, detailed_positions):
    summary_data = {
        'Metric': [
            'Portfolio Value','Total Return','Annualized Return','Volatility',
            'Sharpe Ratio','Sortino Ratio','Max Drawdown','Beta','VaR (95%)'
        ],
        'Value': [
            f"${portfolio_value:,.2f}",
            f"{metrics['total_return']*100:.2f}%",
            f"{metrics['annualized_return']*100:.2f}%",
            f"{metrics['volatility']*100:.2f}%",
            f"{metrics['sharpe_ratio']:.3f}",
            f"{metrics['sortino_ratio']:.3f}",
            f"{metrics['max_drawdown']*100:.2f}%",
            f"{metrics['beta']:.2f}",
            f"${metrics['var_dollar']:,.2f}",
        ]
    }
    output = io.StringIO()
    output.write("PORTFOLIO SUMMARY\n")
    pd.DataFrame(summary_data).to_csv(output, index=False)

    output.write("\n\nPORTFOLIO ALLOCATION\n")
    pd.DataFrame([{'Asset': k, 'Weight': f"{v}%"} for k, v in portfolio.items()]).to_csv(output, index=False)

    if detailed_positions:
        output.write("\n\nDETAILED POSITIONS\n")
        pd.DataFrame(detailed_positions).to_csv(output, index=False)

    return output.getvalue()

def create_pdf_report(metrics, portfolio, portfolio_value):
    if not PDF_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Portfolio Analytics Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))

    table_data = [
        ['Metric', 'Value'],
        ['Portfolio Value', f"${portfolio_value:,.2f}"],
        ['Total Return', f"{metrics['total_return']*100:.2f}%"],
        ['Annualized Return', f"{metrics['annualized_return']*100:.2f}%"],
        ['Volatility', f"{metrics['volatility']*100:.2f}%"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
        ['Sortino Ratio', f"{metrics['sortino_ratio']:.3f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"],
        ['Beta', f"{metrics['beta']:.2f}"],
        ['VaR (95%)', f"${metrics['var_dollar']:,.2f}"]
    ]
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12),
        ('BOTTOMPADDING',(0,0),(-1,0),10),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),0.5,colors.black)
    ]))
    story.append(t)
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# -------------------------- UK Economic Indicators --------------------------
@st.cache_data(ttl=3600)
def get_uk_indicators():
    # Static placeholders for macro (to avoid unreliable scraping)
    tiles = [
        {"label": "Bank Rate (BoE)", "value": "5.25%", "ok": True},
        {"label": "UK CPI (YoY)", "value": "3.0%", "ok": True},
        {"label": "UK Unemployment", "value": "4.2%", "ok": True},
    ]
    # Live proxies via Yahoo Finance
    try:
        ftse = yf.Ticker("^FTSE").history(period="5d")
        if not ftse.empty:
            tiles.append({"label": "FTSE 100 (^FTSE)", "value": f"{ftse['Close'].iloc[-1]:,.1f}", "ok": True})
        else:
            tiles.append({"label": "FTSE 100 (^FTSE)", "value": "–", "ok": False})
    except Exception:
        tiles.append({"label": "FTSE 100 (^FTSE)", "value": "–", "ok": False})

    try:
        gbp = yf.Ticker("GBPUSD=X").history(period="5d")
        if not gbp.empty:
            tiles.append({"label": "GBP/USD (GBPUSD=X)", "value": f"{gbp['Close'].iloc[-1]:.2f}", "ok": True})
        else:
            tiles.append({"label": "GBP/USD (GBPUSD=X)", "value": "–", "ok": False})
    except Exception:
        tiles.append({"label": "GBP/USD (GBPUSD=X)", "value": "–", "ok": False})
    return tiles

# -------------------------- Attribution --------------------------
def calculate_performance_attribution(data, portfolio, portfolio_returns):
    out = []
    if portfolio_returns is None or len(portfolio_returns) == 0:
        return out
    for sym, wt in portfolio.items():
        if sym in data and not data[sym].empty:
            r = (data[sym]['Close'].iloc[-1] / data[sym]['Close'].iloc[0]) - 1.0
            contrib = (wt / 100.0) * r * 100.0
            out.append({
                "Asset": sym,
                "Weight": f"{wt}%",
                "Asset_Return": f"{r*100:.2f}%",
                "Contribution": contrib,
                "Contribution_Display": f"{contrib:.2f}pp"
            })
    out.sort(key=lambda x: x["Contribution"], reverse=True)
    return out

# -------------------------- Rebalancing --------------------------
def calculate_rebalancing(current_portfolio, target_portfolio, portfolio_value, current_prices):
    actions = []
    for sym in set(current_portfolio.keys()).union(target_portfolio.keys()):
        cw = float(current_portfolio.get(sym, 0))
        tw = float(target_portfolio.get(sym, 0))
        if sym not in current_prices:
            continue
        cur_val = portfolio_value * (cw / 100.0)
        tgt_val = portfolio_value * (tw / 100.0)
        diff = tgt_val - cur_val
        if abs(diff) <= 1:
            continue
        price = float(current_prices[sym])
        shares = diff / price if price > 0 else 0.0
        actions.append({
            "Asset": sym,
            "Current_Weight": f"{cw:.1f}%",
            "Target_Weight": f"{tw:.1f}%",
            "Action": "BUY" if diff > 0 else "SELL",
            "Shares": f"{abs(shares):.2f}",
            "Value": f"${abs(diff):,.2f}",
            "value_num": abs(diff),
            "Current_Price": f"${price:.2f}"
        })
    return actions

# -------------------------- Portfolio Metrics --------------------------
@st.cache_data
def calculate_portfolio_metrics(data, portfolio, initial_investment, benchmark_symbol="SPY"):
    def clean_close(df):
        s = df['Close'].copy()
        idx = pd.to_datetime(s.index)
        try: idx = idx.tz_convert(None)
        except Exception:
            try: idx = idx.tz_localize(None)
            except Exception: pass
        s.index = idx.normalize()
        s = s[~s.index.duplicated(keep='last')]
        return s

    symbols = [s for s in portfolio.keys() if s in data and not data[s].empty]
    if not symbols:
        return {
            'total_return':0.0,'annualized_return':0.0,'volatility':0.0,'sharpe_ratio':0.0,
            'sortino_ratio':0.0,'max_drawdown':0.0,'beta':1.0,'var_95':0.0,'var_dollar':0.0,
            'portfolio_returns': pd.Series(dtype=float), 'portfolio_values': pd.Series(dtype=float)
        }

    series = [clean_close(data[s]).rename(s) for s in symbols]
    prices = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if prices.shape[0] < 2:
        return {
            'total_return':0.0,'annualized_return':0.0,'volatility':0.0,'sharpe_ratio':0.0,
            'sortino_ratio':0.0,'max_drawdown':0.0,'beta':1.0,'var_95':0.0,'var_dollar':0.0,
            'portfolio_returns': pd.Series(dtype=float, index=prices.index),
            'portfolio_values': pd.Series(dtype=float, index=prices.index)
        }

    start_prices = prices.iloc[0]
    shares = pd.Series(index=prices.columns, dtype=float)
    for s in prices.columns:
        alloc = initial_investment * (portfolio.get(s, 0) / 100.0)
        sp = float(start_prices[s])
        shares[s] = alloc / sp if sp > 0 else 0.0

    port_vals = prices.dot(shares)
    port_rets = port_vals.pct_change().dropna()
    n = len(port_rets)

    total_return = float(port_vals.iloc[-1] / port_vals.iloc[0] - 1.0) if port_vals.iloc[0] > 0 else 0.0
    annualized_return = (1 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0
    volatility = float(port_rets.std(ddof=0) * np.sqrt(252.0)) if n > 1 else 0.0
    rf = 0.02
    sharpe = (annualized_return - rf) / volatility if volatility > 0 else 0.0
    downside = port_rets[port_rets < 0]
    ddev = float(downside.std(ddof=0) * np.sqrt(252.0)) if len(downside) > 1 else 0.0
    sortino = (annualized_return - rf) / ddev if ddev > 0 else 0.0

    cum = port_vals / port_vals.iloc[0]
    dd = (cum / cum.cummax()) - 1.0
    mdd = float(dd.min()) if not dd.empty else 0.0

    # Beta
    beta = 1.0
    try:
        bench = yf.Ticker(benchmark_symbol).history(
            start=prices.index.min() - timedelta(days=2),
            end=prices.index.max() + timedelta(days=2)
        )['Close']
        bench.index = pd.to_datetime(bench.index)
        try: bench.index = bench.index.tz_convert(None)
        except Exception:
            try: bench.index = bench.index.tz_localize(None)
            except Exception: pass
        bench.index = bench.index.normalize()
        bench = bench[~bench.index.duplicated(keep='last')]
        bench = bench.reindex(port_vals.index).ffill().dropna()
        b_rets = bench.pct_change().dropna()
        aligned = pd.concat([port_rets, b_rets], axis=1, join="inner")
        aligned.columns = ["portfolio","benchmark"]
        if len(aligned) > 1:
            cov = aligned['portfolio'].cov(aligned['benchmark'])
            var_b = aligned['benchmark'].var()
            beta = float(cov / var_b) if var_b > 0 else 1.0
    except Exception:
        beta = 1.0

    var_95 = float(np.percentile(port_rets, 5)) if n > 0 else 0.0
    var_dollar = abs(var_95) * float(port_vals.iloc[-1]) if n > 0 else 0.0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': mdd,
        'beta': beta,
        'var_95': var_95,
        'var_dollar': var_dollar,
        'portfolio_returns': port_rets,
        'portfolio_values': port_vals
    }

# -------------------------- Monte Carlo --------------------------
@st.cache_data
def monte_carlo_simulation(portfolio_returns, start_value, days_ahead=252, simulations=1000):
    if portfolio_returns is None or len(portfolio_returns) < 10:
        return None
    mu = float(portfolio_returns.mean())
    sigma = float(portfolio_returns.std(ddof=0))
    if sigma == 0:
        return None
    rng = np.random.default_rng(42)
    draws = rng.normal(mu, sigma, size=(simulations, days_ahead))
    paths = np.prod(1.0 + draws, axis=1) * start_value
    return {
        "results": paths,
        "percentiles": {
            "10th": float(np.percentile(paths, 10)),
            "50th": float(np.percentile(paths, 50)),
            "90th": float(np.percentile(paths, 90)),
        },
        "expected_value": float(paths.mean())
    }

# -------------------------- Mapping & Suggestions --------------------------
ASSET_SECTORS = {
    'AAPL':'Technology','MSFT':'Technology','GOOGL':'Technology','TSLA':'Consumer Cyclical',
    'NVDA':'Technology','BTC-USD':'Cryptocurrency','GLD':'Commodities','SPY':'ETF - Broad Market',
    'QQQ':'ETF - Technology','VTI':'ETF - Total Market','BND':'Bonds','AMZN':'Consumer Cyclical',
    'META':'Technology','NFLX':'Communication','AMD':'Technology','JPM':'Financial Services',
    'WMT':'Consumer Defensive','PG':'Consumer Defensive','JNJ':'Healthcare','V':'Financial Services',
    'MA':'Financial Services','DIS':'Communication','PYPL':'Financial Services'
}

def suggest_portfolio_improvements(portfolio, metrics, sector_allocation):
    out = []
    tech = sector_allocation.get('Technology', 0)
    if tech > 60:
        out.append({
            "title":"High Technology Concentration","priority":"High",
            "description":f"Your portfolio is {tech}% technology. Consider adding healthcare, utilities, or consumer staples for diversification."
        })
    if metrics.get("sharpe_ratio", 0) < 0.5:
        out.append({
            "title":"Low Risk-Adjusted Returns","priority":"Medium",
            "description":"Add bonds (BND) or defensive stocks to improve risk-adjusted returns."
        })
    if not any(ASSET_SECTORS.get(s,"") == "Bonds" or s == "BND" for s in portfolio.keys()):
        out.append({
            "title":"Missing Bond Allocation","priority":"Medium",
            "description":"Add bonds (BND, TLT) for stability and downside protection."
        })
    if not any(ASSET_SECTORS.get(s,"") == "Commodities" for s in portfolio.keys()):
        out.append({
            "title":"Missing Commodity Exposure","priority":"Low",
            "description":"Consider gold (GLD) or commodity ETFs for inflation protection."
        })
    return out

# -------------------------- Header --------------------------
st.markdown('<h1 class="main-header">Smart Portfolio Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Professional Asset Management & Risk Analysis")

# -------------------------- Sidebar --------------------------
st.sidebar.title("Portfolio Configuration")
st.sidebar.markdown("---")

st.sidebar.subheader("Add Custom Assets")
new_asset = st.sidebar.text_input("Add Asset (e.g., AMZN, META, JPM):", "").upper()
if new_asset and st.sidebar.button("Add Asset"):
    st.session_state.setdefault("custom_assets", [])
    if new_asset not in st.session_state["custom_assets"]:
        st.session_state["custom_assets"].append(new_asset)

default_portfolio = {'AAPL':25,'MSFT':20,'GOOGL':15,'TSLA':10,'NVDA':10,'BTC-USD':10,'GLD':10}
available_assets = list(default_portfolio.keys())
if "custom_assets" in st.session_state:
    available_assets.extend(st.session_state["custom_assets"])

st.sidebar.subheader("Asset Allocation (%)")
portfolio = {}
total_allocation = 0
for sym in available_assets:
    w = st.sidebar.number_input(
        f"{sym} - {ASSET_SECTORS.get(sym,'Other')}",
        min_value=0, max_value=100, value=default_portfolio.get(sym,0), step=1, key=f"w_{sym}"
    )
    if w > 0:
        portfolio[sym] = w
        total_allocation += w

if total_allocation != 100:
    st.sidebar.error(f"Total allocation: {total_allocation}%. Must equal 100%")
else:
    st.sidebar.success(f"✅ Total allocation: {total_allocation}%")

initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, max_value=1_000_000, value=10000, step=1000)

time_periods = {"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"2 Years":730}
selected_period = st.sidebar.selectbox("Analysis Period", list(time_periods.keys()), index=3)
days_back = time_periods[selected_period]

benchmark = st.sidebar.selectbox("Benchmark", ["SPY","QQQ","VTI"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")
show_monte_carlo = st.sidebar.checkbox("Enable Monte Carlo Projections", value=True)
show_suggestions = st.sidebar.checkbox("Show AI Portfolio Suggestions", value=True)
show_rebalancing = st.sidebar.checkbox("Enable Rebalancing Tool", value=True)
show_economic = st.sidebar.checkbox("Show UK Economic Indicators", value=True)

# -------------------------- Main --------------------------
if total_allocation == 100 and portfolio:
    with st.spinner("Fetching market data & computing metrics..."):
        data = {}
        current_prices = {}
        for sym in list(portfolio.keys()) + [benchmark]:
            try:
                hist = yf.Ticker(sym).history(period=f"{days_back}d")
                if not hist.empty:
                    data[sym] = hist
                    current_prices[sym] = float(hist['Close'].iloc[-1])
            except Exception:
                st.warning(f"Could not fetch data for {sym}")

    if data and any(s in data for s in portfolio.keys()):
        metrics = calculate_portfolio_metrics({k:v for k,v in data.items() if k in portfolio}, portfolio, initial_investment, benchmark)

        pv = metrics['portfolio_values']
        if pv is not None and len(pv) > 0:
            portfolio_value = float(pv.iloc[-1])
            start_value = float(pv.iloc[0])
        else:
            portfolio_value = sum((initial_investment * w/100.0) for w in portfolio.values())
            start_value = initial_investment

        gain_loss = portfolio_value - start_value
        total_return_pct = (portfolio_value / start_value - 1.0) * 100.0

        # UK Macro tiles
        if show_economic:
            st.markdown("## 🌍 Economic Environment (UK Focus)")
            eco_tiles = get_uk_indicators()
            cols = st.columns(3)
            for i, tile in enumerate(eco_tiles):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="economic-tile">
                        <h4>{tile['label']}</h4>
                        <h3>{tile['value']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.caption("Note: Macro tiles mix static placeholders with best-effort market proxies (FTSE, GBP/USD).")
            st.markdown("---")

        # Overview
        st.markdown("## 📊 Portfolio Overview")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-container"><h3>Portfolio Value</h3><h2>${portfolio_value:,.2f}</h2></div>""", unsafe_allow_html=True)
        with c2:
            color = "green" if total_return_pct >= 0 else "red"
            st.markdown(f"""<div class="metric-container"><h3>Total Return</h3><h2 style="color:{color}">{total_return_pct:+.2f}%</h2></div>""", unsafe_allow_html=True)
        with c3:
            color = "green" if gain_loss >= 0 else "red"
            st.markdown(f"""<div class="metric-container"><h3>Gain/Loss</h3><h2 style="color:{color}">${gain_loss:+,.2f}</h2></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-container"><h3>Assets</h3><h2>{len(portfolio)}</h2></div>""", unsafe_allow_html=True)

        # Attribution
        st.markdown("## 🎯 Performance Attribution Analysis")
        attribution = calculate_performance_attribution(data, portfolio, metrics['portfolio_returns'])
        if attribution:
            df_attr = pd.DataFrame(attribution)
            fig_attr = px.bar(
                df_attr, x="Asset", y="Contribution",
                title="Individual Asset Contribution (percentage points)",
                color="Contribution", color_continuous_scale=['red','white','green']
            )
            fig_attr.update_layout(height=380)
            st.plotly_chart(fig_attr, use_container_width=True)

            show_attr = df_attr[['Asset','Weight','Asset_Return','Contribution_Display']].rename(
                columns={'Contribution_Display':'Contribution'}
            )
            st.dataframe(show_attr, use_container_width=True, hide_index=True)

        # Rebalancing
        if show_rebalancing:
            st.markdown("## ⚖️ Portfolio Rebalancing Tool")
            cA, cB = st.columns(2)
            target = {}
            with cA:
                st.markdown("**Current Allocation**")
                for s,w in portfolio.items():
                    st.write(f"{s}: {w}%")
            with cB:
                st.markdown("**Target Allocation**")
                tgt_total = 0
                for s in portfolio.keys():
                    tw = st.number_input(f"Target {s} (%)", min_value=0, max_value=100, value=portfolio[s], step=1, key=f"tgt_{s}")
                    target[s] = tw
                    tgt_total += tw
            if tgt_total == 100:
                st.success(f"✅ Target allocation totals: {tgt_total}%")
                actions = calculate_rebalancing(portfolio, target, portfolio_value, current_prices)
                if actions:
                    st.markdown("### 📋 Rebalancing Actions Required")
                    st.dataframe(pd.DataFrame(actions).drop(columns=["value_num"]), use_container_width=True, hide_index=True)
                    total_buy = sum(a["value_num"] for a in actions if a["Action"] == "BUY")
                    total_sell = sum(a["value_num"] for a in actions if a["Action"] == "SELL")
                    st.markdown(f"**Rebalancing Summary:**  \n- Total to Buy: ${total_buy:,.2f}  \n- Total to Sell: ${total_sell:,.2f}  \n- Net Cash Flow: ${total_buy - total_sell:,.2f}")
                else:
                    st.success("✅ Portfolio is already balanced to the target allocation!")
            else:
                st.error(f"❌ Target allocation totals {tgt_total}%. Must equal 100%")

        # Risk
        st.markdown("## ⚠️ Risk Analysis")
        r1,r2,r3,r4 = st.columns(4)
        with r1:
            st.markdown(f"""<div class="risk-metric"><h4>Sharpe Ratio</h4><h3>{metrics['sharpe_ratio']:.3f}</h3><small>Risk-adjusted return</small></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="risk-metric"><h4>Volatility</h4><h3>{metrics['volatility']*100:.1f}%</h3><small>Annual price swings</small></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="risk-metric"><h4>Max Drawdown</h4><h3>{metrics['max_drawdown']*100:.1f}%</h3><small>Worst peak-to-trough loss</small></div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""<div class="risk-metric"><h4>Value at Risk (95%)</h4><h3>${metrics['var_dollar']:,.2f}</h3><small>Daily loss estimate</small></div>""", unsafe_allow_html=True)

        # Performance
        st.markdown("## 📈 Performance Analysis")
        p1,p2,p3,p4 = st.columns(4)
        with p1:
            st.markdown(f"""<div class="performance-metric"><h4>Annualized Return</h4><h3>{metrics['annualized_return']*100:+.1f}%</h3><small>Year-over-year growth</small></div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class="performance-metric"><h4>Sortino Ratio</h4><h3>{metrics['sortino_ratio']:.3f}</h3><small>Downside risk-adjusted</small></div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""<div class="performance-metric"><h4>Beta vs {benchmark}</h4><h3>{metrics['beta']:.2f}</h3><small>Market sensitivity</small></div>""", unsafe_allow_html=True)
        with p4:
            if benchmark in data:
                bench_ret = ((data[benchmark]['Close'].iloc[-1] / data[benchmark]['Close'].iloc[0]) - 1.0) * 100
                outperf = total_return_pct - bench_ret
                color = "green" if outperf >= 0 else "red"
            else:
                outperf, color = 0, "gray"
            st.markdown(f"""<div class="performance-metric"><h4>vs {benchmark}</h4><h3 style="color:{color}">{outperf:+.1f}%</h3><small>Outperformance</small></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # AI Suggestions
        if show_suggestions:
            st.markdown("## 🤖 AI Portfolio Recommendations")
            sector_alloc = {}
            for s,w in portfolio.items():
                sector = ASSET_SECTORS.get(s,"Other")
                sector_alloc[sector] = sector_alloc.get(sector, 0) + w
            for s in suggest_portfolio_improvements(portfolio, metrics, sector_alloc):
                bar = {'High':'#ff4444','Medium':'#ffaa00','Low':'#44aa44'}.get(s['priority'], '#666')
                st.markdown(f"""
                <div class="alert-box" style="border-left-color:{bar}">
                    <h4>🎯 {s['title']} — {s['priority']} Priority</h4>
                    <p>{s['description']}</p>
                </div>
                """, unsafe_allow_html=True)

        # Monte Carlo
        if show_monte_carlo and metrics["portfolio_returns"] is not None and len(metrics["portfolio_returns"]) > 10:
            st.markdown("## 🎲 Monte Carlo Projections (1 Year)")
            mc = monte_carlo_simulation(metrics["portfolio_returns"], portfolio_value, 252, 1000)
            if mc:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class="performance-metric"><h4>Expected Value (1 Year)</h4><h3>${mc['expected_value']:,.0f}</h3><small>Average of 1000 simulations</small></div>""", unsafe_allow_html=True)
                with c2:
                    p10, p50, p90 = mc["percentiles"]["10th"], mc["percentiles"]["50th"], mc["percentiles"]["90th"]
                    st.markdown(f"""<div class="performance-metric"><h4>80% Confidence Range</h4><h3>${p10:,.0f} - ${p90:,.0f}</h3><small>Median: ${p50:,.0f}</small></div>""", unsafe_allow_html=True)

                fig_mc = px.histogram(x=mc["results"], nbins=50, title="Portfolio Value Distribution (1000 Simulations)",
                                      labels={'x':'Portfolio Value ($)', 'y':'Frequency'})
                fig_mc.add_vline(x=p10, line_dash="dash", line_color="red", annotation_text="10th %")
                fig_mc.add_vline(x=p50, line_dash="dash", line_color="yellow", annotation_text="Median")
                fig_mc.add_vline(x=p90, line_dash="dash", line_color="green", annotation_text="90th %")
                fig_mc.update_layout(height=400)
                st.plotly_chart(fig_mc, use_container_width=True)

        # Sector tiles
        st.markdown("## 🏢 Sector Allocation")
        sector_alloc = {}
        for s,w in portfolio.items():
            sector = ASSET_SECTORS.get(s, "Other")
            sector_alloc[sector] = sector_alloc.get(sector, 0) + w
        if sector_alloc:
            cols = st.columns(len(sector_alloc))
            for i,(sec,wt) in enumerate(sector_alloc.items()):
                with cols[i]:
                    st.markdown(f"""<div class="sector-box"><div>{sec}</div><div style="font-size:1.5em;">{wt}%</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Allocation & Risk/Return
        a1, a2 = st.columns(2)
        with a1:
            st.subheader("Portfolio Allocation")
            fig_pie = px.pie(
                values=list(portfolio.values()),
                names=[f"{s}<br>{ASSET_SECTORS.get(s,'Other')}" for s in portfolio.keys()],
                title="Asset Allocation by Sector",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=500, showlegend=True, title_x=0.5)
            st.plotly_chart(fig_pie, use_container_width=True)
        with a2:
            st.subheader("Risk-Return Analysis")
            rows = []
            for s in portfolio.keys():
                if s in data:
                    r = data[s]['Close'].pct_change().dropna()
                    rows.append({
                        "Asset": s,
                        "Annual_Return": (r.mean()*252)*100,
                        "Volatility": (r.std(ddof=0)*np.sqrt(252))*100,
                        "Weight": portfolio[s]
                    })
            if rows:
                df_rr = pd.DataFrame(rows)
                fig_scatter = px.scatter(df_rr, x="Volatility", y="Annual_Return", size="Weight", text="Asset",
                                         title="Risk vs Return Profile",
                                         labels={"Volatility":"Annual Volatility (%)","Annual_Return":"Annual Return (%)"})
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Portfolio vs Benchmark
        st.subheader("Portfolio vs Benchmark Performance")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=metrics['portfolio_values'].index, y=metrics['portfolio_values'].values,
                                     mode='lines', name='Your Portfolio', line=dict(color='#1f77b4', width=3)))
        if benchmark in data:
            bench = data[benchmark]['Close'].copy()
            idx = pd.to_datetime(bench.index)
            try: idx = idx.tz_convert(None)
            except Exception:
                try: idx = idx.tz_localize(None)
                except Exception: pass
            bench.index = idx.normalize()
            bench = bench[~bench.index.duplicated(keep='last')]
            bench = bench.reindex(metrics['portfolio_values'].index).ffill().dropna()
            if len(bench) > 0:
                bench_norm = bench / bench.iloc[0] * metrics['portfolio_values'].iloc[0]
                fig_cmp.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values,
                                             mode='lines', name=f'{benchmark} Benchmark',
                                             line=dict(color='#ff7f0e', width=2, dash='dash')))
        fig_cmp.add_hline(y=initial_investment, line_dash="dot", line_color="gray", annotation_text="Initial Investment")
        fig_cmp.update_layout(title="Portfolio Performance vs Benchmark", xaxis_title="Date", yaxis_title="Value ($)",
                              height=500, hovermode='x unified')
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ---------------- Robust Correlation Heatmap (Business-Day Aligned) ----------------
        st.subheader("Asset Correlation Analysis")
        syms = [s for s in portfolio.keys() if s in data and not data[s].empty]
        if len(syms) > 1:
            def clean_close(df, name):
                s = df['Close'].copy()
                idx = pd.to_datetime(s.index)
                try: idx = idx.tz_convert(None)
                except Exception:
                    try: idx = idx.tz_localize(None)
                    except Exception: pass
                s.index = idx.normalize()
                s = s[~s.index.duplicated(keep='last')]
                return s.rename(name)

            series = [clean_close(data[s], s) for s in syms]
            start = max(s.index.min() for s in series)
            end = min(s.index.max() for s in series)
            if pd.notna(start) and pd.notna(end) and start < end:
                bidx = pd.date_range(start, end, freq="B")          # business days only
                aligned = pd.concat([s.reindex(bidx).ffill() for s in series], axis=1)
                rets = aligned.pct_change().dropna(how="any")       # drop rows with any NaN
                if rets.shape[1] > 1 and len(rets) > 2:
                    corr = rets.corr()
                    fig_heat = px.imshow(corr, title="Asset Correlation Matrix (Business-Day Aligned)",
                                         color_continuous_scale="RdBu", aspect="auto", zmin=-1, zmax=1)
                    fig_heat.update_layout(height=500)
                    st.plotly_chart(fig_heat, use_container_width=True)
                    st.info("💡 Values near +1 move together; near −1 move opposite. Lower correlations improve diversification.")
                else:
                    st.info("Not enough overlapping data to compute correlations.")
            else:
                st.info("Time ranges don't overlap enough to compute correlations.")
        else:
            st.info("Need at least two assets to compute correlation.")

        # Detailed positions
        st.subheader("Detailed Position Analysis")
        detailed = []
        # Aligned start prices for fair shares calculation
        try:
            def cc(df):
                s = df['Close'].copy()
                idx = pd.to_datetime(s.index)
                try: idx = idx.tz_convert(None)
                except Exception:
                    try: idx = idx.tz_localize(None)
                    except Exception: pass
                s.index = idx.normalize()
                s = s[~s.index.duplicated(keep='last')]
                return s
            aligned_series = [cc(data[s]).rename(s) for s in syms]
            aligned_prices = pd.concat(aligned_series, axis=1, join='inner').dropna(how='any')
            start_prices = aligned_prices.iloc[0]
        except Exception:
            start_prices = pd.Series(dtype=float)

        for s,w in portfolio.items():
            if s in current_prices:
                cur = float(current_prices[s])
                alloc = (w/100.0) * initial_investment
                if s in start_prices.index:
                    sp = float(start_prices[s])
                else:
                    try: sp = float(data[s]['Close'].iloc[0])
                    except Exception: sp = 0.0
                shares = alloc / sp if sp > 0 else 0.0
                pos_val = shares * cur
                try:
                    aset_vol = (data[s]['Close'].pct_change().std(ddof=0) * np.sqrt(252)) * 100.0
                except Exception:
                    aset_vol = 0.0
                aset_ret = ((cur / sp) - 1.0) * 100.0 if sp > 0 else 0.0
                detailed.append({
                    "Asset": s, "Sector": ASSET_SECTORS.get(s,"Other"), "Weight": f"{w}%",
                    "Position Value": f"${pos_val:,.2f}", "Current Price": f"${cur:.2f}",
                    "Shares/Units": f"{shares:.4f}", "Return": f"{aset_ret:+.2f}%", "Volatility": f"{aset_vol:.1f}%"
                })
        if detailed:
            st.dataframe(pd.DataFrame(detailed), use_container_width=True, hide_index=True)

        # Export
        st.markdown("---")
        st.markdown("## 📊 Export Dashboard Data")
        e1, e2 = st.columns(2)
        with e1:
            if st.button("📄 Export as CSV", use_container_width=True):
                csv_data = create_csv_export(metrics, portfolio, portfolio_value, detailed)
                st.download_button(
                    label="Download CSV Report",
                    data=csv_data,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        with e2:
            if st.button("📋 Export as PDF", use_container_width=True):
                if PDF_AVAILABLE:
                    try:
                        pdf_data = create_pdf_report(metrics, portfolio, portfolio_value)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")
                else:
                    st.error("PDF export requires reportlab. Install with: pip install reportlab")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;color:#666;font-size:0.9em;">
            <h4>Methodology & Data Sources</h4>
            <p><strong>Data:</strong> Yahoo Finance | <strong>Risk-free rate:</strong> 2% assumed</p>
            <p><strong>Sharpe:</strong> (Annualized Return − RF) / Annualized Vol | <strong>Sortino:</strong> Downside-only volatility</p>
            <p><strong>VaR (95%):</strong> 5th percentile daily loss (scaled to $ current portfolio) | <strong>Beta:</strong> vs benchmark</p>
            <p><strong>Monte Carlo:</strong> 1000 sims using mean/std of daily returns</p>
            <p><strong>Attribution:</strong> Asset percentage-point contribution to portfolio performance</p>
            <br><em>Smart Portfolio Analytics Dashboard — UK Macro, Rebalancing, Export & Pro Charts</em>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Please ensure your portfolio allocation totals 100% to view the dashboard.")
