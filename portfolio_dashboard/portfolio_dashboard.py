import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta, datetime
import numpy as np
import io

# Optional: PDF export (install with: pip install reportlab)
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
    .rebalance-box {
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .economic-indicator {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
        text-align: center;
        border: 2px solid #ff9500;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------- Export Functions --------------------------
def create_csv_export(metrics, portfolio, portfolio_value, detailed_positions):
    """Create CSV data for export"""
    summary_data = {
        'Metric': ['Portfolio Value', 'Total Return', 'Annualized Return', 'Volatility',
                   'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Beta', 'VaR (95%)'],
        'Value': [f"${portfolio_value:,.2f}", f"{metrics['total_return']*100:.2f}%",
                  f"{metrics['annualized_return']*100:.2f}%", f"{metrics['volatility']*100:.2f}%",
                  f"{metrics['sharpe_ratio']:.3f}", f"{metrics['sortino_ratio']:.3f}",
                  f"{metrics['max_drawdown']*100:.2f}%", f"{metrics['beta']:.2f}",
                  f"${metrics['var_dollar']:,.2f}"]
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

def create_pdf_report(metrics, portfolio, portfolio_value, sector_allocation):
    """Create PDF report (if reportlab installed)"""
    if not PDF_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Portfolio Analytics Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))

    summary_data = [
        ['Metric', 'Value'],
        ['Portfolio Value', f"${portfolio_value:,.2f}"],
        ['Total Return', f"{metrics['total_return']*100:.2f}%"],
        ['Annualized Return', f"{metrics['annualized_return']*100:.2f}%"],
        ['Volatility', f"{metrics['volatility']*100:.2f}%"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"],
    ]
    table = Table(summary_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12), ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige), ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------------- UK Economic Indicators (clean) --------------------------
@st.cache_data(ttl=3600)
def get_uk_economic_indicators():
    """
    Minimal UK macro tiles (no FTSE volatility, no 10Y gilt).
    Bank Rate/CPI/Unemployment are placeholders (adjust if desired).
    FTSE100 & GBP/USD are live from Yahoo Finance.
    """
    tiles = {
        "BOE_RATE": {"title": "Bank Rate (BoE)", "value": "5.25%"},
        "UK_CPI":   {"title": "UK CPI (YoY)",  "value": "3.0%"},
        "UK_UNEMP": {"title": "UK Unemployment", "value": "4.2%"},
        "FTSE":     {"title": "FTSE 100 (^FTSE)", "value": "-"},
        "GBPUSD":   {"title": "GBP/USD (GBPUSD=X)", "value": "-"},
    }
    # FTSE 100 (level)
    try:
        ftse_hist = yf.Ticker("^FTSE").history(period="10d")
        if not ftse_hist.empty:
            tiles["FTSE"]["value"] = f"{float(ftse_hist['Close'].iloc[-1]):,.1f}"
    except Exception:
        pass
    # GBP/USD
    try:
        fx_hist = yf.Ticker("GBPUSD=X").history(period="5d")
        if not fx_hist.empty:
            tiles["GBPUSD"]["value"] = f"{float(fx_hist['Close'].iloc[-1]):.2f}"
    except Exception:
        pass
    return tiles

# -------------------------- Performance Attribution --------------------------
def calculate_performance_attribution(data, portfolio, metrics):
    """Which assets contributed most to performance (percentage points)"""
    attribution = []
    if metrics.get('portfolio_returns') is None or metrics['portfolio_returns'].empty:
        return attribution
    for symbol, weight in portfolio.items():
        if symbol in data and not data[symbol].empty:
            asset_total_return = (data[symbol]['Close'].iloc[-1] / data[symbol]['Close'].iloc[0] - 1.0)
            contribution = (weight / 100.0) * asset_total_return * 100.0
            attribution.append({
                'Asset': symbol,
                'Weight': f"{weight}%",
                'Asset_Return': f"{asset_total_return*100:.2f}%",
                'Contribution': contribution,
                'Contribution_Display': f"{contribution:.2f}pp"
            })
    attribution.sort(key=lambda x: x['Contribution'], reverse=True)
    return attribution

# -------------------------- Rebalancing Tool --------------------------
def calculate_rebalancing(current_portfolio, target_portfolio, portfolio_value, current_prices):
    """What to BUY/SELL to reach target weights. Also returns numeric totals."""
    actions = []
    for symbol in set(list(current_portfolio.keys()) + list(target_portfolio.keys())):
        cur_w = current_portfolio.get(symbol, 0.0)
        tgt_w = target_portfolio.get(symbol, 0.0)
        if symbol in current_prices:
            cur_val = portfolio_value * (cur_w / 100.0)
            tgt_val = portfolio_value * (tgt_w / 100.0)
            diff_val = tgt_val - cur_val
            if abs(diff_val) > 1:  # ignore tiny
                px = float(current_prices[symbol])
                shares_to_trade = diff_val / px if px > 0 else 0.0
                action = "BUY" if diff_val > 0 else "SELL"
                actions.append({
                    'Asset': symbol,
                    'Current_Weight': f"{cur_w:.1f}%",
                    'Target_Weight': f"{tgt_w:.1f}%",
                    'Action': action,
                    'Shares': f"{abs(shares_to_trade):.2f}",
                    'Value': f"${abs(diff_val):,.2f}",
                    'ValueNum': float(abs(diff_val)),       # numeric for totals
                    'Current_Price': f"${px:.2f}",
                })
    return actions

# -------------------------- Portfolio Metrics (robust) --------------------------
@st.cache_data
def calculate_portfolio_metrics(data, portfolio, initial_investment, benchmark_symbol="SPY"):
    """Robust metrics using shares allocated at the start date."""
    def _close_series(df, name):
        s = df['Close'].copy()
        idx = pd.to_datetime(s.index)
        try: idx = idx.tz_convert(None)
        except Exception:
            try: idx = idx.tz_localize(None)
            except Exception: pass
        s.index = idx.normalize()
        s = s[~s.index.duplicated(keep='last')]
        s.name = name
        return s

    in_universe = [s for s in portfolio.keys() if s in data and not data[s].empty]
    if not in_universe:
        return {k: 0.0 for k in [
            'total_return','annualized_return','volatility','sharpe_ratio',
            'sortino_ratio','max_drawdown','beta','var_95','var_dollar'
        ]} | {'portfolio_returns': pd.Series(dtype=float), 'portfolio_values': pd.Series(dtype=float)}

    series_list = [_close_series(data[s], s) for s in in_universe]
    prices = pd.concat(series_list, axis=1, join='inner').dropna(how='any')
    if prices.shape[0] < 2:
        return {k: 0.0 for k in [
            'total_return','annualized_return','volatility','sharpe_ratio',
            'sortino_ratio','max_drawdown','beta','var_95','var_dollar'
        ]} | {'portfolio_returns': pd.Series(dtype=float, index=prices.index),
              'portfolio_values': pd.Series(dtype=float, index=prices.index)}

    start_prices = prices.iloc[0]
    shares = pd.Series(dtype=float)
    for s in in_universe:
        alloc = initial_investment * (portfolio[s] / 100.0)
        sp = float(start_prices[s])
        shares.loc[s] = alloc / sp if sp > 0 else 0.0

    portfolio_values = prices.dot(shares)
    portfolio_returns = portfolio_values.pct_change().dropna()
    n = len(portfolio_returns)

    total_return = float(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0) if portfolio_values.iloc[0] > 0 else 0.0
    annualized_return = (1.0 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0
    volatility = float(portfolio_returns.std(ddof=0) * np.sqrt(252.0)) if n > 1 else 0.0

    rf = 0.02
    sharpe_ratio = (annualized_return - rf) / volatility if volatility > 0 else 0.0
    downside = portfolio_returns[portfolio_returns < 0]
    downside_dev = float(downside.std(ddof=0) * np.sqrt(252.0)) if len(downside) > 1 else 0.0
    sortino_ratio = (annualized_return - rf) / downside_dev if downside_dev > 0 else 0.0

    cum = portfolio_values / portfolio_values.iloc[0]
    roll_max = cum.cummax()
    drawdown = (cum / roll_max) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    # Beta vs benchmark
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
        bench = bench.reindex(prices.index).ffill().dropna()
        bench_returns = bench.pct_change().dropna()
        aligned = pd.concat([portfolio_returns, bench_returns], axis=1, join='inner').dropna()
        aligned.columns = ['portfolio', 'benchmark']
        if len(aligned) > 1:
            cov = aligned['portfolio'].cov(aligned['benchmark'])
            var_b = aligned['benchmark'].var()
            beta = float(cov / var_b) if var_b > 0 else 1.0
        else:
            beta = 1.0
    except Exception:
        beta = 1.0

    var_95 = float(np.percentile(portfolio_returns, 5)) if n > 0 else 0.0
    var_dollar = abs(var_95) * float(portfolio_values.iloc[-1]) if n > 0 else 0.0

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'max_drawdown': float(max_drawdown),
        'beta': float(beta),
        'var_95': float(var_95),
        'var_dollar': float(var_dollar),
        'portfolio_returns': portfolio_returns,
        'portfolio_values': portfolio_values
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
    rand = rng.normal(mu, sigma, size=(simulations, days_ahead))
    paths = np.prod(1.0 + rand, axis=1) * start_value
    return {
        'results': paths,
        'percentiles': {
            '10th': float(np.percentile(paths, 10)),
            '50th': float(np.percentile(paths, 50)),
            '90th': float(np.percentile(paths, 90)),
        },
        'expected_value': float(paths.mean())
    }

# -------------------------- Sectors & Suggestions --------------------------
ASSET_SECTORS = {
    'AAPL':'Technology','MSFT':'Technology','GOOGL':'Technology','TSLA':'Consumer Cyclical',
    'NVDA':'Technology','BTC-USD':'Cryptocurrency','GLD':'Commodities','SPY':'ETF - Broad Market',
    'QQQ':'ETF - Technology','VTI':'ETF - Total Market','BND':'Bonds','AMZN':'Consumer Cyclical',
    'META':'Technology','NFLX':'Communication','AMD':'Technology','JPM':'Financial Services',
    'WMT':'Consumer Defensive','PG':'Consumer Defensive','JNJ':'Healthcare','V':'Financial Services',
    'MA':'Financial Services','DIS':'Communication','PYPL':'Financial Services'
}

def suggest_portfolio_improvements(portfolio, metrics, sector_allocation):
    suggestions = []
    tech_weight = sector_allocation.get('Technology', 0)
    if tech_weight > 60:
        suggestions.append({
            'type': 'warning', 'title': 'High Technology Concentration',
            'description': f'Your portfolio is {tech_weight}% technology. Consider adding healthcare, utilities, or consumer staples for diversification.',
            'priority': 'High'
        })
    if metrics.get('sharpe_ratio', 0) < 0.5:
        suggestions.append({
            'type': 'improvement', 'title': 'Low Risk-Adjusted Returns',
            'description': 'Add bonds (BND) or defensive stocks to improve risk-adjusted returns.',
            'priority': 'Medium'
        })
    has_bonds = any(ASSET_SECTORS.get(sym, '') == 'Bonds' or sym == 'BND' for sym in portfolio.keys())
    has_commodities = any(ASSET_SECTORS.get(sym, '') == 'Commodities' for sym in portfolio.keys())
    if not has_bonds:
        suggestions.append({
            'type': 'diversification', 'title': 'Missing Bond Allocation',
            'description': 'Add bonds (BND, TLT) for stability and downside protection.',
            'priority': 'Medium'
        })
    if not has_commodities:
        suggestions.append({
            'type': 'diversification', 'title': 'Missing Commodity Exposure',
            'description': 'Consider gold (GLD) or broad commodity ETFs for inflation protection.',
            'priority': 'Low'
        })
    return suggestions

# -------------------------- Header --------------------------
st.markdown('<h1 class="main-header">Smart Portfolio Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Professional Asset Management & Risk Analysis")

# -------------------------- Sidebar --------------------------
st.sidebar.title("Portfolio Configuration")
st.sidebar.markdown("---")

st.sidebar.subheader("Add Custom Assets")
new_asset = st.sidebar.text_input("Add Asset (e.g., AMZN, META, JPM):", "").upper()
if new_asset and st.sidebar.button("Add Asset"):
    st.session_state.setdefault('custom_assets', [])
    if new_asset not in st.session_state['custom_assets']:
        st.session_state['custom_assets'].append(new_asset)

default_portfolio = {'AAPL':25,'MSFT':20,'GOOGL':15,'TSLA':10,'NVDA':10,'BTC-USD':10,'GLD':10}
available_assets = list(default_portfolio.keys())
if 'custom_assets' in st.session_state:
    available_assets.extend(st.session_state['custom_assets'])

st.sidebar.subheader("Asset Allocation (%)")
portfolio = {}
total_allocation = 0
for symbol in available_assets:
    default_weight = default_portfolio.get(symbol, 0)
    weight = st.sidebar.number_input(
        f"{symbol} - {ASSET_SECTORS.get(symbol, 'Other')}",
        min_value=0, max_value=100, value=default_weight, step=1, key=f"weight_{symbol}"
    )
    if weight > 0:
        portfolio[symbol] = weight
        total_allocation += weight

if total_allocation != 100:
    st.sidebar.error(f"Total allocation: {total_allocation}%. Must equal 100%")
else:
    st.sidebar.success(f"✅ Total allocation: {total_allocation}%")

initial_investment = st.sidebar.number_input(
    "Initial Investment ($)", min_value=1000, max_value=1_000_000, value=10000, step=1000
)

time_periods = {"1 Month":30, "3 Months":90, "6 Months":180, "1 Year":365, "2 Years":730}
selected_period = st.sidebar.selectbox("Analysis Period", list(time_periods.keys()), index=3)
days_back = time_periods[selected_period]

benchmark_options = ["SPY", "QQQ", "VTI"]
benchmark = st.sidebar.selectbox("Benchmark", benchmark_options, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")
show_monte_carlo = st.sidebar.checkbox("Enable Monte Carlo Projections", value=True)
show_suggestions = st.sidebar.checkbox("Show AI Portfolio Suggestions", value=True)
show_rebalancing = st.sidebar.checkbox("Enable Rebalancing Tool", value=True)
show_economic_data = st.sidebar.checkbox("Show Economic Indicators", value=True)

# -------------------------- Main --------------------------
if total_allocation == 100 and portfolio:
    with st.spinner("Fetching market data & computing metrics..."):
        data, current_prices = {}, {}
        for symbol in list(portfolio.keys()) + [benchmark]:
            try:
                hist = yf.Ticker(symbol).history(period=f"{days_back}d")
                if not hist.empty:
                    data[symbol] = hist
                    current_prices[symbol] = float(hist['Close'].iloc[-1])
            except Exception:
                st.warning(f"Could not fetch data for {symbol}")

    if data and any(s in data for s in portfolio.keys()):
        # Metrics
        portfolio_data = {k: v for k, v in data.items() if k in portfolio}
        metrics = calculate_portfolio_metrics(portfolio_data, portfolio, initial_investment, benchmark)
        pv_series = metrics['portfolio_values']
        if pv_series is not None and len(pv_series) > 0:
            portfolio_value = float(pv_series.iloc[-1])
            start_value = float(pv_series.iloc[0])
        else:
            portfolio_value, start_value = initial_investment, initial_investment

        gain_loss = portfolio_value - start_value
        total_return_pct = (portfolio_value / start_value - 1.0) * 100.0

        # 1) Economic Environment (UK)  ---------------------------------------
        if show_economic_data:
            st.markdown("## 🌍 Economic Environment")
            uk_tiles = get_uk_economic_indicators()
            eco_cols = st.columns(4)
            order = ["BOE_RATE", "UK_CPI", "UK_UNEMP", "FTSE", "GBPUSD"]
            for i, key in enumerate(order):
                t = uk_tiles[key]
                with eco_cols[i % 4]:
                    st.markdown(f"""
                        <div class="economic-indicator">
                            <h4>{t['title']}</h4>
                            <h3>{t['value']}</h3>
                        </div>
                    """, unsafe_allow_html=True)
            st.info("ℹ️ These tiles mix static placeholders (BoE rate, CPI, unemployment) with live market proxies (FTSE, GBP/USD).")
            st.markdown("---")

        # 2) Overview ----------------------------------------------------------
        st.markdown("## 📊 Portfolio Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-container"><h3>Portfolio Value</h3>
                            <h2>${portfolio_value:,.2f}</h2></div>""", unsafe_allow_html=True)
        with c2:
            color = "green" if total_return_pct >= 0 else "red"
            st.markdown(f"""<div class="metric-container"><h3>Total Return</h3>
                            <h2 style="color:{color}">{total_return_pct:+.2f}%</h2></div>""",
                        unsafe_allow_html=True)
        with c3:
            color = "green" if gain_loss >= 0 else "red"
            st.markdown(f"""<div class="metric-container"><h3>Gain/Loss</h3>
                            <h2 style="color:{color}">${gain_loss:+,.2f}</h2></div>""",
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-container"><h3>Assets</h3>
                            <h2>{len(portfolio)}</h2></div>""", unsafe_allow_html=True)

        # 3) Performance Attribution ------------------------------------------
        st.markdown("## 🎯 Performance Attribution Analysis")
        attribution = calculate_performance_attribution(data, portfolio, metrics)
        if attribution:
            attr_df = pd.DataFrame(attribution)
            fig_attr = px.bar(attr_df, x='Asset', y='Contribution',
                              title="Individual Asset Contribution (pp)",
                              labels={'Contribution': 'Contribution (percentage points)'},
                              color='Contribution', color_continuous_scale=['red','white','green'])
            fig_attr.update_layout(height=400)
            st.plotly_chart(fig_attr, use_container_width=True)
            st.dataframe(attr_df[['Asset','Weight','Asset_Return','Contribution_Display']],
                         use_container_width=True, hide_index=True)

        # 4) Rebalancing Tool --------------------------------------------------
        if show_rebalancing:
            st.markdown("## ⚖️ Portfolio Rebalancing Tool")
            st.markdown("### Set Target Allocation")
            target_portfolio = {}
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Current Allocation**")
                for symbol, wt in portfolio.items():
                    st.write(f"{symbol}: {wt}%")
            with colB:
                st.markdown("**Target Allocation**")
                tgt_total = 0
                for symbol in portfolio.keys():
                    tgt = st.number_input(f"Target {symbol} (%)",
                                          min_value=0, max_value=100,
                                          value=portfolio[symbol], step=1,
                                          key=f"target_{symbol}")
                    target_portfolio[symbol] = tgt
                    tgt_total += tgt

            if tgt_total == 100:
                st.success(f"✅ Target allocation totals: {tgt_total}%")
                actions = calculate_rebalancing(portfolio, target_portfolio, portfolio_value, current_prices)
                if actions:
                    st.markdown("### 📋 Rebalancing Actions Required")
                    df_reb = pd.DataFrame(actions)
                    st.dataframe(df_reb.drop(columns=['ValueNum']), use_container_width=True, hide_index=True)

                    total_buy  = sum(a['ValueNum'] for a in actions if a['Action'] == 'BUY')
                    total_sell = sum(a['ValueNum'] for a in actions if a['Action'] == 'SELL')
                    st.markdown(f"""**Rebalancing Summary:**
                    - Total to Buy: ${total_buy:,.2f}
                    - Total to Sell: ${total_sell:,.2f}
                    - Net Cash Flow: ${total_buy - total_sell:,.2f}""")
                else:
                    st.success("✅ Portfolio is already balanced to the target allocation!")
            else:
                st.error(f"❌ Target allocation totals {tgt_total}%. Must equal 100%")

        # 5) Risk Analysis -----------------------------------------------------
        st.markdown("## ⚠️ Risk Analysis")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""<div class="risk-metric"><h4>Sharpe Ratio</h4>
                            <h3>{metrics['sharpe_ratio']:.3f}</h3>
                            <small>Risk-adjusted return</small></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="risk-metric"><h4>Volatility</h4>
                            <h3>{metrics['volatility']*100:.1f}%</h3>
                            <small>Annual price swings</small></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="risk-metric"><h4>Max Drawdown</h4>
                            <h3>{metrics['max_drawdown']*100:.1f}%</h3>
                            <small>Worst peak-to-trough loss</small></div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""<div class="risk-metric"><h4>Value at Risk (95%)</h4>
                            <h3>${metrics['var_dollar']:,.2f}</h3>
                            <small>Daily loss estimate</small></div>""", unsafe_allow_html=True)

        # 6) Performance Analysis ---------------------------------------------
        st.markdown("## 📈 Performance Analysis")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(f"""<div class="performance-metric"><h4>Annualized Return</h4>
                            <h3>{metrics['annualized_return']*100:+.1f}%</h3>
                            <small>Year-over-year growth</small></div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class="performance-metric"><h4>Sortino Ratio</h4>
                            <h3>{metrics['sortino_ratio']:.3f}</h3>
                            <small>Downside risk-adjusted</small></div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""<div class="performance-metric"><h4>Beta vs {benchmark}</h4>
                            <h3>{metrics['beta']:.2f}</h3>
                            <small>Market sensitivity</small></div>""", unsafe_allow_html=True)
        with p4:
            if benchmark in data:
                bench_total_return = ((data[benchmark]['Close'].iloc[-1] / data[benchmark]['Close'].iloc[0]) - 1) * 100
                outperf = total_return_pct - bench_total_return
                color = "green" if outperf >= 0 else "red"
            else:
                outperf, color = 0, "gray"
            st.markdown(f"""<div class="performance-metric"><h4>vs {benchmark}</h4>
                            <h3 style="color:{color}">{outperf:+.1f}%</h3>
                            <small>Outperformance</small></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # 7) AI Suggestions ----------------------------------------------------
        sector_allocation = {}
        for sym, wt in portfolio.items():
            sector = ASSET_SECTORS.get(sym, 'Other')
            sector_allocation[sector] = sector_allocation.get(sector, 0) + wt
        if show_suggestions:
            st.markdown("## 🤖 AI Portfolio Recommendations")
            suggestions = suggest_portfolio_improvements(portfolio, metrics, sector_allocation)
            if suggestions:
                for s in suggestions:
                    color = {'High':'#ff4444','Medium':'#ffaa00','Low':'#44aa44'}.get(s['priority'],'#666')
                    st.markdown(f"""
                    <div class="alert-box" style="border-left-color:{color};">
                        <h4>🎯 {s['title']} - {s['priority']} Priority</h4>
                        <p>{s['description']}</p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ Your portfolio looks well-balanced! No major suggestions right now.")

        # 8) Monte Carlo -------------------------------------------------------
        if show_monte_carlo and metrics['portfolio_returns'] is not None and len(metrics['portfolio_returns']) > 10:
            st.markdown("## 🎲 Monte Carlo Projections (1 Year)")
            mc = monte_carlo_simulation(metrics['portfolio_returns'], portfolio_value, 252, 1000)
            if mc:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class="performance-metric"><h4>Expected Value (1 Year)</h4>
                                    <h3>${mc['expected_value']:,.0f}</h3>
                                    <small>Average of 1000 simulations</small></div>""", unsafe_allow_html=True)
                with c2:
                    p10, p50, p90 = mc['percentiles']['10th'], mc['percentiles']['50th'], mc['percentiles']['90th']
                    st.markdown(f"""<div class="performance-metric"><h4>80% Confidence Range</h4>
                                    <h3>${p10:,.0f} - ${p90:,.0f}</h3>
                                    <small>Median: ${p50:,.0f}</small></div>""", unsafe_allow_html=True)
                fig_mc = px.histogram(x=mc['results'], nbins=50,
                                      title="Portfolio Value Distribution (1000 Simulations)",
                                      labels={'x':'Portfolio Value ($)','y':'Frequency'})
                fig_mc.add_vline(x=p10, line_dash="dash", line_color="red", annotation_text="10th %")
                fig_mc.add_vline(x=p50, line_dash="dash", line_color="yellow", annotation_text="Median")
                fig_mc.add_vline(x=p90, line_dash="dash", line_color="green", annotation_text="90th %")
                fig_mc.update_layout(height=400)
                st.plotly_chart(fig_mc, use_container_width=True)

        # 9) Sector Allocation -------------------------------------------------
        st.markdown("## 🏢 Sector Allocation")
        if sector_allocation:
            boxes = st.columns(len(sector_allocation))
            for i, (sec, wt) in enumerate(sector_allocation.items()):
                with boxes[i]:
                    st.markdown(f"""<div class="sector-box"><div>{sec}</div>
                                    <div style="font-size:1.5em;">{wt}%</div></div>""",
                                unsafe_allow_html=True)

        st.markdown("---")

        # 10) Charts -----------------------------------------------------------
        a1, a2 = st.columns(2)
        with a1:
            st.subheader("Portfolio Allocation")
            fig_pie = px.pie(values=list(portfolio.values()),
                             names=[f"{s}<br>{ASSET_SECTORS.get(s,'Other')}" for s in portfolio.keys()],
                             title="Asset Allocation by Sector",
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=500, showlegend=True, title_x=0.5)
            st.plotly_chart(fig_pie, use_container_width=True)
        with a2:
            st.subheader("Risk-Return Analysis")
            rr = []
            for sym in portfolio.keys():
                if sym in data:
                    rets = data[sym]['Close'].pct_change().dropna()
                    rr.append({'Asset':sym,
                               'Annual_Return': (rets.mean()*252)*100,
                               'Volatility': (rets.std(ddof=0)*np.sqrt(252))*100,
                               'Weight': portfolio[sym]})
            if rr:
                df_rr = pd.DataFrame(rr)
                fig_scatter = px.scatter(df_rr, x='Volatility', y='Annual_Return', size='Weight', text='Asset',
                                         title="Risk vs Return Profile",
                                         labels={'Volatility':'Annual Volatility (%)', 'Annual_Return':'Annual Return (%)'})
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Portfolio vs Benchmark Performance")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=metrics['portfolio_values'].index,
                                     y=metrics['portfolio_values'].values,
                                     mode='lines', name='Your Portfolio',
                                     line=dict(color='#1f77b4', width=3)))
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
        fig_cmp.update_layout(title="Portfolio Performance vs Benchmark", xaxis_title="Date",
                              yaxis_title="Value ($)", height=500, hovermode='x unified')
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Asset Correlation Analysis")
        ret_map = {}
        for sym in portfolio.keys():
            if sym in data:
                ret_map[sym] = data[sym]['Close'].pct_change().fillna(0)
        if len(ret_map) > 1:
            corr = pd.DataFrame(ret_map).corr()
            fig_heatmap = px.imshow(corr, title="Asset Correlation Matrix",
                                    color_continuous_scale="RdBu", aspect="auto")
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.info("💡 Values near +1 move together; near -1 move opposite. Lower correlations improve diversification.")

        # 11) Detailed Positions ----------------------------------------------
        st.subheader("Detailed Position Analysis")
        detailed = []
        # Align start prices nicely
        try:
            aligned_syms = [s for s in portfolio.keys() if s in data and not data[s].empty]
            def _s(df):
                s = df['Close'].copy()
                idx = pd.to_datetime(s.index)
                try: idx = idx.tz_convert(None)
                except Exception:
                    try: idx = idx.tz_localize(None)
                    except Exception: pass
                s.index = idx.normalize()
                s = s[~s.index.duplicated(keep='last')]
                return s
            aligned_prices = pd.concat([_s(data[s]).rename(s) for s in aligned_syms],
                                       axis=1, join='inner').dropna(how='any')
            start_prices_aligned = aligned_prices.iloc[0]
        except Exception:
            start_prices_aligned = pd.Series(dtype=float)

        for sym, wt in portfolio.items():
            if sym in current_prices:
                current_price = current_prices[sym]
                alloc = (wt/100.0) * initial_investment
                if sym in start_prices_aligned.index:
                    sp = float(start_prices_aligned[sym])
                else:
                    sp = float(data[sym]['Close'].iloc[0]) if sym in data else 0.0
                shares = alloc / sp if sp > 0 else 0.0
                position_value = shares * current_price
                asset_return = ((current_price / sp) - 1.0) * 100.0 if sp > 0 else 0.0
                try:
                    asset_vol = (data[sym]['Close'].pct_change().std(ddof=0) * np.sqrt(252)) * 100.0
                except Exception:
                    asset_vol = 0.0
                detailed.append({
                    'Asset': sym, 'Sector': ASSET_SECTORS.get(sym, 'Other'), 'Weight': f"{wt}%",
                    'Position Value': f"${position_value:,.2f}", 'Current Price': f"${current_price:.2f}",
                    'Shares/Units': f"{shares:.4f}", 'Return': f"{asset_return:+.2f}%",
                    'Volatility': f"{asset_vol:.1f}%"
                })
        if detailed:
            st.dataframe(pd.DataFrame(detailed), use_container_width=True, hide_index=True)

        # 12) Export -----------------------------------------------------------
        st.markdown("---")
        st.markdown("## 📊 Export Dashboard Data")
        exp1, exp2 = st.columns(2)
        with exp1:
            if st.button("📄 Export as CSV", use_container_width=True):
                csv_data = create_csv_export(metrics, portfolio, portfolio_value, detailed)
                st.download_button("Download CSV Report", data=csv_data,
                                   file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")
        with exp2:
            if st.button("📋 Export as PDF", use_container_width=True):
                if PDF_AVAILABLE:
                    try:
                        pdf_data = create_pdf_report(metrics, portfolio, portfolio_value, sector_allocation)
                        if pdf_data:
                            st.download_button("Download PDF Report", data=pdf_data,
                                               file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                               mime="application/pdf")
                        else:
                            st.error("Failed to generate PDF report")
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")
                else:
                    st.error("PDF export requires reportlab. Run: pip install reportlab")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            <h4>Methodology & Data Sources</h4>
            <p><strong>Data:</strong> Yahoo Finance | <strong>Risk-free rate:</strong> 2% assumed</p>
            <p><strong>Sharpe:</strong> (Annualized Return − RF) / Annualized Vol | <strong>Sortino:</strong> Downside-only volatility</p>
            <p><strong>VaR (95%):</strong> 5th percentile daily loss (scaled to $ current portfolio) | <strong>Beta:</strong> vs selected benchmark</p>
            <p><strong>Monte Carlo:</strong> 1000 sims using mean/std of daily returns</p>
            <p><strong>Performance Attribution:</strong> Individual asset contribution to total returns</p>
            <br><em>Smart Portfolio Analytics Dashboard</em>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Please ensure your portfolio allocation totals 100% to view the dashboard.")

