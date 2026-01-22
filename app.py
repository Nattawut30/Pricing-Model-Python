"""

Professional Options Pricing Calculator

Created By: Nattawut Boonnoon
GitHub: https://github.com/Nattawut30
Linkedin: www.linkedin.com/in/nattawut-bn
Email: nattawut.boonnoon@hotmail.com
Phone: (+66) 92 271 6680
Locations: Bangkok, Thailand

A Black-Scholes calculator combining simplicity with powerful features.
Perfect for education AND real-world trading analysis.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

# Import my pricing model right awayyyy
from pricing_model import (
    OptionsPricingModel, 
    calculate_implied_volatility,
    calculate_historical_volatility,
    quick_price
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Options Pricing Calculator Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for coolness styling
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success/Error boxes */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALIZATION
# ============================================================================

if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = []

# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes ugh!
def fetch_stock_data(ticker, period='1mo'):
    """Fetch real-time stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None
        
        current_price = hist['Close'].iloc[-1]
        hist_vol = calculate_historical_volatility(hist['Close'].values)
        
        return current_price, hist_vol
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None


def create_price_surface_heatmap(S, K, T, r, sigma_range, S_range):
    """Create 3D price surface visualization"""
    
    call_prices = np.zeros((len(sigma_range), len(S_range)))
    put_prices = np.zeros((len(sigma_range), len(S_range)))
    
    for i, sigma in enumerate(sigma_range):
        for j, stock_price in enumerate(S_range):
            try:
                model = OptionsPricingModel(stock_price, K, T, r, sigma)
                call_prices[i, j] = model.call_price()
                put_prices[i, j] = model.put_price()
            except:
                call_prices[i, j] = np.nan
                put_prices[i, j] = np.nan
    
    return call_prices, put_prices


def create_strategy_payoff(strategy_name, S_current, K, premium_call, premium_put):
    """Generate payoff diagram for common strategies"""
    
    S_range = np.linspace(S_current * 0.5, S_current * 1.5, 100)
    
    strategies = {
        'Long Call': np.maximum(S_range - K, 0) - premium_call,
        'Long Put': np.maximum(K - S_range, 0) - premium_put,
        'Covered Call': (S_range - S_current) - np.maximum(S_range - K, 0) + premium_call,
        'Protective Put': (S_range - S_current) + np.maximum(K - S_range, 0) - premium_put,
        'Long Straddle': np.maximum(S_range - K, 0) + np.maximum(K - S_range, 0) - (premium_call + premium_put),
        'Short Straddle': (premium_call + premium_put) - np.maximum(S_range - K, 0) - np.maximum(K - S_range, 0),
    }
    
    return S_range, strategies.get(strategy_name, np.zeros_like(S_range))


def export_to_csv(data, filename):
    """Export calculation results to CSV"""
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv

# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">📊 Options Pricing Calculator Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Black-Scholes Analysis | Educational + Real-World Ready</div>', unsafe_allow_html=True)

# SIDEBAR - INPUT PARAMETERS
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Mode selection
calc_mode = st.sidebar.radio(
    "Calculation Mode",
    ["Quick Calculator", "Advanced Analysis", "Strategy Builder"],
    help="Choose your analysis depth"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Input Method")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Entry", "Fetch Live Data"],
    help="Manual for custom scenarios, Live for real market data"
)

# Live data fetching
if input_method == "Fetch Live Data":
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, TSLA, MSFT)"
    ).upper()
    
    if st.sidebar.button("📊 Fetch Real-Time Data"):
        with st.spinner(f"Fetching data for {ticker}..."):
            current_price, hist_vol = fetch_stock_data(ticker)
            
            if current_price:
                st.session_state.fetched_price = current_price
                st.session_state.fetched_vol = hist_vol * 100  # Convert to percentage as feedback requested
                st.sidebar.success(f"✅ Loaded {ticker}: ${current_price:.2f}")
            else:
                st.sidebar.error("❌ Failed to fetch data. Check ticker symbol.")

# Parameters input
st.sidebar.markdown("### 💰 Option Parameters")

# Stock price
if input_method == "Fetch Live Data" and 'fetched_price' in st.session_state:
    stock_price = st.sidebar.number_input(
        "Current Stock Price ($)",
        min_value=0.01,
        max_value=100000.0,
        value=float(st.session_state.fetched_price),
        step=0.01,
        format="%.2f"
    )
else:
    stock_price = st.sidebar.number_input(
        "Current Stock Price ($)",
        min_value=0.01,
        max_value=100000.0,
        value=100.0,
        step=1.0,
        format="%.2f"
    )

strike_price = st.sidebar.number_input(
    "Strike Price ($)",
    min_value=0.01,
    max_value=100000.0,
    value=100.0,
    step=1.0,
    format="%.2f"
)

# Calculate moneyness
moneyness = stock_price / strike_price
if moneyness > 1.05:
    st.sidebar.info("📈 In-The-Money (ITM) for Calls")
elif moneyness < 0.95:
    st.sidebar.info("📉 In-The-Money (ITM) for Puts")
else:
    st.sidebar.info("📊 At-The-Money (ATM)")

st.sidebar.markdown("### 📅 Time & Market")

days_to_expiry = st.sidebar.slider(
    "Days to Expiration",
    min_value=1,
    max_value=730,
    value=30,
    help="Trading days until option expires"
)

# Volatility input
if input_method == "Fetch Live Data" and 'fetched_vol' in st.session_state:
    volatility = st.sidebar.slider(
        "Volatility (% Annual)",
        min_value=1.0,
        max_value=200.0,
        value=float(st.session_state.fetched_vol),
        step=0.5,
        help="Historical volatility pre-loaded"
    )
else:
    volatility = st.sidebar.slider(
        "Volatility (% Annual)",
        min_value=1.0,
        max_value=200.0,
        value=25.0,
        step=0.5,
        help="Standard deviation of returns"
    )

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (% Annual)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
    help="Treasury rate or LIBOR"
)

st.sidebar.markdown("---")

# Calculate button
calculate_btn = st.sidebar.button(
    "🚀 Calculate Options",
    type="primary",
    use_container_width=True
)

# MAIN CALCULATION
# ============================================================================

if calculate_btn or 'results' in st.session_state:
    
    try:
        # Perform calculation
        results = quick_price(
            S=stock_price,
            K=strike_price,
            T_days=days_to_expiry,
            r_pct=risk_free_rate,
            sigma_pct=volatility
        )
        
        st.session_state.results = results
        st.session_state.params = {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'days_to_expiry': days_to_expiry,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate
        }
        
        # Add to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.calculation_history.append({
            'timestamp': timestamp,
            'S': stock_price,
            'K': strike_price,
            'T': days_to_expiry,
            'sigma': volatility,
            'r': risk_free_rate,
            'call': results['call_price'],
            'put': results['put_price']
        })
        
        # Keep only last 10 calculations
        if len(st.session_state.calculation_history) > 10:
            st.session_state.calculation_history.pop(0)
        
    except Exception as e:
        st.error(f"❌ Calculation Error: {str(e)}")
        st.stop()

# DISPLAY RESULTS
# ============================================================================

if 'results' in st.session_state:
    results = st.session_state.results
    params = st.session_state.params
    
    
    # KEY METRICS DISPLAY
    # ========================================================================
    
    st.markdown("## 💰 Option Prices")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📞 Call Option",
            f"${results['call_price']:.2f}",
            delta=f"IV: ${results['call_intrinsic']:.2f}",
            help="European Call Option Price"
        )
    
    with col2:
        st.metric(
            "📉 Put Option",
            f"${results['put_price']:.2f}",
            delta=f"IV: ${results['put_intrinsic']:.2f}",
            help="European Put Option Price"
        )
    
    with col3:
        st.metric(
            "⏱️ Call Time Value",
            f"${results['call_time_value']:.2f}",
            help="Extrinsic value of call"
        )
    
    with col4:
        st.metric(
            "⏱️ Put Time Value",
            f"${results['put_time_value']:.2f}",
            help="Extrinsic value of put"
        )
    
    # Put-Call Parity Check
    parity = results['parity_check']
    if parity['is_valid']:
        st.markdown('<div class="success-box">✅ <b>Put-Call Parity Verified</b> - Prices are mathematically consistent</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-box">⚠️ <b>Parity Check Failed</b> - Difference: ${parity["difference"]:.4f}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
        # TABS FOR DIFFERENT ANALYSES
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Greeks Analysis",
        "📈 Price Sensitivity",
        "🎯 Strategy Builder",
        "🔥 Advanced Analysis",
        "📜 History & Export"
    ])
    
        # TAB 1: GREEKS ANALYSIS
    # ========================================================================
    
    with tab1:
        st.markdown("### 📊 Option Greeks - Risk Metrics")
        
        greeks = results['greeks']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📞 Call Option Greeks")
            
            greeks_data_call = {
                'Greek': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
                'Value': [
                    f"{greeks['call_delta']:.4f}",
                    f"{greeks['call_gamma']:.4f}",
                    f"{greeks['call_theta']:.4f}",
                    f"{greeks['call_vega']:.4f}",
                    f"{greeks['call_rho']:.4f}"
                ],
                'Meaning': [
                    f'${greeks["call_delta"]:.2f} per $1 stock move',
                    f'Delta changes by {greeks["call_gamma"]:.4f} per $1 move',
                    f'Loses ${abs(greeks["call_theta"]):.2f} per day',
                    f'${greeks["call_vega"]:.2f} per 1% volatility change',
                    f'${greeks["call_rho"]:.2f} per 1% rate change'
                ]
            }
            
            df_call = pd.DataFrame(greeks_data_call)
            st.dataframe(df_call, use_container_width=True, hide_index=True)
            
            # Visual gauge for Delta
            fig_delta_call = go.Figure(go.Indicator(
                mode="gauge+number",
                value=greeks['call_delta'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Call Delta"},
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkgreen"},
                      'steps': [
                          {'range': [0, 0.3], 'color': "lightgray"},
                          {'range': [0.3, 0.7], 'color': "gray"},
                          {'range': [0.7, 1], 'color': "darkgray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 0.5}}))
            fig_delta_call.update_layout(height=250)
            st.plotly_chart(fig_delta_call, use_container_width=True)
        
        with col2:
            st.markdown("#### 📉 Put Option Greeks")
            
            greeks_data_put = {
                'Greek': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
                'Value': [
                    f"{greeks['put_delta']:.4f}",
                    f"{greeks['put_gamma']:.4f}",
                    f"{greeks['put_theta']:.4f}",
                    f"{greeks['put_vega']:.4f}",
                    f"{greeks['put_rho']:.4f}"
                ],
                'Meaning': [
                    f'${greeks["put_delta"]:.2f} per $1 stock move',
                    f'Delta changes by {greeks["put_gamma"]:.4f} per $1 move',
                    f'Loses ${abs(greeks["put_theta"]):.2f} per day',
                    f'${greeks["put_vega"]:.2f} per 1% volatility change',
                    f'${greeks["put_rho"]:.2f} per 1% rate change'
                ]
            }
            
            df_put = pd.DataFrame(greeks_data_put)
            st.dataframe(df_put, use_container_width=True, hide_index=True)
            
            # Visual gauge for Delta
            fig_delta_put = go.Figure(go.Indicator(
                mode="gauge+number",
                value=abs(greeks['put_delta']),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Put Delta (Absolute)"},
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkred"},
                      'steps': [
                          {'range': [0, 0.3], 'color': "lightgray"},
                          {'range': [0.3, 0.7], 'color': "gray"},
                          {'range': [0.7, 1], 'color': "darkgray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 0.5}}))
            fig_delta_put.update_layout(height=250)
            st.plotly_chart(fig_delta_put, use_container_width=True)
        
        # Educational section
        with st.expander("📚 Understanding Greeks"):
            st.markdown("""
            ### What Each Greek Tells You:
            
            **Delta (Δ)** - Directional Risk
            - Measures price change per $1 stock move
            - Call: 0 to 1 | Put: -1 to 0
            - ~0.5 = At-the-money
            
            **Gamma (Γ)** - Delta Risk
            - How fast Delta changes
            - Highest at-the-money
            - Important for hedging
            
            **Theta (Θ)** - Time Decay
            - Value lost each day
            - Always negative for long positions
            - Accelerates near expiration
            
            **Vega (ν)** - Volatility Risk
            - Price change per 1% volatility move
            - Higher for longer-dated options
            - Always positive for long positions
            
            **Rho (ρ)** - Interest Rate Risk
            - Price change per 1% rate change
            - Usually smallest Greek
            - More important for long-dated options
            """)
    
    
    # TAB 2: PRICE SENSITIVITY
    # ========================================================================
    
    with tab2:
        st.markdown("### 📈 Price Sensitivity Analysis")
        
        # Generate price ranges
        S_range = np.linspace(stock_price * 0.7, stock_price * 1.3, 100)
        vol_range = np.linspace(max(1, volatility - 20), volatility + 20, 100)
        time_range = np.arange(1, min(days_to_expiry + 1, 180))
        
        # Calculate prices across ranges
        call_prices_S = []
        put_prices_S = []
        call_prices_vol = []
        put_prices_vol = []
        call_prices_time = []
        put_prices_time = []
        
        for s in S_range:
            try:
                model = OptionsPricingModel(s, strike_price, days_to_expiry/365, 
                                          risk_free_rate/100, volatility/100)
                call_prices_S.append(model.call_price())
                put_prices_S.append(model.put_price())
            except:
                call_prices_S.append(np.nan)
                put_prices_S.append(np.nan)
        
        for v in vol_range:
            try:
                model = OptionsPricingModel(stock_price, strike_price, days_to_expiry/365,
                                          risk_free_rate/100, v/100)
                call_prices_vol.append(model.call_price())
                put_prices_vol.append(model.put_price())
            except:
                call_prices_vol.append(np.nan)
                put_prices_vol.append(np.nan)
        
        for t in time_range:
            try:
                model = OptionsPricingModel(stock_price, strike_price, t/365,
                                          risk_free_rate/100, volatility/100)
                call_prices_time.append(model.call_price())
                put_prices_time.append(model.put_price())
            except:
                call_prices_time.append(np.nan)
                put_prices_time.append(np.nan)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Price vs Stock Price', 'Price vs Volatility', 'Price vs Time to Expiry'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Stock price sensitivity
        fig.add_trace(go.Scatter(x=S_range, y=call_prices_S, name='Call', line=dict(color='green', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=S_range, y=put_prices_S, name='Put', line=dict(color='red', width=3)), row=1, col=1)
        fig.add_vline(x=stock_price, line_dash="dash", line_color="blue", annotation_text="Current", row=1, col=1)
        
        # Volatility sensitivity
        fig.add_trace(go.Scatter(x=vol_range, y=call_prices_vol, name='Call', line=dict(color='green', width=3), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=vol_range, y=put_prices_vol, name='Put', line=dict(color='red', width=3), showlegend=False), row=1, col=2)
        fig.add_vline(x=volatility, line_dash="dash", line_color="blue", annotation_text="Current", row=1, col=2)
        
        # Time decay
        fig.add_trace(go.Scatter(x=time_range, y=call_prices_time, name='Call', line=dict(color='green', width=3), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=time_range, y=put_prices_time, name='Put', line=dict(color='red', width=3), showlegend=False), row=1, col=3)
        
        fig.update_xaxes(title_text="Stock Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_xaxes(title_text="Days to Expiry", row=1, col=3)
        fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
        
        fig.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: STRATEGY BUILDER
    # ========================================================================
    
    with tab3:
        st.markdown("### 🎯 Option Strategy Analyzer")
        
        strategy = st.selectbox(
            "Select Strategy",
            ['Long Call', 'Long Put', 'Covered Call', 'Protective Put', 
             'Long Straddle', 'Short Straddle']
        )
        
        # Generate payoff
        S_range, payoff = create_strategy_payoff(
            strategy, 
            stock_price, 
            strike_price,
            results['call_price'],
            results['put_price']
        )
        
        # Plot payoff diagram
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=S_range,
            y=payoff,
            mode='lines',
            name=strategy,
            line=dict(width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vline(x=stock_price, line_dash="dash", line_color="blue", annotation_text="Current Price")
        
        fig.update_layout(
            title=f"{strategy} Payoff Diagram at Expiration",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy explanation
        strategy_info = {
            'Long Call': "Bullish strategy. Unlimited upside, limited downside (premium paid).",
            'Long Put': "Bearish strategy. Profit from stock decline, limited risk.",
            'Covered Call': "Income generation. Sell call against owned stock.",
            'Protective Put': "Downside protection. Insurance for long stock position.",
            'Long Straddle': "Volatility play. Profit from large moves in either direction.",
            'Short Straddle': "Sell volatility. Profit if stock stays near strike."
        }
        
        st.info(f"**Strategy:** {strategy_info[strategy]}")
    
    
    # TAB 4: ADVANCED ANALYSIS
    # ========================================================================
    
    with tab4:
        st.markdown("### 🔥 Advanced Analysis Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌡️ Implied Volatility Calculator")
            
            market_price_input = st.number_input(
                "Observed Market Price ($)",
                min_value=0.01,
                value=results['call_price'],
                step=0.01
            )
            
            option_type_iv = st.radio("Option Type", ['call', 'put'])
            
            if st.button("Calculate Implied Volatility"):
                with st.spinner("Calculating..."):
                    iv = calculate_implied_volatility(
                        market_price_input,
                        stock_price,
                        strike_price,
                        days_to_expiry / 365,
                        risk_free_rate / 100,
                        option_type_iv
                    )
                    
                    if iv:
                        st.success(f"**Implied Volatility:** {iv*100:.2f}%")
                        st.metric("IV vs Current Vol", f"{iv*100:.2f}%", 
                                delta=f"{(iv*100 - volatility):.2f}%")
                    else:
                        st.error("Could not calculate IV. Check inputs.")
        
        with col2:
            st.markdown("#### 📊 Price Surface Heatmap")
            
            if st.button("Generate 3D Surface"):
                with st.spinner("Generating heatmap..."):
                    sigma_range = np.linspace(volatility*0.5/100, volatility*1.5/100, 20)
                    S_range_heat = np.linspace(stock_price*0.8, stock_price*1.2, 20)
                    
                    call_surface, put_surface = create_price_surface_heatmap(
                        stock_price, strike_price, days_to_expiry/365,
                        risk_free_rate/100, sigma_range, S_range_heat
                    )
                    
                    fig = go.Figure(data=[go.Surface(
                        x=S_range_heat,
                        y=sigma_range*100,
                        z=call_surface,
                        colorscale='Viridis'
                    )])
                    
                    fig.update_layout(
                        title='Call Option Price Surface',
                        scene=dict(
                            xaxis_title='Stock Price',
                            yaxis_title='Volatility (%)',
                            zaxis_title='Option Price'
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    
    # TAB 5: HISTORY & EXPORT
    # ========================================================================
    
    with tab5:
        st.markdown("### 📜 Calculation History")
        
        if st.session_state.calculation_history:
            history_df = pd.DataFrame(st.session_state.calculation_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                csv = export_to_csv(st.session_state.calculation_history, 'history.csv')
                st.download_button(
                    "📥 Download History (CSV)",
                    csv,
                    "options_history.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with col2:
                # Export current results
                current_export = {
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Stock_Price': [stock_price],
                    'Strike_Price': [strike_price],
                    'Days_to_Expiry': [days_to_expiry],
                    'Volatility': [volatility],
                    'Risk_Free_Rate': [risk_free_rate],
                    'Call_Price': [results['call_price']],
                    'Put_Price': [results['put_price']],
                    'Call_Delta': [greeks['call_delta']],
                    'Put_Delta': [greeks['put_delta']],
                    'Gamma': [greeks['call_gamma']],
                    'Vega': [greeks['call_vega']],
                    'Call_Theta': [greeks['call_theta']],
                    'Put_Theta': [greeks['put_theta']]
                }
                
                csv_current = export_to_csv(current_export, 'current.csv')
                st.download_button(
                    "📥 Download Current Results",
                    csv_current,
                    "current_calculation.csv",
                    "text/csv",
                    key='download-current'
                )
        else:
            st.info("No calculations in history yet. Run a calculation to see it here.")


# JUST FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <h3>📊 Options Pricing Calculator Pro</h3>
        <p>Built with Streamlit • Black-Scholes Model • European Options</p>
        <p style='font-size: 0.9em;'><b>Educational Tool + Real-World Ready</b></p>
        <p style='font-size: 0.8em; color: #999;'>
            ⚠️ For educational and analytical purposes. Not financial advice. 
            Always consult professionals before trading.
        </p>
        <p style='font-size: 0.8em; color: #999; margin-top: 1rem;'>
            Created with ❤️ | 50% Beginner • 30% Intermediate • 20% Advanced
        </p>
    </div>
""", unsafe_allow_html=True)
