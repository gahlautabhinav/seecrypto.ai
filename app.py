import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path

# ========== Configuration ==========
st.set_page_config(
    page_title="🚀 CryptoAI Oracle", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for amazing UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .prediction-price {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0,255,136,0.5);
        margin: 1rem 0;
    }
    
    .crypto-selector {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
    }
    
    .sidebar-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    .loading-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        text-align: center;
        color: #667eea;
    }
    
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        color: white;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .glow-effect {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        to { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
    }
</style>
""", unsafe_allow_html=True)

# ========== Cryptocurrency Configuration ==========
CRYPTO_CONFIG = {
    "bitcoin": {"name": "Bitcoin", "symbol": "BTC", "color": "#f7931a", "icon": "₿"},
    "ethereum": {"name": "Ethereum", "symbol": "ETH", "color": "#627eea", "icon": "Ξ"},
    "cardano": {"name": "Cardano", "symbol": "ADA", "color": "#0033ad", "icon": "₳"},
    "ripple": {"name": "XRP", "symbol": "XRP", "color": "#23292f", "icon": "◉"},
    "solana": {"name": "Solana", "symbol": "SOL", "color": "#9945ff", "icon": "◎"},
    "dogecoin": {"name": "Dogecoin", "symbol": "DOGE", "color": "#c2a633", "icon": "Ð"},
    "litecoin": {"name": "Litecoin", "symbol": "LTC", "color": "#bfbbbb", "icon": "Ł"},
    "chainlink": {"name": "Chainlink", "symbol": "LINK", "color": "#375bd2", "icon": "⬡"},
    "polkadot": {"name": "Polkadot", "symbol": "DOT", "color": "#e6007a", "icon": "●"},
    "uniswap": {"name": "Uniswap", "symbol": "UNI", "color": "#ff007a", "icon": "🦄"},
    "avalanche-2": {"name": "Avalanche", "symbol": "AVAX", "color": "#e84142", "icon": "▲"},
    "stellar": {"name": "Stellar", "symbol": "XLM", "color": "#7d00ff", "icon": "✦"},
    "cosmos": {"name": "Cosmos", "symbol": "ATOM", "color": "#2e3148", "icon": "⚛"},
    "tron": {"name": "TRON", "symbol": "TRX", "color": "#ff060a", "icon": "◈"},
    "vechain": {"name": "VeChain", "symbol": "VET", "color": "#15bdff", "icon": "♦"},
    "the-graph": {"name": "The Graph", "symbol": "GRT", "color": "#6f4cff", "icon": "📊"},
    "aave": {"name": "Aave", "symbol": "AAVE", "color": "#b6509e", "icon": "👻"},
    "optimism": {"name": "Optimism", "symbol": "OP", "color": "#ff0420", "icon": "🔴"},
    "arbitrum": {"name": "Arbitrum", "symbol": "ARB", "color": "#28a0f0", "icon": "🔵"},
    "aptos": {"name": "Aptos", "symbol": "APT", "color": "#000000", "icon": "⬟"},
    "sui": {"name": "Sui", "symbol": "SUI", "color": "#4da2ff", "icon": "💧"},
    "algorand": {"name": "Algorand", "symbol": "ALGO", "color": "#000000", "icon": "△"},
    "hedera-hashgraph": {"name": "Hedera", "symbol": "HBAR", "color": "#000000", "icon": "ℏ"},
    "near": {"name": "NEAR Protocol", "symbol": "NEAR", "color": "#000000", "icon": "◇"},
    "render-token": {"name": "Render", "symbol": "RNDR", "color": "#000000", "icon": "🎬"},
    "mina-protocol": {"name": "Mina", "symbol": "MINA", "color": "#e18b44", "icon": "○"},
    "fantom": {"name": "Fantom", "symbol": "FTM", "color": "#1969ff", "icon": "👻"},
    "theta-token": {"name": "Theta Network", "symbol": "THETA", "color": "#2ab8e6", "icon": "θ"},
    "axie-infinity": {"name": "Axie Infinity", "symbol": "AXS", "color": "#0055d4", "icon": "🎮"},
    "flow": {"name": "Flow", "symbol": "FLOW", "color": "#00ef8b", "icon": "🌊"},
    "immutable-x": {"name": "Immutable X", "symbol": "IMX", "color": "#000000", "icon": "⚡"},
    "lido-dao": {"name": "Lido DAO", "symbol": "LDO", "color": "#00a3ff", "icon": "🏛"},
    "gala": {"name": "Gala", "symbol": "GALA", "color": "#000000", "icon": "🎲"},
    "chiliz": {"name": "Chiliz", "symbol": "CHZ", "color": "#cc0000", "icon": "🌶"},
    "oasis-network": {"name": "Oasis Network", "symbol": "ROSE", "color": "#0092f5", "icon": "🌹"},
    "iota": {"name": "IOTA", "symbol": "MIOTA", "color": "#000000", "icon": "ι"},
    "zilliqa": {"name": "Zilliqa", "symbol": "ZIL", "color": "#49c1bf", "icon": "◊"},
    "basic-attention-token": {"name": "Basic Attention Token", "symbol": "BAT", "color": "#ff5000", "icon": "🦇"},
    "enjincoin": {"name": "Enjin Coin", "symbol": "ENJ", "color": "#624dbf", "icon": "⚔"}
}

# ========== Utility Functions ==========
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=600)
def fetch_crypto_data(coin_id, days=200):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        r = requests.get(url, params=params)
        data = r.json()

        prices = data["prices"]
        volumes = data["total_volumes"]

        records = []
        for i in range(len(prices)):
            ts = datetime.utcfromtimestamp(prices[i][0] / 1000)
            close = prices[i][1]
            volume = volumes[i][1]
            records.append({"Date": ts, "Close": close, "Volume": volume})

        df = pd.DataFrame(records)
        df["Open"] = df["Close"].shift(1)
        df["High"] = df["Close"].rolling(2).max()
        df["Low"] = df["Close"].rolling(2).min()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {coin_id}: {str(e)}")
        return None

def prepare_features(df):
    df = df.copy()
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['ema12'] = df['Close'].ewm(span=12).mean()
    df['ema26'] = df['Close'].ewm(span=26).mean()
    df['rsi'] = compute_rsi(df['Close'], 14)
    df.dropna(inplace=True)
    return df

def load_model_and_scalers(coin_id):
    try:
        model_path = f"models/lstm_close_model_{coin_id}.h5"
        scaler_x_path = f"models/scaler_X_{coin_id}.save"
        scaler_y_path = f"models/scaler_y_close_{coin_id}.save"
        
        if not all(os.path.exists(path) for path in [model_path, scaler_x_path, scaler_y_path]):
            return None, None, None
            
        model = load_model(model_path, compile=False)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model for {coin_id}: {str(e)}")
        return None, None, None

def create_prediction_chart(df, coin_id, predicted_price):
    config = CRYPTO_CONFIG[coin_id]
    
    # Use last 30 days
    past_30 = df[-30:]
    dates = past_30['Date'].tolist()
    prices = past_30['Close'].tolist()
    
    # Add tomorrow's prediction
    tomorrow = dates[-1] + timedelta(days=1)
    dates.append(tomorrow)
    prices.append(predicted_price)
    
    # Confidence interval
    confidence_margin = 0.05 * predicted_price  # 5% confidence interval
    lower_bound = predicted_price - confidence_margin
    upper_bound = predicted_price + confidence_margin
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=dates[:-1],
        y=prices[:-1],
        mode='lines+markers',
        name='Historical Price',
        line=dict(color=config['color'], width=3),
        marker=dict(size=6),
        hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # Predicted price
    fig.add_trace(go.Scatter(
        x=[tomorrow],
        y=[predicted_price],
        mode='markers+text',
        name='AI Prediction',
        marker=dict(size=20, color='#00ff88', symbol='star', line=dict(color='white', width=2)),
        text=[f"${predicted_price:,.2f}"],
        textposition="top center",
        textfont=dict(size=14, family='Orbitron'),
        hovertemplate='<b>Predicted: $%{y:,.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=[tomorrow, tomorrow, tomorrow],
        y=[lower_bound, predicted_price, upper_bound],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,255,136,0.2)',
        line=dict(color='rgba(0,255,136,0.5)', dash='dash'),
        name='Confidence Band (±5%)',
        hovertemplate='<b>Range: $%{y:,.2f}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"📈 {config['name']} ({config['symbol']}) - 30-Day Trend + AI Prediction",
            font=dict(family='Space Grotesk', size=20, color='white')
        ),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        ),
        hovermode="x unified",
        height=500
    )
    
    return fig

def create_feature_chart(features_df, coin_id):
    config = CRYPTO_CONFIG[coin_id]
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=['Open', 'High', 'Low', 'Volume', 'MA7', 'MA21', 'EMA12', 'EMA26', 'RSI'],
        specs=[[{"secondary_y": False}]*3 for _ in range(3)]
    )
    
    features = ['Open', 'High', 'Low', 'Volume', 'ma7', 'ma21', 'ema12', 'ema26', 'rsi']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8', '#f7dc6f', '#bb8fce']
    
    row_col_pairs = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
    
    for i, (feature, color, (row, col)) in enumerate(zip(features, colors, row_col_pairs)):
        value = features_df[feature].iloc[0]
        
        fig.add_trace(
            go.Bar(x=[feature], y=[value], marker_color=color, name=feature, showlegend=False),
            row=row, col=col
        )
        
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        
        # Format the y-axis labels properly
        if feature == 'Volume':
            if value >= 1e9:
                formatted_value = f"${value/1e9:.1f}B"
            elif value >= 1e6:
                formatted_value = f"${value/1e6:.1f}M"
            elif value >= 1e3:
                formatted_value = f"${value/1e3:.1f}K"
            else:
                formatted_value = f"${value:,.0f}"
        elif feature == 'rsi':
            formatted_value = f"{value:.1f}"
        else:
            formatted_value = f"${value:,.2f}"
            
        fig.update_yaxes(title_text=formatted_value, row=row, col=col)
    
    fig.update_layout(
        title=dict(
            text=f"📊 {config['name']} - Current Technical Features",
            font=dict(family='Space Grotesk', size=18, color='white')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600
    )
    
    return fig

# ========== Main App ==========
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 CryptoAI Oracle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced LSTM Neural Network for Cryptocurrency Price Prediction</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">🎯 Select Cryptocurrency</h2>', unsafe_allow_html=True)
        
        # Crypto selector with improved styling
        crypto_options = {f"{config['icon']} {config['name']} ({config['symbol']})": coin_id 
                         for coin_id, config in CRYPTO_CONFIG.items()}
        
        selected_crypto_display = st.selectbox(
            "Choose your crypto:",
            options=list(crypto_options.keys()),
            index=0,
            help="Select a cryptocurrency to get AI-powered price predictions"
        )
        
        selected_crypto = crypto_options[selected_crypto_display]
        config = CRYPTO_CONFIG[selected_crypto]
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🤖 Model Information")
        st.info(f"""
        **Architecture:** LSTM Neural Network  
        **Features:** 9 technical indicators  
        **Training:** 200+ days of historical data  
        **Update:** Real-time via CoinGecko API
        """)
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Loading animation
        with st.spinner("🔮 Consulting the AI Oracle..."):
            # Load model and scalers
            model, scaler_X, scaler_y = load_model_and_scalers(selected_crypto)
            
            if model is None:
                st.markdown(f'''
                <div class="warning-box">
                    ⚠️ Model not found for {config['name']}. Please ensure the model is trained and saved in the models/ directory.
                </div>
                ''', unsafe_allow_html=True)
                return
            
            # Fetch data
            df = fetch_crypto_data(selected_crypto)
            
            if df is None:
                st.error("Failed to fetch cryptocurrency data. Please try again later.")
                return
            
            # Prepare features
            df_features = prepare_features(df)
            latest = df_features[-1:]
            
            features = ['Open', 'High', 'Low', 'Volume', 'ma7', 'ma21', 'ema12', 'ema26', 'rsi']
            X = latest[features].values
            X_scaled = scaler_X.transform(X).reshape(1, 1, -1)
            
            # Make prediction
            pred_scaled = model.predict(X_scaled, verbose=0)
            predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            # Current price for comparison
            current_price = latest['Close'].iloc[0]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
    
    # Prediction display
    st.markdown(f'''
    <div class="prediction-card glow-effect">
        <h2 style="color: white; font-family: 'Space Grotesk', sans-serif; margin-bottom: 1rem;">
            {config['icon']} {config['name']} ({config['symbol']})
        </h2>
        <div class="prediction-price">${predicted_price:,.2f}</div>
        <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">
            Next-Day Prediction
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <div>
                <div style="color: {'#00ff88' if price_change >= 0 else '#ff4757'}; font-size: 1.5rem; font-weight: 700;">
                    {'↗' if price_change >= 0 else '↘'} ${abs(price_change):,.2f}
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Change</div>
            </div>
            <div>
                <div style="color: {'#00ff88' if price_change_pct >= 0 else '#ff4757'}; font-size: 1.5rem; font-weight: 700;">
                    {'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}%
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Percentage</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${current_price:,.2f}</div>
            <div class="metric-label">Current Price</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{latest['rsi'].iloc[0]:.1f}</div>
            <div class="metric-label">RSI</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        # Format volume in millions/billions for better display
        volume = latest['Volume'].iloc[0]
        if volume >= 1e9:
            volume_display = f"${volume/1e9:.1f}B"
        elif volume >= 1e6:
            volume_display = f"${volume/1e6:.1f}M"
        elif volume >= 1e3:
            volume_display = f"${volume/1e3:.1f}K"
        else:
            volume_display = f"${volume:,.0f}"
            
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{volume_display}</div>
            <div class="metric-label">Volume (24h)</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        ma_trend = "📈" if latest['ma7'].iloc[0] > latest['ma21'].iloc[0] else "📉"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{ma_trend}</div>
            <div class="metric-label">MA Trend</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    
    # Price prediction chart
    fig_price = create_prediction_chart(df_features, selected_crypto, predicted_price)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Feature analysis
    with st.expander("📊 Technical Analysis Deep Dive", expanded=True):
        fig_features = create_feature_chart(latest[features], selected_crypto)
        st.plotly_chart(fig_features, use_container_width=True)
        
        # Feature insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Price Indicators")
            st.markdown(f"""
            - **Moving Average 7:** ${latest['ma7'].iloc[0]:,.2f}
            - **Moving Average 21:** ${latest['ma21'].iloc[0]:,.2f}
            - **EMA 12:** ${latest['ema12'].iloc[0]:,.2f}
            - **EMA 26:** ${latest['ema26'].iloc[0]:,.2f}
            """)
        
        with col2:
            st.markdown("### 🎯 Technical Signals")
            rsi_signal = "Overbought" if latest['rsi'].iloc[0] > 70 else "Oversold" if latest['rsi'].iloc[0] < 30 else "Neutral"
            ma_signal = "Bullish" if latest['ma7'].iloc[0] > latest['ma21'].iloc[0] else "Bearish"
            
            st.markdown(f"""
            - **RSI Signal:** {rsi_signal} ({latest['rsi'].iloc[0]:.1f})
            - **MA Signal:** {ma_signal}
            - **Volume:** {volume_display}
            - **Volatility:** {((latest['High'].iloc[0] - latest['Low'].iloc[0]) / latest['Close'].iloc[0] * 100):.2f}%
            """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem; padding: 1rem; 
                background: rgba(255,255,255,0.05); border-radius: 10px; margin: 2rem 0;'>
        ⚠️ <strong>Disclaimer:</strong> This AI prediction is for educational purposes only. 
        Cryptocurrency investments are highly volatile and risky. Always do your own research 
        and consult with financial advisors before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()