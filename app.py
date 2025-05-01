import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

st.set_page_config(page_title="Crypto Spread Dashboard", layout="wide")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    # Hyperparameter tuning
    n_est = st.slider("n_estimators", 10, 500, value=100, step=10)
    max_d = st.select_slider("max_depth", options=[None, 5, 10, 20, 30], value=None)
    retrain = st.button("🔄 Refresh & Retrain")

    st.markdown("---")
    st.header("📂 Upload CSVs")
    btc_file = st.file_uploader("BTC CSV", type="csv")
    eth_file = st.file_uploader("ETH CSV", type="csv")

# --- Load & Preprocess ---
if btc_file and eth_file:
    btc = pd.read_csv(btc_file).rename(str.lower, axis=1)
    eth = pd.read_csv(eth_file).rename(str.lower, axis=1)

    # parse dates
    btc['date'] = pd.to_datetime(btc['date'], format="%Y-%m-%d", errors='coerce')
    eth['date'] = pd.to_datetime(eth['date'], format="%Y-%m-%d", errors='coerce')
    btc.dropna(subset=['date'], inplace=True)
    eth.dropna(subset=['date'], inplace=True)

    # merge
    df = pd.merge(btc, eth, on='date', suffixes=('_btc','_eth')).sort_values('date')
    df['spread'] = df['close_btc'] - df['close_eth']
    df['signal'] = (df['spread'].shift(-1) > df['spread']).astype(int)
    df.dropna(subset=['signal'], inplace=True)

    # features & target
    X = df[['close_btc','close_eth','spread']]
    y = df['signal']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # retrain on demand
    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Tabs Layout ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Candlestick & Spread", 
        "🤖 Model Metrics",
        "📊 Equity Curve",
        "📅 Custom Prediction"
    ])

    # --- Tab 1: Candlestick & Spread ---
    with tab1:
        st.subheader("BTC Price (Candlestick)")
        fig1 = go.Figure(data=[go.Candlestick(
            x=btc['date'], open=btc['open'], high=btc['high'],
            low=btc['low'], close=btc['close']
        )])
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Spread vs 30-day MA")
        spread_df = df[['date','spread']].copy()
        spread_df['ma30'] = spread_df['spread'].rolling(30).mean()
        fig2 = px.line(
            spread_df, x='date', y=['spread','ma30'],
            labels={'value':'Spread','date':'Date'},
            title="Daily Spread & 30-day MA"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 2: Model Metrics ---
    with tab2:
        st.subheader("🔍 Performance Metrics")
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        mse  = -cross_val_score(
            model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        ).mean()

        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{acc:.2%}")
        cols[1].metric("Precision", f"{prec:.2%}")
        cols[2].metric("Recall", f"{rec:.2%}")
        cols[3].metric("F1 Score", f"{f1:.2%}")
        cols[4].metric("MSE", f"{mse:.4f}")

    # --- Tab 3: Backtesting Equity Curve ---
    with tab3:
        st.subheader("🔄 Strategy vs Buy-and-Hold")
        # calculate returns
        df['btc_ret'] = df['close_btc'].pct_change().fillna(0)
        df['strat_ret'] = df['btc_ret'] * (df['signal'].shift(1).fillna(0)*2 - 1)
        df['cum_btc'] = (1 + df['btc_ret']).cumprod()
        df['cum_strat'] = (1 + df['strat_ret']).cumprod()

        ec_df = df[['date','cum_btc','cum_strat']].melt('date', var_name='strategy', value_name='equity')
        fig3 = px.line(ec_df, x='date', y='equity', color='strategy', 
                       labels={'equity':'Cumulative Returns','strategy':'Legend'},
                       title="Equity Curve")
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tab 4: Custom Prediction Form ---
    with tab4:
        st.subheader("📅 Predict on Custom Inputs")
        with st.form("predict_form"):
            d = st.date_input("Date", value=df['date'].iloc[-1])
            btc_price = st.number_input("BTC Close Price", float(df['close_btc'].iloc[-1]))
            eth_price = st.number_input("ETH Close Price", float(df['close_eth'].iloc[-1]))
            submitted = st.form_submit_button("Predict")
            if submitted:
                spread = btc_price - eth_price
                pred = model.predict([[btc_price, eth_price, spread]])[0]
                action = "Buy" if pred==1 else "Sell"
                st.markdown(f"## Model recommends: **{action}** on {d}")

else:
    st.warning("Upload both BTC and ETH CSV files to proceed.")
