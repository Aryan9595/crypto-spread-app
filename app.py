import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Page Config ---
st.set_page_config(page_title="Crypto Spread Dashboard", layout="wide")
st.title("📈 Crypto Spread Classifier & Strategy Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Hyperparameters")
    n_est = st.slider("n_estimators", 10, 500, value=100, step=10)
    max_d = st.select_slider("max_depth", options=[None, 5, 10, 20, 30], value=None)
    st.markdown("---")
    st.header("📂 Upload CSVs")
    btc_file = st.file_uploader("Upload BTC CSV", type="csv")
    eth_file = st.file_uploader("Upload ETH CSV", type="csv")

# --- Main Logic ---
if btc_file and eth_file:
    # Load & normalize
    btc = pd.read_csv(btc_file).rename(str.lower, axis=1)
    eth = pd.read_csv(eth_file).rename(str.lower, axis=1)

    # Parse dates
    btc['date'] = pd.to_datetime(btc['date'], format="%Y-%m-%d", errors='coerce')
    eth['date'] = pd.to_datetime(eth['date'], format="%Y-%m-%d", errors='coerce')
    btc.dropna(subset=['date'], inplace=True)
    eth.dropna(subset=['date'], inplace=True)

    # Merge
    df = pd.merge(btc, eth, on='date', suffixes=('_btc','_eth')).sort_values('date').reset_index(drop=True)

    # Signals & features
    df['spread'] = df['close_btc'] - df['close_eth']
    df['signal'] = (df['spread'].shift(-1) > df['spread']).astype(int)
    df.dropna(subset=['signal'], inplace=True)

    X = df[['close_btc','close_eth','spread']]
    y = df['signal']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model training
    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Candlestick & Spread", 
        "🔍 Model Metrics",
        "📊 Backtest & KPIs",
        "📝 Custom Prediction"
    ])

    # --- Tab 1: Candlestick & Spread ---
    with tab1:
        st.subheader("BTC Price (Interactive Candlestick)")
        fig1 = go.Figure(data=[go.Candlestick(
            x=btc['date'], open=btc['open'], high=btc['high'],
            low=btc['low'], close=btc['close']
        )])
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Spread vs 30-Day MA")
        spread_df = df[['date','spread']].copy()
        spread_df['ma30'] = spread_df['spread'].rolling(30).mean()
        fig2 = px.line(
            spread_df, x='date', y=['spread','ma30'],
            labels={'value':'Spread','date':'Date'},
            title="Daily Spread & 30-Day Moving Average"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 2: Model Metrics ---
    with tab2:
        st.subheader("Model Performance Metrics")
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
        cols[4].metric("MSE (CV)", f"{mse:.4f}")

    # --- Tab 3: Backtesting & KPIs ---
    with tab3:
        st.subheader("Strategy vs Buy-and-Hold Equity Curve")
        df['btc_ret']   = df['close_btc'].pct_change().fillna(0)
        df['strat_ret'] = df['btc_ret'] * (df['signal'].shift(1).fillna(0)*2 - 1)
        df['cum_btc']   = (1 + df['btc_ret']).cumprod()
        df['cum_strat'] = (1 + df['strat_ret']).cumprod()

        # Compute KPIs
        max_dd = (df['cum_strat'] / df['cum_strat'].cummax() - 1).min()
        annual_ret = (df['cum_strat'].iloc[-1] ** (252/len(df)) - 1)
        sharpe   = annual_ret / df['strat_ret'].std() * np.sqrt(252)

        st.metric("Max Drawdown", f"{max_dd:.2%}")
        st.metric("Annualized Return", f"{annual_ret:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        ec_df = df[['date','cum_btc','cum_strat']].melt(
            'date', var_name='strategy', value_name='equity'
        )
        fig3 = px.line(
            ec_df, x='date', y='equity', color='strategy',
            labels={'equity':'Cumulative Returns','strategy':'Strategy'},
            title="Equity Curve Comparison"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tab 4: Custom Prediction ---
    with tab4:
        st.subheader("Predict Buy/Sell on Custom Inputs")
        with st.form("predict_form"):
            d         = st.date_input("Date", value=df['date'].iloc[-1])
            btc_price = st.number_input("BTC Close Price", float(df['close_btc'].iloc[-1]))
            eth_price = st.number_input("ETH Close Price", float(df['close_eth'].iloc[-1]))
            submitted = st.form_submit_button("Predict")

            if submitted:
                spread = btc_price - eth_price
                pred   = model.predict([[btc_price, eth_price, spread]])[0]
                action = "Buy" if pred==1 else "Sell"
                st.markdown(f"## Recommendation: **{action}** on {d}")
                st.write(f"- BTC Close: ${btc_price:.2f}")
                st.write(f"- ETH Close: ${eth_price:.2f}")
                st.write(f"- Spread: ${spread:.2f}")

else:
    st.warning("⚠️ Please upload both BTC and ETH CSV files to proceed.")
