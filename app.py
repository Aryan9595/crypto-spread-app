import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# App title and layout
st.set_page_config(page_title="Crypto Spread Classifier", layout="wide")
st.title("📊 Crypto Spread Classifier Dashboard")
st.markdown("This app predicts BTC-ETH spread movement using a Random Forest model.")

# File upload section
st.sidebar.header("Upload Datasets")
btc_file = st.sidebar.file_uploader("Upload BTC CSV", type=["csv"])
eth_file = st.sidebar.file_uploader("Upload ETH CSV", type=["csv"])

if btc_file and eth_file:
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)

    # Ensure consistent datetime format
    btc["date"] = pd.to_datetime(btc["date"], format="%Y-%m-%d", errors="coerce")
    eth["date"] = pd.to_datetime(eth["date"], format="%Y-%m-%d", errors="coerce")
    btc.dropna(subset=["date"], inplace=True)
    eth.dropna(subset=["date"], inplace=True)

    # Merge datasets on date
    df = pd.merge(btc, eth, on="date", suffixes=('_btc', '_eth'))
    df["spread"] = df["close_btc"] - df["close_eth"]
    df["spread_direction"] = (df["spread"].diff().shift(-1) > 0).astype(int)

    features = [
        "close_btc", "volume_btc", "high_btc", "low_btc", "open_btc",
        "close_eth", "volume_eth", "high_eth", "low_eth", "open_eth"
    ]
    df = df.dropna()
    X = df[features]
    y = df["spread_direction"]

    # Split and preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display metrics
    st.subheader("📈 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

    mse = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    st.metric("MSE (CV)", f"{mse:.4f}")

    # Training progress plot
    st.subheader("📊 Training Accuracy Over Time")
    train_acc, test_acc, steps = [], [], []
    for i in range(10, len(X_train), 10):
        model.fit(X_train[:i], y_train[:i])
        train_acc.append(accuracy_score(y_train[:i], model.predict(X_train[:i])))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))
        steps.append(i)

    fig, ax = plt.subplots()
    ax.plot(steps, train_acc, label="Train Accuracy")
    ax.plot(steps, test_acc, label="Test Accuracy")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Show raw data
    with st.expander("📂 View Merged Data"):
        st.dataframe(df.head(20))

else:
    st.warning("⚠️ Please upload both BTC and ETH CSV files to proceed.")
